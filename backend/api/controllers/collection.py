"""Is the data still arriving, and is the loop still cycling? Two questions.

**They are not the same question, and conflating them cost thirty-one days.**
`venue_ladder` stopped writing on 2026-09-16 and nobody noticed until
2026-10-17, because every component heartbeat stayed green the whole time: the
recorder's loop ran happily and produced nothing. So this surface reports both,
labelled so they cannot be read as one:

* **freshness** — are rows landing in each dataset. Measured trader-side
  (`core/freshness.py` reads Parquet footer statistics) and published to a JSON
  document on the shared data volume, because this process has no pyarrow and
  cannot import the trader package.
* **liveness** — did each component finish a pass. One stamp file per component
  under `core/heartbeat.py`'s root, read live here because an age written into
  a document an hour ago would be an hour wrong.

**Nothing here substitutes.** An absent status document is not "everything is
fine", a missing stamp is not "zero seconds ago", and a dataset nobody has a
cadence for is not "fresh". Each comes back null with a reason, which is the
standing rule on this API and is doubly load-bearing on a health surface: the
failure being designed against is a dead feed that renders as a healthy one.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: How old the trader's published measurement may be before the datasets it
#: describes stop being evidence about now. The live loop republishes hourly
#: from `store_sync`, so two missed passes — the same tolerance
#: `core.heartbeat` gives that component.
STATUS_MAX_AGE_SECONDS = 7200.0


def _data_dir() -> Path:
    """The trader's `data/` directory as this container sees it.

    Compose bind-mounts it read-only at `/app/trader-data`. The fallback is the
    repo layout, so the surface also works for `uvicorn app:app` run from a
    checkout — which is how the tests and the dev loop run it.
    """
    configured = os.getenv('TRADER_DATA_DIR')
    if configured:
        return Path(configured)
    # controllers/ -> api/ -> backend/, then the trader beside this API.
    return Path(__file__).resolve().parents[2] / 'trader' / 'data'


def _status_path() -> Path:
    override = os.getenv('COLLECTION_STATUS_PATH')
    return Path(override) if override else _data_dir() / 'status' / 'collection.json'


def _heartbeat_dir() -> Path:
    override = os.getenv('HEARTBEAT_ROOT')
    return Path(override) if override else _data_dir() / 'alive'


def _read_status() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The published document, or None and why not."""
    path = _status_path()
    try:
        raw = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None, (
            f'the trader has not published a freshness measurement at {path} — '
            'it is written hourly by the live loop\'s store_sync component, or '
            'on demand by `python -m scripts.check_freshness --publish`')
    except OSError as exc:
        return None, f'could not read {path}: {exc}'
    try:
        document = json.loads(raw)
    except ValueError as exc:
        return None, f'{path} is not readable JSON: {exc}'
    if not isinstance(document, dict):
        return None, f'{path} does not hold a status document'
    if document.get('schema') != 1:
        # Refused rather than best-effort parsed. A document whose shape this
        # does not understand has fields it would read as absent, and absent
        # renders as "no expectation" — the one direction a health surface
        # must not be wrong in.
        return None, (f'{path} carries schema {document.get("schema")!r}, '
                      'which this API does not understand')
    return document, None


def _age_of(stamp: Optional[str], now: datetime) -> Optional[float]:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now - parsed).total_seconds()


def dataset_freshness(*, now: Optional[datetime] = None) -> dict[str, Any]:
    """Per-dataset newest row, its age against THIS clock, and the limit.

    The age is recomputed from `newest_event_time` rather than served from the
    document's own `age_seconds`, which was correct only at the instant it was
    written. What the document's age does bound is how much the reader is
    allowed to conclude: rows may have arrived since it was taken, so a stale
    document makes every age an upper bound and says so.
    """
    now = now or datetime.now(timezone.utc)
    document, reason = _read_status()
    if document is None:
        return {'available': False, 'reason': reason, 'measured_at': None,
                'measurement_age_seconds': None, 'measurement_stale': None,
                'datasets': []}

    measured_at = document.get('generated_at')
    measurement_age = _age_of(measured_at, now)
    measurement_stale = (None if measurement_age is None
                         else measurement_age > STATUS_MAX_AGE_SECONDS)

    rows = []
    for row in document.get('datasets') or []:
        if not isinstance(row, dict) or not row.get('dataset'):
            continue
        limit = row.get('limit_seconds')
        newest = row.get('newest_event_time')
        age = _age_of(newest, now)
        if age is None:
            # Two different absences, and they must not share a reason: the
            # dataset has no rows, or it has rows and this process could not
            # read the timestamp. The second is a bug here, not a dead feed.
            rows.append({
                'dataset': row['dataset'],
                'newest_event_time': newest,
                'age_seconds': None, 'limit_seconds': limit, 'stale': None,
                'reason': (row.get('reason') or 'no rows found in the store')
                if not newest else
                f'could not read {newest!r} as a timestamp',
            })
            continue
        rows.append({
            'dataset': row['dataset'],
            'newest_event_time': newest,
            'age_seconds': age,
            'limit_seconds': limit,
            # No limit means no continuous producer, so there is no verdict to
            # give. `None`, never False.
            'stale': None if limit is None else age > limit,
            'reason': (None if limit is not None else
                       'no continuous producer, so no age is expected'),
        })
    rows.sort(key=lambda r: r['dataset'])
    return {
        'available': True, 'reason': None,
        'measured_at': measured_at,
        'measurement_age_seconds': measurement_age,
        'measurement_stale': measurement_stale,
        'datasets': rows,
    }


def component_liveness(*, now: Optional[float] = None) -> dict[str, Any]:
    """Per-component seconds since its loop last finished a pass.

    Read from the stamp files directly: they are one float each and always
    current, where the published document's copy would be as old as the
    document. The LIMITS come from the document, because they are
    `core.heartbeat.MAX_AGE` and this process cannot import it — copying that
    dict here is exactly the two-definitions-of-one-number defect this
    repository keeps paying for. Without the document the ages are still
    measured and the verdict is null with a reason, which is the honest split:
    the age is a measurement, "is that too old" is a judgement needing a number
    nobody published.
    """
    now = time.time() if now is None else now
    document, document_reason = _read_status()
    limits = (document or {}).get('component_limits') or {}

    directory = _heartbeat_dir()
    try:
        seen = sorted(p.name for p in directory.iterdir() if p.is_file()
                      and not p.name.startswith('.'))
    except OSError as exc:
        return {'available': False, 'components': [],
                'reason': (f'no component heartbeats at {directory}: {exc}. '
                           'The live loop writes one stamp file per component '
                           'there; an unreadable directory is not evidence '
                           'that the loop is healthy.')}

    rows = []
    # Everything with a stamp, plus everything the trader declared and has
    # never stamped. "Not started yet" and "died before its first cycle" look
    # identical from outside, and the second is the one worth catching.
    for name in sorted(set(seen) | set(limits)):
        limit = limits.get(name)
        age: Optional[float] = None
        reason: Optional[str] = None
        try:
            age = max(0.0, now - float((directory / name).read_text().strip()))
        except (OSError, ValueError):
            reason = 'never reported a completed cycle'
        if age is not None and limit is None:
            reason = ('no limit published — run the live loop so it writes a '
                      'status document, or the staleness verdict is a guess')
        rows.append({
            'component': name,
            'age_seconds': age,
            'limit_seconds': limit,
            'stale': None if (age is None or limit is None) else age > limit,
            'reason': reason,
        })
    return {
        'available': True,
        'components': rows,
        'reason': (None if document is not None else
                   f'component limits are unavailable: {document_reason}'),
    }


def collection_health() -> dict[str, Any]:
    """Both halves, in one request, never merged into one verdict."""
    return {
        'freshness': dataset_freshness(),
        'liveness': component_liveness(),
    }
