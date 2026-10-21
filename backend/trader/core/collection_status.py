"""The freshness measurement, written where a process that cannot import us
can read it.

`core.freshness` answers "is each dataset still receiving rows" and
`core.heartbeat` answers "is each component still cycling" -- two different
questions, and the gap between them cost thirty-one days of `venue_ladder`.
Both answers lived only in the trader's log, which means a dead feed was
something only somebody reading logs would notice. Nobody was.

**The API cannot import the trader.** It is a separate image with a separate
requirements file and no pyarrow, and `newest_event_time` is a Parquet footer
read over the whole store. So the measurement stays here, on the side that can
make it, and is published as one small JSON document the dashboard's API reads
off the shared data volume.

Two things it deliberately does NOT publish:

* **A staleness verdict with no timestamp behind it.** Every row carries
  `newest_event_time` as an absolute instant, so the reader computes the age
  against its own clock rather than trusting an age frozen at write time. A
  document written an hour ago still yields a correct age; what it cannot do is
  know about rows that arrived since, which is why `generated_at` is published
  beside it and the reader is expected to say how old the measurement is.
* **Heartbeat ages.** They would be wrong within seconds of being written. Only
  the LIMITS travel in this file; the stamps themselves are one small file per
  component under `core.heartbeat.ROOT`, which any process sharing the volume
  can read directly and cheaply without importing anything.

Nothing here raises. A status writer that can take down the loop it reports on
is worse than no status writer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core import freshness, heartbeat

logger = logging.getLogger('collection-status')

#: Bumped when the shape changes, so a reader can refuse a document it does not
#: understand rather than reporting its absent fields as healthy.
SCHEMA = 1

#: Where the document lands. Under `data/`, for the same reason
#: `core.heartbeat.ROOT` is: `logs/` is a named volume the container writes as
#: root, while `data/` is the bind mount every side already shares.
DEFAULT_PATH = Path(
    os.getenv('COLLECTION_STATUS_PATH')
    or Path(__file__).resolve().parents[1] / 'data' / 'status' / 'collection.json')


def snapshot(store_root: str, *, datasets=None,
             now: Optional[datetime] = None) -> dict[str, Any]:
    """Measure every dataset's newest row. One pass over the footers.

    `age_seconds` is included for a human reading the file by hand; a reader
    serving it onward should recompute from `newest_event_time`, which does not
    rot.
    """
    now = now or datetime.now(timezone.utc)
    wanted = list(datasets) if datasets else list(freshness.EXPECTED_MAX_AGE)

    rows: list[dict[str, Any]] = []
    for dataset in wanted:
        limit = freshness.EXPECTED_MAX_AGE.get(dataset)
        seen = freshness.newest_event_time(store_root, dataset)
        if seen is None:
            rows.append({
                'dataset': dataset, 'newest_event_time': None,
                'age_seconds': None, 'limit_seconds': limit,
                # Not "stale: false". Nothing was measured, so there is no
                # verdict to give -- the reader renders the reason.
                'stale': None,
                'reason': 'no rows found in the store for this dataset',
            })
            continue
        age = (now - seen).total_seconds()
        rows.append({
            'dataset': dataset,
            'newest_event_time': seen.isoformat(),
            'age_seconds': age,
            'limit_seconds': limit,
            # `None`, not False: a dataset with no expectation (`limit is
            # None`) is one nothing writes continuously, and reporting it as
            # fresh would be a claim nobody measured.
            'stale': None if limit is None else age > limit,
            'reason': (None if limit is not None else
                       'no continuous producer, so no age is expected'),
        })

    return {
        'schema': SCHEMA,
        'generated_at': now.isoformat(),
        'store_root': str(store_root),
        'datasets': rows,
        # The limits only. See the module docstring: a heartbeat AGE written
        # here would be wrong seconds later, so the stamps are read live.
        'component_limits': dict(heartbeat.MAX_AGE),
        'heartbeat_root': str(heartbeat.ROOT),
    }


def late(document: dict[str, Any]) -> list[tuple[str, Optional[float], Optional[float]]]:
    """`(dataset, age_or_None, limit)` for everything behind, from a snapshot.

    The same tuples `core.freshness.stale` returns, so the caller logs one
    thing whichever side computed it -- and without a second pass over 17 GB of
    footers.
    """
    out = []
    for row in document.get('datasets', ()):
        if row.get('limit_seconds') is None:
            continue
        if row.get('newest_event_time') is None or row.get('stale'):
            out.append((row['dataset'], row.get('age_seconds'),
                        row.get('limit_seconds')))
    return out


def write(document: dict[str, Any], path: Optional[Path] = None) -> Optional[Path]:
    """Publish atomically. Returns the path written, or None.

    `os.replace` over a temp file in the same directory, so a reader polling
    this on a timer never catches a half-written document and reports every
    dataset missing.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            'w', dir=str(target.parent), prefix='.collection-', suffix='.json',
            delete=False, encoding='utf-8')
        try:
            with handle:
                json.dump(document, handle, indent=1, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(handle.name)
            raise
        return target
    except Exception as exc:                              # noqa: BLE001
        logger.warning('could not publish the collection status: %s',
                       str(exc)[:200])
        return None


def publish(store_root: str, *, path: Optional[Path] = None,
            datasets=None) -> dict[str, Any]:
    """Measure, then publish. Returns the document whether or not it landed."""
    document = snapshot(store_root, datasets=datasets)
    write(document, path)
    return document
