"""Is each DATASET still receiving rows? Not: is each process still cycling.

**`core.heartbeat` cannot catch the failure that actually happened.** It proves
a component's loop reached the end of a pass, which is the right question for a
wedged coroutine and the wrong one for a loop that runs happily and writes
nothing. `venue_ladder` stopped recording on 2026-09-16 and was dead for
**thirty-one days** before anyone noticed, while the raw frames it is built
from stayed current and every heartbeat stayed green. The cost was not the
ladder: `scripts/build_depth.py` reads it for `source='live'`, so `venue_depth`
lost its full minute grid too and carried only the four decision offsets from
2026-10-14 to 2026-10-18 -- which is precisely the sampling CLAUDE.md warns
"would foreclose the question" of which offsets to trade.

So this asserts the thing that was actually missing: rows arrived recently.

**It reads only the NEWEST file of each dataset.** The store holds 17 GB, and a
check that scans it cannot run on a health interval. The newest partition's
`max(event_time)` answers the question and costs one file read.

**It is deliberately NOT wired into the container healthcheck.** A dataset can
be legitimately behind -- a venue outage, a backfill not yet run -- and failing
the healthcheck restarts a process that is trading real money. Staleness here
is a thing to be told about, not a thing to restart for. Run it from
`scripts/check_freshness.py`.
"""

from __future__ import annotations

import glob
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger('freshness')

#: How far behind each dataset may fall before it is worth saying out loud.
#: Generous multiples of each producer's own cadence -- the point is to catch a
#: month, not a slow hour. `None` means "no expectation": a dataset nothing
#: writes continuously, which must not be reported as broken.
EXPECTED_MAX_AGE: dict[str, Optional[float]] = {
    # **Calibrated to how each dataset is WRITTEN, not to how often rows
    # happen.** A check that cries wolf is one nobody reads, and the first
    # version of this did exactly that: it flagged three datasets at 1h15m
    # against a 10-minute limit, when the limit was simply wrong.
    #
    # Spool-backed (`core.spool`): the raw frames land in a spool and
    # `compact_all` folds only the CLOSED hours into Parquet, from the hourly
    # `store_sync` loop. So the newest Parquet row legitimately trails by the
    # open hour plus a cycle. Three hours is the first figure that is really
    # late.
    'venue_book_events': 3 * 3600.0,
    'spot_trades': 3 * 3600.0,
    'index_proxy_1s': 3 * 3600.0,
    # Folded hourly by `build_depth` / `scrape` / `collect_settlements`, all
    # inside the same `store_sync` loop. Two missed passes.
    'venue_depth': 3 * 3600.0,
    'minute_bars': 3 * 3600.0,
    'venue_settlements': 3 * 3600.0,
    # Minute-grid recorders writing straight through. Fifteen minutes of
    # silence here is one whole market window with nothing recorded -- this is
    # the check that `venue_ladder` needed and did not have.
    'venue_ladder': 900.0,
    'pm_ladder': 900.0,
    'venue_implied_vol': 3600.0,
    # **It has a live producer now** — `scripts/record_trades.py`, the `trades`
    # component. This read `None` ("written only by the backfill, so an age
    # here is expected") until 2026-10-20, which was true and is the reason the
    # dataset went 22 days with an interior hole and nobody was told.
    #
    # The recorder deliberately runs about two minutes behind the tape: it
    # writes a market-minute only once that minute has fully elapsed AND the
    # poll's lookback covered all of it, because a partial minute's
    # `min_price` understates how deep an aggressor went. So the newest row
    # legitimately trails by a few minutes, and the limit is the same fifteen
    # as the other minute-grid recorders — one whole market window with
    # nothing recorded.
    #
    # It will read STALE until the live process is restarted with the new
    # component. That is the true answer, not a false alarm.
    'venue_trades': 900.0,
}


def newest_event_time(store_root: str, dataset: str) -> Optional[datetime]:
    """`max(event_time)` from the most recently written file, or None.

    **Read from the Parquet row-group STATISTICS, not from the rows.** Every
    writer records a per-column min/max in the footer, so the newest timestamp
    costs a metadata read whatever the file weighs -- which matters when one
    dataset is 17 GB. Reading the column instead was the first attempt and it
    also pulled in duckdb's timezone handling, which needs `pytz` and is not
    installed here; the footer is both cheaper and free of that.

    Never raises: a freshness probe that can take the caller down is worse than
    one that reports nothing.
    """
    try:
        import pyarrow.parquet as pq

        files = glob.glob(os.path.join(store_root, dataset, '**', '*.parquet'),
                          recursive=True)
        if not files:
            return None
        # **Every file, not the newest few by mtime.** The store partitions by
        # venue, symbol and month, so the most recently TOUCHED file is not
        # reliably the one holding the most recent row -- a first version read
        # the newest twelve and reported `venue_settlements`, the LABEL, as
        # seven days stale when it was current to the minute. A footer read is
        # metadata only, so scanning them all is cheap even at 489 files.
        best: Optional[datetime] = None
        for path in files:
            meta = pq.ParquetFile(path).metadata
            names = meta.schema.names
            column = next((n for n in ('event_time', 'created_time',
                                       'settled_time', 'timestamp',
                                       'window_open') if n in names), None)
            if column is None:
                continue
            index = names.index(column)
            for group in range(meta.num_row_groups):
                stats = meta.row_group(group).column(index).statistics
                if stats is None or stats.max is None:
                    continue
                value = stats.max
                stamp = (value if isinstance(value, datetime)
                         else datetime.fromisoformat(str(value)))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                if best is None or stamp > best:
                    best = stamp
        return best
    except Exception:                                   # noqa: BLE001
        logger.debug('could not read the newest %s row', dataset, exc_info=True)
        return None


def stale(store_root: str, *, datasets=None,
          now: Optional[datetime] = None) -> list[tuple[str, Optional[float], float]]:
    """`(dataset, age_seconds_or_None, limit)` for everything behind."""
    now = now or datetime.now(timezone.utc)
    wanted = list(datasets) if datasets else [
        d for d, limit in EXPECTED_MAX_AGE.items() if limit is not None]
    late = []
    for dataset in wanted:
        limit = EXPECTED_MAX_AGE.get(dataset)
        if limit is None:
            continue
        seen = newest_event_time(store_root, dataset)
        if seen is None:
            late.append((dataset, None, limit))
            continue
        age = (now - seen).total_seconds()
        if age > limit:
            late.append((dataset, age, limit))
    return late


def describe(age: Optional[float]) -> str:
    if age is None:
        return 'no rows found'
    return str(timedelta(seconds=int(age)))
