"""Rebuild the `venue_ladder` minutes nobody sampled, by replaying the frames.

`record_ladder` died on 2026-09-16 and nothing noticed for 32 days: its
`/markets` query had no `max_close_ts`, so once Kalshi began publishing more
than eight windows ahead every row it got back opened ~24h out, the in-window
filter rejected all of them, and a recorder that writes nothing also logs
nothing. `venue_ladder` has a 31-day hole. The stream recorder beside it never
stopped.

**So the hole is not all lost, and this is the tool that says how much isn't.**
`venue_book_events` holds every frame the live `BookCache` folded, so a minute
inside the raw tier's retention can be sampled now exactly as it would have
been then — same applier (`core.stream_book.BookCache`), same row builder
(`scripts.record_ladder.ws_row`), a clock moved to the sample instant instead
of read off the wall.

    python -m scripts.replay_ladder                      # dry run: the plan only
    python -m scripts.replay_ladder --end 2026-10-19T04:00 --write

**That the rebuild is faithful is MEASURED, not argued.**
`research/validate/_validate_replay.py` folds the archive against every live
`transport='ws'` row the recorder wrote once it came back, and over the whole
overlap — 8,169 minutes, 2026-10-19 03:50 to 2026-10-21 01:55 — the rebuilt
ladder is bit-for-bit the recorded one on 8,169 of 8,169. Run it again before
believing this tool; it is cheap and it is the only thing standing between a
backfill and a plausible fiction.

**Three limits, and the first is a deadline.**

* `venue_book_events` is a BOUNDED tier — 60 days (`RAW_RETENTION_DAYS`), and
  `core.spool.prune` enforces it on the hour in each file's name. Whatever this
  can recover, it can recover until the frames age out, and not after. As of
  2026-10-20 the oldest archived hour is 2026-09-30 13:00 UTC, so the front of
  the recoverable range starts being deleted on **2026-11-29 14:00 UTC**, and
  the rest follows it hour by hour. `core.spool.guard_disk` can also shed this
  tier FIRST and early if the volume fills, so the deadline is a ceiling rather
  than a promise.
* The archive starts where it starts. `venue_ladder` is dark from 2026-09-17
  00:21 to 2026-10-19 03:50 — 32 days — and the frames only reach back to
  2026-09-30 13:00, so **18.6 of those 32 days can be rebuilt and the first
  13.5 are gone for good.** `--write` will not invent a row for a minute it
  cannot fold to.
* A replayed row is labelled `transport='ws_replay'`, not `'ws'`. `transport`
  is part of `venue_ladder`'s event key, so a reconstruction lands BESIDE a
  live recording instead of superseding it — the same separation
  `venue_depth.source` already keeps between a book somebody recorded and the
  same book rebuilt afterwards, and what makes
  `research/validate/_validate_replay.py` possible at all. It also means
  re-running over a period the recorder covered cannot destroy what it wrote.

**Nothing is merged into `data.parquet`.** `ResearchStore.write` reads a whole
`(venue, symbol, month)` partition, concatenates and rewrites it — and the live
`record_ladder` is writing that same October partition every minute. Two
read-modify-write cycles over one file is a lost write, silently, on the table
whose entire justification is that a minute lost is gone. So each day is
written as its own immutable `replay-YYYY-MM-DD.parquet` beside the recorder's
file, which `ResearchStore.read` still finds because it globs `**/*.parquet`
and does not depend on the layout — exactly how `core.spool.compact` already
lands the raw tier. Re-running a day replaces that one file, so the tool is
idempotent without ever reading what the live recorder is holding.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from core.config import DEFAULT_CONFIG, series_to_symbol
from core.datastore import ResearchStore
from core.replay_book import (archive_events, hour_files, replay_cache,
                              sampled_books, window_open)
from core.stream_book import BookCache
from data_collection.stream.kalshi import VENUE
from scripts.record_ladder import ws_row

logger = logging.getLogger('replay-ladder')

DATASET = 'venue_ladder'
#: Labelled as reconstructed, never as recorded. See the module docstring.
DEFAULT_TRANSPORT = 'ws_replay'


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

class Sampler:
    """Turns a folded cache at an instant into `venue_ladder` rows.

    One row per market whose own fifteen minutes contain the instant, which is
    the same filter `record_ladder` applies — `0 <= minute <= window_minutes`,
    judged on the window rather than on the venue's `status`, which lags by up
    to forty seconds.
    """

    def __init__(self, *, window_minutes: int = 15,
                 transport: str = DEFAULT_TRANSPORT, grace_minutes: float = 2.0):
        self.window_minutes = window_minutes
        self.transport = transport
        self.grace = grace_minutes
        self.symbols = series_to_symbol()
        self._open: dict[str, Optional[pd.Timestamp]] = {}
        self.skipped_unparsed = 0
        self.suspicious = 0

    def _opened(self, ticker: str) -> Optional[pd.Timestamp]:
        if ticker not in self._open:
            self._open[ticker] = window_open(
                ticker, window_minutes=self.window_minutes)
            if self._open[ticker] is None:
                self.skipped_unparsed += 1
        return self._open[ticker]

    def _symbol(self, ticker: str) -> Optional[str]:
        return self.symbols.get(ticker.split('-', 1)[0])

    def rows(self, cache: BookCache, when: float) -> list[dict]:
        stamp = datetime.fromtimestamp(when, tz=timezone.utc)
        out = []
        for ticker in cache.tickers():
            opened = self._opened(ticker)
            if opened is None:
                continue
            minute = (pd.Timestamp(stamp) - opened).total_seconds() / 60.0
            if minute > self.window_minutes + self.grace:
                # Settled, and it will never be in window again. Dropped for
                # the reason `record_stream.retire` gives: a book left in the
                # cache goes quietly stale and still reads as a market with
                # something resting in it.
                cache.forget(ticker)
                continue
            if not (0 <= minute <= self.window_minutes):
                continue
            symbol = self._symbol(ticker)
            if symbol is None:
                continue
            row = ws_row(cache, ticker=ticker, symbol=symbol, now=stamp,
                         open_time=opened, minute=minute, read_at=stamp)
            if row is None:
                continue
            row['transport'] = self.transport
            # **A condemned book is written, and written as condemned.** A
            # sequence gap marks every book on the connection until the next
            # snapshot rebuilds it, and so does an outage in the recorder
            # itself. The live sampler recorded those minutes anyway, because
            # for the archive a doubtful book is still data. `quality` is how
            # this store says so: `read` defaults to `min_quality='valid'`, so
            # these are excluded from every ordinary read and still on disk for
            # anyone who asks for them.
            if cache.gapped(ticker):
                row['quality'] = 'suspicious'
                self.suspicious += 1
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def day_file(store: ResearchStore, *, venue: str, symbol: str,
             day: str) -> Path:
    return (store.root / DATASET / f'venue={venue}' / f'symbol={symbol}'
            / f'month={day[:7]}' / f'replay-{day}.parquet')


def existing_days(store: ResearchStore) -> set[str]:
    """Days already written by a previous run, by filename."""
    root = store.root / DATASET
    if not root.exists():
        return set()
    return {p.stem[len('replay-'):] for p in root.rglob('replay-*.parquet')}


def write_day(store: ResearchStore, rows: list[dict], day: str) -> int:
    """One immutable file per (symbol, day). Replaces its own, never merges."""
    frame = store._prepare(DATASET, pd.DataFrame(rows))   # noqa: SLF001
    written = 0
    for (venue, symbol), chunk in frame.groupby(['venue', 'symbol'], sort=True):
        path = day_file(store, venue=str(venue), symbol=str(symbol), day=day)
        path.parent.mkdir(parents=True, exist_ok=True)
        chunk = chunk.sort_values('event_time').reset_index(drop=True)
        tmp = path.with_suffix('.parquet.tmp')
        chunk.to_parquet(tmp, index=False, compression='zstd')
        os.replace(tmp, path)
        written += len(chunk)
    return written


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def replay(store: ResearchStore, *, start: pd.Timestamp | None,
           end: pd.Timestamp | None, phase: float, window_minutes: int,
           transport: str, max_event_gap: float, write: bool,
           warmup_minutes: float = 30.0, max_age_seconds: float = 10.0,
           on_day=None) -> dict:
    """Fold the archive and emit one day of rows at a time.

    `start`/`end` bound the SAMPLE instants, not the frames. The fold opens
    `warmup_minutes` earlier and throws those samples away, because a cold
    cache has no book at all until the venue's next snapshot and every minute
    before it would simply be missing. Kalshi snapshots each market on every
    resubscription, which `record_stream` does at least once per fifteen-minute
    boundary, so half an hour is comfortably more than one.
    """
    warmup = pd.Timedelta(minutes=warmup_minutes)
    paths = hour_files(store.root, start=None if start is None else start - warmup,
                       end=end)
    cache, clock = replay_cache(max_age_seconds)
    sampler = Sampler(window_minutes=window_minutes, transport=transport)
    stats = {'files': len(paths), 'samples': 0, 'rows': 0, 'written': 0,
             'days': 0, 'first': None, 'last': None}
    buffer: list[dict] = []
    current: Optional[str] = None

    def flush():
        nonlocal buffer, current
        if current is None or not buffer:
            buffer, current = [], None
            return
        if write:
            stats['written'] += write_day(store, buffer, current)
        stats['days'] += 1
        if on_day is not None:
            on_day(current, len(buffer), write)
        buffer, current = [], None

    seen = {'path': None}

    def note(path: Path) -> None:
        seen['path'] = path
        logger.info('folding %s', path.name)

    events = archive_events(paths, venue=VENUE, on_file=note)
    for mark in sampled_books(events, cache, clock, phase=phase,
                              max_event_gap=max_event_gap, venue=VENUE):
        stamp = pd.Timestamp(mark, unit='s', tz='UTC')
        if start is not None and stamp < start:
            continue
        if end is not None and stamp >= end:
            continue
        stats['samples'] += 1
        stats['first'] = stats['first'] or stamp
        stats['last'] = stamp
        day = stamp.strftime('%Y-%m-%d')
        if day != current:
            flush()
            current = day
        rows = sampler.rows(cache, mark)
        stats['rows'] += len(rows)
        buffer.extend(rows)
    flush()
    stats['suspicious'] = sampler.suspicious
    stats['unparsed_tickers'] = sampler.skipped_unparsed
    return stats


def _bound(value: str | None) -> Optional[pd.Timestamp]:
    if not value:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def resume_point(done, start, *, force: bool = False):
    """Where to resume, given the days already on disk and an explicit start.

    Returns `(start, holes)`. `holes` are days that have files but sit AFTER an
    unwritten gap, so a caller can say so out loud.

    **An explicit `start` is returned untouched.** This used to move any start
    earlier than the resume point, so
    `--start 2026-10-02 --end 2026-10-02T23:59` became the inverted window
    `[2026-10-19, 2026-10-02)` -- zero hour files, zero rows, and an exit that
    logged success.

    **And the resume point is the end of the leading RUN, not `max + 1`.** The
    old form assumed the finished days were contiguous. A parallel run on
    2026-10-21 left 2026-10-02 unwritten while 2026-10-18 succeeded, so the
    resume point jumped past the hole and a plain re-run could never fill it.
    """
    if force or not done or start is not None:
        return start, []
    ordered = sorted(pd.Timestamp(d, tz='UTC') for d in done)
    resume = ordered[0]
    for day in ordered:
        if day > resume:
            break                       # a hole: stop, do not skip it
        resume = day + pd.Timedelta(days=1)
    return resume, [d for d in ordered if d > resume]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    parser.add_argument('--start', help='UTC bound on the sample instants')
    parser.add_argument('--end', help='UTC bound, exclusive')
    parser.add_argument('--phase', type=float, default=25.0,
                        help='seconds past the minute to sample at, matching '
                             "record_ladder's own default (25)")
    parser.add_argument('--transport', default=DEFAULT_TRANSPORT,
                        help='the label these rows carry. Anything but "ws" '
                             'or "rest", which belong to the live recorders.')
    parser.add_argument('--warmup-minutes', type=float, default=30.0,
                        help='how much archive to fold before --start without '
                             'emitting it, so the cache has a snapshot behind '
                             'it when the first sample lands (default 30)')
    parser.add_argument('--max-event-gap', type=float, default=60.0,
                        help='seconds of silence that count as the recorder '
                             'having been down rather than a quiet market')
    parser.add_argument('--write', action='store_true',
                        help='write the rows. Without it this prints the plan '
                             'and touches nothing.')
    parser.add_argument('--dry-run', action='store_true',
                        help='the default; accepted so it can be said out loud')
    parser.add_argument('--force', action='store_true',
                        help='re-do days a previous run already wrote. Without '
                             'it the run resumes at the first day that has no '
                             'replay file.')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    if args.dry_run and args.write:
        logger.error('--dry-run and --write say opposite things; pick one')
        return 2
    if args.transport in ('ws', 'rest'):
        logger.error('--transport %r is a live recorder\'s label. A '
                     'reconstruction that calls itself a recording cannot be '
                     'checked against one.', args.transport)
        return 2

    store = ResearchStore(args.research_store)
    start, end = _bound(args.start), _bound(args.end)

    done = existing_days(store)
    start, holes = resume_point(done, start, force=bool(args.force))
    if done and not args.force and holes is not None and start is not None:
        logger.info('%d day(s) already replayed; resuming at %s '
                    '(--force to redo them)', len(done), start.date())
    if holes:
        logger.warning(
            'these day(s) have files AFTER an unwritten gap and are left '
            'alone: %s. Re-run a missing day explicitly with --start and '
            '--end; an explicit start is honoured.',
            ', '.join(str(d.date()) for d in holes[:8]))

    write = bool(args.write)
    logger.info('store=%s  window=[%s, %s)  phase=%.1fs  transport=%s  %s',
                store.root, start or 'archive start', end or 'archive end',
                args.phase, args.transport,
                'WRITING' if write else 'dry run — nothing will be written')

    def on_day(day: str, rows: int, wrote: bool) -> None:
        logger.info('%s: %6d rows %s', day, rows,
                    'written' if wrote else '(dry run)')

    stats = replay(store, start=start, end=end, phase=args.phase,
                   window_minutes=DEFAULT_CONFIG.window_minutes,
                   transport=args.transport, max_event_gap=args.max_event_gap,
                   write=write, warmup_minutes=args.warmup_minutes,
                   on_day=on_day)

    logger.info('%d hour file(s), %d sample instants, %d rows over %d day(s)',
                stats['files'], stats['samples'], stats['rows'], stats['days'])
    if stats['first'] is not None:
        logger.info('covered %s -> %s', stats['first'], stats['last'])
    logger.info('%d row(s) marked quality=suspicious (a gapped or '
                'awaiting-snapshot book: on disk, excluded from a default '
                'read)', stats['suspicious'])
    if stats['unparsed_tickers']:
        logger.warning('%d ticker(s) did not parse as a window and were '
                       'skipped', stats['unparsed_tickers'])
    if write:
        logger.info('wrote %d row(s)', stats['written'])
    else:
        logger.info('dry run: nothing written. Add --write.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
