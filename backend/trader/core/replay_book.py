"""Rebuild the book the live path folded, from the frames it was folded from.

`venue_ladder` is the table `record_ladder` calls irrecoverable: no venue
endpoint serves resting size at a past instant, so a minute not sampled is a
minute gone. That argument has one hole in it, and this module is the hole:
since 2026-08-27 every frame that built the live book is archived in
`venue_book_events`, so a ladder inside the raw tier's retention is not gone —
it is unsampled, which is a different thing.

**The fold is not re-implemented here.** `core.stream_book.BookCache` is the one
applier, in the live path and in this one, for the reason `core/decide.py`
records about `decide()`: two implementations of one rule disagree eventually
and nothing says when. All this module adds is the INVERSE of
`core.spool.event_rows` — the regroup from one-row-per-level back into the
frames — and a clock the sampler can move, so a replayed `Ladder` carries the
same `age_seconds` the live cache would have reported at that instant.

**Three things the archive cannot give back**, and they are the reason a replay
is checked rather than assumed:

* **A snapshot with both sides empty leaves no rows at all.** `event_rows`
  emits one record per level, so a book cleared to nothing is unrepresentable
  and the replay keeps the last non-empty ladder where the live cache would
  have emptied it. Kalshi serves exactly that at settlement — which is why
  nothing is sampled outside a market's own window.
* **A dark stretch is not an idle market.** At ~800 frames a second a minute
  with no rows means the recorder was down, not that nobody quoted. Folding
  across it would carry a pre-outage book forward under a post-outage
  timestamp, so `sampled_books` invalidates every book on a gap and the sampler
  skips the dark minutes entirely.
* **A sequence gap condemns the whole connection**, not one market — `seq` is
  contiguous per SUBSCRIPTION. `BookCache` already marks every book on the
  socket; a replay must not quietly launder that into a clean row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent, Level

#: The raw tier the replay reads. Bounded by `RAW_RETENTION_DAYS` (60), which
#: is the deadline on everything this module can recover.
RAW_DATASET = 'venue_book_events'

#: What a replay needs off an archived row, and nothing else. `symbol` is
#: deliberately not among them: it is 3.5M strings an hour to recover something
#: the ticker's own series prefix already says, and the archive writes
#: 'UNKNOWN' into it whenever the stream carried a ticker the recorder's map had
#: not caught up with.
ARCHIVE_COLUMNS = ('event_time', 'market_ticker', 'seq', 'kind', 'side',
                   'price', 'size', 'absolute')

#: The columns of one archived level row, in the order `events_from_rows` wants
#: them. A tuple rather than a mapping because the caller has 1.5e9 of these:
#: building a dict per row costs more than the fold itself, measured at 2.2s per
#: 500k rows against 9s for a whole hour of folding.
FIELDS = ('market_ticker', 'received', 'seq', 'kind', 'side', 'price', 'size',
          'absolute')

#: Kalshi stamps a ticker in EXCHANGE local time, so the window a ticker names
#: is a wall-clock reading in New York and the UTC instant it denotes moves with
#: daylight saving. Reading it as UTC would shift every window four or five
#: hours; hardcoding -04:00 would be right until the first Sunday in November.
VENUE_TZ = ZoneInfo('America/New_York')

#: `KXBTC15M-26OCT180800-00`: series, then the window's CLOSE as YYMMMDDHHMM in
#: venue-local time, then a suffix that repeats the close minute. The suffix is
#: recent — `CLAUDE.md` still cites its ABSENCE as the tell for an up/down
#: market — so it is optional here rather than required, and nothing downstream
#: reads it.
_TICKER = re.compile(
    r'^KX[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})(?:-\d+)?$')
_MONTHS = {m: i + 1 for i, m in enumerate(
    ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
     'NOV', 'DEC'))}


def window_open(ticker: str, *, window_minutes: int = 15) -> Optional[pd.Timestamp]:
    """The UTC open of the window `ticker` names, or None if it is not one.

    **The ticker names the CLOSE.** Verified against every
    `(market_ticker, window_open)` pair the live recorder wrote — 7,059 distinct
    tickers over 26 days — where `close - window_minutes` reproduces the venue's
    own `open_time` on all of them and nothing else does.

    This is the one place in the repository that reads a market out of a string
    rather than asking the venue, and it is confined to the replay on purpose.
    `CLAUDE.md` is right that a pattern is a guess that keeps working until the
    venue renames a series — but a replay cannot ask the venue about a window
    that settled weeks ago, and the pattern is checked here against recorded
    truth rather than assumed. A ticker that does not match is SKIPPED, never
    guessed at: a mis-parsed window would stamp a real book onto the wrong
    fifteen minutes, which is the one error that reads as data.
    """
    match = _TICKER.match(ticker or '')
    if match is None:
        return None
    year, month, day, hour, minute = match.groups()
    if month not in _MONTHS:
        return None
    try:
        close = datetime(2000 + int(year), _MONTHS[month], int(day),
                         int(hour), int(minute), tzinfo=VENUE_TZ)
    except ValueError:
        return None
    opened = pd.Timestamp(close).tz_convert('UTC') - pd.Timedelta(
        minutes=window_minutes)
    # A 15-minute market opens on a quarter-hour boundary. Anything else is a
    # ticker shaped like one of ours that is not one of ours.
    if opened.second or opened.microsecond or opened.minute % window_minutes:
        return None
    return opened


def row_tuple(record: Mapping) -> tuple:
    """One spooled or compacted level row in `FIELDS` order.

    Accepts both shapes the row ever has: the spool's JSONL record, whose
    receipt time is the float `t`, and the compacted Parquet row, whose
    `event_time` is the same instant as a Timestamp.
    """
    received = record.get('t')
    if received is None:
        received = pd.Timestamp(record['event_time']).timestamp()
    return (record['market_ticker'], float(received), record.get('seq'),
            record['kind'], record['side'], float(record['price']),
            float(record['size']), bool(record['absolute']))


def _seq(value) -> Optional[int]:
    if value is None or value != value:            # NaN from a nullable column
        return None
    return int(value)


def _event(head: tuple, yes: list[Level], no: list[Level],
           venue: str) -> BookEvent:
    ticker, received, seq, kind, absolute = head
    return BookEvent(venue=venue, market_ticker=str(ticker), kind=str(kind),
                     received=float(received), seq=_seq(seq),
                     yes=sorted(yes), no=sorted(no), absolute=bool(absolute))


def events_from_rows(rows: Iterable[Sequence], *,
                     venue: str = 'kalshi') -> Iterator[BookEvent]:
    """The inverse of `core.spool.event_rows`: level rows back into frames.

    **Regrouped on `(market_ticker, received, seq)`, never `(market_ticker,
    seq)`.** `core/spool.py` records why at length: `seq` restarts at 1 on every
    resubscription, which `record_stream` does at least once per fifteen-minute
    boundary, so the narrower pair is not unique across a day and merges two
    ladders minutes apart into one impossible frame. Those are exactly the
    snapshots a replay bootstraps from.

    Grouping is by ADJACENCY, because `event_rows` emits a frame's levels
    consecutively and `core.spool.compact` preserves that order. So read the
    Parquet files directly and in order — `ResearchStore.read` sorts by
    `(symbol, event_time)` and would interleave frames that share a
    millisecond, of which there are hundreds a second.
    """
    key = head = None
    yes: list[Level] = []
    no: list[Level] = []
    for row in rows:
        ticker, received, seq, kind, side, price, size, absolute = row
        here = (ticker, received, seq, kind, absolute)
        if here != key:
            if key is not None:
                yield _event(head, yes, no, venue)
            key = head = here
            yes, no = [], []
        (yes if side == 'yes' else no).append((price, size))
    if key is not None:
        yield _event(head, yes, no, venue)


def hour_files(root: Path, *, start: pd.Timestamp | None = None,
               end: pd.Timestamp | None = None) -> list[Path]:
    """Every compacted hour of frames under `root`, in time order.

    Bounded on the hour in the file's NAME — the same key `core.spool.prune`
    ages on — rather than on mtime, which a restore or a copy resets. An hour
    is kept when it can contain a sample in `[start, end)`, so the fold opens
    on the hour `start` falls inside rather than at the top of it.
    """
    paths = []
    for path in (Path(root) / RAW_DATASET).rglob('*.parquet'):
        try:
            hour = pd.Timestamp(datetime.strptime(path.stem, '%Y%m%dT%H'),
                                tz='UTC')
        except ValueError:
            continue
        if start is not None and hour + pd.Timedelta(hours=1) <= start:
            continue
        if end is not None and hour >= end:
            continue
        paths.append((hour, path))
    return [p for _, p in sorted(paths)]


def file_rows(path: Path) -> Iterator[Sequence]:
    """One archived hour as `FIELDS` tuples, in the order it was written.

    Columnar on purpose. The same hour through `Table.to_pylist()` costs ~15s
    of dict building against ~9s for the entire fold, and there are hundreds of
    hours of it.
    """
    table = pq.read_table(path, columns=list(ARCHIVE_COLUMNS))
    received = (table.column('event_time').to_numpy(zero_copy_only=False)
                .astype('datetime64[ns]').astype('int64') / 1e9)
    return zip(table.column('market_ticker').to_pylist(),
               received,
               table.column('seq').to_numpy(zero_copy_only=False),
               table.column('kind').to_pylist(),
               table.column('side').to_pylist(),
               table.column('price').to_numpy(zero_copy_only=False),
               table.column('size').to_numpy(zero_copy_only=False),
               table.column('absolute').to_numpy(zero_copy_only=False))


def archive_events(paths: Iterable[Path], *, venue: str = 'kalshi',
                   on_file=None) -> Iterator[BookEvent]:
    """The frames in `paths`, in order, as one continuous stream."""
    for path in paths:
        if on_file is not None:
            on_file(path)
        yield from events_from_rows(file_rows(path), venue=venue)


@dataclass
class ReplayClock:
    """`time.time()` for a cache being folded through history.

    `BookCache` dates every read against this, so moving it to the sample
    instant makes `Ladder.age_seconds` exactly `sample - last_frame` — the same
    quantity the live recorder measured, without its clock skew.
    """
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


def replay_cache(max_age_seconds: float = 10.0) -> tuple[BookCache, ReplayClock]:
    """A cache and the clock it dates every read against, already wired.

    **Built together because building them apart fails silently.** A
    `BookCache()` left on `time.time` still folds perfectly and still answers
    every query — it just reports an age of however long ago the archive is, so
    `ws_row` stamps `read_at - age` before the window even opened and
    `available_time` collapses onto `event_time`. The ladder is right, the
    timestamps are nonsense, and nothing raises.
    """
    clock = ReplayClock()
    return BookCache(max_age_seconds=max_age_seconds, now=clock), clock


def next_mark(after: float, *, phase: float, period: float) -> float:
    """The first sampling instant strictly after `after`."""
    phase = float(phase) % period
    return (after - phase) // period * period + phase + period


def sampled_books(events: Iterable[BookEvent], cache: BookCache,
                  clock: ReplayClock, *, phase: float = 25.0,
                  period: float = 60.0, max_event_gap: float = 60.0,
                  venue: str = 'kalshi') -> Iterator[float]:
    """Fold `events`, pausing at each aligned instant to hand back the clock.

    Yields the sample epoch once every book is folded up to and including it;
    the caller then reads `cache` and writes whatever rows it wants. The cache
    is handed over rather than copied because copying a ~200-level ladder per
    market per minute over 18 days is most of the work for none of the answer.

    **A sample is emitted only when a later frame proves the fold reached it.**
    The marks after the last event are not emitted at all: a book dated by a
    clock with no frames behind it is an assertion, not an observation, and at
    the end of an archive that is precisely the row somebody would read as
    "the market was quiet".

    **`max_event_gap` seconds of silence invalidates every book.** The live
    recorder treats fifteen seconds of it as a dead socket; here it means the
    recorder itself was down, and the repair is the same one `record_stream`
    makes — every book is awaiting a snapshot until one arrives. Marks inside
    the dark stretch are skipped rather than filled.
    """
    if getattr(cache, '_now', None) is not clock:
        raise ValueError(
            'the cache must be dated by this clock — build both with '
            'replay_cache(). A cache still on time.time() folds correctly and '
            'reports an age of however long ago the archive is, which is how '
            'every replayed timestamp goes wrong without anything raising.')
    mark: Optional[float] = None
    previous: Optional[float] = None
    for event in events:
        received = event.received
        if previous is not None and received - previous > max_event_gap:
            cache.invalidate(venue)
            cache.reset_sequence(venue)
            mark = next_mark(received, phase=phase, period=period)
        elif mark is None:
            mark = next_mark(received, phase=phase, period=period)
        previous = received
        while mark < received:
            clock.now = mark
            yield mark
            mark += period
        cache.apply(event)
