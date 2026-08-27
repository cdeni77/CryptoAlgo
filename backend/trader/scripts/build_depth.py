"""Every BOOK source, into one table at every minute.

**The problem this solves, and the one it does not.** The book lived in three
incompatible places: Kalshi live in `venue_ladder` (raw levels, every minute),
Polymarket live in `pm_ladder`, and the Kalshi Predexon backfill in a JSONL file
outside the research store entirely (packed thirteen-number series). This module
unifies those three into `venue_depth`, keyed the same way, at **every minute**
of the window — because the offset grid is itself under test and a table sampled
where the model currently scores would foreclose the question.

**`venue_quotes` is a fourth, separate table this does NOT read, and an earlier
version of this docstring claimed it did.** It holds the Kalshi Predexon
backfill at an irregular seven offsets (2, 3, 4, 6, 9, 12, 14) and is written by
`scripts/backfill_quotes.py`, not by this module. As of this writing twelve
files still read it directly — `retro_economics.py`, `refit_market_init.py`,
`retro_forecast_test.py`, `diagnose_market_mid.py`, `quote_coverage.py`,
`_status.py`, `_book_analysis.py`, `_offset_vs_market.py`, `_collect_book.py`
among them — so "one table" was true of the book and not yet true of the whole
pipeline. Folding `venue_quotes` into `venue_depth` too is a real migration
(different offset grid, different producer, a dozen consumers to repoint) and
was deliberately left undone here rather than done under time pressure.

`quote_age_seconds` says how stale each row is. Predexon serves book CHANGES,
so the state at minute `m` is the last change at or before it and a quiet book
carries forward; a forward fill that cannot be told from an observation lets a
fresh forecast "beat" a stale price. That is not hypothetical — see the note in
`core/datastore.SCHEMAS`.

`source` distinguishes provenance: `live` is a book somebody recorded while the
market was open; `backfill` is the same book from Predexon after the fact. They
should agree, and where they overlap the disagreement is a free measurement of
both. A table that could not tell them apart would make a reconstruction look
like an observation.

**Denomination is the invariant, not the schema.** On both venues `no_levels`
prices are NO-denominated, so `1 - price` is the YES ask. Kalshi serves that
natively (its orderbook is two bid stacks); Polymarket's asks are converted at
write time in `scripts/record_pm_ladder._no_levels`. The packed backfill series
stores integer cents, so it is divided by 100 here.

Idempotent: the store merges partitions on write, so re-running only adds.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Iterable, Optional

import pandas as pd

from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore

logger = logging.getLogger('build-depth')

# The packed field order written by `_collect_book.py` / `_collect_pm.py`.
PACKED = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')
PM_ASSETS = {'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD'}


def _row(**kw) -> dict:
    base = {'quality': 'valid', 'seq': float('nan'), 'gaps': 0.0}
    base.update(kw)
    return base


def _cumulative(levels, *, best: Optional[float], within: float,
                invert: bool) -> float:
    """Resting size at prices at least as good as `best +/- within`.

    The same rule `scripts/live._cumulative` applies, kept identical on purpose:
    the live path and this one must agree or the joined table is not one table.
    """
    if best is None or not math.isfinite(best):
        return 0.0
    total = 0.0
    for entry in levels or []:
        try:
            price, size = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        effective = (1.0 - price) if invert else price
        if (effective <= best + within) if invert else (effective >= best - within):
            total += size
    return total


def _ladder_source(record: dict, source: str) -> str:
    """`source`, qualified by the transport that fetched the ladder.

    **Since the WebSocket migration `venue_ladder` holds TWO rows for the same
    minute** — the REST poll and the stream cache — and both belong there:
    comparing them is the evidence the stream reproduces the book. Stamping both
    `source='live'` would produce two `venue_depth` rows with an identical event
    key, so `read` would keep whichever carried the later `available_time` and
    `venue_depth` would silently become a mix of the two transports, varying row
    by row.

    `source` sits in `EVENT_KEY_EXTRA['venue_depth']` for precisely this
    situation: it already separates a book somebody recorded from the same book
    reconstructed afterwards, and a second live observer is the same kind of
    thing. So REST keeps `source='live'` and the existing series stays exactly
    what it was, while the stream lands beside it as `live_ws`.

    A missing transport means a row written before the column existed, and every
    one of those was a REST poll.
    """
    transport = record.get('transport')
    if transport is None or not isinstance(transport, str) or transport == 'rest':
        return source
    return f'{source}_{transport}'


def _from_ladder(frame: pd.DataFrame, *, source: str) -> list[dict]:
    """Raw recorded ladders -> the summarised row, one per minute per observer."""
    rows: list[dict] = []
    for record in frame.to_dict('records'):
        try:
            yes = json.loads(record.get('yes_levels') or '[]')
            no = json.loads(record.get('no_levels') or '[]')
        except (TypeError, ValueError):
            continue
        if not yes and not no:
            continue
        yes_bid = max((float(p) for p, _ in yes), default=None)
        best_no = max((float(p) for p, _ in no), default=None)
        yes_ask = (1.0 - best_no) if best_no is not None else None
        minute = record.get('minute_into_window')
        if minute is None or not math.isfinite(float(minute)):
            continue
        # **Floor, never round.** The recorder polls on its own cadence and lands
        # at 9m59s as often as at 10m00s. Rounding that to minute 10 stamps the
        # row as describing a moment AFTER it was observed, and the store
        # rejects it — correctly, because a point-in-time read at t would then
        # return a book nobody had yet seen. Flooring says what is true: nine
        # minutes had elapsed when this was recorded.
        offset = int(math.floor(float(minute)))
        window_open = pd.Timestamp(record['window_open'])
        event_time = window_open + pd.Timedelta(minutes=offset)
        observed = record.get('available_time') or record.get('event_time')
        observed = pd.Timestamp(observed) if observed is not None else event_time
        rows.append(_row(
            venue=record.get('venue'), symbol=record.get('symbol'),
            event_time=event_time,
            available_time=max(observed, event_time),
            quote_age_seconds=max(0.0, (observed - event_time).total_seconds()),
            market_ticker=record.get('market_ticker'), window_open=window_open,
            offset_minutes=offset,
            yes_bid=yes_bid, yes_ask=yes_ask,
            yes_bid_size=sum(float(s) for p, s in yes if float(p) == yes_bid) if yes_bid is not None else 0.0,
            yes_ask_size=sum(float(s) for p, s in no if float(p) == best_no) if best_no is not None else 0.0,
            depth_bid_1c=_cumulative(yes, best=yes_bid, within=0.01, invert=False),
            depth_bid_5c=_cumulative(yes, best=yes_bid, within=0.05, invert=False),
            depth_ask_1c=_cumulative(no, best=yes_ask, within=0.01, invert=True),
            depth_ask_5c=_cumulative(no, best=yes_ask, within=0.05, invert=True),
            depth_bid_total=sum(float(s) for _, s in yes),
            depth_ask_total=sum(float(s) for _, s in no),
            levels_bid=float(len(yes)), levels_ask=float(len(no)),
            source=_ladder_source(record, source)))
    return rows


def _from_packed(path: str, *, venue: str, window_minutes: int,
                 symbol_of=None) -> list[dict]:
    """A packed tick series -> the state at each minute of the window.

    The row for minute `m` is the last snapshot at or before `window_open + m`,
    which is the book a decision there would actually have priced against. A
    minute with no prior snapshot is skipped rather than forward-filled from
    nothing: Polymarket books often begin mid-window, and inventing a quote for
    the minutes before that would be fabricating the very thing under test.
    """
    if not os.path.exists(path):
        logger.info('no %s; skipping', path)
        return []
    rows: list[dict] = []
    index = {name: i for i, name in enumerate(PACKED)}
    with open(path) as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            series = record.get('series') or []
            if not series:
                continue
            window_open = pd.Timestamp(record['window_open'])
            if window_open.tzinfo is None:
                window_open = window_open.tz_localize('UTC')
            symbol = record.get('symbol') or (symbol_of(record) if symbol_of else None)
            if not symbol:
                continue
            ticker = record.get('market_ticker') or record.get('market_slug')
            stamps = [s[index['ts']] for s in series]
            cursor = 0
            for offset in range(0, window_minutes + 1):
                mark = int((window_open + pd.Timedelta(minutes=offset)).timestamp() * 1000)
                while cursor + 1 < len(series) and (stamps[cursor + 1] or 0) <= mark:
                    cursor += 1
                if (stamps[cursor] or 0) > mark:
                    continue                      # no book yet at this minute
                snap = series[cursor]

                def value(name):
                    raw = snap[index[name]]
                    return None if raw is None else float(raw)

                bid, ask = value('best_bid'), value('best_ask')
                stamp = stamps[cursor]
                age = (mark - stamp) / 1000.0 if stamp else float('nan')
                rows.append(_row(
                    venue=venue, symbol=symbol,
                    event_time=window_open + pd.Timedelta(minutes=offset),
                    available_time=window_open + pd.Timedelta(minutes=offset),
                    market_ticker=ticker, window_open=window_open,
                    offset_minutes=offset,
                    # Packed prices are integer cents on both venues.
                    yes_bid=None if bid is None else bid / 100.0,
                    yes_ask=None if ask is None else ask / 100.0,
                    yes_bid_size=value('bid_at_touch') or 0.0,
                    yes_ask_size=value('ask_at_touch') or 0.0,
                    depth_bid_1c=value('bid_1c') or 0.0,
                    depth_ask_1c=value('ask_1c') or 0.0,
                    depth_bid_5c=value('bid_5c') or 0.0,
                    depth_ask_5c=value('ask_5c') or 0.0,
                    depth_bid_total=value('bid_vol') or 0.0,
                    depth_ask_total=value('ask_vol') or 0.0,
                    levels_bid=value('bid_levels') or 0.0,
                    levels_ask=value('ask_levels') or 0.0,
                    # How old the snapshot was at this minute mark. The endpoint
                    # serves changes, so a quiet book carries forward, and an
                    # un-aged forward fill is indistinguishable from an
                    # observation.
                    quote_age_seconds=age,
                    source='backfill'))
    return rows


def _pm_symbol(record: dict) -> Optional[str]:
    slug = str(record.get('market_slug') or '')
    return PM_ASSETS.get(slug.split('-', 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sources', default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')

    window_minutes = DEFAULT_CONFIG.window_minutes
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    want = args.sources.split(',') if args.sources != 'all' else [
        'kalshi_live', 'pm_live', 'kalshi_backfill', 'pm_backfill']

    total = 0
    for name in want:
        rows: list[dict] = []
        if name == 'kalshi_live':
            try:
                rows = _from_ladder(store.read('venue_ladder'), source='live')
            except Exception as exc:                      # noqa: BLE001
                logger.warning('venue_ladder: %s', str(exc)[:90])
        elif name == 'pm_live':
            try:
                rows = _from_ladder(store.read('pm_ladder'), source='live')
            except Exception as exc:                      # noqa: BLE001
                logger.warning('pm_ladder: %s', str(exc)[:90])
        elif name == 'kalshi_backfill':
            rows = _from_packed('data/book_full.jsonl', venue='kalshi',
                                window_minutes=window_minutes)
        elif name == 'pm_backfill':
            rows = _from_packed('data/pm_prices.jsonl', venue='polymarket',
                                window_minutes=window_minutes,
                                symbol_of=_pm_symbol)
        if not rows:
            logger.info('%-16s no rows', name)
            continue
        written = store.write('venue_depth', pd.DataFrame(rows))
        total += len(rows)
        logger.info('%-16s %6d rows -> venue_depth (partition total %d)',
                    name, len(rows), written)

    logger.info('built %d rows', total)
    depth = store.read('venue_depth')
    summary = depth.groupby(['venue', 'source']).agg(
        rows=('offset_minutes', 'size'),
        offsets=('offset_minutes', lambda s: f'{int(s.min())}..{int(s.max())}'),
        windows=('window_open', 'nunique'))
    logger.info('venue_depth now:\n%s', summary.to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
