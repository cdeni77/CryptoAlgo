"""Fold every book source into `venue_depth`, at every minute of the window.

Sources: Kalshi `venue_ladder`, Polymarket `pm_ladder`, and packed backfill
series. Every minute is kept because the offset grid is itself under test.
`venue_quotes` is a separate table that this module does not read.

`quote_age_seconds` says how stale each row is: the backfill serves book
changes, so a quiet book carries forward. `source` separates a book recorded
live from the same book reconstructed later.

On both venues `no_levels` prices are NO-denominated, so `1 - price` is the
YES ask. Packed backfill prices are integer cents.

Idempotent: the store merges partitions on write.
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
    """Return one `venue_depth` row, with `quality` derived from the quote.

    A crossed or out-of-range quote is 'suspicious': kept in the store, but
    out of the default `min_quality='valid'` read.
    """
    base = {'quality': 'valid', 'seq': float('nan'), 'gaps': 0.0}
    base.update(kw)
    bid, ask = base.get('yes_bid'), base.get('yes_ask')
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        bid = ask = None
    bad_range = ((bid is not None and not 0.0 <= bid <= 1.0)
                 or (ask is not None and not 0.0 <= ask <= 1.0))
    crossed = bid is not None and ask is not None and ask < bid
    if bad_range or crossed:
        base['quality'] = 'suspicious'
    return base


def _cumulative(levels, *, best: Optional[float], within: float,
                invert: bool) -> float:
    """Return resting size at prices at least as good as `best +/- within`.

    Must match the live path's `_cumulative`.
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
    """Return `source`, qualified by the transport that fetched the ladder.

    `venue_ladder` holds a REST row and a stream row for one minute. `source`
    is part of the `venue_depth` event key, so REST stays 'live' and the stream
    becomes 'live_ws'; otherwise one would overwrite the other. A missing
    transport means a REST poll.
    """
    transport = record.get('transport')
    if transport is None or not isinstance(transport, str) or transport == 'rest':
        return source
    return f'{source}_{transport}'


def _from_ladder(frame: pd.DataFrame, *, source: str) -> list[dict]:
    """Summarise raw recorded ladders: one row per minute per observer."""
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
        # Floor, never round. Rounding up stamps a row after it was observed,
        # which breaks point-in-time reads.
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
    """Return the book state at each minute of the window from a packed series.

    Minute `m` uses the last snapshot at or before `window_open + m`. A minute
    with no prior snapshot is skipped, never filled.
    """
    if not os.path.exists(path):
        # The research/collect pipeline does not write this file; its books
        # reach venue_depth through load_store.
        logger.warning('no %s, so no %s backfill is built here. Books from '
                       'research/collect load with `python -m '
                       'research.collect.load_store --what depth --month '
                       'YYYY-MM`.', path, venue)
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
                    # The snapshot's age at this minute. A quiet book carries
                    # forward, so the age separates a fill from an observation.
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
