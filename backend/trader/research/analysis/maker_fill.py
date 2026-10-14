"""Would a passive (maker) order have filled, and was the fill adversely selected?

At each decision instant (window open + offset minutes + a delay) the tool posts
a hypothetical limit order to buy one side of a Kalshi 15-minute market, rests it
for a fixed time, then cancels it. It replays a recorded book to find whether
the order fills, then marks the fill against the later mid and the venue's own
settlement. The same decisions are also priced as a taker at the ask.

Two inputs, one engine:

* `stream` -- `venue_book_events` (WebSocket snapshots and signed deltas),
  folded through `BookCache`, the one applier. About 14 days of history.
* `snapshots` -- the Predexon order-book snapshots that `research/collect`
  writes (`COLLECT_DATA/derived`, or `COLLECT_DATA/archive` for a month not yet
  consolidated). Back to mid-May. The snapshots are change-driven and carry
  only the touch, so only the trade-through fill rule applies.

Fill rules. The queue position is unknown, so:

* `conservative` -- fill only when the ask on our side moves to our price or
  better. In the real book (which does not hold our order) that is a seller
  who found no buyer at our price; with our order resting there, the seller
  fills us.
* `queue` (stream only) -- also fill when the cumulative decrease of the
  resting size at our price is more than the size ahead of us at post time.
  Every decrease counts as queue progress, cancels too, so this is an upper
  bound.

Fills are all or nothing, and our order does not change the book.

Point in time: a frame or snapshot stamped at or before the decision instant
sets the post; only frames after the post and at or before the cancel can fill.

    python research/analysis/maker_fill.py --start 2026-10-01 --end 2026-10-14
    python research/analysis/maker_fill.py --input snapshots --start 2026-05-15
    python research/analysis/maker_fill.py --select all --rest-seconds 20
    python research/analysis/maker_fill.py --select model --model-file p.parquet
"""

from __future__ import annotations

# `core` is rooted at backend/trader/, which Python does not add to sys.path for
# a script run from a subdirectory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import dataclasses
import gzip
import heapq
import itertools
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.costs import round_to_tick, tick_at, trade_fee
from core.stream_book import MIN_SIZE, BookCache
from data_collection.stream.base import BookEvent

logger = logging.getLogger('maker_fill')

VENUE = 'kalshi'
INPUTS = ('stream', 'snapshots')
SELECTS = ('bias', 'all', 'model')
SIDES = ('UP', 'DOWN')
CONSERVATIVE, QUEUE = 'conservative', 'queue'
EPS = 1e-9
EPOCH = pd.Timestamp(0, tz='UTC')
CENTS = 100.0
DEFAULT_HORIZONS = (10, 30, 60)
MID_EDGES = (0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
STREAM_COLUMNS = ('symbol', 'event_time', 'market_ticker', 'seq', 'kind',
                  'side', 'price', 'size', 'absolute')
SNAPSHOT_COLUMNS = ('symbol', 'market_id', 'window_open', 'ts',
                    'best_bid', 'best_ask')


def _seconds(stamps) -> np.ndarray:
    """Epoch seconds as float, for any datetime unit."""
    index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True))
    return np.asarray((index - EPOCH) / pd.Timedelta(seconds=1), dtype=float)


def _utc(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


# ---- the book, as the engine sees it -----------------------------------------

@dataclass(frozen=True)
class Top:
    """YES best bid and YES best ask. NaN where that side is empty."""

    yes_bid: float
    yes_ask: float

    def side(self, side: str) -> tuple[float, float]:
        """(bid, ask) for the contract `side` buys. DOWN buys NO."""
        if side == 'UP':
            return self.yes_bid, self.yes_ask
        return 1.0 - self.yes_ask, 1.0 - self.yes_bid

    def mid(self, side: str) -> float:
        bid, ask = self.side(side)
        if not (math.isfinite(bid) and math.isfinite(ask)) or ask < bid - EPS:
            return math.nan
        return (bid + ask) / 2.0


class StreamBook:
    """One market's `BookCache`, read as a touch and as a size per level."""

    queue = True

    def __init__(self, ticker: str, max_age: float) -> None:
        self.ticker = ticker
        self.max_age = max_age
        self.last = -math.inf
        self._cache = BookCache(max_age_seconds=math.inf)
        self._ladder = None

    def apply(self, t: float, event: BookEvent) -> None:
        self._cache.apply(event)
        self.last = t
        self._ladder = None

    def _read(self):
        if self._ladder is None:
            self._ladder = self._cache.ladder(self.ticker)
        return self._ladder

    def top(self, now: float) -> Optional[Top]:
        """The touch, or None when there is no book or it is older than max_age."""
        ladder = self._read()
        if ladder is None or now - self.last > self.max_age:
            return None
        yes_bid = ladder.yes[-1][0] if ladder.yes else math.nan
        yes_ask = 1.0 - ladder.no[-1][0] if ladder.no else math.nan
        return Top(yes_bid, yes_ask)

    def level(self, side: str, price: float) -> float:
        """Resting size at `price` on the bid stack of `side`."""
        ladder = self._read()
        if ladder is None:
            return 0.0
        stack = ladder.yes if side == 'UP' else ladder.no
        for level_price, size in stack:
            if abs(level_price - price) < 1e-6:
                return size
        return 0.0


class SnapshotBook:
    """The touch from a change-driven snapshot series. No sizes per level."""

    queue = False

    def __init__(self, max_age: float) -> None:
        self.max_age = max_age
        self.last = -math.inf
        self._top: Optional[Top] = None

    def apply(self, t: float, top: Top) -> None:
        self._top = top
        self.last = t

    def top(self, now: float) -> Optional[Top]:
        if self._top is None or now - self.last > self.max_age:
            return None
        return self._top

    def level(self, side: str, price: float) -> float:
        return 0.0


# ---- the policy and the simulated order --------------------------------------

@dataclass(frozen=True)
class Policy:
    select: str = 'bias'
    offsets: tuple[int, ...] = tuple(DEFAULT_CONFIG.decision_offsets)
    delay_seconds: float = 3.0
    rest_seconds: float = 20.0
    contracts: int = 1
    improve: bool = False
    band: tuple[float, float] = (0.65, 0.95)
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    max_book_age: float = 60.0
    min_edge: float = 0.0
    maker_fee_rate: float = DEFAULT_CONFIG.maker_fee_rate
    # The venue's maker formula: round_up(M * 0.0175 * C * P * (1 - P)).
    # When set, it replaces the flat `maker_fee_rate`.
    maker_fee_multiplier: Optional[float] = None

    def __post_init__(self) -> None:
        if self.select not in SELECTS:
            raise ValueError(f'select must be one of {SELECTS}, got {self.select!r}')
        lo, hi = self.band
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError(f'band must satisfy 0 <= lo <= hi <= 1, got {self.band}')
        if self.contracts < 1 or self.rest_seconds <= 0:
            raise ValueError('contracts must be >= 1 and rest_seconds > 0')

    def config(self, base: Config = DEFAULT_CONFIG) -> Config:
        return dataclasses.replace(base, maker_fee_rate=self.maker_fee_rate)


@dataclass
class Order:
    id: int
    input: str
    symbol: str
    market_ticker: str
    window_open: pd.Timestamp
    offset: int
    side: str
    decided: float
    cancel: float
    yes_mid: float
    bid: float
    ask: float
    price: float
    improved: bool
    ahead: float
    model_p: float = math.nan
    level: float = 0.0
    depleted: float = 0.0
    fills: dict = field(default_factory=dict)       # mode -> fill time
    mid_fill: dict = field(default_factory=dict)    # mode -> side mid at fill
    marks: dict = field(default_factory=dict)       # (mode, horizon) -> side mid

    @property
    def side_mid(self) -> float:
        return self.yes_mid if self.side == 'UP' else 1.0 - self.yes_mid


def post_price(bid: float, ask: float, improve: bool) -> tuple[float, bool]:
    """Our bid, and whether it improves the touch. Never crosses the ask."""
    if improve:
        candidate = float(round_to_tick(bid + float(tick_at(bid))))
        if bid + EPS < candidate < ask - EPS:
            return candidate, True
    return bid, False


MAKER_FEE_COEFFICIENT = 0.0175


def maker_fee(price: float, contracts: int, config: Config,
              multiplier: Optional[float] = None) -> float:
    """Maker fee per contract, rounded per order as the venue charges.

    With `multiplier`, the venue's quadratic maker schedule; otherwise the flat
    `config.maker_fee_rate`.
    """
    if multiplier is not None:
        raw = (multiplier * MAKER_FEE_COEFFICIENT * contracts
               * price * (1.0 - price))
        return math.ceil(round(raw * 10_000, 6)) / 10_000 / contracts
    return float(trade_fee(contracts, price, config, maker=True)) / contracts


def taker_fee(price: float, contracts: int, config: Config) -> float:
    """Taker fee per contract: 0.07 * p * (1 - p), rounded per order."""
    return float(trade_fee(contracts, price, config, maker=False)) / contracts


def choose_sides(policy: Policy, top: Top, *, model_p: float = math.nan,
                 config: Config = DEFAULT_CONFIG) -> list[str]:
    """The sides to post at one decision."""
    if policy.select == 'all':
        return list(SIDES)
    if policy.select == 'bias':
        lo, hi = policy.band
        for side in SIDES:
            mid = top.mid(side)
            if math.isfinite(mid) and lo - EPS <= mid <= hi + EPS:
                return [side]
        return []
    if not math.isfinite(model_p):
        return []
    best, best_edge = None, -math.inf
    for side, p in (('UP', model_p), ('DOWN', 1.0 - model_p)):
        bid, ask = top.side(side)
        if not (math.isfinite(bid) and math.isfinite(ask)) or ask <= bid + EPS:
            continue
        price, _ = post_price(bid, ask, policy.improve)
        edge = p - price - maker_fee(price, policy.contracts, config,
                                     policy.maker_fee_multiplier)
        if edge > best_edge:
            best, best_edge = side, edge
    return [best] if best is not None and best_edge > policy.min_edge else []


# ---- the engine ----------------------------------------------------------------

def simulate_market(updates: Iterable[tuple[float, object]], book, *, input: str,
                    symbol: str, ticker: str, window_open: pd.Timestamp,
                    policy: Policy, model: Optional[dict] = None,
                    skips: Optional[Counter] = None,
                    ids: Optional[Iterator[int]] = None) -> list[Order]:
    """Replay one market and return its simulated orders.

    `updates` is (time, payload) in arrival order; `book.apply` takes each. An
    instant (decision, cancel, mark) runs after every update stamped at or
    before it, and before the first update stamped after it.
    """
    skips = Counter() if skips is None else skips
    ids = itertools.count() if ids is None else ids
    config = policy.config()
    heap: list = []
    tiebreak = itertools.count()
    orders: list[Order] = []
    active: list[Order] = []
    opened = window_open.timestamp()
    modes = (CONSERVATIVE, QUEUE) if book.queue else (CONSERVATIVE,)

    def push(when: float, kind: str, payload) -> None:
        heapq.heappush(heap, (when, next(tiebreak), kind, payload))

    for offset in policy.offsets:
        push(opened + offset * 60.0 + policy.delay_seconds, 'decide', offset)

    def decide(now: float, offset: int) -> None:
        top = book.top(now)
        if top is None:
            skips['no current book'] += 1
            return
        if not (math.isfinite(top.yes_bid) and math.isfinite(top.yes_ask)) \
                or top.yes_ask <= top.yes_bid + EPS:
            skips['one-sided or crossed book'] += 1
            return
        key = (symbol, window_open, int(offset))
        model_p = math.nan
        if policy.select == 'model':
            model_p = (model or {}).get(key, math.nan)
            if not math.isfinite(model_p):
                skips['no model probability'] += 1
                return
        sides = choose_sides(policy, top, model_p=model_p, config=config)
        if not sides:
            skips['no side selected'] += 1
            return
        for side in sides:
            bid, ask = top.side(side)
            price, improved = post_price(bid, ask, policy.improve)
            ahead = 0.0 if improved else book.level(side, price)
            order = Order(id=next(ids), input=input, symbol=symbol,
                          market_ticker=ticker, window_open=window_open,
                          offset=int(offset), side=side, decided=now,
                          cancel=now + policy.rest_seconds,
                          yes_mid=(top.yes_bid + top.yes_ask) / 2.0,
                          bid=bid, ask=ask, price=price, improved=improved,
                          ahead=ahead, model_p=model_p, level=ahead)
            orders.append(order)
            active.append(order)
            push(order.cancel, 'cancel', order)

    def fill(order: Order, mode: str, now: float, top: Top) -> None:
        order.fills[mode] = now
        order.mid_fill[mode] = top.mid(order.side)
        for horizon in policy.horizons:
            push(now + horizon, 'mark', (order, mode, horizon))

    def check(now: float) -> None:
        top = book.top(now)
        if top is None:
            return
        for order in list(active):
            _, ask = top.side(order.side)
            through = math.isfinite(ask) and ask <= order.price + EPS
            if through and CONSERVATIVE not in order.fills:
                fill(order, CONSERVATIVE, now, top)
            if book.queue and QUEUE not in order.fills:
                size = book.level(order.side, order.price)
                order.depleted += max(0.0, order.level - size)
                order.level = size
                if through or order.depleted > order.ahead + MIN_SIZE:
                    fill(order, QUEUE, now, top)
            if all(m in order.fills for m in modes):
                active.remove(order)

    def run_due(before: float) -> None:
        while heap and heap[0][0] < before:
            when, _, kind, payload = heapq.heappop(heap)
            if kind == 'decide':
                decide(when, payload)
            elif kind == 'cancel':
                if payload in active:
                    active.remove(payload)
            else:
                order, mode, horizon = payload
                top = book.top(when)
                order.marks[(mode, horizon)] = (top.mid(order.side) if top is not None
                                                else math.nan)

    for t, payload in updates:
        run_due(t)
        book.apply(t, payload)
        if active:
            check(t)
    run_due(math.inf)
    return orders


# ---- input 1: the WebSocket frames ---------------------------------------------

def stream_updates(rows: pd.DataFrame) -> Iterator[tuple[str, str, list]]:
    """(market_ticker, symbol, [(time, BookEvent)]) per market, in arrival order.

    Rows regroup on (market_ticker, event_time, seq); `core/spool` gives the
    reason. `seq` goes to None: one market replayed alone is not a connection,
    so the per-connection gap check does not apply.
    """
    if rows is None or not len(rows):
        return
    frame = rows.copy()
    frame['t'] = _seconds(frame['event_time'])
    frame['seq'] = pd.to_numeric(frame['seq'], errors='coerce')
    frame = frame.sort_values(['market_ticker', 't', 'seq'], kind='stable')
    for ticker, part in frame.groupby('market_ticker', sort=True):
        times = part['t'].to_numpy(dtype=float)
        seqs = part['seq'].to_numpy(dtype=float)
        kinds = part['kind'].astype(str).to_numpy()
        sides = part['side'].astype(str).to_numpy()
        prices = pd.to_numeric(part['price'], errors='coerce').to_numpy(dtype=float)
        sizes = pd.to_numeric(part['size'], errors='coerce').to_numpy(dtype=float)
        absolute = (part['absolute'].astype(bool).to_numpy()
                    if 'absolute' in part.columns else np.ones(len(part), bool))
        same_seq = (seqs[1:] == seqs[:-1]) | (np.isnan(seqs[1:]) & np.isnan(seqs[:-1]))
        starts = np.flatnonzero(np.concatenate(
            [[True], (times[1:] != times[:-1]) | ~same_seq]))
        ends = np.append(starts[1:], len(part))
        updates = []
        for start, end in zip(starts, ends):
            yes = [(prices[i], sizes[i]) for i in range(start, end) if sides[i] == 'yes']
            no = [(prices[i], sizes[i]) for i in range(start, end) if sides[i] == 'no']
            updates.append((times[start], BookEvent(
                venue=VENUE, market_ticker=ticker, kind=kinds[start],
                received=times[start], seq=None, yes=yes, no=no,
                absolute=bool(absolute[start]))))
        yield ticker, str(part['symbol'].iloc[0]), updates


# ---- input 2: the Predexon snapshots ---------------------------------------------

def snapshot_updates(rows: pd.DataFrame) -> Iterator[tuple[str, str, pd.Timestamp, list]]:
    """(market_id, symbol, window_open, [(time, Top)]) per market, by `ts`.

    `ts` is the snapshot time in ms; prices are integer cents, YES-denominated.
    """
    if rows is None or not len(rows):
        return
    frame = rows.copy()
    frame['t'] = pd.to_numeric(frame['ts'], errors='coerce') / 1000.0
    frame = frame.dropna(subset=['t']).sort_values(['market_id', 't'], kind='stable')
    frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
    for ticker, part in frame.groupby('market_id', sort=True):
        bids = pd.to_numeric(part['best_bid'], errors='coerce').to_numpy(dtype=float)
        asks = pd.to_numeric(part['best_ask'], errors='coerce').to_numpy(dtype=float)
        times = part['t'].to_numpy(dtype=float)
        updates = [(times[i], Top(round(bids[i] / CENTS, 4), round(asks[i] / CENTS, 4)))
                   for i in range(len(part))]
        yield ticker, str(part['symbol'].iloc[0]), part['window_open'].iloc[0], updates


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [p.strftime('%Y-%m') for p in
            pd.period_range(start.tz_localize(None), end.tz_localize(None), freq='M')]


def _archive_rows(path: Path, symbol: str, start, end) -> pd.DataFrame:
    """Packed snapshots from a not-yet-consolidated archive partition."""
    from research.collect.fetchers import FIELDS

    best: dict = {}
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
                opened = pd.Timestamp(rec['window_open'])
                key = (rec['market_id'], opened)
                series = rec.get('series') or []
            except (ValueError, KeyError, TypeError):
                continue
            if not start <= opened < end:
                continue
            if key not in best or len(series) > len(best[key]):
                best[key] = series
    rows = []
    for (market_id, opened), series in best.items():
        for snap in series:
            row = dict(zip(FIELDS, snap))
            rows.append({'symbol': symbol, 'market_id': market_id,
                         'window_open': opened, 'ts': row.get('ts'),
                         'best_bid': row.get('best_bid'),
                         'best_ask': row.get('best_ask')})
    return pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))


def load_snapshots(root: Path | str, symbols: Sequence[str], start: pd.Timestamp,
                   end: pd.Timestamp) -> pd.DataFrame:
    """Kalshi snapshots with window_open in [start, end), derived layer first.

    A month with no `derived` partition falls back to the gzipped archive.
    """
    root = Path(root)
    parts = []
    for symbol in symbols:
        for month in _months(start, end):
            derived = (root / 'derived' / f'venue={VENUE}' / f'symbol={symbol}'
                       / f'month={month}' / 'data.parquet')
            archive = (root / 'archive' / f'venue={VENUE}' / f'symbol={symbol}'
                       / f'month={month}')
            if derived.exists():
                import pyarrow.parquet as pq
                table = pq.read_table(
                    derived, columns=[c for c in SNAPSHOT_COLUMNS if c != 'symbol'],
                    filters=[('window_open', '>=', start.to_pydatetime()),
                             ('window_open', '<', end.to_pydatetime())])
                frame = table.to_pandas()
                frame['symbol'] = symbol
                parts.append(frame)
            else:
                for path in sorted(archive.glob('windows.jsonl*')):
                    parts.append(_archive_rows(path, symbol, start, end))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=list(SNAPSHOT_COLUMNS))
    return pd.concat(parts, ignore_index=True)[list(SNAPSHOT_COLUMNS)]


# ---- settlement, windows, model ----------------------------------------------------

@dataclass(frozen=True)
class Settlements:
    settled_up: dict      # market_ticker -> bool
    window_open: dict     # market_ticker -> pd.Timestamp

    @classmethod
    def of(cls, frame: Optional[pd.DataFrame]) -> 'Settlements':
        if frame is None or not len(frame):
            return cls({}, {})
        rows = frame
        if 'venue' in rows.columns:
            rows = rows[rows['venue'].astype(str).str.lower() == VENUE]
        rows = rows.dropna(subset=['market_ticker'])
        up = {t: bool(v) for t, v in zip(rows['market_ticker'], rows['settled_up'])
              if v is not None and not (isinstance(v, float) and math.isnan(v))}
        opens = dict(zip(rows['market_ticker'],
                         pd.to_datetime(rows['window_open'], utc=True)))
        return cls(up, opens)

    def window_of(self, ticker: str) -> Optional[pd.Timestamp]:
        """The window a ticker settles, from the store or else from its name."""
        if ticker in self.window_open:
            return self.window_open[ticker]
        from research.collect.fetchers import kalshi_window_open
        try:
            return pd.Timestamp(kalshi_window_open(ticker))
        except ValueError:
            return None


def load_model(path: Path | str, column: str = 'p_up') -> dict:
    """{(symbol, window_open, offset): P(up)} from a CSV or Parquet file."""
    path = Path(path)
    frame = (pd.read_parquet(path) if path.suffix in ('.parquet', '.pq')
             else pd.read_csv(path))
    offset = 'offset' if 'offset' in frame.columns else 'offset_minutes'
    missing = {'symbol', 'window_open', offset, column} - set(frame.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    opens = pd.to_datetime(frame['window_open'], utc=True)
    return {(s, w, int(o)): float(p) for s, w, o, p in
            zip(frame['symbol'], opens, frame[offset], frame[column])}


# ---- per-order rows ------------------------------------------------------------

def order_rows(orders: Sequence[Order], settlements: Settlements, policy: Policy,
               config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per (order, fill mode). P&L in dollars per contract."""
    config = policy.config(config)
    rows = []
    for o in orders:
        settled = settlements.settled_up.get(o.market_ticker)
        payout = (math.nan if settled is None
                  else float(settled if o.side == 'UP' else not settled))
        t_fee = taker_fee(o.ask, policy.contracts, config)
        m_fee = maker_fee(o.price, policy.contracts, config,
                          policy.maker_fee_multiplier)
        modes = (CONSERVATIVE, QUEUE) if o.input == 'stream' else (CONSERVATIVE,)
        for mode in modes:
            filled = mode in o.fills
            mid_fill = o.mid_fill.get(mode, math.nan)
            row = {
                'input': o.input, 'mode': mode, 'symbol': o.symbol,
                'market_ticker': o.market_ticker, 'window_open': o.window_open,
                'offset': o.offset, 'side': o.side,
                'decided': pd.Timestamp(o.decided, unit='s', tz='UTC'),
                'yes_mid': o.yes_mid, 'side_mid': o.side_mid, 'bid': o.bid,
                'ask': o.ask, 'price': o.price, 'improved': o.improved,
                'ahead': o.ahead, 'model_p': o.model_p, 'filled': filled,
                'fill_delay': (o.fills[mode] - o.decided) if filled else math.nan,
                'mid_fill': mid_fill, 'payout': payout,
                'maker_fee': m_fee,
                'pnl': (payout - o.price - m_fee) if filled else math.nan,
                'taker_fee': t_fee, 'taker_pnl': payout - o.ask - t_fee,
            }
            for horizon in policy.horizons:
                mid = o.marks.get((mode, horizon), math.nan)
                row[f'mid_{horizon}s'] = mid
                row[f'adverse_{horizon}s'] = mid_fill - mid
            row['adverse_settle'] = mid_fill - payout
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame):
        frame['mid_bucket'] = pd.cut(frame['side_mid'], MID_EDGES, right=False)
    return frame


# ---- statistics --------------------------------------------------------------------

def cluster_bootstrap(values, clusters, *, n_boot: int = 5000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float, float]:
    """Mean, and a percentile CI from resampling whole clusters with replacement."""
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if not keep.any():
        return math.nan, math.nan, math.nan
    values = values[keep]
    codes, _ = pd.factorize(pd.Series(np.asarray(clusters)[keep]))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    mean = float(values.mean())
    k = len(sums)
    if k < 2 or n_boot <= 0:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = []
    for done in range(0, n_boot, 500):
        draw = rng.integers(0, k, size=(min(500, n_boot - done), k))
        stats.append(sums[draw].sum(axis=1) / counts[draw].sum(axis=1))
    lo, hi = np.quantile(np.concatenate(stats), [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(lo), float(hi)


def summarise(rows: pd.DataFrame, by: Optional[str] = None, *,
              horizons: Sequence[int] = DEFAULT_HORIZONS, n_boot: int = 5000,
              seed: int = 0) -> pd.DataFrame:
    """Fill rate, adverse selection and P&L (cents per contract) per group.

    One fill mode at a time. Adverse selection is mid at fill minus mid later,
    for the side bought: positive means the price moved against us. Maker P&L
    is over filled, settled orders. Taker P&L is over every settled order, and
    `taker_filled_c` over the maker-filled ones only, which separates the
    price improvement from the selection of which orders fill.
    """
    if not len(rows):
        return pd.DataFrame()
    if rows['mode'].nunique() > 1:
        raise ValueError('summarise one fill mode at a time')
    groups = ([('all', rows)] if by is None
              else [(str(k), g) for k, g in rows.groupby(by, observed=True, sort=True)])
    out = []
    for name, g in groups:
        filled = g[g['filled']]
        maker = cluster_bootstrap(filled['pnl'], filled['window_open'],
                                  n_boot=n_boot, seed=seed)
        taker = cluster_bootstrap(g['taker_pnl'], g['window_open'],
                                  n_boot=n_boot, seed=seed)
        row = {by or 'group': name, 'orders': len(g), 'fills': len(filled),
               'fill_rate': len(filled) / len(g) if len(g) else math.nan}
        for horizon in horizons:
            row[f'adv_{horizon}s_c'] = filled[f'adverse_{horizon}s'].mean() * CENTS
        row['adv_settle_c'] = filled['adverse_settle'].mean() * CENTS
        row['maker_n'] = int(filled['pnl'].notna().sum())
        row['maker_c'], row['maker_lo_c'], row['maker_hi_c'] = (v * CENTS for v in maker)
        row['taker_n'] = int(g['taker_pnl'].notna().sum())
        row['taker_c'], row['taker_lo_c'], row['taker_hi_c'] = (v * CENTS for v in taker)
        row['taker_filled_c'] = filled['taker_pnl'].mean() * CENTS
        out.append(row)
    return pd.DataFrame(out)


def report(rows: pd.DataFrame, policy: Policy, *, input: str, skips: Counter,
           n_boot: int = 5000, seed: int = 0) -> str:
    """Every table, each headed by the input and the fill mode that produced it."""
    lines = []
    if skips:
        lines.append('decisions skipped: ' + ', '.join(
            f'{k} {v:,}' for k, v in sorted(skips.items())))
    if not len(rows):
        lines.append('no simulated orders')
        return '\n'.join(lines)
    for mode in rows['mode'].unique():
        part = rows[rows['mode'] == mode]
        for by in (None, 'mid_bucket', 'offset', 'symbol', 'side'):
            table = summarise(part, by, horizons=policy.horizons, n_boot=n_boot,
                              seed=seed)
            title = 'all orders' if by is None else f'by {by}'
            lines.append(f'\n== input={input}  fill={mode}  {title} ==')
            lines.append(table.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    lines.append('\nP&L and adverse selection in cents per contract; CI is a '
                 f'{n_boot:,}-resample block bootstrap by window_open.')
    return '\n'.join(lines)


# ---- driver --------------------------------------------------------------------------

def simulate_stream(events: pd.DataFrame, settlements: Settlements, policy: Policy,
                    *, model: Optional[dict] = None, skips: Optional[Counter] = None,
                    ids: Optional[Iterator[int]] = None) -> list[Order]:
    skips = Counter() if skips is None else skips
    orders = []
    for ticker, symbol, updates in stream_updates(events):
        window_open = settlements.window_of(ticker)
        if window_open is None:
            skips['market with no window'] += 1
            continue
        orders += simulate_market(
            updates, StreamBook(ticker, policy.max_book_age), input='stream',
            symbol=symbol, ticker=ticker, window_open=window_open, policy=policy,
            model=model, skips=skips, ids=ids)
    return orders


def simulate_snapshots(snapshots: pd.DataFrame, policy: Policy, *,
                       model: Optional[dict] = None, skips: Optional[Counter] = None,
                       ids: Optional[Iterator[int]] = None) -> list[Order]:
    skips = Counter() if skips is None else skips
    orders = []
    for ticker, symbol, window_open, updates in snapshot_updates(snapshots):
        orders += simulate_market(
            updates, SnapshotBook(policy.max_book_age), input='snapshots',
            symbol=symbol, ticker=ticker, window_open=window_open, policy=policy,
            model=model, skips=skips, ids=ids)
    return orders


def _chunks(start: pd.Timestamp, end: pd.Timestamp, minutes: int):
    edge = start.floor('15min')
    while edge < end:
        stop = min(edge + pd.Timedelta(minutes=minutes), end)
        yield edge, stop
        edge = stop


def run(args) -> tuple[pd.DataFrame, Counter, Policy]:
    from core.datastore import DataStoreError, ResearchStore

    policy = Policy(
        select=args.select, offsets=tuple(args.offsets), delay_seconds=args.delay_seconds,
        rest_seconds=args.rest_seconds, contracts=args.contracts, improve=args.improve,
        band=tuple(args.band), horizons=tuple(args.horizons),
        max_book_age=args.max_book_age, min_edge=args.min_edge,
        maker_fee_rate=args.maker_fee,
        maker_fee_multiplier=args.maker_fee_multiplier)
    end = _utc(args.end) or pd.Timestamp.now(tz='UTC').floor('15min')
    start = _utc(args.start) or end - pd.Timedelta(days=1)
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    store = ResearchStore(args.store or os.getenv('RESEARCH_STORE'))
    try:
        settled = store.read('venue_settlements', venue=VENUE, symbols=symbols,
                             start=start - pd.Timedelta(hours=1),
                             end=end + pd.Timedelta(hours=1), min_quality=None)
    except DataStoreError as exc:
        logger.warning('venue_settlements unavailable (%s): P&L will be empty', exc)
        settled = pd.DataFrame()
    settlements = Settlements.of(settled)
    model = load_model(args.model_file, args.model_column) if args.model_file else None
    if policy.select == 'model' and model is None:
        raise SystemExit('--select model needs --model-file')

    skips: Counter = Counter()
    ids = itertools.count()
    orders: list[Order] = []
    for lo, hi in _chunks(start, end, args.chunk_minutes):
        if args.input == 'stream':
            events = store.read('venue_book_events', venue=VENUE, symbols=symbols,
                                start=lo, end=hi - pd.Timedelta(microseconds=1),
                                columns=STREAM_COLUMNS, min_quality=None)
            orders += simulate_stream(events, settlements, policy, model=model,
                                      skips=skips, ids=ids)
        else:
            snaps = load_snapshots(args.collect_data, symbols, lo, hi)
            orders += simulate_snapshots(snaps, policy, model=model, skips=skips,
                                         ids=ids)
        logger.info('%s .. %s: %d orders so far', lo, hi, len(orders))
    return order_rows(orders, settlements, policy), skips, policy


def build_parser() -> argparse.ArgumentParser:
    def ints(text):
        return [int(v) for v in text.split(',') if v.strip()]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', choices=INPUTS, default='stream')
    parser.add_argument('--store', default=None, help='defaults to $RESEARCH_STORE')
    parser.add_argument('--collect-data', default=None,
                        help='snapshots root; defaults to $COLLECT_DATA')
    parser.add_argument('--symbols', default=','.join(DEFAULT_CONFIG.symbols))
    parser.add_argument('--start', default=None, help='ISO time (default: 24h before --end)')
    parser.add_argument('--end', default=None, help='ISO time, exclusive (default now)')
    parser.add_argument('--chunk-minutes', type=int, default=None,
                        help='read size, a multiple of 15 (default: 60 for stream, '
                             '1440 for snapshots)')
    parser.add_argument('--select', choices=SELECTS, default='bias')
    parser.add_argument('--band', type=float, nargs=2, default=(0.65, 0.95),
                        metavar=('LO', 'HI'),
                        help='bias mode: side mid band; the mirror band buys the other side')
    parser.add_argument('--model-file', default=None, help='CSV or Parquet')
    parser.add_argument('--model-column', default='p_up')
    parser.add_argument('--min-edge', type=float, default=0.0,
                        help='model mode: P(side) - post price - maker fee must exceed this')
    parser.add_argument('--offsets', type=ints,
                        default=list(DEFAULT_CONFIG.decision_offsets))
    parser.add_argument('--delay-seconds', type=float, default=3.0)
    parser.add_argument('--rest-seconds', type=float, default=20.0)
    parser.add_argument('--contracts', type=int, default=1)
    parser.add_argument('--improve', action='store_true',
                        help='post one tick above the best bid when that does not cross')
    parser.add_argument('--horizons', type=ints, default=list(DEFAULT_HORIZONS),
                        help='mark-out seconds after a fill')
    parser.add_argument('--max-book-age', type=float, default=None,
                        help='seconds since the last update before a book is not used '
                             '(default: 60 for stream, 900 for snapshots, which only '
                             'change when the book changes)')
    parser.add_argument('--maker-fee', type=float, default=DEFAULT_CONFIG.maker_fee_rate,
                        help='dollars per contract (default: config maker_fee_rate)')
    parser.add_argument('--maker-fee-multiplier', type=float, default=None,
                        metavar='M',
                        help='use the venue schedule round_up(M * 0.0175 * C * P '
                             '* (1-P)) instead of the flat --maker-fee')
    parser.add_argument('--boot', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None, help='per-order rows, .csv or .parquet')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)-7s %(message)s')
    stream = args.input == 'stream'
    if args.chunk_minutes is None:
        args.chunk_minutes = 60 if stream else 1440
    if args.max_book_age is None:
        args.max_book_age = 60.0 if stream else 900.0
    if args.chunk_minutes % 15:
        raise SystemExit('--chunk-minutes must be a multiple of 15')
    if args.input == 'snapshots' and args.collect_data is None:
        args.collect_data = (os.getenv('COLLECT_DATA')
                             or str(Path(__file__).resolve().parents[2] / 'data' / 'collection'))
    rows, skips, policy = run(args)
    print(f'maker fill simulation: input={args.input} select={policy.select} '
          f'offsets={list(policy.offsets)} delay={policy.delay_seconds:g}s '
          f'rest={policy.rest_seconds:g}s contracts={policy.contracts} '
          f'improve={policy.improve} maker_fee='
          + (f'quadratic M={policy.maker_fee_multiplier:g}'
             if policy.maker_fee_multiplier is not None
             else f'${policy.maker_fee_rate:.4f} flat'))
    if args.input == 'snapshots':
        print('snapshots: change-driven touch only; conservative fill rule only; '
              'tail prices on a whole-cent grid')
    print(report(rows, policy, input=args.input, skips=skips, n_boot=args.boot,
                 seed=args.seed))
    if args.out and len(rows):
        out = rows.assign(mid_bucket=rows['mid_bucket'].astype(str))
        if args.out.endswith('.parquet'):
            out.to_parquet(args.out, index=False)
        else:
            out.to_csv(args.out, index=False)
        print(f'\nwrote {len(rows):,} rows to {args.out}')
    return 0 if len(rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
