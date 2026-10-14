"""The books: reconcile with the venue, settle, adopt its balance, record equity.

Live, the venue is the account of record. Its balance, settlements and fills
win over our own arithmetic, which is kept alongside to show the disagreement.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from core import venue_ledger
from core.config import Config, SERIES_BY_SYMBOL
from core.pg_writer import PgWriter
from core.windows import bar_mean
from data_collection.kalshi_client import (
    KalshiClient, KalshiError, parse_fill, parse_settlement,
)
from livetrade import state as live_state
from livetrade.util import _db

logger = logging.getLogger('live')

# Minute prices older than this are dropped from the serving store each cycle.
PRICE_RETENTION_HOURS = 48
# Venue balance samples older than this are dropped on the same cadence. It
# matches the Account page's 30-day chart. Settlements are never pruned.
BALANCE_RETENTION_DAYS = 30

# The last shard observed on a book. The balance query needs it before this
# cycle's book is read.
_LAST_EXCHANGE_INDEX: Optional[int] = None


def record_minute_prices(writer: PgWriter, bars: dict[str, pd.DataFrame],
                         *, hours: int = 6) -> int:
    """Store the last few hours of bars so the dashboard can draw the path."""
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
    rows = []
    for symbol, frame in bars.items():
        recent = frame.loc[frame['event_time'] >= cutoff]
        for bar in recent.itertuples():
            rows.append({
                'symbol': symbol, 'minute': bar.event_time.to_pydatetime(),
                'open': float(bar.open), 'high': float(bar.high),
                'low': float(bar.low), 'close': float(bar.close),
            })
    written = writer.write_minute_prices(rows) if rows else 0
    now = pd.Timestamp.now(tz='UTC')
    writer.prune_minute_prices(
        (now - pd.Timedelta(hours=PRICE_RETENTION_HOURS)).to_pydatetime())
    writer.prune_venue_balances(
        (now - pd.Timedelta(days=BALANCE_RETENTION_DAYS)).to_pydatetime())
    return written


def as_utc(value) -> pd.Timestamp:
    """Return a UTC timestamp from an aware or naive value.

    Postgres returns aware values and SQLite naive ones. Everything stored is
    UTC, so a naive value is localised.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tz is None else stamp.tz_convert('UTC')


def venue_settled_up(row: dict, side: str) -> Optional[bool]:
    """Return whether the up side won, per the venue's settlement row.

    Reads an explicit result if present, else infers it from revenue and the
    side held. Returns None when neither is legible.
    """
    for key in ('market_result', 'result', 'settlement_result'):
        value = row.get(key)
        if isinstance(value, str) and value.strip().lower() in ('yes', 'no'):
            return value.strip().lower() == 'yes'
    revenue = row.get('revenue_dollars', row.get('revenue'))
    try:
        revenue = float(revenue)
    except (TypeError, ValueError):
        return None
    # Revenue is what the position paid out. A paid-out `up` holding means yes
    # won; a paid-out `down` holding means it lost.
    won = revenue > 0.0
    if side == 'up':
        return won
    if side == 'down':
        return not won
    return None


def resolve_window(strike: float, settle_time, symbol: str,
                   bars: dict[str, pd.DataFrame]) -> tuple[Optional[bool], float]:
    """Return whether the window settled up under the training rule, and the value.

    Uses the `(O+H+L+C)/4` mean of the minute ending at `settle_time`, compared
    with `>=` because the venue's `strike_type` is `greater_or_equal`. This
    must match `core/windows.py`. Returns `(None, nan)` when the bar is missing.
    """
    frame = bars.get(symbol)
    if frame is None:
        return None, float('nan')
    minute = as_utc(settle_time) - pd.Timedelta(minutes=1)
    row = frame.loc[frame['event_time'] == minute]
    if row.empty:
        return None, float('nan')
    value = float(bar_mean(row).iloc[0])
    if not np.isfinite(value):
        return None, float('nan')
    return value >= strike, value


def settle_predictions(writer: PgWriter, bars: dict[str, pd.DataFrame],
                       *, now: Optional[datetime] = None) -> int:
    """Record the realised outcome on every settled window, traded or not.

    The market benchmark needs outcomes on all windows, not only the traded
    ones that `settle_due` touches.
    """
    now = now or datetime.now(timezone.utc)
    filled = 0
    for symbol, window_open, settle_time, strike in writer.windows_awaiting_outcome(now):
        settled_up, _ = resolve_window(strike, settle_time, symbol, bars)
        if settled_up is None:
            continue
        filled += writer.set_window_outcome(symbol, window_open, settled_up=settled_up)
    if filled:
        logger.info('recorded the outcome on %d prediction row(s)', filled)
    return filled


def settle_due(writer: PgWriter, bars: dict[str, pd.DataFrame],
               *, venue_settlements: Optional[dict[str, dict]] = None,
               ) -> list[tuple[int, float]]:
    """Settle every matured position on the venue's rule.

    The venue's settlement wins where available. Ours uses Coinbase bars as a
    proxy for CF Benchmarks BRTI, so every disagreement is logged.
    """
    now = datetime.now(timezone.utc)
    venue_settlements = venue_settlements or {}
    settled: list[tuple[int, float]] = []
    for position in writer.positions_due(now):
        strike = _strike_for(writer, position)
        if strike is None:
            logger.error('%s window %s: no strike recorded, cannot settle',
                         position.symbol, position.window_open)
            continue

        # The venue first, where it has settled.
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        from_venue = None
        if ticker and ticker in venue_settlements:
            from_venue = venue_settled_up(venue_settlements[ticker], position.side)

        # Ours: the mean over the minute ending at settle_time, and `>=`.
        settle_minute = as_utc(position.settle_time) - pd.Timedelta(minutes=1)
        from_bars, settle_price = resolve_window(
            strike, position.settle_time, position.symbol, bars)

        if from_venue is not None and from_bars is not None and from_venue != from_bars:
            logger.warning(
                '%s window %s: the venue settled %s and our bars say %s '
                '(mean %.4f vs strike %.4f). Taking the venue. A persistent '
                'disagreement here is the Coinbase-vs-BRTI basis, measured.',
                position.symbol, position.window_open,
                'up' if from_venue else 'down', 'up' if from_bars else 'down',
                settle_price, strike)

        settled_up = from_venue if from_venue is not None else from_bars
        if settled_up is None:
            logger.warning(
                '%s window %s: neither the venue nor a bar at %s can settle this '
                'yet, waiting', position.symbol, position.window_open,
                settle_minute)
            continue

        pnl = writer.settle_position(position.id, settled_up=bool(settled_up))
        if pnl is not None:
            settled.append((position.id, pnl))
            logger.info('settled %s %s: %s (%s) vs strike %.4f -> %+.2f',
                        position.symbol, position.window_open,
                        'up' if settled_up else 'down',
                        'venue' if from_venue is not None else f'mean {settle_price:.4f}',
                        strike, pnl)
    return settled


def _strike_for(writer: PgWriter, position) -> Optional[float]:
    from core.pg_writer import Prediction
    with writer._session() as session:  # noqa: SLF001 - same package, one query
        row = (session.query(Prediction)
               .filter(Prediction.symbol == position.symbol,
                       Prediction.window_open == position.window_open)
               .order_by(Prediction.offset_minutes)
               .first())
        return float(row.strike) if row is not None else None


def _ticket_for(writer: PgWriter, position):
    from core.pg_writer import OrderTicket

    with writer._session() as session:  # noqa: SLF001 - same package, one query
        return (session.query(OrderTicket)
                .filter(OrderTicket.symbol == position.symbol,
                        OrderTicket.window_open == position.window_open)
                .one_or_none())


class VenueState(NamedTuple):
    """The venue's settlements and balance, read before we settle.

    The balance is adopted only after settlement (see `adopt_venue_balance`).
    """

    settlements: dict[str, dict]
    balance: float


def adopt_venue_balance(writer: PgWriter, venue_balance: float, *,
                        exchange_index: Optional[int] = None,
                        credited_this_cycle: float = 0.0) -> None:
    """Adopt the venue's balance as ours. Call this after settling.

    The venue credits a settlement at once, so adopting before `settle_due`
    counts a payout twice. Any drift from our figure is logged before the
    overwrite; a growing drift means an unrecorded fill, a partial, or a
    mispriced fee.
    """
    if not np.isfinite(venue_balance):
        # Never write an unreadable balance over a readable one.
        logger.error('the venue did not return a readable balance; keeping our own '
                     'figure this cycle rather than overwriting it')
        return
    account = writer.account()
    if account is None:
        return
    ours = float(account.bankroll)
    drift = venue_balance - ours
    # A payout we credited this cycle is not a drift: the venue balance is
    # sampled at the top of the cycle, one settlement behind.
    explained = abs(abs(drift) - abs(float(credited_this_cycle or 0.0))) <= 0.01
    if abs(drift) > 0.01 and explained:
        logger.info('balance lags by %+.2f, which is exactly what we credited '
                    'this cycle — the venue posts settlements one cycle behind. '
                    'Adopting theirs.', drift)
    elif abs(drift) > 0.01:
        logger.warning(
            'balance drift: ours $%.2f, venue $%.2f (%+.2f). The venue is the '
            'account of record — writing theirs. A drift that grows means a '
            'fill we did not record, a partial, or a mispriced fee.',
            ours, venue_balance, drift)
    # Sample before the overwrite, so the row keeps both figures at one instant.
    try:
        writer.write_venue_balance(
            timestamp=datetime.now(timezone.utc), balance=venue_balance,
            exchange_index=exchange_index, our_bankroll=ours)
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not sample the venue balance this cycle')
    writer.update_account(bankroll=venue_balance)


def remember_exchange_index(quotes: dict) -> Optional[int]:
    """Remember the shard from the book just read, and warn on a change.

    Balances are local to a shard, so a stale shard reads the wrong balance.
    """
    global _LAST_EXCHANGE_INDEX
    seen = venue_exchange_index(quotes)
    if seen is None:
        return _LAST_EXCHANGE_INDEX
    if _LAST_EXCHANGE_INDEX is not None and seen != _LAST_EXCHANGE_INDEX:
        logger.warning(
            'the venue moved these markets from exchange shard %s to %s. This '
            'cycle reconciled against the old one; balances are per-shard, so '
            'treat this cycle\'s balance as suspect.',
            _LAST_EXCHANGE_INDEX, seen)
    _LAST_EXCHANGE_INDEX = seen
    return seen


def venue_exchange_index(quotes: dict) -> Optional[int]:
    """Return the exchange shard the traded markets live on, per the venue.

    Balances are local to a shard, so an order draws only on its market's
    shard. Read from the quotes, never configured. Returns None when no book
    was read, which means the whole-account total.
    """
    # `fetch_quotes` stores `(Quote, ticker)`, not a bare Quote; accept both.
    candidates = [v[0] if isinstance(v, tuple) else v for v in quotes.values()]
    seen = {int(q.exchange_index) for q in candidates
            if getattr(q, 'exchange_index', None) is not None}
    if not seen:
        return None
    if len(seen) > 1:
        # Narrowing to one shard would understate the others' balance.
        logger.warning('markets span exchange shards %s; using the account total',
                       sorted(seen))
        return None
    return seen.pop()


def persist_venue_ledger(writer: PgWriter, *, fills: list[dict],
                         settlements: list[dict]) -> tuple[int, int]:
    """Store the venue's fills and settlements. Idempotent on the venue's keys.

    Covers only the live tier; `scripts.sync_venue` backfills the historical
    tier. Logs exceptions instead of raising, so the cycle still trades.
    """
    written_fills = written_settlements = 0
    try:
        # Write only new rows. A settlement stored with a null `pnl` is not in
        # the skip set, so an incomplete parse is retried.
        known_fills, known_settlements = writer.venue_ledger_keys()
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not read the stored venue ledger; writing all rows')
        known_fills, known_settlements = set(), set()

    try:
        rows = []
        for raw in fills:
            parsed = parse_fill(raw)
            # Only markets this system trades.
            if not _is_ours(getattr(parsed, 'ticker', '') or ''):
                continue
            if parsed.trade_id and parsed.trade_id not in known_fills:
                rows.append(venue_ledger.fill_row(parsed))
        written_fills = writer.upsert_venue_fills(rows)
    except Exception:  # noqa: BLE001
        logger.exception('could not store the venue fills this cycle')
    try:
        rows = []
        for raw in settlements:
            parsed = parse_settlement(raw)
            if not parsed.ticker or parsed.ticker in known_settlements:
                continue
            # Only markets this system trades. The account may hold others,
            # which must not enter this equity curve.
            if not _is_ours(parsed.ticker):
                continue
            rows.append(venue_ledger.settlement_row(
                parsed, position=writer.position_for_ticker(parsed.ticker)))
        written_settlements = writer.upsert_venue_settlements(rows)
    except Exception:  # noqa: BLE001
        logger.exception('could not store the venue settlements this cycle')
    if written_fills or written_settlements:
        logger.debug('venue ledger: %d fill(s), %d settlement(s) stored',
                     written_fills, written_settlements)
    return written_fills, written_settlements


def _is_ours(ticker: str) -> bool:
    """Return whether the ticker is in a series this system trades."""
    return venue_ledger.is_ours(ticker, SERIES_BY_SYMBOL.values())


async def reconcile_with_venue(writer: PgWriter, kalshi: KalshiClient, *,
                               exchange_index: Optional[int] = None) -> VenueState:
    """Read and store the venue's ledger, and compare open positions with ours.

    Returns the balance and settlements; `adopt_venue_balance` writes the
    balance after settlement.
    """
    state = await kalshi.reconcile(exchange_index=exchange_index)
    venue_balance = float(state['balance'])
    persist_venue_ledger(writer, fills=list(state.get('fills') or []),
                         settlements=list(state.get('settlements') or []))

    # Settlements keyed on market ticker, for `settle_due`.
    resolved: dict[str, dict] = {}
    for row in state.get('settlements', []):
        ticker = str(row.get('ticker', ''))
        if not ticker:
            continue
        resolved[ticker] = row
    if resolved:
        logger.debug('venue reports %d settlement(s) to reconcile', len(resolved))

    # The size field is `position_fp`, a fixed-point string, so use
    # `KalshiClient.position_size`. Only the series this loop trades count.
    venue_open = {str(p.get('ticker', '')) for p in state.get('positions', [])
                  if KalshiClient.position_size(p) != 0
                  and _is_ours(str(p.get('ticker', '')))}
    for position in writer.open_positions():
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        if ticker and ticker not in venue_open and ticker not in resolved:
            logger.warning(
                '%s window %s: we hold %d contracts the venue does not report. '
                'Most likely the order never filled — a fill_or_kill that killed '
                'leaves a ticket and no position.',
                position.symbol, position.window_open, position.contracts)

    # The reverse: a position the venue holds and we do not. A POST that timed
    # out after acceptance leaves this.
    ours = set()
    for position in writer.open_positions():
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        if ticker:
            ours.add(ticker)
    for ticker in sorted(venue_open - ours):
        logger.error(
            'the venue reports an open position in %s that we have no record of. '
            'An order was filled and not booked — most likely a POST that timed '
            'out after being accepted. Reconcile by hand before trading again.',
            ticker)
    return VenueState(settlements=resolved, balance=venue_balance)


async def bookkeeping(args, config: Config, writer: PgWriter, kalshi,
                      bars: dict, *, phase=None) -> list:
    """Reconcile with the venue, settle, record outcomes, adopt its balance.

    Settles first and adopts the balance last, so a payout is not counted twice.
    """
    venue_settlements: dict[str, dict] = {}
    venue_balance = float('nan')
    shard = _LAST_EXCHANGE_INDEX
    if kalshi is not None and args.reconcile:
        try:
            venue = await reconcile_with_venue(writer, kalshi, exchange_index=shard)
            venue_settlements, venue_balance = venue.settlements, venue.balance
            live_state._LAST_RECONCILED = time.monotonic()
        except KalshiError as exc:
            logger.error('reconciliation failed (%s); falling back to our own '
                         'bookkeeping for this cycle', exc)
        if phase:
            phase('reconcile')
    settled = await _db(settle_due, writer, bars,
                        venue_settlements=venue_settlements)
    # Record outcomes on every window, held or not.
    await _db(settle_predictions, writer, bars)
    # Our credits are in, so any remaining drift is a real disagreement.
    if kalshi is not None and args.reconcile:
        await _db(adopt_venue_balance, writer, venue_balance, exchange_index=shard,
                  credited_this_cycle=sum(p for _, p in settled))
    if phase:
        phase('settle')
    return settled


def _write_equity_point(writer: PgWriter, now) -> None:
    account = writer.account()
    open_now = writer.open_positions()
    writer.write_equity_point(
        timestamp=now,
        equity=(account.bankroll + sum(p.outlay for p in open_now)) if account else 0.0,
        bankroll=account.bankroll if account else 0.0,
        staked=sum(p.outlay for p in open_now),
        open_positions=len(open_now),
        realized_pnl=account.realized_pnl if account else 0.0,
    )
