"""The books: reconcile with the venue, settle, adopt its balance, record equity.

Live, the venue is the account of record. Its balance, settlements and fills
are read every cycle and preferred over our own arithmetic, which is kept
alongside so the disagreement stays visible.
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
# Venue balance samples older than this are dropped on the same cadence. One a
# cycle is ~1,440 rows a day, so without it `venue_balance` grows forever. Kept
# as long as the Account page's 30-day chart reads; the settlements are the
# record and are never pruned.
BALANCE_RETENTION_DAYS = 30

# The last shard actually observed on a book. The balance query needs it, and
# the book is read after reconciliation, so it cannot come from this cycle's
# quotes.
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
    """A UTC timestamp, whether the source remembered the timezone or not.

    Postgres returns tz-aware values from `DateTime(timezone=True)` and SQLite
    naive ones, on which `tz_convert` raises. Everything stored is UTC, so
    localise when the tz is missing.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tz is None else stamp.tz_convert('UTC')


def venue_settled_up(row: dict, side: str) -> Optional[bool]:
    """Did the *up* side win, according to the venue's settlement row?

    The venue is the account of record, so this is preferred over any bar-derived
    answer. Kalshi has published more than one shape for this, so read an
    explicit result where there is one and fall back to inferring from revenue
    and the side we held. Returns None when neither is legible — the caller then
    falls back to bars and says so, rather than guessing.
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
    """Did this window settle up, per the trained rule? Plus the value used.

    The one implementation `settle_due` and `settle_predictions` share, so live
    settlement cannot drift from the training label again.

    The rule is `core/windows.py`'s: the `(O+H+L+C)/4` mean of the minute *ending*
    at `settle_time`, compared with `>=` because the venue's `strike_type` is
    `greater_or_equal`. Returns `(None, nan)` when the bar is not there yet.
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
    """Fill in the realised outcome on every settled window, traded or not.

    Without this the market benchmark cannot exist. `market_probability` is
    recorded on each decision — the venue's own mid — and scoring it needs the
    answer beside it. `settle_due` only touches windows we *hold*, which is the
    ~6% we chose to trade: exactly the selected sample that cannot answer "is the
    market's probability better than ours".
    """
    # Injectable so a test can place a window in the past without waiting for the
    # wall clock to agree.
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
    """Settle every matured position, on the venue's rule.

    **This must agree with `core/windows.py:build_windows` exactly**, because the
    model is trained against that label and the money is booked against this
    one — hence the shared `resolve_window`.

    The venue's own settlement wins wherever it is available. Ours is an OHLC
    mean of Coinbase standing in for sixty seconds of CF Benchmarks BRTI, which
    is a close proxy and not the same number; every disagreement is logged
    because a persistent one is the basis risk becoming measurable.
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

        # The venue first, where it knows.
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        from_venue = None
        if ticker and ticker in venue_settlements:
            from_venue = venue_settled_up(venue_settlements[ticker], position.side)

        # Ours: the mean over the minute ENDING at settle_time, and `>=`.
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

        # **The venue's own fee, not the published schedule's estimate.**
        # `Position.fee` is set at DECISION time, where `assume_maker` makes it
        # 0.00 -- correct for an order that rests, wrong for the ~19% that
        # cross. The settlement row knows what was actually charged.
        #
        # Parsed rather than read straight off the dict: the venue serves
        # `revenue` in integer cents and `fee_cost` in dollars, both unsuffixed,
        # and reading the fee through the cents fallback divides it by a
        # hundred -- a bug this repo has already had once.
        venue_fee = None
        if ticker and ticker in venue_settlements:
            try:
                venue_fee = parse_settlement(venue_settlements[ticker]).fee_cost
            except Exception:                              # noqa: BLE001
                logger.warning('%s: could not read the venue fee; keeping ours',
                               ticker)
        # **The venue's `fee_cost` is PER MARKET, and multi-entry puts several
        # positions on one.** Handing the whole figure to each of them charged
        # it once per position: measured 2026-10-20 on
        # KXBTC15M-26OCT201130-30, our fee came to 0.0714 against the venue's
        # 0.0357 -- exactly double, across two positions -- and that 3.5c was
        # the entire residual `pnl_gap` after everything else reconciled.
        #
        # Split by contracts, which is how the venue charges it. A market with
        # one position is unaffected: the share is 1.0.
        if venue_fee is not None:
            venue_fee = float(venue_fee) * _fee_share(writer, position)
        pnl = writer.settle_position(position.id, settled_up=bool(settled_up),
                                     fee=venue_fee)
        if pnl is not None:
            settled.append((position.id, pnl))
            logger.info('settled %s %s: %s (%s) vs strike %.4f -> %+.2f',
                        position.symbol, position.window_open,
                        'up' if settled_up else 'down',
                        'venue' if from_venue is not None else f'mean {settle_price:.4f}',
                        strike, pnl)
    return settled


def _fee_share(writer: PgWriter, position) -> float:
    """This position's share of its market's fee, by contracts.

    One position in the market -> 1.0, which is every row before multi-entry.
    A market whose positions hold no contracts at all splits evenly rather than
    dividing by zero; that only arises from the sub-contract rounding already
    fixed in `scripts/live.py`, and an even split is still closer than
    charging each of them the whole fee.
    """
    from core.pg_writer import Position

    with writer._session() as session:                     # noqa: SLF001
        rows = (session.query(Position)
                .filter(Position.symbol == position.symbol,
                        Position.window_open == position.window_open)
                .all())
    if len(rows) <= 1:
        return 1.0
    total = sum(float(r.contracts or 0.0) for r in rows)
    if total <= 0:
        return 1.0 / len(rows)
    return float(position.contracts or 0.0) / total


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
        # Matched on the offset too: a window holds one ticket per decision
        # offset once multi-entry is on, and `(symbol, window)` alone would
        # raise `MultipleResultsFound` inside the reconcile.
        return (session.query(OrderTicket)
                .filter(OrderTicket.symbol == position.symbol,
                        OrderTicket.window_open == position.window_open,
                        OrderTicket.offset_minutes == position.offset_minutes)
                .one_or_none())


class VenueState(NamedTuple):
    """What the venue says, read before we settle anything ourselves.

    The balance travels with the settlements rather than being written on the
    spot, because it must be adopted only after settlement — see
    `adopt_venue_balance`.
    """

    settlements: dict[str, dict]
    balance: float
    #: The cross-shard total, which is the figure the venue's own app shows.
    #: `balance` above is the trading shard alone. Defaulted so every existing
    #: construction site and test keeps working unchanged.
    total_balance: float = float('nan')
    #: The venue's OWN mark of the open book, from `/portfolio/balance`. Kept
    #: beside the cash so the account value on screen is Kalshi's arithmetic
    #: rather than ours marked at the bid.
    portfolio_value: float = float('nan')


def adopt_venue_balance(writer: PgWriter, venue_balance: float, *,
                        exchange_index: Optional[int] = None,
                        credited_this_cycle: float = 0.0,
                        total_balance: float = float('nan'),
                        portfolio_value: float = float('nan')) -> None:
    """Make the venue's balance the one we report. **Call this after settling.**

    Our running figure against theirs: a gap that grows is an unrecorded fill, a
    partial, or a fee we mispriced. Logged rather than silently overwritten,
    because the venue is right either way and a silent overwrite hides how wrong
    we were.

    **Ordering.** The venue credits a settlement the moment it settles, so a
    balance adopted before `settle_due` would have the same payout credited
    twice — inflating the bankroll Kelly sizes off for a cycle and filling the
    drift log with benign pairs that would hide a real unrecorded fill.
    Settling first makes the drift mean what it says.
    """
    if not np.isfinite(venue_balance):
        # The venue is the account of record only when it actually answered.
        # Writing an unreadable balance over a correct one is worse than keeping
        # ours and saying so.
        logger.error('the venue did not return a readable balance; keeping our own '
                     'figure this cycle rather than overwriting it')
        return
    account = writer.account()
    if account is None:
        return
    ours = float(account.bankroll)
    drift = venue_balance - ours
    # A payout we booked this cycle and the venue has not posted yet is not a
    # drift: `venue_balance` is sampled at the top of the cycle, and the venue
    # posts settlements one cycle behind. Subtract what we just credited; a
    # drift that survives that is an unrecorded fill, a partial, or a
    # mispriced fee.
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
    # Sampled *before* the overwrite, so the row keeps both figures as they stood
    # at the same instant; afterwards it would record a drift of zero forever.
    # The trend is the diagnosis: a drift that stays put is a starting-balance
    # mismatch, one that grows is an unrecorded fill.
    try:
        writer.write_venue_balance(
            timestamp=datetime.now(timezone.utc), balance=venue_balance,
            exchange_index=exchange_index, our_bankroll=ours,
            total_balance=total_balance, portfolio_value=portfolio_value)
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not sample the venue balance this cycle')
    writer.update_account(bankroll=venue_balance)


def remember_exchange_index(quotes: dict) -> Optional[int]:
    """Learn the shard from the book that was just read.

    **A change here is loud, not silent.** Balances are local to a shard, so a
    stale one reads the wrong balance (every order refused
    `insufficient_balance` while the funds sit elsewhere). A move is rare, costs
    at most the one cycle that already reconciled, and is corrected before the
    next.
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
    """Which exchange shard the traded markets live on, per the venue.

    Kalshi shards its exchange by category and **balances are local to a
    shard**, so the only balance an order can draw on is the one on the shard
    holding its market.

    Read off the quotes rather than configured, for the same reason markets are
    resolved by asking rather than by building a ticker: a constant is a guess
    that keeps working until the venue moves, and then it is silently wrong.
    Returns None when no book was read, which means the whole-account total.
    """
    # `fetch_quotes` stores `(Quote, ticker)`, not a bare Quote; accept both.
    candidates = [v[0] if isinstance(v, tuple) else v for v in quotes.values()]
    seen = {int(q.exchange_index) for q in candidates
            if getattr(q, 'exchange_index', None) is not None}
    if not seen:
        return None
    if len(seen) > 1:
        # Never observed. Narrowing to one shard would understate the balance
        # available to the others, so decline to narrow at all.
        logger.warning('markets span exchange shards %s; using the account total',
                       sorted(seen))
        return None
    return seen.pop()


def persist_venue_ledger(writer: PgWriter, *, fills: list[dict],
                         settlements: list[dict]) -> tuple[int, int]:
    """Store the venue's own fills and settlements. Idempotent on the venue's keys.

    The account curve is drawn from what this writes. It adds no request:
    `reconcile` already fetched both. It covers only what the live tier
    returned; `scripts.sync_venue` does the deep backfill across the historical
    tier.

    Exceptions are caught and logged rather than raised. A cycle that cannot write
    telemetry must still trade and settle — the store is a record, not the
    account.
    """
    written_fills = written_settlements = 0
    try:
        # Two queries, then only the rows that are actually new: the venue
        # returns the same ~200 fills and settlements every cycle. A settlement
        # stored with a null `pnl` is not in the skip set, so an incomplete
        # parse is retried and heals.
        known_fills, known_settlements = writer.venue_ledger_keys()
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not read the stored venue ledger; writing all rows')
        known_fills, known_settlements = set(), set()

    try:
        rows = []
        for raw in fills:
            parsed = parse_fill(raw)
            # Same scope as the settlements below: a fill in a market this
            # system never traded is somebody's manual punt, not ours.
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
            # Only markets this system traded. The Kalshi account is a
            # person's and may hold anything; another strategy's settlements
            # summed into this equity curve would be a wrong P&L.
            if not _is_ours(parsed.ticker):
                continue
            rows.append(venue_ledger.settlement_row(
                parsed, position=writer.position_for_ticker(parsed.ticker)))
        written_settlements = writer.upsert_venue_settlements(rows)
    except Exception:  # noqa: BLE001
        logger.exception('could not store the venue settlements this cycle')
    # **Correct any position the venue has since filled further.**
    # `book_resting_fills` books what has filled SO FAR and then resolves the
    # ticket, which drops it out of `open_tickets()` -- so an order that keeps
    # filling afterwards leaves us holding fewer contracts than the venue does.
    # Measured 2026-10-20 on the 15:00 SOL window: three entries recorded 2, 1
    # and 0 contracts against the venue's 6, which under-credited the payout at
    # settlement and surfaced as a $1.00 balance drift.
    #
    # Driven off the venue's own fills, grouped by order, and applied only to
    # PENDING positions -- a settled one already has a payout computed from its
    # size, and changing it here would contradict that.
    try:
        _reconcile_sizes(writer, fills)
    except Exception:                                  # noqa: BLE001
        logger.exception('could not reconcile position sizes this cycle')
    if written_fills or written_settlements:
        logger.debug('venue ledger: %d fill(s), %d settlement(s) stored',
                     written_fills, written_settlements)
    return written_fills, written_settlements


def _reconcile_sizes(writer: PgWriter, fills: list[dict]) -> int:
    """Bring each pending position up to the venue's fill total for its order."""
    from core.pg_writer import OrderTicket

    by_order: dict[str, list[dict]] = {}
    for raw in fills:
        parsed = parse_fill(raw)
        oid = str(getattr(parsed, 'order_id', '') or raw.get('order_id') or '')
        if oid and _is_ours(str(getattr(parsed, 'ticker', '') or '')):
            by_order.setdefault(oid, []).append(raw)
    if not by_order:
        return 0

    with writer._session() as session:                 # noqa: SLF001
        tickets = {
            str(t.venue_order_id): t
            for t in session.query(OrderTicket)
            .filter(OrderTicket.venue_order_id.isnot(None))
            .all()
        }
    corrected = 0
    for oid, rows in by_order.items():
        ticket = tickets.get(oid)
        if ticket is None:
            continue
        side = str(ticket.side)
        total = 0.0
        cost = 0.0
        for row in rows:
            n = float(row.get('count_fp') or row.get('count') or 0) or 0.0
            yes = row.get('yes_price_dollars') or row.get('yes_price')
            try:
                yes = float(yes)
            except (TypeError, ValueError):
                continue
            if yes > 1.0:                               # integer cents
                yes /= 100.0
            # The price of the side BOUGHT: a NO fill costs 1 - the yes price.
            cost += (yes if side == 'up' else 1.0 - yes) * n
            total += n
        if total <= 0:
            continue
        if writer.reconcile_position_size(
                symbol=ticket.symbol, window_open=ticket.window_open,
                offset_minutes=ticket.offset_minutes,
                contracts=total, cost=cost, fee=0.0):
            corrected += 1
    return corrected


def _is_ours(ticker: str) -> bool:
    """Delegates to `core.venue_ledger.is_ours`, so `scripts/sync_venue.py`
    filters exactly the same way."""
    return venue_ledger.is_ours(ticker, SERIES_BY_SYMBOL.values())


async def reconcile_with_venue(writer: PgWriter, kalshi: KalshiClient, *,
                               exchange_index: Optional[int] = None) -> VenueState:
    """Read the venue's ledger, and keep it.

    Three comparisons, each of which has a specific failure it catches:

    * **balance** — carried out in `VenueState` and written by
      `adopt_venue_balance` *after* settlement, not here.
    * **settlements** — what a position actually paid. Ours are settled from an
      OHLC mean of Coinbase standing in for CF Benchmarks BRTI, which will
      sometimes disagree.
    * **open positions** — a position we think is open and the venue does not is
      an order that never filled.

    It also **stores** the fills and settlements, via `persist_venue_ledger`.
    """
    state = await kalshi.reconcile(exchange_index=exchange_index)
    venue_balance = float(state['balance'])
    persist_venue_ledger(writer, fills=list(state.get('fills') or []),
                         settlements=list(state.get('settlements') or []))

    # Settlements keyed on the market ticker we stored, returned so `run_cycle`
    # can hand them to `settle_due`.
    resolved: dict[str, dict] = {}
    for row in state.get('settlements', []):
        ticker = str(row.get('ticker', ''))
        if not ticker:
            continue
        resolved[ticker] = row
    if resolved:
        logger.debug('venue reports %d settlement(s) to reconcile', len(resolved))

    # `KalshiClient.position_size`, not `p['position']`: the field is
    # `position_fp`, a fixed-point string.
    # Only the series this loop trades: the account may hold positions nobody
    # here placed, and reporting those as unbooked fills every cycle would bury
    # a real one. Scoped by series prefix rather than an ignore-list, so an
    # unbooked fill in KXBTC15M / KXETH15M / KXSOL15M still fires.
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

    # The reverse direction: a position the venue holds and we do not. That is
    # what an order POST that timed out after the venue accepted it leaves
    # behind, and it is the one discrepancy that costs money silently.
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
    return VenueState(settlements=resolved, balance=venue_balance,
                      total_balance=float(state.get('balance_total', float('nan'))),
                      portfolio_value=float(state.get('portfolio_value',
                                                      float('nan'))))


async def bookkeeping(args, config: Config, writer: PgWriter, kalshi,
                      bars: dict, *, phase=None) -> list:
    """Reconcile with the venue, settle, record outcomes, adopt its balance.

    The order inside is the one `adopt_venue_balance` documents: settle our own
    credits first, then take the venue's balance, so a payout is not counted
    twice. The shard is the one the latest book reported.
    """
    venue_settlements: dict[str, dict] = {}
    venue_balance = float('nan')
    venue_total = float('nan')
    venue_portfolio_value = float('nan')
    shard = _LAST_EXCHANGE_INDEX
    if shard is None:
        # **Seed from the store rather than reconciling against the total.**
        # With no shard, `reconcile` reads the cross-shard balance and
        # `adopt_venue_balance` writes it onto the account -- $495.15 adopted
        # against a true shard-2 $494.05 on 2026-10-20, a $1.11 overstatement
        # that sizes Kelly until the first book is read. The shard practically
        # never moves and the last one is already recorded.
        try:
            shard = writer.last_exchange_index()
            if shard is not None:
                globals()['_LAST_EXCHANGE_INDEX'] = shard
                logger.info('seeded the exchange shard as %s from the last '
                            'recorded balance', shard)
        except Exception:                          # noqa: BLE001
            logger.exception('could not seed the exchange shard')
    if kalshi is not None and args.reconcile:
        try:
            venue = await reconcile_with_venue(writer, kalshi, exchange_index=shard)
            venue_settlements, venue_balance = venue.settlements, venue.balance
            venue_total = venue.total_balance
            venue_portfolio_value = venue.portfolio_value
            live_state._LAST_RECONCILED = time.monotonic()
        except KalshiError as exc:
            logger.error('reconciliation failed (%s); falling back to our own '
                         'bookkeeping for this cycle', exc)
        if phase:
            phase('reconcile')
    settled = await _db(settle_due, writer, bars,
                        venue_settlements=venue_settlements)
    # Independent of whether anything was held: this is what turns the recorded
    # market quotes into a scoreable sample.
    await _db(settle_predictions, writer, bars)
    # Now that our own credits are in, the drift is a real disagreement.
    if kalshi is not None and args.reconcile:
        await _db(adopt_venue_balance, writer, venue_balance, exchange_index=shard,
                  credited_this_cycle=sum(p for _, p in settled),
                  total_balance=venue_total,
                  portfolio_value=venue_portfolio_value)
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
