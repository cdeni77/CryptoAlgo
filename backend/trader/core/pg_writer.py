"""The serving store: what the API and the frontend read.

Two stores, two jobs. `core/datastore.py` is the research store — immutable
Parquet, point-in-time reads, the thing feature builds and backtests run on.
This is PostgreSQL, mutable, and holds only what a dashboard needs to show:
predictions we actually made, positions we actually took, and the account.

**A binary needs almost none of what a perpetual future needed.** The previous
schema carried mark price, unrealised PnL, funding accrual, take-profit and
stop-loss levels, notional and leverage. None of them exists here. A contract is
bought once at a known price, cannot lose more than its stake, accrues nothing,
and settles once at $1 or $0. Carrying those columns would mean writing zeros
into them forever and inviting a dashboard to render a zero as a measurement.

**Refusals are stored, not just trades.** A dashboard showing only trades cannot
show that the system declined 99% of windows because the forecast did not cover
the fee — which is the most informative thing it could say, and was on the perp
system.

**These models are duplicated in `backend/api/models/` for container isolation,
and `tests/test_orm_parity.py` fails when they diverge** in columns, types,
nullability, defaults or the migration list. That test exists because
`wallet.balance` once differed by a factor of ten between the two copies, so
whichever container created the row decided the account's starting balance.
"""

from __future__ import annotations

import enum
import logging
from contextlib import contextmanager
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text,
    UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import func

from core.book import won as _won

logger = logging.getLogger(__name__)

Base = declarative_base()


def _int_or_none(value) -> Optional[int]:
    """An integer where the venue sent one, None where it sent nothing."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_or_none(value: Optional[float]) -> Optional[float]:
    """A number only when it is one. NaN and inf become NULL.

    `balance_breakdown` reports NaN when the venue did not answer, and NaN is a
    float that Postgres will happily store in a DOUBLE PRECISION column — where
    it then poisons every AVG, MAX and comparison downstream, silently. NULL is
    the honest encoding of "the venue did not say", and the serving layer
    already renders a null as a reason rather than a zero.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float('inf'), float('-inf')) else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Outcome(str, enum.Enum):
    """A position's terminal state. There is no third possibility."""

    PENDING = 'pending'
    WON = 'won'
    LOST = 'lost'


class Prediction(Base):
    """One scored decision point: (symbol, window, offset), traded or refused."""

    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False, index=True)
    offset_minutes = Column(Integer, nullable=False)
    decision_time = Column(DateTime(timezone=True), nullable=False)

    # the barrier, as observed
    strike = Column(Float, nullable=False)
    last_price = Column(Float, nullable=False)
    displacement = Column(Float, nullable=False)
    sigma_remaining = Column(Float, nullable=True)
    z_score = Column(Float, nullable=True)

    # the two probabilities, and never one without the other
    baseline_probability = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)

    # What the forecast was actually judged against, and where that came from.
    # A backtest has no quotes and stands the calibrated baseline in for the
    # market; a live decision reads the real book. Those are different claims and
    # a row that does not distinguish them makes a backtest look like a fill.
    # The venue's own view, and the realised answer. These three exist for one
    # purpose: to eventually score the MARKET's probability against the outcome,
    # on every window, traded or refused.
    #
    # That is the only economically meaningful benchmark and nothing in this
    # system could compute it. `market_probability` was written and read by
    # nothing, there was no `outcome` column at all, and `positions` only covers
    # the ~6% of windows that traded — a selected sample. Beating F(x/sigma) says
    # nothing about beating the price you would actually pay.
    #
    # `market_probability` is the MID (the venue's belief). The two asks are what
    # a trade would cost, which is a different question and needed for expected
    # value rather than for calibration.
    market_probability = Column(Float, nullable=True)
    market_ask_up = Column(Float, nullable=True)
    market_ask_down = Column(Float, nullable=True)
    outcome = Column(Integer, nullable=True)
    price_source = Column(String, nullable=False, default='baseline')

    # the decision
    reason = Column(String, nullable=False, index=True)
    traded = Column(Boolean, nullable=False, default=False)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=True)
    effective_cost = Column(Float, nullable=True)
    edge = Column(Float, nullable=True)
    contracts = Column(Integer, nullable=True)

    model_version = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'window_open', 'offset_minutes',
                         name='uq_prediction_point'),
        Index('ix_predictions_traded_window', 'traded', 'window_open'),
    )


class MinutePrice(Base):
    """One minute of price per symbol, for the window chart.

    The prediction rows carry `last_price` at four offsets, which is enough to
    decide with and far too sparse to draw. A barrier problem's natural picture
    is the path against the line it has to finish above, and that needs every
    minute.

    Cheap: three symbols at one row a minute is 4,320 rows a day. The research
    store keeps the authoritative bars; this is a rolling window for the screen,
    and `prune` drops what has scrolled off.
    """

    __tablename__ = 'minute_prices'

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    minute = Column(DateTime(timezone=True), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'minute', name='uq_minute_price'),
        Index('ix_minute_prices_symbol_minute', 'symbol', 'minute'),
    )


class Position(Base):
    """One contract purchase, held to settlement. No marking, no exit."""

    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False, index=True)
    offset_minutes = Column(Integer, nullable=False)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    # **Cost only: `contracts * price`. The fee is NOT in here**, despite what
    # this comment said until 2026-10-20 ("everything paid, fee included").
    # Verified against every row with a non-zero fee: `outlay == contracts *
    # price` exactly, and `settle_position` subtracts `fee` separately --
    # which it would not if the fee were already included. The false comment
    # mattered because `at risk` on the dashboard is this column and reads as
    # cash committed, and `unrealized_pnl` is computed against it.
    outlay = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)        # charged at entry, subtracted at settle
    model_probability = Column(Float, nullable=False)
    baseline_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)

    outcome = Column(String, nullable=False, default=Outcome.PENDING.value, index=True)
    settled_up = Column(Boolean, nullable=True)
    # **Mark to the VENUE's bid, not to our forecast.** `at risk` on the
    # dashboard is cost, deliberately: marking a position to the model's own
    # probability books conviction as profit. The venue's bid is a different
    # claim -- it is what someone will actually pay for this contract right
    # now, so an open position can be shown moving without inventing P&L.
    # `bid_for(side)` because that is where a SALE lands; the ask is what it
    # would cost to buy more, which is not what this position is worth.
    #
    # Nullable, and stale-able: `mark_at` is when the quote was read, so the
    # surface can say how old the number is rather than implying it is live.
    mark_price = Column(Float, nullable=True)
    mark_at = Column(DateTime(timezone=True), nullable=True)
    payout = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # **Keyed on the OFFSET too, since 2026-10-20.** It was
        # (symbol, window_open), which made one position per window a database
        # invariant -- correct while `already_entered` allowed exactly one
        # entry, and wrong the moment `max_entries_per_symbol_window` could
        # exceed 1: entries 2..N could not be booked at all, so our books
        # recorded 1 contract where the venue held 4 and `pnl_gap` drifted by
        # the difference.
        #
        # The protection it was written for is kept intact. That was a
        # duplicate order for the SAME decision surviving a restart -- the
        # process died after the order was on the wire and `restart:
        # unless-stopped` brought it back to send it again. A decision is
        # (symbol, window, offset), so keying on all three still refuses the
        # repeat while allowing the four offsets to be distinct bets.
        UniqueConstraint('symbol', 'window_open', 'offset_minutes',
                         name='uq_position_window_offset'),
        Index('ix_positions_open', 'outcome', 'settle_time'),
    )


class Account(Base):
    """The single account row: the bankroll, and what it is now.

    `mode` is 'paper' or 'live' and it is not cosmetic. A live account holds real
    money, so every surface that shows a number from this row has to be able to
    say which kind it is — a live account that renders identically to a paper one
    is the single worst failure this schema could permit.
    """

    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, nullable=False, default='paper', index=True)
    starting_bankroll = Column(Float, nullable=False, default=100.0)
    bankroll = Column(Float, nullable=False, default=100.0)
    staked = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    # **The epoch the dashboard reads from. NOT a deletion.**
    #
    # Restarting the P&L view must not destroy the evidence. `predictions` is
    # the only out-of-sample measurement of whether the model beats the PRICE --
    # `scored_against_market` reads nothing else, and `model_minus_market` and
    # `calibration_vs_market` are computed from it -- and `venue_fills` /
    # `venue_settlements` hold windows Kalshi itself no longer serves. Truncating
    # for a clean chart would trade the one number that is not self-graded for a
    # cosmetic reset.
    #
    # So the API filters on or after this instant and the rows stay. Clearing it
    # restores the full history, and `scripts/evaluate` and `scripts/promote`
    # never consult it: research always reads everything.
    reset_at = Column(DateTime(timezone=True), nullable=True)
    fees_paid = Column(Float, nullable=False, default=0.0)
    halted = Column(Boolean, nullable=False, default=False)
    halted_reason = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())


class EquityPoint(Base):
    """Equity at a settlement. Never marked to a model probability.

    Marking an open binary at our own forecast books the edge we believe in as
    profit we have not received, which is how a losing system draws a rising
    equity curve. `staked` is carried at cost.
    """

    __tablename__ = 'equity_curve'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    bankroll = Column(Float, nullable=False)
    staked = Column(Float, nullable=False, default=0.0)
    open_positions = Column(Integer, nullable=False, default=0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRun(Base):
    """A promotion attempt. Blocked ones are kept: they are the trial count."""

    __tablename__ = 'model_runs'

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    installed = Column(Boolean, nullable=False, default=False)
    forced = Column(Boolean, nullable=False, default=False)
    force_reason = Column(Text, nullable=True)

    folds = Column(Integer, nullable=True)
    windows_evaluated = Column(Integer, nullable=True)
    log_loss_skill = Column(Float, nullable=True)
    log_loss_skill_se = Column(Float, nullable=True)
    folds_positive = Column(Integer, nullable=True)
    calibration_error = Column(Float, nullable=True)
    residual_scale = Column(Float, nullable=True)
    control_gain_share = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    total_return = Column(Float, nullable=True)
    gates = Column(JSON, nullable=True)
    failed_gates = Column(Text, nullable=True)
    provenance = Column(JSON, nullable=True)


class CalibrationBin(Base):
    """A row of the reliability table, for the one chart that cannot be faked.

    A model can hit the base rate exactly while being wrong at every level of
    confidence. Since this system only trades its confident predictions, a
    miscalibration in the 0.85-0.95 band matters far more than the headline.
    """

    __tablename__ = 'calibration'

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)        # 'model' or 'baseline'
    bin_low = Column(Float, nullable=False)
    bin_high = Column(Float, nullable=False)
    predicted = Column(Float, nullable=True)
    observed = Column(Float, nullable=True)
    count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('model_version', 'source', 'bin_low',
                         name='uq_calibration_bin'),
    )


class OrderTicket(Base):
    """A live decision, as something a person can act on.

    This repository has no Kalshi API client and no Kalshi credentials, so in
    live mode the engine does not place orders — it writes a ticket carrying
    everything needed to place one, and records what came back. Being explicit
    about that boundary is the point: a system that silently did nothing while
    appearing to trade would be worse than one that says a human is in the loop.

    `status` moves new -> placed -> filled, or new -> skipped. Nothing here
    advances it automatically; the dashboard does, or a future order client will.
    """

    __tablename__ = 'order_tickets'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False)
    offset_minutes = Column(Integer, nullable=False)

    # The venue's own identifier for the market, resolved by asking the venue
    # which market opens and closes on this window rather than by building a
    # ticker from a pattern. A pattern is a guess that keeps working until the
    # venue renames a series.
    market_ticker = Column(String, nullable=True, index=True)
    venue_order_id = Column(String, nullable=True)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    # The price the decision was sized at, and the worst price still worth
    # paying. A ticket without a limit is an instruction to pay anything.
    limit_price = Column(Float, nullable=False)
    max_price = Column(Float, nullable=False)
    expected_cost = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)

    status = Column(String, nullable=False, default='new', index=True)
    filled_contracts = Column(Integer, nullable=True)
    filled_price = Column(Float, nullable=True)
    filled_at = Column(DateTime(timezone=True), nullable=True)
    note = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # **Keyed on the OFFSET too, since 2026-10-20** — the same widening
        # `positions` needed, and missing it left a worse hole. The order for
        # the second offset was still PLACED and FILLED at the venue (two
        # distinct `order_id`s on KXETH15M-26OCT201000-00, 13:48:09 @ 0.680 and
        # 13:51:05 @ 0.750), but `write_ticket` returned the first ticket's id
        # instead of recording it. No ticket means no position, so our books
        # showed 1 contract against the venue's 2 and `pnl_gap` moved every
        # time a window took a second entry.
        #
        # The protection it was written for — one order per decision surviving
        # a restart — is intact, because a decision is (symbol, window, offset).
        UniqueConstraint('symbol', 'window_open', 'offset_minutes',
                         name='uq_ticket_window_offset'),
        Index('ix_order_tickets_status_created', 'status', 'created_at'),
    )


class VenueFill(Base):
    """One execution, as the VENUE recorded it. Not our belief about it.

    `Position` says what we thought we bought: the price `decide()` sized at and
    the fee it predicted from the published schedule. This says what the account
    of record actually shows. The two differ on every partial fill, every fill
    better than the limit, and every fee we mispriced — 16 of the first 323 live
    fills were partial — and where they differ, this row is right.

    Keyed on the venue's `trade_id`, so a re-sync is idempotent. That matters more
    here than anywhere else in this schema: the live loop and
    `scripts/sync_venue.py` both write these, the live and historical tiers
    overlap around the cutoff, and a fill counted twice doubles a cost basis.

    `raw` keeps the whole payload. A parser reads the fields it knows about, and
    the fields it does not are the ones a venue change turns out to hinge on —
    the `position_fp` and `yes_bid_dollars` lessons both cost real money and both
    would have been visible in a stored payload.
    """

    __tablename__ = 'venue_fills'

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, nullable=False, unique=True, index=True)
    order_id = Column(String, nullable=True, index=True)
    ticker = Column(String, nullable=False, index=True)

    # 'up' or 'down', this project's vocabulary for the venue's yes/no. The
    # translation happens in the parser, once, because a mislabelled side inverts
    # a trade's P&L rather than merely mis-displaying it.
    side = Column(String, nullable=False)
    action = Column(String, nullable=True)
    # Fractional because the wire encoding is a fixed-point string (`count_fp`:
    # "5.00"). Whole contracts in practice; an Integer column would silently
    # truncate the day that stops being true.
    contracts = Column(Float, nullable=False)
    # Dollars paid per contract, for `side`. Nullable: a fill whose price did not
    # parse is a fill at an unknown price, which is not a fill at zero.
    price = Column(Float, nullable=True)
    is_taker = Column(Boolean, nullable=True)
    created_time = Column(DateTime(timezone=True), nullable=True, index=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_fills_ticker_time', 'ticker', 'created_time'),
    )


class VenueSettlement(Base):
    """One market resolved, as the venue paid it. **This is the P&L.**

    A binary bought once and held to expiry pays one fee at entry and settles once
    at $1 or $0, so a settlement row is a position's complete economic history:
    cost basis, payout, fee. Our own `Position.pnl` recomputes all three — the
    outcome from an OHLC mean of Coinbase standing in for sixty seconds of CF
    Benchmarks BRTI, the fee from the published schedule — and both are estimates
    of the numbers in this row.

    `pnl` is stored rather than derived on read, and that is a deliberate
    exception to keeping derived values out of this schema. The API runs in another
    container and cannot import the trader, so a P&L computed at read time would
    have to be written twice, in two languages' worth of null handling, and
    `test_orm_parity` cannot compare arithmetic — only columns. Storing it keeps
    one definition, in the one process that writes here, and leaves the API doing
    what it is supposed to do: serving a measurement it did not invent.

    Null `pnl` means the venue left one of the three fields absent. That is not
    break-even, and nothing downstream may read it as zero.
    """

    __tablename__ = 'venue_settlements'

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, unique=True, index=True)
    event_ticker = Column(String, nullable=True)
    # 'yes' or 'no', as the venue resolved it — the answer our bars only estimate.
    market_result = Column(String, nullable=True)

    yes_contracts = Column(Float, nullable=False, default=0.0)
    no_contracts = Column(Float, nullable=False, default=0.0)
    yes_cost = Column(Float, nullable=True)
    no_cost = Column(Float, nullable=True)
    revenue = Column(Float, nullable=True)
    fee_cost = Column(Float, nullable=True)
    # revenue - (yes_cost + no_cost) - fee_cost. See the class docstring for why
    # it is a column, and `Settlement.pnl` for why the fee is subtracted.
    pnl = Column(Float, nullable=True)

    settled_time = Column(DateTime(timezone=True), nullable=True, index=True)

    # Our own books, alongside the venue's, for the one comparison that matters:
    # a gap that grows is a fee we mispriced or a settlement our Coinbase proxy
    # got wrong. Null when no position of ours maps to this ticker — which is
    # itself informative, since it means the venue settled something we never
    # recorded buying.
    position_id = Column(Integer, nullable=True, index=True)
    our_pnl = Column(Float, nullable=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_settlements_settled', 'settled_time'),
    )


class VenuePosition(Base):
    """The venue's own open positions, as it reports them, replaced each sample.

    **Our `positions` table is not the same thing and never will be.** A row
    there exists once the live loop has booked a fill; the venue has the
    position the instant it fills. The gap is one cycle at best, and at worst
    it is permanent -- an order POST that timed out after being accepted
    leaves a position the venue holds and we have no record of, which is the
    one discrepancy that costs money silently.

    So the dashboard shows these, and ours are joined on where they exist to
    supply what the venue does not know: which decision offset opened it, what
    the model thought, what edge was claimed. A venue row with no match is
    displayed as exactly that rather than hidden.

    Replaced wholesale on every sample rather than appended, because this is
    current state and not a series. The series is `venue_balance`, which
    carries the account value a second at a time.
    """

    __tablename__ = 'venue_positions'

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    exchange_index = Column(Integer, nullable=True)
    #: SIGNED, as the venue serves it: negative is a NO position. Stored as
    #: served rather than normalised, because converting here would put the
    #: same sign convention in two places and they would drift.
    position = Column(Float, nullable=False)
    #: Cost basis, net of any netting the venue has already applied.
    market_exposure = Column(Float, nullable=True)
    fees_paid = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    sampled_at = Column(DateTime(timezone=True), nullable=False, index=True)


class VenueBalance(Base):
    """The venue's balance, sampled. What makes the account chart live.

    Settlements step the curve every fifteen minutes at most, and only when
    something settled. This is read every cycle, so the chart has a present
    rather than ending at the last resolved market.

    It is cash on the shard, not equity: entering a position debits it and
    settlement credits it back, so on its own it sawtooths. That is why the
    account curve plots realised P&L as its primary series and carries this
    beside it — the balance is the real money, and the P&L is how the portfolio
    is doing. `exchange_index` is recorded because **balances are local to a
    shard**: a total across shards is not money any single order can draw on, and
    reading one number as the other is what made every order come back
    `insufficient_balance` against an apparently healthy $107.96.
    """

    __tablename__ = 'venue_balance'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    balance = Column(Float, nullable=False)
    exchange_index = Column(Integer, nullable=True)
    # **The cross-shard total, which is what the venue's own app shows.**
    # `balance` above is the TRADING shard alone -- the only cash an order on a
    # crypto market can draw on -- and the two differ by whatever is stranded
    # elsewhere. Recording only the shard made the dashboard read $482.46
    # against the app's $483.57 and look out of sync when drift was exactly
    # 0.00. Nullable because rows written before 2026-10-18 do not have it, and
    # a backfilled zero would draw a cliff in the series that never happened.
    total_balance = Column(Float, nullable=True)
    #: The VENUE's own mark of the open book at this instant, from
    #: `/portfolio/balance.portfolio_value`. Null when the venue did not answer
    #: -- never a zero, because a flat book is a real zero and the two must not
    #: be confused. Recorded per cycle because Kalshi serves only the current
    #: value: there is no history endpoint, so an equity curve has to be
    #: sampled or it cannot be built at all.
    portfolio_value = Column(Float, nullable=True)
    # What our own arithmetic thought at the same instant, and the difference.
    # Stored rather than only logged because a drift that GROWS is the signal —
    # an unrecorded fill, a partial, a mispriced fee — and a single log line
    # cannot show a trend.
    our_bankroll = Column(Float, nullable=True)
    drift = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_balance_timestamp_desc', 'timestamp'),
    )

# Indexes and constraints added after a table first shipped. Applied by
# `_run_migrations` and asserted identical to the API's list by test_orm_parity —
# an index that exists in one container and not the other is a query that is
# fast in development and a table scan in production.
MIGRATIONS: tuple[str, ...] = (
    'CREATE INDEX IF NOT EXISTS ix_predictions_reason_window '
    'ON predictions (reason, window_open DESC)',
    'CREATE INDEX IF NOT EXISTS ix_positions_settled_at '
    'ON positions (settled_at DESC)',
    'CREATE INDEX IF NOT EXISTS ix_equity_curve_timestamp_desc '
    'ON equity_curve (timestamp DESC)',
    'CREATE INDEX IF NOT EXISTS ix_minute_prices_recent '
    'ON minute_prices (symbol, minute DESC)',
    'CREATE INDEX IF NOT EXISTS ix_order_tickets_window '
    'ON order_tickets (window_open DESC)',
    # The venue ledger. Every read of these is "newest first" or "since a
    # timestamp", which is one index each; the settlements one carries the P&L
    # aggregate and the account chart, so it is the hottest query on the surface.
    'CREATE INDEX IF NOT EXISTS ix_venue_fills_created_desc '
    'ON venue_fills (created_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_settlements_settled_desc '
    'ON venue_settlements (settled_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_balance_recent '
    'ON venue_balance (timestamp DESC)',
    # NOTE: the market-benchmark columns on `predictions` (market_ask_up,
    # market_ask_down, outcome) deliberately have NO migration here.
    #
    # `ADD COLUMN IF NOT EXISTS` is not parseable on SQLite, so it would be
    # skipped by the dialect guard in `_run_migrations` and production would be
    # running a statement the test suite had never executed — which is exactly
    # the trap `tests/test_orm_parity.py` was written to close, and exactly what
    # the previous all-ADD-COLUMN list did.
    #
    # The ORM gives them to a fresh database via `create_all`, and the serving
    # store is regenerated telemetry rather than a record of account, so an
    # existing one is wiped: `docker compose down -v`. The research store and the
    # scraped SQLite are the irreplaceable parts and neither is touched.
)


class TraderAlreadyRunning(RuntimeError):
    """Another process holds the trading lock on this database."""


class AccountModeMismatch(RuntimeError):
    """The stored account was opened in a different mode than this run asks for."""


class PgWriter:
    """Everything the trader writes to the serving store."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.getenv('DATABASE_URL')
        if not url:
            raise ValueError('DATABASE_URL is not set')
        self._engine = create_engine(url, pool_pre_ping=True, future=True)
        self._sessions = sessionmaker(bind=self._engine, future=True)
        Base.metadata.create_all(self._engine)
        self._ensure_columns()
        self._ensure_constraints()
        self._run_migrations()

    #: Columns added to an existing table after it was first created.
    #: `(table, column, type)`, applied only when genuinely absent.
    ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
        ('account', 'reset_at', 'TIMESTAMP WITH TIME ZONE'),
        ('venue_balance', 'total_balance', 'DOUBLE PRECISION'),
        ('venue_balance', 'portfolio_value', 'DOUBLE PRECISION'),
        ('positions', 'mark_price', 'DOUBLE PRECISION'),
        ('positions', 'mark_at', 'TIMESTAMP WITH TIME ZONE'),
    )

    def _ensure_columns(self) -> None:
        """Add columns that `create_all` cannot.

        **`create_all` creates missing TABLES, never missing columns**, so a
        database that predates a field reports `UndefinedColumn` on every read
        of that table -- loudly, but only at runtime.

        This is NOT in `MIGRATIONS`. That tuple is asserted by
        `tests/test_orm_parity.py` to hold only portable, idempotent
        `CREATE INDEX IF NOT EXISTS` statements, for a stated reason: anything
        else needs a dialect guard, and a guard means the SQLite test suite
        stops exercising the statement production runs. `ADD COLUMN IF NOT
        EXISTS` is Postgres-only.

        So the existence check is done in Python via the dialect-neutral
        inspector, and the `ALTER` that follows is bare `ADD COLUMN`, which
        SQLite and Postgres both accept. Idempotent by inspection rather than
        by syntax, and exercised identically on both.
        """
        from sqlalchemy import inspect as _inspect

        inspector = _inspect(self._engine)
        existing = set(inspector.get_table_names())
        for table, column, type_ in self.ADDED_COLUMNS:
            if table not in existing:
                continue
            have = {c['name'] for c in inspector.get_columns(table)}
            if column in have:
                continue
            with self._engine.begin() as con:
                con.exec_driver_sql(
                    f'ALTER TABLE {table} ADD COLUMN {column} {type_}')
            logger.info('added %s.%s', table, column)

    #: Unique constraints that had to CHANGE, as (table, old_name, new_name,
    #: columns). `create_all` gives a fresh database the new one straight from
    #: the ORM; only an existing database needs the swap.
    CHANGED_UNIQUES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
        ('positions', 'uq_position_window', 'uq_position_window_offset',
         ('symbol', 'window_open', 'offset_minutes')),
        ('order_tickets', 'uq_ticket_window', 'uq_ticket_window_offset',
         ('symbol', 'window_open', 'offset_minutes')),
    )

    def _ensure_constraints(self) -> None:
        """Swap unique constraints that widened, on databases that predate them.

        Separate from `_ensure_columns` because the statement is not portable:
        SQLite cannot `ALTER TABLE ... DROP CONSTRAINT` at all. That is
        tolerable here in a way the repo's `MIGRATIONS` note says it generally
        is not, for one specific reason -- the ORM is the source of truth, so
        `create_all` builds the NEW constraint on any fresh database, which is
        every SQLite test database. There is no path where SQLite needs the
        swap, so skipping it there does not leave production running a
        statement the tests never exercise; it leaves production catching up to
        what the tests already have.

        Existence is checked through the dialect-neutral inspector and the swap
        runs in ONE transaction, so a crash between the drop and the create
        cannot leave the table with no uniqueness at all -- which on this table
        would mean duplicate positions for one decision, the exact failure the
        constraint exists to prevent.
        """
        from sqlalchemy import inspect as _inspect

        if self._engine.dialect.name == 'sqlite':
            return
        inspector = _inspect(self._engine)
        tables = set(inspector.get_table_names())
        for table, old_name, new_name, columns in self.CHANGED_UNIQUES:
            if table not in tables:
                continue
            names = {c['name'] for c in inspector.get_unique_constraints(table)}
            if new_name in names:
                continue
            cols = ', '.join(columns)
            with self._engine.begin() as con:
                if old_name in names:
                    con.exec_driver_sql(
                        f'ALTER TABLE {table} DROP CONSTRAINT {old_name}')
                con.exec_driver_sql(
                    f'ALTER TABLE {table} ADD CONSTRAINT {new_name} '
                    f'UNIQUE ({cols})')
            logger.info('%s: %s -> %s (%s)', table, old_name, new_name, cols)

    def _run_migrations(self) -> None:
        """Apply the additive migrations, tolerating only SQLite's dialect gaps.

        This used to catch bare `Exception` and log at `debug`, so *any* migration
        failure against real Postgres — a type mismatch, a lock timeout, a
        permissions problem — vanished at a level most deployments do not
        capture, and the failure resurfaced later as a confusing runtime error on
        the accounting database instead of a loud one at startup.

        The tolerance it was written for is real but narrow: `MIGRATIONS` uses
        `ADD COLUMN IF NOT EXISTS` and similar, which SQLite does not parse, and
        the tests run on SQLite. So tolerate it on SQLite and nowhere else.
        """
        sqlite = self._engine.dialect.name == 'sqlite'
        with self._engine.begin() as connection:
            for statement in MIGRATIONS:
                try:
                    connection.execute(text(statement))
                except Exception as exc:  # noqa: BLE001 - re-raised unless SQLite
                    if not sqlite:
                        logger.error('migration failed on %s: %s\n  %s',
                                     self._engine.dialect.name, exc, statement)
                        raise
                    logger.debug('migration skipped on sqlite (%s): %s', exc, statement)

    def _session(self) -> Session:
        return self._sessions()

    # ---- predictions ----------------------------------------------------
    def write_prediction(self, **fields) -> int:
        """Upsert one decision point. Idempotent on (symbol, window, offset).

        Idempotent because the orchestrator can re-run a cycle — a retry after a
        network failure must not double-count a refusal or, worse, a trade.
        """
        with self._session() as session:
            existing = session.query(Prediction).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open'],
                offset_minutes=fields['offset_minutes'],
            ).one_or_none()
            if existing is not None:
                for key, value in fields.items():
                    setattr(existing, key, value)
                session.commit()
                return existing.id
            row = Prediction(**fields)
            session.add(row)
            session.commit()
            return row.id

    # ---- minute prices --------------------------------------------------
    def write_minute_prices(self, rows: Iterable[dict]) -> int:
        """Upsert minute bars for the chart. Idempotent on (symbol, minute)."""
        written = 0
        with self._session() as session:
            for row in rows:
                existing = session.query(MinutePrice).filter_by(
                    symbol=row['symbol'], minute=row['minute']).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                else:
                    session.add(MinutePrice(**row))
                written += 1
            session.commit()
        return written

    def prune_minute_prices(self, before: datetime) -> int:
        """Drop rows that have scrolled off the chart. Called by the engine."""
        with self._session() as session:
            n = session.query(MinutePrice).filter(MinutePrice.minute < before).delete()
            session.commit()
            return int(n)

    # ---- single-writer guard --------------------------------------------
    @contextmanager
    def exclusive_trader_lock(self):
        """Hold a database-wide lock for the duration of a trading process.

        There was no singleton of any kind. `docker-compose` runs a `trader`
        service and nothing stopped an operator also running
        `python -m scripts.live` by hand — two processes, one account, one
        window. Measured, both sized a full position off the same bankroll.

        A Postgres session-level advisory lock is the right primitive: it is
        released automatically when the connection drops, so a killed trader does
        not strand the lock. On SQLite (tests) there is no advisory lock and no
        concurrency to guard, so this yields with a warning rather than pretending.
        """
        if self._engine.dialect.name != 'postgresql':
            logger.warning('no advisory lock on %s; single-writer is unenforced',
                           self._engine.dialect.name)
            yield True
            return
        # An arbitrary but stable key. Distinct from the API's schema-bootstrap
        # lock in backend/api/app.py so the two cannot block each other.
        key = 0x51545241  # 'QTRA'
        connection = self._engine.connect()
        try:
            got = bool(connection.execute(
                text('SELECT pg_try_advisory_lock(:k)'), {'k': key}).scalar())
            if not got:
                raise TraderAlreadyRunning(
                    'another process already holds the trading lock on this '
                    'database. Two traders against one account double every '
                    'position and race on the bankroll. Stop the other one — '
                    'most likely the compose `trader` service — or point this '
                    'run at a different DATABASE_URL.'
                )
            yield True
        finally:
            connection.close()

    # ---- order tickets ---------------------------------------------------
    def write_ticket(self, **fields) -> int:
        """Record a live decision as something a person can place."""
        with self._session() as session:
            # Matched on the decision, not the window: returning the first
            # offset's ticket here silently discarded every later entry.
            existing = session.query(OrderTicket).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open'],
                offset_minutes=fields['offset_minutes']).one_or_none()
            if existing is not None:
                return existing.id
            row = OrderTicket(**fields)
            session.add(row)
            session.commit()
            return row.id

    #: Ticket states that are still live at the venue and therefore still the
    #: loop's problem. 'resting' joins 'new' because a `good_till_canceled`
    #: maker order sits on the book until it fills or is cancelled — and a
    #: resting order nothing sweeps is an open position nobody decided to hold.
    OPEN_TICKET_STATUSES = ('new', 'resting')

    def open_tickets(self, limit: int = 50) -> list[OrderTicket]:
        with self._session() as session:
            return list(
                session.query(OrderTicket)
                .filter(OrderTicket.status.in_(self.OPEN_TICKET_STATUSES))
                .order_by(OrderTicket.created_at.desc()).limit(limit)
            )

    def resolve_ticket(self, ticket_id: int, *, status: str,
                       filled_contracts: Optional[int] = None,
                       filled_price: Optional[float] = None,
                       note: Optional[str] = None,
                       venue_order_id: Optional[str] = None) -> None:
        with self._session() as session:
            row = session.get(OrderTicket, ticket_id)
            if row is None:
                return
            row.status = status
            if venue_order_id:
                row.venue_order_id = str(venue_order_id)
            # Only overwrite a recorded fill with another recorded fill. These
            # default to None, so a later call — e.g. the venue rejecting a
            # duplicate submission as `skipped` — used to erase the
            # filled_contracts / filled_price / filled_at of a real, filled,
            # real-money order, destroying the only local record of it.
            if filled_contracts is not None:
                row.filled_contracts = filled_contracts
            if filled_price is not None:
                row.filled_price = filled_price
            if note is not None:
                row.note = note
            if status == 'filled':
                row.filled_at = utcnow()
            session.commit()

    # ---- positions ------------------------------------------------------
    def open_position(self, **fields) -> Optional[int]:
        """Book a position, once, for a (symbol, window). Returns None if it exists.

        A bare insert against `uq_position_window` used to raise `IntegrityError`
        on the second cycle of any window, and `scripts/live.py`'s loop catches
        only `KeyboardInterrupt` — so the process died, *after* the duplicate
        order was already on the wire, and `restart: unless-stopped` brought it
        back to do it again. The constraint was doing its job; the caller was
        not. Get-or-create makes the second attempt a no-op the caller can see,
        matching `write_ticket`.
        """
        with self._session() as session:
            existing = session.query(Position).filter_by(
                symbol=fields['symbol'], window_open=fields['window_open'],
                offset_minutes=fields['offset_minutes']).one_or_none()
            if existing is not None:
                logger.warning(
                    '%s window %s offset +%sm already holds a position; '
                    'not booking a second',
                    fields['symbol'], fields['window_open'],
                    fields['offset_minutes'])
                return None
            row = Position(outcome=Outcome.PENDING.value, **fields)
            session.add(row)
            session.commit()
            return row.id

    def positions_due(self, now: datetime) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome == Outcome.PENDING.value)
                .filter(Position.settle_time <= now)
                .order_by(Position.settle_time)
            )

    def settle_position(self, position_id: int, *, settled_up: bool,
                        fee: Optional[float] = None) -> Optional[float]:
        """Resolve one position, credit the account, and return its PnL.

        Returns None if the position was already settled, which is what makes a
        re-run of `settle_due` idempotent.

        **The payout is credited here, in the same transaction as the outcome.**
        It used to be nowhere: `act_on` debited the stake and nothing ever
        credited a win, so the only two writers of `Account.bankroll` were the
        entry debit and the live venue overwrite. In paper mode there is no venue,
        so the bankroll fell by the stake on every trade and rose never — the
        equity curve decayed to the ruin floor regardless of the win rate, and
        `realized_pnl` was a hard-coded zero rendered as a measurement. The
        backtest (`core/book.py`) always did this correctly, which is exactly why
        no gate caught it: the two accounting paths had diverged and only the
        simulated one was right.

        Crediting inside the position's own transaction is deliberate. A payout
        applied in a second transaction can be lost to a crash between the two,
        and then the position says "won" while the bankroll never saw the money.

        **`fee` is the venue's own charge, and the P&L subtracts it.** Two
        defects, measured 2026-10-20 and fixed together:

          * `pnl` was `payout - outlay`, and `outlay` is `contracts * price`
            exactly. The fee was never in the arithmetic at all, so our books
            reported GROSS where the venue reports net. Measured against 419
            settled markets, 409 of them differed from the venue by precisely
            the fee and nothing else.
          * `Position.fee` was 0.0000 on every row, because `maker_entry`
            implies `assume_maker` and the estimate is made at DECISION time --
            before it is known whether the order rests or crosses. About 19% of
            fills cross and pay roughly 1.18c per contract.

        Passing the venue's figure fixes both: it is what was actually charged,
        not what the published schedule predicts. Omitted (paper mode, or a
        settlement the venue has not posted), the stored estimate is used, so
        the arithmetic is still net rather than gross.

        `Account.fees_paid` moves by the DIFFERENCE between the new figure and
        whatever was booked at entry, so a fee is never counted twice.
        """
        with self._session() as session:
            row = session.get(Position, position_id)
            if row is None or row.outcome != Outcome.PENDING.value:
                return None
            won = _won(side=row.side, settled_up=settled_up)
            payout = float(row.contracts) if won else 0.0
            booked_fee = float(row.fee or 0.0)
            if fee is not None and float(fee) == float(fee):  # not NaN
                row.fee = float(fee)
            charged = float(row.fee or 0.0)
            row.settled_up = settled_up
            row.payout = payout
            row.pnl = payout - row.outlay - charged
            row.outcome = Outcome.WON.value if won else Outcome.LOST.value
            row.settled_at = utcnow()
            pnl = float(row.pnl)
            fee_delta = charged - booked_fee
            # Relative UPDATE rather than read-modify-write: the arithmetic
            # happens in the database, so a concurrent debit cannot be lost.
            session.query(Account).update(
                {Account.bankroll: Account.bankroll + payout,
                 Account.realized_pnl: Account.realized_pnl + pnl,
                 Account.fees_paid: Account.fees_paid + fee_delta},
                synchronize_session=False,
            )
            session.commit()
            return pnl

    def entries_for_window(self, window_open: datetime) -> tuple[frozenset[str], float, int]:
        """What is already committed for one window: symbols, stake, count.

        The live loop used to build `WindowExposure()` empty on every cycle, so
        `ALREADY_ENTERED`, `POSITION_LIMIT` and `WINDOW_EXPOSURE` were evaluated
        against nothing and a single window could be entered once per cycle per
        offset — up to twelve times where the backtest permits one. Exposure has
        to come from the durable store, because the process restarts and the
        window does not.

        Tickets count as well as positions: a ticket exists from the moment an
        order is sent, so a crash between sending and booking still shows up.
        That includes 'unknown' — a POST that failed in flight may well have been
        accepted, and re-entering on top of it could double the position.

        **A kill is the exception, because the venue has told us it took
        nothing.** Live on 2026-08-26 two `immediate_or_cancel` orders were
        cancelled with `fill_count_fp: "0.00"` and no position anywhere, and the
        next cycle refused the window as already holding $8.55 — which was
        exactly the two decision stakes. A window with a dead order in it is a
        window with no exposure, and holding the slot shut costs the entry
        outright.
        """
        with self._session() as session:
            symbols: set[str] = set()
            stake = 0.0
            for row in (session.query(Position)
                        .filter(Position.window_open == window_open)):
                symbols.add(str(row.symbol))
                stake += float(row.outlay or 0.0)
            dead = ('skipped', 'killed')
            for row in (session.query(OrderTicket)
                        .filter(OrderTicket.window_open == window_open)
                        .filter(OrderTicket.status.notin_(dead))):
                if str(row.symbol) not in symbols:
                    symbols.add(str(row.symbol))
                    stake += float(row.expected_cost or 0.0)
            return frozenset(symbols), stake, len(symbols)

    def reconcile_position_size(self, *, symbol: str, window_open,
                                offset_minutes: int, contracts: float,
                                cost: float, fee: float) -> bool:
        """Correct a PENDING position to the venue's own fill totals.

        **A position is booked from the fills visible at the time**, and an
        order can keep filling afterwards: `book_resting_fills` books the
        partial, resolves the ticket, and the ticket then drops out of
        `open_tickets()` so later fills are never added. Measured live
        2026-10-20 on the 15:00 SOL window -- three entries recorded 2, 1 and 0
        contracts against the venue's 6, which under-credited the settlement
        payout and showed up as a $1.00 balance drift.

        Only PENDING positions, and only UPWARDS: a settled position already
        has a payout computed from its size, and shrinking a position would
        discard contracts the venue says we hold. Returns whether anything
        changed, so the caller can log a correction rather than a no-op.
        """
        target = int(round(float(contracts)))
        if target < 1:
            return False
        stamp = pd.Timestamp(window_open)
        stamp = (stamp.tz_localize('UTC') if stamp.tzinfo is None
                 else stamp.tz_convert('UTC'))
        with self._session() as session:
            row = (session.query(Position)
                   .filter(Position.symbol == str(symbol),
                           Position.offset_minutes == int(offset_minutes),
                           Position.outcome == Outcome.PENDING.value)
                   .all())
            match = None
            for candidate in row:
                when = pd.Timestamp(candidate.window_open)
                when = (when.tz_localize('UTC') if when.tzinfo is None
                        else when.tz_convert('UTC'))
                if when == stamp:
                    match = candidate
                    break
            if match is None or int(match.contracts or 0) >= target:
                return False
            before = int(match.contracts or 0)
            # The bankroll was debited for what we booked, so only the
            # DIFFERENCE is taken now -- debiting the whole cost again would
            # double-charge the contracts already recorded.
            extra = float(cost) - float(match.outlay or 0.0)
            match.contracts = target
            match.outlay = float(cost)
            match.price = float(cost) / target if target else match.price
            match.fee = float(fee)
            session.query(Account).update(
                {Account.bankroll: Account.bankroll - extra},
                synchronize_session=False,
            )
            session.commit()
            logger.warning(
                '%s window %s offset +%sm: the venue filled %d contract(s), we '
                'booked %d. Corrected, and the bankroll debited $%.4f more.',
                symbol, window_open, offset_minutes, target, before, extra)
            return True

    def tickers_for_open_positions(self) -> dict:
        """(symbol, window_open, offset) -> market ticker, for pending positions.

        `positions` carries no ticker column -- it is keyed on the decision
        because the paper engine predates the venue client -- so anything that
        needs to look a position up at the venue goes through the ticket that
        placed it.
        """
        out: dict = {}
        with self._session() as session:
            open_keys = {
                (str(p.symbol), p.window_open, int(p.offset_minutes))
                for p in session.query(Position)
                .filter(Position.outcome == Outcome.PENDING.value)
            }
            if not open_keys:
                return out
            for ticket in (session.query(OrderTicket)
                           .filter(OrderTicket.market_ticker.isnot(None))):
                key = (str(ticket.symbol), ticket.window_open,
                       int(ticket.offset_minutes))
                if key in open_keys:
                    out[key] = str(ticket.market_ticker)
        return out

    def mark_open_positions(self, marks: dict, *, at: datetime) -> int:
        """Record what the venue's book says each OPEN position is worth now.

        `marks` maps (symbol, window_open) -> price, the bid for the side held.
        Only pending positions are touched: a settled one has a payout, and a
        mark on top of that would be a second, contradictory answer to the same
        question.

        Returns the number updated. Telemetry, not accounting -- nothing sizes
        or decides off this, so the caller treats a failure here as something to
        log rather than something to stop the cycle for.
        """
        if not marks:
            return 0

        def key(symbol, when):
            """A key that survives the representation the row came back in.

            **SQLite hands back naive datetimes even for a
            `DateTime(timezone=True)` column**, while callers pass tz-aware
            `pandas.Timestamp`s, and the two never compare equal -- so a direct
            dict lookup silently matched nothing and marked zero positions.
            Normalising to a UTC epoch makes the lookup indifferent to naive
            vs aware and to Timestamp vs datetime. Naive values are read as
            UTC, which is what every stored timestamp in this schema is.
            """
            stamp = pd.Timestamp(when)
            stamp = (stamp.tz_localize('UTC') if stamp.tzinfo is None
                     else stamp.tz_convert('UTC'))
            return (str(symbol), stamp.value)

        normalised = {key(sym, when): price for (sym, when), price in marks.items()}
        updated = 0
        with self._session() as session:
            for row in (session.query(Position)
                        .filter(Position.outcome == Outcome.PENDING.value)):
                price = normalised.get(key(row.symbol, row.window_open))
                if price is None:
                    continue
                try:
                    value = float(price)
                except (TypeError, ValueError):
                    continue
                # A zero BID means nobody is bidding, which is a real state and
                # a real mark -- distinct from "no quote", which leaves the
                # previous mark and its timestamp alone so the surface can show
                # it going stale instead of jumping to zero.
                if value != value:               # NaN
                    continue
                row.mark_price = value
                row.mark_at = at
                updated += 1
            session.commit()
        return updated

    def entry_counts_for_window(self, window_open: datetime):
        """Per-symbol entry COUNTS, total stake and total entries for a window.

        `entries_for_window` answers "which symbols are in this window" and
        deduplicates by symbol, which is exactly right while one entry per
        (symbol, window) is the rule. Once `max_entries_per_symbol_window`
        exceeds 1 that dedupe silently UNDER-COUNTS: a second entry on a symbol
        already seen contributes no stake, so `max_window_exposure_fraction` --
        the notional backstop that is doing the real work once the per-symbol
        gate is relaxed -- would be computed against less than is at risk.

        Double counting is avoided by pairing on `prediction_id` rather than on
        symbol: a ticket is counted only when no position exists for the same
        decision. A ticket with no `prediction_id` is counted on its own, since
        there is nothing to pair it with and missing real exposure is the worse
        error. Dead tickets ('skipped', 'killed') are excluded for the reason
        `entries_for_window` documents -- the venue has said it took nothing.
        """
        counts: dict[str, int] = {}
        sides: dict[str, str] = {}
        stake = 0.0
        entries = 0
        with self._session() as session:
            claimed: set[int] = set()
            # Oldest first, so `setdefault` records the side actually entered
            # FIRST. That is the side the window is committed to; a later
            # opposite decision must be refused against it, not the reverse.
            for row in (session.query(Position)
                        .filter(Position.window_open == window_open)
                        .order_by(Position.id)):
                symbol = str(row.symbol)
                counts[symbol] = counts.get(symbol, 0) + 1
                if row.side is not None:
                    sides.setdefault(symbol, str(row.side))
                stake += float(row.outlay or 0.0)
                entries += 1
                if row.prediction_id is not None:
                    claimed.add(int(row.prediction_id))
            dead = ('skipped', 'killed')
            for row in (session.query(OrderTicket)
                        .filter(OrderTicket.window_open == window_open)
                        .filter(OrderTicket.status.notin_(dead))
                        .order_by(OrderTicket.id)):
                if row.prediction_id is not None and int(row.prediction_id) in claimed:
                    continue
                symbol = str(row.symbol)
                counts[symbol] = counts.get(symbol, 0) + 1
                if row.side is not None:
                    sides.setdefault(symbol, str(row.side))
                stake += float(row.expected_cost or 0.0)
                entries += 1
        # **The side must come from the STORE, not only from this cycle's
        # exposure.** A restart, or simply the next cycle, rebuilds
        # `WindowExposure` from here; without the side a window entered UP at
        # +3 would happily be entered DOWN at +9, which is the -$6.29 defect.
        return counts, stake, entries, sides

    def first_position_time(self) -> Optional[datetime]:
        """When our books start, or None if they hold nothing.

        `sync_venue` needs this to say how much of the P&L gap is simply a
        shorter book. The venue remembers every settlement it ever made; this
        store is wiped between experiments on purpose, so the two routinely
        total over different periods. Measured on the first live run: 365 venue
        settlements over four days against a store reset the night before, and
        the entire gap was coverage rather than error.

        Returned tz-aware in UTC. SQLite hands back naive datetimes even for a
        `DateTime(timezone=True)` column, and comparing one of those against the
        venue's aware timestamp raises rather than misreporting — which is the
        better failure, but not one to ship.
        """
        with self._session() as session:
            earliest = session.query(func.min(Position.window_open)).scalar()
        if earliest is None:
            return None
        if earliest.tzinfo is None:
            return earliest.replace(tzinfo=timezone.utc)
        return earliest.astimezone(timezone.utc)

    def open_positions(self) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome == Outcome.PENDING.value)
                .order_by(Position.settle_time)
            )

    def settled_positions_since(self, since: datetime) -> list[Position]:
        with self._session() as session:
            return list(
                session.query(Position)
                .filter(Position.outcome != Outcome.PENDING.value)
                .filter(Position.settled_at >= since)
                .order_by(Position.settled_at.desc())
            )

    def realised_high_water(self) -> float:
        """The highest cumulative realised PnL this account has ever reached.

        Derived from the settlements rather than stored on the account, so it
        needs no column and cannot drift out of sync with the ledger it
        summarises. `account.realized_pnl` is the current value of the same
        series, so the pair gives peak and current without a second source of
        truth.

        **Both sides must span the same period, and they did not.**
        `scripts/reset_dashboard.py --rebase` zeroes `account.realized_pnl` at a
        new epoch, while this walked every settlement ever recorded — so the
        drawdown breaker compared a high-water from BEFORE the reset against a
        P&L counted AFTER it. Measured 2026-09-24: peak $322.38 against a
        current $173.02, a phantom 46.3% drawdown that halted a live account
        whose realised P&L since its epoch was -$26.98. A halt is sticky and
        cleared by hand, so a false one costs real trading days.

        Bounded by `account.reset_at` when one is set, which makes the pair
        commensurable again. Research never reads this — it is a live-account
        breaker only.
        """
        with self._session() as session:
            epoch = (session.query(Account.reset_at)
                     .order_by(Account.id).limit(1).scalar())
            rows = (session.query(Position.pnl)
                    .filter(Position.outcome != Outcome.PENDING.value)
                    .filter(Position.settled_at.isnot(None)))
            if epoch is not None:
                rows = rows.filter(Position.settled_at >= epoch)
            rows = rows.order_by(Position.settled_at.asc())
            peak = running = 0.0
            for (value,) in rows:
                running += float(value or 0.0)
                peak = max(peak, running)
            return peak

    # ---- the market benchmark -------------------------------------------
    def scored_against_market(self) -> list[tuple]:
        """Every settled window where the venue's own probability was recorded.

        `(symbol, window_open, offset_minutes, market_probability,
        baseline_probability, model_probability, outcome, decision_time)`.

        Traded *and* refused, which is the point — the traded subset is selected by
        our own model's opinion, and it is our own opinion under test.

        `decision_time` is carried because it is not `window_open + offset`. The
        features are built for the nominal offset while the quote is read whenever
        the cycle got there — measured, +3.62m on average for a +3m decision. One
        minute of information is worth ~0.027 nats against a total model edge of
        +0.002, so that lag is comparable to the whole effect and runs against us.
        Reporting it is how the comparison stops being silently unfair.
        """
        with self._session() as session:
            return list(
                session.query(
                    Prediction.symbol, Prediction.window_open,
                    Prediction.offset_minutes, Prediction.market_probability,
                    Prediction.baseline_probability, Prediction.model_probability,
                    Prediction.outcome, Prediction.decision_time)
                .filter(Prediction.market_probability.isnot(None))
                .filter(Prediction.outcome.isnot(None))
                .order_by(Prediction.window_open)
            )


    def windows_awaiting_outcome(self, now: datetime, *, limit: int = 500
                                ) -> list[tuple[str, datetime, datetime, float]]:
        """Settled windows whose predictions still have no realised outcome.

        One row per (symbol, window), with the strike, newest first. This is what
        makes the market benchmark possible: `market_probability` is recorded on
        every window including the refused ones, and without the outcome beside it
        the venue's probability can never be scored. `positions` cannot substitute
        — it only covers the ~6% of windows that traded, which is a selected
        sample and the wrong one.
        """
        with self._session() as session:
            rows = (session.query(Prediction.symbol, Prediction.window_open,
                                  Prediction.settle_time, Prediction.strike)
                    .filter(Prediction.settle_time <= now)
                    .filter(Prediction.outcome.is_(None))
                    .group_by(Prediction.symbol, Prediction.window_open,
                              Prediction.settle_time, Prediction.strike)
                    .order_by(Prediction.window_open.desc())
                    .limit(limit))
            return [(str(a), b, c, float(d)) for a, b, c, d in rows]

    def set_window_outcome(self, symbol: str, window_open: datetime,
                           *, settled_up: bool) -> int:
        """Record the realised outcome on every offset of one window."""
        with self._session() as session:
            n = (session.query(Prediction)
                 .filter(Prediction.symbol == symbol,
                         Prediction.window_open == window_open,
                         Prediction.outcome.is_(None))
                 .update({Prediction.outcome: 1 if settled_up else 0},
                         synchronize_session=False))
            session.commit()
            return int(n)

    def correct_positions_from_venue(self, labels) -> tuple[int, int, float]:
        """Re-settle positions the VENUE graded differently, and fix the P&L.

        **`settle_due` prefers the venue's settlement, but only if it has one
        at that moment.** When the venue's row has not arrived by the time the
        window matures it falls back to our Coinbase bars, and nothing ever
        re-grades it — the same "only fills what is empty" shape that
        `set_window_outcome` had for predictions.

        Measured 2026-10-03 over 216 settled trades since the epoch: our books
        showed **130 winners against the venue's 123**, a 3.2% disagreement
        that matches the documented Coinbase-vs-BRTI rate almost exactly, all
        of it in near-ties. It was worth $21.06 — our books read +$8.80 where
        the venue read -$12.26.

        That is not merely a reporting gap. `account.realized_pnl` is what the
        daily-loss limit and the drawdown breaker read, so both were keyed to a
        P&L optimistic by that amount: the safety brakes under-react by exactly
        the margin our own label flatters us.

        **`bankroll` is deliberately NOT adjusted here.** `adopt_venue_balance`
        writes the venue's balance every cycle and the venue is the account of
        record; correcting a derived figure the venue already owns would be a
        second source of truth. `realized_pnl` has no such owner, which is why
        it drifted.

        `labels` is an iterable of `(symbol, window_open, settled_up)`.
        Returns `(examined, corrected, pnl_delta)`.
        """
        examined = corrected = 0
        delta_total = 0.0
        with self._session() as session:
            for symbol, window_open, settled_up in labels:
                if settled_up is None:
                    continue
                examined += 1
                rows = (session.query(Position)
                        .filter(Position.symbol == symbol,
                                Position.window_open == window_open,
                                Position.settled_at.isnot(None))
                        .all())
                for row in rows:
                    if bool(row.settled_up) == bool(settled_up):
                        continue
                    won = _won(side=row.side, settled_up=bool(settled_up))
                    payout = float(row.contracts) if won else 0.0
                    before = float(row.pnl or 0.0)
                    row.settled_up = bool(settled_up)
                    row.payout = payout
                    # **Net of the fee, like `settle_position`.** This read
                    # `payout - outlay` while the settle path reads
                    # `payout - outlay - fee`, so any position the venue
                    # re-graded silently became fee-free -- and
                    # `account.realized_pnl` is restated as `sum(positions.pnl)`
                    # a few lines below, so the error propagated straight to
                    # the dashboard's headline.
                    row.pnl = (payout - float(row.outlay or 0.0)
                               - float(row.fee or 0.0))
                    row.outcome = Outcome.WON.value if won else Outcome.LOST.value
                    delta_total += float(row.pnl) - before
                    corrected += 1
            # **The invariant, restated every pass rather than only after a
            # correction.** It is a statement about what `realized_pnl` IS,
            # so enforcing it only when something changed leaves any drift
            # already present in place -- which is exactly what happened on
            # the first run here.
            # **Recomputed from the positions, not nudged by the delta.**
            # `account.realized_pnl` covers the period since
            # `account.reset_at` -- `reset_dashboard --rebase` zeroes it
            # there and `settle_position` accumulates forward from it. The
            # labels this walks reach further back than the epoch, so
            # applying the raw delta leaks pre-epoch corrections into a
            # post-epoch total: on the first real run that put it at
            # -$42.20 against post-epoch positions summing to -$12.20.
            #
            # Restating the invariant is both the fix and a repair for any
            # drift already there, from this or anything else:
            #
            #     realized_pnl == sum(positions.pnl where settled_at >= epoch)
            epoch = (session.query(Account.reset_at)
                     .order_by(Account.id).limit(1).scalar())
            total = session.query(func.coalesce(func.sum(Position.pnl), 0.0)) \
                .filter(Position.settled_at.isnot(None))
            if epoch is not None:
                total = total.filter(Position.settled_at >= epoch)
            session.query(Account).update(
                {Account.realized_pnl: float(total.scalar() or 0.0)},
                synchronize_session=False,
            )
            session.commit()
        return examined, corrected, delta_total

    def correct_outcomes_from_venue(self, labels) -> tuple[int, int]:
        """Overwrite `Prediction.outcome` where the VENUE settled differently.

        **The market gates are scored on this column, and it was filled from
        Coinbase bars only.** `settle_due` prefers the venue's own settlement
        for the money; `settle_predictions`, eleven lines away in the same
        function with the venue data already in scope, did not -- and
        `scored_against_market()` can return no other label. So
        `model_minus_market` and `calibration_vs_market` were computed on our
        proxy while the backtest branch of the same gate substitutes
        `venue_outcome`. Two label conventions under one gate name.

        Measured 2026-09-16 on 5,471 rows at +12m: the labels disagree on 2.85%,
        all of it in near-ties as expected, and the statistic moves
        **+0.000924 -> +0.000188**. Four fifths of the apparent live edge was
        self-grading.

        `set_window_outcome` deliberately fills only NULLs, so a venue label
        arriving after ours could never correct it. This one overwrites, and
        ONLY where the two differ, so a re-run is free and the count is the
        disagreement rather than the row count.

        `labels` is an iterable of `(symbol, window_open, settled_up)`.
        Returns `(examined, corrected)`.
        """
        examined = corrected = 0
        with self._session() as session:
            for symbol, window_open, settled_up in labels:
                if settled_up is None:
                    continue
                examined += 1
                value = 1 if settled_up else 0
                n = (session.query(Prediction)
                     .filter(Prediction.symbol == symbol,
                             Prediction.window_open == window_open,
                             Prediction.outcome.isnot(None),
                             Prediction.outcome != value)
                     .update({Prediction.outcome: value},
                             synchronize_session=False))
                corrected += int(n)
            session.commit()
        return examined, corrected

    # ---- account --------------------------------------------------------
    def ensure_account(self, starting_bankroll: float,
                       *, mode: str = 'paper') -> Account:
        if mode not in ('paper', 'live'):
            raise ValueError(f"account mode must be 'paper' or 'live', got {mode!r}")
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                row = Account(mode=mode, starting_bankroll=starting_bankroll,
                              bankroll=starting_bankroll)
                session.add(row)
                session.commit()
                session.refresh(row)
                return row
            # `mode` used to be set only on creation. Paper is the compose
            # default, so the first run created a paper account and every later
            # `--mode live` run kept rendering as paper on every dashboard
            # surface — the failure this schema's own comment calls the worst it
            # could permit. Switching mode is not a display change either: the
            # bankroll history belongs to the other mode, so refuse rather than
            # silently inherit it.
            if row.mode != mode:
                raise AccountModeMismatch(
                    f"this account was opened in {row.mode!r} mode and holds its "
                    f"bankroll and settled history. Running in {mode!r} mode "
                    f"against it would report one mode's money under the other's "
                    f"name. Use a separate DATABASE_URL for {mode!r}, or reset "
                    f"this one deliberately."
                )
            session.refresh(row)
            return row

    def account(self) -> Optional[Account]:
        with self._session() as session:
            return session.query(Account).order_by(Account.id).first()

    def update_account(self, **fields) -> None:
        """Set absolute fields on the account.

        For anything that is an *increment* — a stake debited, a payout credited,
        a fee accrued — use `adjust_account` instead. This method reads the row
        into Python and writes it back, so two overlapping callers lose one
        another's change.
        """
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                raise ValueError('no account row; call ensure_account first')
            for key, value in fields.items():
                setattr(row, key, value)
            session.commit()

    def adjust_account(self, *, bankroll_delta: float = 0.0,
                       fees_delta: float = 0.0,
                       realized_delta: float = 0.0) -> None:
        """Apply increments to the account in one atomic statement.

        `bankroll = bankroll - stake` used to be computed in Python between two
        separate transactions. Nothing enforces one trader process, so a
        supervisor restart overlap or an operator running paper and live against
        one `DATABASE_URL` silently dropped one of the two debits. Doing the
        arithmetic in SQL removes the read entirely, so there is no window to
        lose.
        """
        with self._session() as session:
            updated = session.query(Account).update(
                {Account.bankroll: Account.bankroll + bankroll_delta,
                 Account.fees_paid: Account.fees_paid + fees_delta,
                 Account.realized_pnl: Account.realized_pnl + realized_delta},
                synchronize_session=False,
            )
            if not updated:
                raise ValueError('no account row; call ensure_account first')
            session.commit()

    def write_equity_point(self, **fields) -> int:
        with self._session() as session:
            row = EquityPoint(**fields)
            session.add(row)
            session.commit()
            return row.id

    # ---- the venue ledger -----------------------------------------------
    #
    # Written by the live loop every cycle (from the reconcile it already
    # performs) and by `scripts.sync_venue` for the deep, paginated backfill.
    # Both paths upsert on the venue's own key, so running them together, or the
    # same one twice, converges rather than double-counting.

    def upsert_venue_fills(self, rows: Iterable[dict]) -> int:
        """Upsert fills on `trade_id`. Returns the number of rows written.

        Idempotent because it has to be: the live and historical tiers overlap
        around the cutoff, so the same fill arrives twice in one sync, and a fill
        counted twice doubles a cost basis.

        A row's fields are overwritten on conflict rather than skipped. The venue
        amends its own records — a `_dollars` field appearing where only cents were
        served before is the documented direction of travel — and a stale row that
        parsed badly the first time should heal on the next sync instead of being
        preserved forever by an INSERT ... DO NOTHING.
        """
        written = 0
        with self._session() as session:
            for row in rows:
                trade_id = row.get('trade_id')
                if not trade_id:
                    logger.warning('a venue fill arrived with no trade_id; '
                                   'skipping rather than inventing a key')
                    continue
                existing = session.query(VenueFill).filter_by(
                    trade_id=trade_id).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    existing.synced_at = utcnow()
                else:
                    session.add(VenueFill(**row))
                written += 1
            session.commit()
        return written

    def upsert_venue_settlements(self, rows: Iterable[dict]) -> int:
        """Upsert settlements on `ticker`. Returns the number of rows written."""
        written = 0
        with self._session() as session:
            for row in rows:
                ticker = row.get('ticker')
                if not ticker:
                    continue
                existing = session.query(VenueSettlement).filter_by(
                    ticker=ticker).one_or_none()
                if existing is not None:
                    for key, value in row.items():
                        setattr(existing, key, value)
                    existing.synced_at = utcnow()
                else:
                    session.add(VenueSettlement(**row))
                written += 1
            session.commit()
        return written

    def venue_ledger_keys(self) -> tuple[set[str], set[str]]:
        """What is already stored, as two sets, in two queries.

        The live loop reconciles every cycle and the venue returns up to 200 fills
        and 200 settlements each time — almost all of them rows it returned last
        cycle too. Upserting all four hundred means four hundred SELECT-then-write
        round trips a minute in the steady state, for no change.

        So the caller filters against these first. The settlement set deliberately
        holds only tickers whose `pnl` is **not** null: a row that parsed
        incompletely is left out, so it is retried and can heal when the venue
        fills the gap in. Filtering on mere presence would freeze the first bad
        parse in place forever.
        """
        with self._session() as session:
            trade_ids = {tid for (tid,) in session.query(VenueFill.trade_id)}
            tickers = {t for (t,) in session.query(VenueSettlement.ticker)
                       .filter(VenueSettlement.pnl.is_not(None))}
        return trade_ids, tickers

    def venue_fills(self, *, since: Optional[datetime] = None,
                    limit: int = 200) -> list[VenueFill]:
        with self._session() as session:
            query = session.query(VenueFill)
            if since is not None:
                query = query.filter(VenueFill.created_time >= since)
            return list(query.order_by(VenueFill.created_time.desc()).limit(limit))

    def venue_settlements(self, *, since: Optional[datetime] = None,
                          limit: Optional[int] = None) -> list[VenueSettlement]:
        """Settled markets, newest first. `limit=None` for the whole ledger.

        The default is unlimited because the P&L total is a sum over everything: a
        limit would silently report the profit of the most recent N trades as the
        account's profit. Callers drawing a table pass one; callers totalling do
        not.
        """
        with self._session() as session:
            query = session.query(VenueSettlement)
            if since is not None:
                query = query.filter(VenueSettlement.settled_time >= since)
            query = query.order_by(VenueSettlement.settled_time.desc())
            if limit is not None:
                query = query.limit(limit)
            return list(query)

    def replace_venue_positions(self, rows, *, at) -> int:
        """Store the venue's open positions, replacing whatever was there.

        Current state, not a series -- a venue that reports nothing open must
        leave an EMPTY table, so the delete runs whether or not there are rows
        to write. Appending instead would leave a settled position on screen
        forever, which is the failure this is meant to prevent.
        """
        with self._session() as session:
            session.query(VenuePosition).delete()
            for row in rows:
                session.add(VenuePosition(
                    ticker=str(row.get('ticker') or ''),
                    exchange_index=_int_or_none(row.get('exchange_index')),
                    position=float(row.get('position') or 0.0),
                    market_exposure=_finite_or_none(row.get('market_exposure')),
                    fees_paid=_finite_or_none(row.get('fees_paid')),
                    realized_pnl=_finite_or_none(row.get('realized_pnl')),
                    sampled_at=at))
            session.commit()
            return len(rows)

    def position_for_ticker(self, ticker: str):
        """Our own position for a venue ticker, via the ticket that placed it.

        `positions` has no ticker column — a position is keyed on (symbol, window)
        because the paper engine predates the venue client — so the join goes
        through `order_tickets`, which records the market the order was actually
        sent to. Returns None when nothing maps, which is what the venue settling
        a market we never booked looks like.
        """
        with self._session() as session:
            ticket = (session.query(OrderTicket)
                      .filter(OrderTicket.market_ticker == ticker)
                      .order_by(OrderTicket.created_at.desc())
                      .first())
            if ticket is None:
                return None
            # **Matched on the offset as well.** A window can hold one position
            # per decision offset, so filtering on (symbol, window) alone would
            # raise `MultipleResultsFound` and take the reconcile down with it.
            # The ticket names the decision, so it names the position.
            return (session.query(Position)
                    .filter(Position.symbol == ticket.symbol,
                            Position.window_open == ticket.window_open,
                            Position.offset_minutes == ticket.offset_minutes)
                    .one_or_none())

    def write_venue_balance(self, *, timestamp: datetime, balance: float,
                            exchange_index: Optional[int] = None,
                            our_bankroll: Optional[float] = None,
                            total_balance: Optional[float] = None,
                            portfolio_value: Optional[float] = None) -> int:
        """Sample the venue's balance. The drift is computed here, once.

        Sampled rather than upserted: the point of the series is the trend, and
        every cycle's reading is a distinct observation of a moving number.
        """
        drift = (None if our_bankroll is None
                 else float(balance) - float(our_bankroll))
        with self._session() as session:
            row = VenueBalance(
                timestamp=timestamp, balance=float(balance),
                exchange_index=exchange_index, our_bankroll=our_bankroll,
                drift=drift, total_balance=_finite_or_none(total_balance),
                portfolio_value=_finite_or_none(portfolio_value),
            )
            session.add(row)
            session.commit()
            return row.id

    def last_exchange_index(self) -> Optional[int]:
        """The shard the last sampled balance came from, or None.

        **Seeds `_LAST_EXCHANGE_INDEX` across a restart.** That global starts
        None, so the first cycles after every restart reconcile with
        `exchange_index=None` -- which reads the CROSS-SHARD TOTAL and adopts
        it as the bankroll. Measured 2026-10-20: $495.1542 adopted against a
        true shard-2 balance of $494.0487, a $1.1055 overstatement that then
        sizes Kelly until a book is read and the shard relearned. The shard
        moves roughly never, and the last one is already recorded here.
        """
        with self._session() as session:
            value = (session.query(VenueBalance.exchange_index)
                     .filter(VenueBalance.exchange_index.isnot(None))
                     .order_by(VenueBalance.timestamp.desc())
                     .limit(1).scalar())
        return int(value) if value is not None else None

    def venue_balances_since(self, since: datetime) -> list[VenueBalance]:
        with self._session() as session:
            return list(
                session.query(VenueBalance)
                .filter(VenueBalance.timestamp >= since)
                .order_by(VenueBalance.timestamp)
            )

    def latest_venue_balance(self) -> Optional[VenueBalance]:
        with self._session() as session:
            return (session.query(VenueBalance)
                    .order_by(VenueBalance.timestamp.desc())
                    .first())

    def prune_venue_balances(self, before: datetime) -> int:
        """Drop balance samples that have scrolled off the chart.

        One row a minute is 1,440 a day, the same order as `minute_prices`, and
        for the same reason it is pruned rather than kept forever: this is a
        rolling window for a screen. The settlements are the record and they are
        never pruned.
        """
        with self._session() as session:
            n = (session.query(VenueBalance)
                 .filter(VenueBalance.timestamp < before).delete())
            session.commit()
            return int(n)

    # ---- model runs and calibration -------------------------------------
    def record_model_run(self, **fields) -> int:
        with self._session() as session:
            existing = session.query(ModelRun).filter_by(
                version=fields['version']).one_or_none()
            if existing is not None:
                for key, value in fields.items():
                    setattr(existing, key, value)
                session.commit()
                return existing.id
            row = ModelRun(**fields)
            session.add(row)
            session.commit()
            return row.id

    def write_calibration(self, model_version: str, source: str,
                          bins: Iterable[dict]) -> int:
        """Replace the reliability table for one (version, source)."""
        written = 0
        with self._session() as session:
            session.query(CalibrationBin).filter_by(
                model_version=model_version, source=source).delete()
            for row in bins:
                session.add(CalibrationBin(model_version=model_version,
                                           source=source, **row))
                written += 1
            session.commit()
        return written
