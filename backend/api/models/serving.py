"""The serving schema, mirrored from `backend/trader/core/pg_writer.py`.

The trader and the API run in separate containers and neither imports the
other, so the ORM exists twice. `backend/trader/tests/test_orm_parity.py` fails
when the two copies diverge in columns, types, nullability, defaults or the
migration list — a note in a doc was not enough, and `wallet.balance` had
already drifted 10,000 against 100,000, which meant whichever container created
the row decided the account's starting balance.

**Everything below the imports is copied verbatim from the trader's module.**
Keep it that way: if a column needs to change, change it there and copy, then
run the parity test. The docstrings travel with it deliberately — the reason a
column does *not* exist (mark price, unrealised PnL, funding accrual, stop
levels) is as load-bearing as the ones that do.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from models.base import Base


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
    outlay = Column(Float, nullable=False)     # everything paid, fee included
    fee = Column(Float, nullable=False)
    model_probability = Column(Float, nullable=False)
    baseline_probability = Column(Float, nullable=False)
    edge = Column(Float, nullable=False)

    outcome = Column(String, nullable=False, default=Outcome.PENDING.value, index=True)
    settled_up = Column(Boolean, nullable=True)
    payout = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('symbol', 'window_open', name='uq_position_window'),
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
        UniqueConstraint('symbol', 'window_open', name='uq_ticket_window'),
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
)
