"""The serving schema, mirrored from `backend/trader/core/pg_writer.py`.

The trader and the API run in separate containers and neither imports the
other, so the ORM exists twice. Everything below the imports is copied from the
trader's module: change it there, copy it here, and run
`backend/trader/tests/test_orm_parity.py`.
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

    # The venue's view and the realised outcome, on every window, traded or
    # refused, so the market's probability can be scored against the outcome.
    #
    # `market_probability` is the mid (the venue's belief). The two asks are
    # what a trade would cost.
    #
    # `price_source` says what priced the decision: 'baseline' in a backtest,
    # which has no quotes, or the real book live.
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

    A rolling window for the screen; `prune` drops old rows. The research store
    keeps the authoritative bars.
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

    `mode` is 'paper' or 'live'. Every surface that shows a number from this
    row must also show its mode.
    """

    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, nullable=False, default='paper', index=True)
    starting_bankroll = Column(Float, nullable=False, default=100.0)
    bankroll = Column(Float, nullable=False, default=100.0)
    # The epoch the dashboard reads from. It filters what is shown and deletes
    # nothing. See `controllers.serving.reset_epoch`.
    reset_at = Column(DateTime(timezone=True), nullable=True)
    staked = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    fees_paid = Column(Float, nullable=False, default=0.0)
    halted = Column(Boolean, nullable=False, default=False)
    halted_reason = Column(String, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())


class EquityPoint(Base):
    """Equity at a settlement. Never marked to a model probability.

    Marking at our own forecast would book unearned edge as profit. `staked` is
    carried at cost.
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
    """One row of the reliability table: predicted against observed, per bin."""

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
    """A live decision: everything needed to place the order, and what came back.

    `status` moves new -> placed -> filled, or new -> skipped.
    """

    __tablename__ = 'order_tickets'

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String, nullable=False, index=True)
    window_open = Column(DateTime(timezone=True), nullable=False, index=True)
    settle_time = Column(DateTime(timezone=True), nullable=False)
    offset_minutes = Column(Integer, nullable=False)

    # The venue's market identifier, resolved by asking the venue, never built
    # from a pattern.
    market_ticker = Column(String, nullable=True, index=True)
    venue_order_id = Column(String, nullable=True)

    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    # The price the decision was sized at, and the worst price still worth
    # paying. Every ticket has a limit.
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
    """One execution, as the venue recorded it. Where it differs from `Position`, this is right.

    Keyed on the venue's `trade_id`, so a re-sync is idempotent: two writers
    and two overlapping ledger tiers must not count a fill twice.

    `raw` keeps the whole payload, including fields the parser does not read.
    """

    __tablename__ = 'venue_fills'

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String, nullable=False, unique=True, index=True)
    order_id = Column(String, nullable=True, index=True)
    ticker = Column(String, nullable=False, index=True)

    # 'up' or 'down' for the venue's yes/no, translated once, in the parser.
    side = Column(String, nullable=False)
    action = Column(String, nullable=True)
    # Float, because the wire encoding is a fixed-point string (`count_fp`:
    # "5.00").
    contracts = Column(Float, nullable=False)
    # Dollars paid per contract, for `side`. Null means unknown, not zero.
    price = Column(Float, nullable=True)
    is_taker = Column(Boolean, nullable=True)
    created_time = Column(DateTime(timezone=True), nullable=True, index=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_fills_ticker_time', 'ticker', 'created_time'),
    )


class VenueSettlement(Base):
    """One market resolved, as the venue paid it. This is the P&L.

    A held binary pays one fee at entry and settles once, so this row holds its
    cost, payout and fee. `Position.pnl` only estimates them.

    `pnl` is stored, not derived on read, because the API cannot import the
    trader; this keeps one definition. Null `pnl` means the venue left a field
    absent. It is not break-even, and nothing may read it as zero.
    """

    __tablename__ = 'venue_settlements'

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, unique=True, index=True)
    event_ticker = Column(String, nullable=True)
    # 'yes' or 'no', as the venue resolved it.
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

    # Our own books beside the venue's. A growing gap is a mispriced fee or a
    # wrong proxy settlement. Null when no position of ours maps to this ticker.
    position_id = Column(Integer, nullable=True, index=True)
    our_pnl = Column(Float, nullable=True)

    raw = Column(JSON, nullable=True)
    synced_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_settlements_settled', 'settled_time'),
    )


class VenueBalance(Base):
    """The venue's balance, sampled every cycle.

    It is cash on one shard, not equity: an entry debits it and settlement
    credits it. Balances are local to a shard (`exchange_index`); a total
    across shards is not spendable by any single order.
    """

    __tablename__ = 'venue_balance'

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    balance = Column(Float, nullable=False)
    exchange_index = Column(Integer, nullable=True)
    # Our own bankroll at the same instant, and the difference. Stored, so a
    # growing drift (unrecorded fill, partial, mispriced fee) is visible.
    our_bankroll = Column(Float, nullable=True)
    drift = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('ix_venue_balance_timestamp_desc', 'timestamp'),
    )

# Indexes and constraints added after a table first shipped. test_orm_parity
# asserts this list equals the trader's.
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
    # The venue ledger: every read is "newest first" or "since a timestamp".
    'CREATE INDEX IF NOT EXISTS ix_venue_fills_created_desc '
    'ON venue_fills (created_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_settlements_settled_desc '
    'ON venue_settlements (settled_time DESC)',
    'CREATE INDEX IF NOT EXISTS ix_venue_balance_recent '
    'ON venue_balance (timestamp DESC)',
)
