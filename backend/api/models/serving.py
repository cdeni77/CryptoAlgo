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
    """The single account row. A $100 bankroll, and what it is now."""

    __tablename__ = 'account'

    id = Column(Integer, primary_key=True, index=True)
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
)
