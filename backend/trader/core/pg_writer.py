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
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Index, Integer, String, Text,
    UniqueConstraint, create_engine, text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)

Base = declarative_base()


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


class PgWriter:
    """Everything the trader writes to the serving store."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.getenv('DATABASE_URL')
        if not url:
            raise ValueError('DATABASE_URL is not set')
        self._engine = create_engine(url, pool_pre_ping=True, future=True)
        self._sessions = sessionmaker(bind=self._engine, future=True)
        Base.metadata.create_all(self._engine)
        self._run_migrations()

    def _run_migrations(self) -> None:
        with self._engine.begin() as connection:
            for statement in MIGRATIONS:
                try:
                    connection.execute(text(statement))
                except Exception as exc:  # noqa: BLE001 - SQLite in tests lacks some syntax
                    logger.debug('migration skipped (%s): %s', exc, statement)

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

    def recent_predictions(self, limit: int = 200, *, traded_only: bool = False) -> list[Prediction]:
        with self._session() as session:
            query = session.query(Prediction)
            if traded_only:
                query = query.filter(Prediction.traded.is_(True))
            return list(query.order_by(Prediction.window_open.desc()).limit(limit))

    def refusal_counts(self, since: Optional[datetime] = None) -> dict[str, int]:
        """The funnel, as the dashboard shows it."""
        with self._session() as session:
            query = session.query(Prediction.reason, func.count(Prediction.id))
            if since is not None:
                query = query.filter(Prediction.window_open >= since)
            return {reason: int(count) for reason, count in query.group_by(Prediction.reason)}

    # ---- positions ------------------------------------------------------
    def open_position(self, **fields) -> int:
        with self._session() as session:
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

    def settle_position(self, position_id: int, *, settled_up: bool) -> Optional[float]:
        """Resolve one position and return its PnL, or None if already settled."""
        with self._session() as session:
            row = session.get(Position, position_id)
            if row is None or row.outcome != Outcome.PENDING.value:
                return None
            won = settled_up if row.side == 'up' else not settled_up
            payout = float(row.contracts) if won else 0.0
            row.settled_up = settled_up
            row.payout = payout
            row.pnl = payout - row.outlay
            row.outcome = Outcome.WON.value if won else Outcome.LOST.value
            row.settled_at = utcnow()
            session.commit()
            return row.pnl

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

    # ---- account --------------------------------------------------------
    def ensure_account(self, starting_bankroll: float) -> Account:
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                row = Account(starting_bankroll=starting_bankroll,
                              bankroll=starting_bankroll)
                session.add(row)
                session.commit()
                session.refresh(row)
            return row

    def account(self) -> Optional[Account]:
        with self._session() as session:
            return session.query(Account).order_by(Account.id).first()

    def update_account(self, **fields) -> None:
        with self._session() as session:
            row = session.query(Account).order_by(Account.id).first()
            if row is None:
                raise ValueError('no account row; call ensure_account first')
            for key, value in fields.items():
                setattr(row, key, value)
            session.commit()

    def write_equity_point(self, **fields) -> int:
        with self._session() as session:
            row = EquityPoint(**fields)
            session.add(row)
            session.commit()
            return row.id

    def equity_curve_since(self, since: datetime) -> list[EquityPoint]:
        with self._session() as session:
            return list(
                session.query(EquityPoint)
                .filter(EquityPoint.timestamp >= since)
                .order_by(EquityPoint.timestamp)
            )

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

    def latest_model_run(self) -> Optional[ModelRun]:
        with self._session() as session:
            return (session.query(ModelRun)
                    .order_by(ModelRun.created_at.desc()).first())

    def model_runs(self, limit: int = 50) -> list[ModelRun]:
        with self._session() as session:
            return list(session.query(ModelRun)
                        .order_by(ModelRun.created_at.desc()).limit(limit))

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

    def calibration(self, model_version: Optional[str] = None) -> list[CalibrationBin]:
        with self._session() as session:
            query = session.query(CalibrationBin)
            if model_version:
                query = query.filter_by(model_version=model_version)
            return list(query.order_by(CalibrationBin.source, CalibrationBin.bin_low))
