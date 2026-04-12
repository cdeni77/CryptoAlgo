"""Postgres writer for trader service, including paper-trading persistence."""

import enum
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# ── Inline model definitions (mirror of the API models) ────────────
# We duplicate the ORM classes here so the trader doesn't need to
# import from the API codebase (they run in separate containers).
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, String, Text, create_engine, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class TradeSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


class PaperOrderStatus(str, enum.Enum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, nullable=False, index=True)
    datetime_open = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    datetime_close = Column(DateTime(timezone=True), nullable=True)
    side = Column(Enum(TradeSide), nullable=False)
    contracts = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    fee_open = Column(Float, nullable=True, default=0.0)
    fee_close = Column(Float, nullable=True, default=0.0)
    net_pnl = Column(Float, nullable=True)
    margin_used = Column(Float, nullable=True)
    leverage = Column(Float, nullable=True)
    reason_entry = Column(Text, nullable=True)
    reason_exit = Column(Text, nullable=True)
    status = Column(Enum(TradeStatus), default=TradeStatus.OPEN, nullable=False)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    direction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    raw_probability = Column(Float, nullable=True)
    model_auc = Column(Float, nullable=True)
    price_at_signal = Column(Float, nullable=True)
    momentum_pass = Column(Boolean, nullable=True)
    trend_pass = Column(Boolean, nullable=True)
    regime_pass = Column(Boolean, nullable=True)
    ml_pass = Column(Boolean, nullable=True)
    contracts_suggested = Column(Integer, nullable=True)
    notional_usd = Column(Float, nullable=True)
    acted_on = Column(Boolean, default=False)
    trade_id = Column(Integer, nullable=True)
    passed_gates = Column(Boolean, nullable=False, default=True)
    gate_failure_reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Wallet(Base):
    __tablename__ = "wallet"
    id = Column(Integer, primary_key=True)
    balance = Column(Float, default=10000.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ModelRun(Base):
    __tablename__ = "model_runs"
    id = Column(Integer, primary_key=True, index=True)
    run_started_at = Column(DateTime(timezone=True), nullable=False, index=True)
    run_finished_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, nullable=False, index=True)
    retrain_window_days = Column(Integer, nullable=False, default=90)
    symbols_total = Column(Integer, nullable=False, default=0)
    symbols_trained = Column(Integer, nullable=False, default=0)
    artifacts_version = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class PaperOrder(Base):
    __tablename__ = "paper_orders"
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, nullable=False, index=True)
    coin = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    target_price = Column(Float, nullable=False)
    status = Column(Enum(PaperOrderStatus), nullable=False, default=PaperOrderStatus.NEW)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PaperFill(Base):
    __tablename__ = "paper_fills"
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, nullable=False, index=True)
    signal_id = Column(Integer, nullable=False, index=True)
    coin = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    fill_price = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    slippage_bps = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    contracts = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    mark_price = Column(Float, nullable=False)
    notional = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    fees_paid = Column(Float, nullable=False, default=0.0)
    opened_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_open = Column(Boolean, nullable=False, default=True)
    # Exit parameters frozen at open time — paper engine uses these to close positions
    tp_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    max_hold_until = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String, nullable=True)


class PaperEquityCurve(Base):
    __tablename__ = "paper_equity_curve"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False, default=0)


class PaperEngineConfig(Base):
    """Single-row table written by the paper engine on startup to expose its runtime config."""
    __tablename__ = "paper_engine_config"
    id = Column(Integer, primary_key=True, default=1)
    active_coins = Column(JSON, nullable=False, default=list)
    tier_map = Column(JSON, nullable=False, default=dict)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PgWriter:
    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL env var is not set.")
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self._run_pg_migrations()

    def _run_pg_migrations(self) -> None:
        """Add columns that post-date create_all(). Idempotent — safe on every startup."""
        stmts = [
            "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION",
            "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS sl_price DOUBLE PRECISION",
            "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS max_hold_until TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS exit_reason VARCHAR",
        ]
        with self.engine.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))

    def _session(self) -> Session:
        return self.SessionLocal()

    # ── Signals ─────────────────────────────────────────────────────
    def write_signal(
        self,
        coin: str,
        timestamp: datetime,
        direction: str,
        confidence: float,
        mode: str = "signals",
        idempotency_key: str | None = None,
        raw_probability: float | None = None,
        model_auc: float | None = None,
        price_at_signal: float | None = None,
        momentum_pass: bool | None = None,
        trend_pass: bool | None = None,
        regime_pass: bool | None = None,
        ml_pass: bool | None = None,
        contracts_suggested: int | None = None,
        notional_usd: float | None = None,
        acted_on: bool = False,
        trade_id: int | None = None,
        passed_gates: bool = True,
        gate_failure_reason: str | None = None,
    ) -> int:
        """Insert one signal row. Returns the signal id (idempotent for coin+timestamp+direction)."""
        _ = mode
        with self._session() as db:
            if idempotency_key:
                # Deduplicate by coin+direction within the same minute window
                minute_start = timestamp.replace(second=0, microsecond=0)
                minute_end = minute_start + timedelta(minutes=1)
                existing = db.query(Signal).filter(
                    Signal.coin == coin,
                    Signal.timestamp >= minute_start,
                    Signal.timestamp < minute_end,
                    Signal.direction == direction,
                ).first()
                if existing:
                    return existing.id

            def _f(v):
                return float(v) if v is not None else None

            sig = Signal(
                coin=coin,
                timestamp=timestamp,
                direction=direction,
                confidence=_f(confidence),
                raw_probability=_f(raw_probability),
                model_auc=_f(model_auc),
                price_at_signal=_f(price_at_signal),
                momentum_pass=momentum_pass,
                trend_pass=trend_pass,
                regime_pass=regime_pass,
                ml_pass=ml_pass,
                contracts_suggested=int(contracts_suggested) if contracts_suggested is not None else None,
                notional_usd=_f(notional_usd),
                acted_on=acted_on,
                trade_id=trade_id,
                passed_gates=passed_gates,
                gate_failure_reason=gate_failure_reason,
            )
            db.add(sig)
            db.commit()
            db.refresh(sig)
            return sig.id

    # ── Trades ──────────────────────────────────────────────────────
    def open_trade(
        self,
        coin: str,
        side: str,
        contracts: float,
        entry_price: float,
        mode: str = "live",
        idempotency_key: str | None = None,
        fee_open: float = 0.0,
        margin_used: float | None = None,
        leverage: float | None = None,
        reason_entry: str | None = None,
    ) -> int:
        """Insert an open trade. Returns trade id."""
        _ = mode
        with self._session() as db:
            if idempotency_key:
                existing = db.query(Trade).filter(
                    Trade.coin == coin,
                    Trade.reason_entry == f"idem:{idempotency_key}",
                ).first()
                if existing:
                    return existing.id

            t = Trade(
                coin=coin,
                side=TradeSide(side),
                contracts=contracts,
                entry_price=entry_price,
                fee_open=fee_open,
                margin_used=margin_used,
                leverage=leverage,
                reason_entry=f"idem:{idempotency_key}" if idempotency_key else reason_entry,
                status=TradeStatus.OPEN,
            )
            db.add(t)
            db.commit()
            db.refresh(t)
            return t.id

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        mode: str = "live",
        idempotency_key: str | None = None,
        fee_close: float = 0.0,
        net_pnl: float | None = None,
        reason_exit: str | None = None,
    ) -> bool:
        """Close an existing trade. Returns True on success."""
        _ = mode
        with self._session() as db:
            if idempotency_key:
                existing = db.query(Trade).filter(
                    Trade.reason_exit == f"idem:{idempotency_key}",
                    Trade.status == TradeStatus.CLOSED,
                ).first()
                if existing:
                    return True

            t = db.query(Trade).filter(Trade.id == trade_id).first()
            if not t:
                return False
            t.exit_price = exit_price
            t.fee_close = fee_close
            t.net_pnl = net_pnl
            t.reason_exit = f"idem:{idempotency_key}" if idempotency_key else reason_exit
            t.datetime_close = datetime.now(timezone.utc)
            t.status = TradeStatus.CLOSED
            db.commit()
            return True

    def update_balance(self, new_balance: float) -> None:
        with self._session() as db:
            w = db.query(Wallet).order_by(Wallet.id.desc()).first()
            if w:
                w.balance = new_balance
            else:
                db.add(Wallet(balance=new_balance))
            db.commit()

    # ── Model runs ──────────────────────────────────────────────────
    def create_model_run(self, retrain_window_days: int, symbols_total: int, artifacts_version: str | None = None) -> int:
        with self._session() as db:
            run = ModelRun(
                run_started_at=datetime.now(timezone.utc),
                status="running",
                retrain_window_days=retrain_window_days,
                symbols_total=symbols_total,
                artifacts_version=artifacts_version,
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run.id

    def complete_model_run(
        self,
        run_id: int,
        success: bool,
        symbols_trained: int,
        metrics: dict | None = None,
        error: str | None = None,
    ) -> bool:
        with self._session() as db:
            run = db.query(ModelRun).filter(ModelRun.id == run_id).first()
            if not run:
                return False
            run.run_finished_at = datetime.now(timezone.utc)
            run.status = "success" if success else "failed"
            run.symbols_trained = symbols_trained
            run.metrics = metrics
            run.error = error
            db.commit()
            return True
    def upsert_wallet_balance(self, new_balance: float) -> None:
        """Alias for wallet balance upsert used by the trader loops."""
        self.update_balance(new_balance)
    def create_paper_order(self, signal_id: int, coin: str, side: str, contracts: int, target_price: float) -> int:
        with self._session() as db:
            order = PaperOrder(
                signal_id=signal_id,
                coin=coin,
                side=side,
                contracts=contracts,
                target_price=target_price,
                status=PaperOrderStatus.NEW,
            )
            db.add(order)
            db.commit()
            db.refresh(order)
            return order.id

    def mark_paper_order_filled(self, order_id: int) -> None:
        with self._session() as db:
            order = db.query(PaperOrder).filter(PaperOrder.id == order_id).first()
            if order:
                order.status = PaperOrderStatus.FILLED
                db.commit()

    def create_paper_fill(self, order_id: int, signal_id: int, coin: str, side: str, contracts: int, fill_price: float, fee: float, notional: float, slippage_bps: float = 0.0) -> int:
        with self._session() as db:
            fill = PaperFill(
                order_id=order_id,
                signal_id=signal_id,
                coin=coin,
                side=side,
                contracts=contracts,
                fill_price=fill_price,
                fee=fee,
                notional=notional,
                slippage_bps=slippage_bps,
            )
            db.add(fill)
            db.commit()
            db.refresh(fill)
            return fill.id

    def get_open_paper_position(self, coin: str) -> Optional[PaperPosition]:
        with self._session() as db:
            return (
                db.query(PaperPosition)
                .filter(PaperPosition.coin == coin, PaperPosition.is_open.is_(True))
                .order_by(PaperPosition.id.desc())
                .first()
            )

    def get_all_open_paper_positions_for_coin(self, coin: str) -> list[PaperPosition]:
        with self._session() as db:
            return (
                db.query(PaperPosition)
                .filter(PaperPosition.coin == coin, PaperPosition.is_open.is_(True))
                .order_by(PaperPosition.id.asc())
                .all()
            )

    def get_all_open_paper_positions(self) -> list[PaperPosition]:
        with self._session() as db:
            return (
                db.query(PaperPosition)
                .filter(PaperPosition.is_open.is_(True))
                .order_by(PaperPosition.id.asc())
                .all()
            )

    def update_paper_position_mark(self, position_id: int, mark_price: float, unrealized_pnl: float) -> None:
        with self._session() as db:
            position = db.query(PaperPosition).filter(PaperPosition.id == position_id).first()
            if position:
                position.mark_price = mark_price
                position.unrealized_pnl = unrealized_pnl
                db.commit()

    def get_latest_signal_price(self, coin: str) -> Optional[float]:
        with self._session() as db:
            sig = (
                db.query(Signal.price_at_signal)
                .filter(Signal.coin == coin, Signal.price_at_signal.isnot(None))
                .order_by(Signal.id.desc())
                .first()
            )
            return float(sig.price_at_signal) if sig else None

    def upsert_paper_position(self, coin: str, side: str, contracts: int, entry_price: float, mark_price: float, notional: float, realized_pnl: float, unrealized_pnl: float, fees_paid: float, is_open: bool = True, tp_price: float | None = None, sl_price: float | None = None, max_hold_until: datetime | None = None) -> int:
        # Always INSERT a new position row. The old position must be explicitly closed via
        # close_paper_position() before this is called. Overwriting in-place would cause
        # "re-entry without close" — the side/entry_price would silently flip on the same row.
        import logging as _logging
        _log = _logging.getLogger("pg_writer")
        with self._session() as db:
            existing = (
                db.query(PaperPosition)
                .filter(PaperPosition.coin == coin, PaperPosition.side == side, PaperPosition.is_open.is_(True))
                .first()
            )
            if existing:
                _log.warning(
                    "upsert_paper_position: same-side open position already exists for %s %s (id=%s) — skipping insert",
                    coin, side, existing.id,
                )
                return existing.id
            position = PaperPosition(
                coin=coin,
                side=side,
                contracts=contracts,
                entry_price=entry_price,
                mark_price=mark_price,
                notional=notional,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                fees_paid=fees_paid,
                is_open=is_open,
                tp_price=tp_price,
                sl_price=sl_price,
                max_hold_until=max_hold_until,
            )
            db.add(position)
            db.commit()
            db.refresh(position)
            return position.id

    def close_paper_position(self, position_id: int, mark_price: float, realized_pnl: float, fees_paid: float, exit_reason: str | None = None) -> None:
        with self._session() as db:
            position = db.query(PaperPosition).filter(PaperPosition.id == position_id).first()
            if not position:
                return
            position.mark_price = mark_price
            position.realized_pnl = realized_pnl
            position.unrealized_pnl = 0.0
            position.fees_paid = fees_paid
            position.is_open = False
            if exit_reason:
                position.exit_reason = exit_reason
            db.commit()

    def get_recent_signal_prices_for_coin(self, coin: str, limit: int = 48) -> list[tuple[datetime, float]]:
        """Return the last `limit` (timestamp, price_at_signal) pairs for a coin, oldest first."""
        with self._session() as db:
            rows = (
                db.query(Signal.timestamp, Signal.price_at_signal)
                .filter(Signal.coin == coin, Signal.price_at_signal.isnot(None))
                .order_by(Signal.id.desc())
                .limit(limit)
                .all()
            )
        return [(r.timestamp, float(r.price_at_signal)) for r in reversed(rows)]

    def write_paper_equity_point(self, equity: float, cash_balance: float, unrealized_pnl: float, realized_pnl: float, open_positions: int, timestamp: Optional[datetime] = None) -> int:
        with self._session() as db:
            row = PaperEquityCurve(
                timestamp=timestamp or datetime.now(timezone.utc),
                equity=equity,
                cash_balance=cash_balance,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                open_positions=open_positions,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return row.id

    def get_unprocessed_signals(self, since_id: int) -> list[Signal]:
        with self._session() as db:
            return (
                db.query(Signal)
                .filter(Signal.id > since_id, Signal.acted_on.is_(False), Signal.passed_gates.is_(True))
                .order_by(Signal.id.asc())
                .all()
            )

    def mark_signal_acted(self, signal_id: int) -> None:
        with self._session() as db:
            sig = db.query(Signal).filter(Signal.id == signal_id).first()
            if sig:
                sig.acted_on = True
                db.commit()

    def count_open_positions(self) -> int:
        with self._session() as db:
            return db.query(PaperPosition).filter(PaperPosition.is_open.is_(True)).count()

    def get_closed_paper_positions_since(self, since: datetime) -> list[PaperPosition]:
        with self._session() as db:
            return (
                db.query(PaperPosition)
                .filter(PaperPosition.is_open.is_(False), PaperPosition.updated_at >= since)
                .order_by(PaperPosition.updated_at.asc())
                .all()
            )

    def get_paper_equity_curve_since(self, since: datetime) -> list[PaperEquityCurve]:
        with self._session() as db:
            return (
                db.query(PaperEquityCurve)
                .filter(PaperEquityCurve.timestamp >= since)
                .order_by(PaperEquityCurve.timestamp.asc())
                .all()
            )

    def upsert_paper_engine_config(self, active_coins: list[str], tier_map: dict[str, str]) -> None:
        """Write (or overwrite) the single paper_engine_config row so the API can expose it."""
        with self._session() as db:
            row = db.query(PaperEngineConfig).filter(PaperEngineConfig.id == 1).first()
            if row:
                row.active_coins = sorted(active_coins)
                row.tier_map = tier_map
            else:
                db.add(PaperEngineConfig(id=1, active_coins=sorted(active_coins), tier_map=tier_map))
            db.commit()

    def compute_paper_state_from_history(self, initial_equity: float = 10_000.0) -> dict:
        """Reconstruct correct cash_balance/realized/unrealized from fill + position history.

        This is used on engine startup to avoid the reset-to-10000 bug after container restarts.
        Formula: cash = initial - sum(open_fees_from_fills) + sum(realized_pnl_from_closed_positions)
        """
        with self._session() as db:
            total_open_fees = db.query(func.sum(PaperFill.fee)).scalar() or 0.0
            total_realized = (
                db.query(func.sum(PaperPosition.realized_pnl))
                .filter(PaperPosition.is_open.is_(False))
                .scalar()
            ) or 0.0
            open_positions = (
                db.query(PaperPosition).filter(PaperPosition.is_open.is_(True)).all()
            )
            total_unrealized = sum(float(p.unrealized_pnl or 0) for p in open_positions)
        cash = initial_equity - float(total_open_fees) + float(total_realized)
        return {
            "cash_balance": cash,
            "realized_pnl": float(total_realized),
            "unrealized_pnl": total_unrealized,
        }
