"""
pg_writer.py — Postgres writer for the trader service.

The trader imports this to persist trades and hourly ML signals to
the same Postgres database that the FastAPI backend reads from.

Usage inside the trader:
    from pg_writer import PgWriter

    writer = PgWriter()                       # reads DATABASE_URL from env
    writer.write_signal(coin="BTC", ...)
    writer.open_trade(coin="BTC", ...)
    writer.close_trade(trade_id=1, ...)
"""

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# ── Inline model definitions (mirror of the API models) ────────────
# We duplicate the ORM classes here so the trader doesn't need to
# import from the API codebase (they run in separate containers).
from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class TradeSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


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


# ====================================================================
class PgWriter:
    """Thin wrapper that the trader uses to write to Postgres."""

    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL env var is not set. "
                "Set it to e.g. postgresql+psycopg2://postgres:yourpassword@db:5432/trades_db"
            )
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        # Ensure tables exist (idempotent)
        Base.metadata.create_all(self.engine)

    def _session(self) -> Session:
        return self.SessionLocal()

    # ── Signals ─────────────────────────────────────────────────────
    def write_signal(
        self,
        coin: str,
        timestamp: datetime,
        direction: str,
        confidence: float,
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
    ) -> int:
        """Insert one signal row. Returns the signal id."""
        with self._session() as db:
            sig = Signal(
                coin=coin,
                timestamp=timestamp,
                direction=direction,
                confidence=confidence,
                raw_probability=raw_probability,
                model_auc=model_auc,
                price_at_signal=price_at_signal,
                momentum_pass=momentum_pass,
                trend_pass=trend_pass,
                regime_pass=regime_pass,
                ml_pass=ml_pass,
                contracts_suggested=contracts_suggested,
                notional_usd=notional_usd,
                acted_on=acted_on,
                trade_id=trade_id,
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
        fee_open: float = 0.0,
        margin_used: float | None = None,
        leverage: float | None = None,
        reason_entry: str | None = None,
    ) -> int:
        """Insert an open trade. Returns trade id."""
        with self._session() as db:
            t = Trade(
                coin=coin,
                side=TradeSide(side),
                contracts=contracts,
                entry_price=entry_price,
                fee_open=fee_open,
                margin_used=margin_used,
                leverage=leverage,
                reason_entry=reason_entry,
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
        fee_close: float = 0.0,
        net_pnl: float | None = None,
        reason_exit: str | None = None,
    ) -> bool:
        """Close an existing trade. Returns True on success."""
        with self._session() as db:
            t = db.query(Trade).filter(Trade.id == trade_id).first()
            if not t:
                return False
            t.exit_price = exit_price
            t.fee_close = fee_close
            t.net_pnl = net_pnl
            t.reason_exit = reason_exit
            t.datetime_close = datetime.now(timezone.utc)
            t.status = TradeStatus.CLOSED
            db.commit()
            return True

    # ── Wallet ──────────────────────────────────────────────────────
    def update_balance(self, new_balance: float) -> None:
        """Upsert the wallet balance."""
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
