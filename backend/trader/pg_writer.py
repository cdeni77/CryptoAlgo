"""Postgres writer for trader service, including paper-trading persistence."""

import enum
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
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
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Wallet(Base):
    __tablename__ = "wallet"
    id = Column(Integer, primary_key=True)
    balance = Column(Float, default=10000.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


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


class PaperEquityCurve(Base):
    __tablename__ = "paper_equity_curve"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False, default=0)


class PgWriter:
    def __init__(self, database_url: Optional[str] = None):
        url = database_url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL env var is not set.")
        self.engine = create_engine(url)
        self.SessionLocal = sessionmaker(bind=self.engine)
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
    ) -> int:
        """Insert one signal row. Returns the signal id (idempotent for coin+timestamp+direction)."""
    def write_signal(self, coin: str, timestamp: datetime, direction: str, confidence: float, raw_probability: float | None = None, model_auc: float | None = None, price_at_signal: float | None = None, momentum_pass: bool | None = None, trend_pass: bool | None = None, regime_pass: bool | None = None, ml_pass: bool | None = None, contracts_suggested: int | None = None, notional_usd: float | None = None, acted_on: bool = False, trade_id: int | None = None) -> int:
        with self._session() as db:
            existing = db.query(Signal).filter(
                Signal.coin == coin,
                Signal.timestamp == timestamp,
                Signal.direction == direction,
            ).first()
            if existing:
                return existing.id

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
        mode: str = "live",
        idempotency_key: str | None = None,
        fee_open: float = 0.0,
        margin_used: float | None = None,
        leverage: float | None = None,
        reason_entry: str | None = None,
    ) -> int:
        """Insert an open trade. Returns trade id."""
    def open_trade(self, coin: str, side: str, contracts: float, entry_price: float, fee_open: float = 0.0, margin_used: float | None = None, leverage: float | None = None, reason_entry: str | None = None) -> int:
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
    def close_trade(self, trade_id: int, exit_price: float, fee_close: float = 0.0, net_pnl: float | None = None, reason_exit: str | None = None) -> bool:
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

    def upsert_paper_position(self, coin: str, side: str, contracts: int, entry_price: float, mark_price: float, notional: float, realized_pnl: float, unrealized_pnl: float, fees_paid: float, is_open: bool = True) -> int:
        with self._session() as db:
            position = (
                db.query(PaperPosition)
                .filter(PaperPosition.coin == coin, PaperPosition.is_open.is_(True))
                .order_by(PaperPosition.id.desc())
                .first()
            )
            if position:
                position.side = side
                position.contracts = contracts
                position.entry_price = entry_price
                position.mark_price = mark_price
                position.notional = notional
                position.realized_pnl = realized_pnl
                position.unrealized_pnl = unrealized_pnl
                position.fees_paid = fees_paid
                position.is_open = is_open
                db.commit()
                db.refresh(position)
                return position.id

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
            )
            db.add(position)
            db.commit()
            db.refresh(position)
            return position.id

    def close_paper_position(self, position_id: int, mark_price: float, realized_pnl: float, fees_paid: float) -> None:
        with self._session() as db:
            position = db.query(PaperPosition).filter(PaperPosition.id == position_id).first()
            if not position:
                return
            position.mark_price = mark_price
            position.realized_pnl = realized_pnl
            position.unrealized_pnl = 0.0
            position.fees_paid = fees_paid
            position.is_open = False
            db.commit()

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
                .filter(Signal.id > since_id)
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
