import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, JSON, String, Text
from sqlalchemy.sql import func

from models.base import Base


class PaperOrderStatus(str, enum.Enum):
    NEW = "new"
    FILLED = "filled"
    CANCELED = "canceled"


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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


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
    opened_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_open = Column(Boolean, nullable=False, default=True, index=True)
    tp_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    max_hold_until = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String, nullable=True)
    # Funding accrued so far, in account currency. On hourly-funding perps this
    # is the largest cost after commission — 2bp/hour over a day is 48bp against
    # a 5-54bp round trip — and the paper engine persists it so a restart does
    # not forget what the position has already cost.
    funding_paid = Column(Float, nullable=False, default=0.0)


class PaperEquityCurve(Base):
    __tablename__ = "paper_equity_curve"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False, default=0)


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


class PaperEngineConfig(Base):
    __tablename__ = "paper_engine_config"

    id = Column(Integer, primary_key=True, default=1)
    active_coins = Column(JSON, nullable=False, default=list)
    tier_map = Column(JSON, nullable=False, default=dict)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaperOrderResponse(BaseModel):
    id: int
    signal_id: int
    coin: str
    side: str
    contracts: int
    target_price: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class PaperFillResponse(BaseModel):
    id: int
    order_id: int
    signal_id: int
    coin: str
    side: str
    contracts: int
    fill_price: float
    fee: float
    notional: float
    slippage_bps: float
    created_at: datetime

    class Config:
        from_attributes = True


class PaperPositionResponse(BaseModel):
    id: int
    coin: str
    side: str
    contracts: int
    entry_price: float
    mark_price: float
    notional: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    opened_at: datetime
    updated_at: Optional[datetime] = None
    is_open: bool
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    max_hold_until: Optional[datetime] = None
    exit_reason: Optional[str] = None
    funding_paid: float = 0.0

    class Config:
        from_attributes = True


class PaperEquityCurveResponse(BaseModel):
    id: int
    timestamp: datetime
    equity: float
    cash_balance: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int

    class Config:
        from_attributes = True
