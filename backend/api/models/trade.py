import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

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


class TradeBase(BaseModel):
    coin: str
    datetime_open: datetime
    side: str
    contracts: float
    entry_price: float
    fee_open: Optional[float] = None
    margin_used: Optional[float] = None
    leverage: Optional[float] = None
    reason_entry: Optional[str] = None
    status: str


class TradeResponse(TradeBase):
    id: int
    datetime_close: Optional[datetime] = None
    exit_price: Optional[float] = None
    fee_close: Optional[float] = None
    net_pnl: Optional[float] = None
    reason_exit: Optional[str] = None

    class Config:
        from_attributes = True


class SignalResponse(BaseModel):
    id: int
    coin: str
    timestamp: datetime
    direction: str
    confidence: float
    raw_probability: Optional[float] = None
    model_auc: Optional[float] = None
    price_at_signal: Optional[float] = None
    momentum_pass: Optional[bool] = None
    trend_pass: Optional[bool] = None
    regime_pass: Optional[bool] = None
    ml_pass: Optional[bool] = None
    contracts_suggested: Optional[int] = None
    notional_usd: Optional[float] = None
    acted_on: bool = False
    trade_id: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


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
