import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.sql import func

from models.base import Base


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
