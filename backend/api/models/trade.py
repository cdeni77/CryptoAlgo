import enum

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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

    def __repr__(self):
        return f"<Trade {self.id} | {self.coin} | {self.side} | {self.status}>"

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

class TradeCreate(TradeBase):
    pass

class TradeResponse(TradeBase):
    id: int
    datetime_close: Optional[datetime] = None
    exit_price: Optional[float] = None
    fee_close: Optional[float] = None
    net_pnl: Optional[float] = None
    reason_exit: Optional[str] = None

    class Config:
        from_attributes = True  