"""
Unified SQLAlchemy models for the trading system.

IMPORTANT: All models share the same Base so that
    Base.metadata.create_all(engine)
creates every table in one shot (trades, wallet, signals).
"""

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Enum, Text, Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

# ── Single shared Base ──────────────────────────────────────────────
Base = declarative_base()


# ====================================================================
# ENUMS
# ====================================================================
class TradeSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatus(str, enum.Enum):
    OPEN = "open"
    CLOSED = "closed"


# ====================================================================
# TRADE
# ====================================================================
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


# ====================================================================
# SIGNAL — hourly ML predictions written by the trader
# ====================================================================
class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    coin = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    direction = Column(String, nullable=False)          # "long", "short", "neutral"
    confidence = Column(Float, nullable=False)           # calibrated probability 0-1
    raw_probability = Column(Float, nullable=True)       # raw model output before calibration
    model_auc = Column(Float, nullable=True)             # validation AUC of the model that produced this
    price_at_signal = Column(Float, nullable=True)       # spot price when signal was generated
    # Gate details — which filters passed
    momentum_pass = Column(Boolean, nullable=True)
    trend_pass = Column(Boolean, nullable=True)
    regime_pass = Column(Boolean, nullable=True)
    ml_pass = Column(Boolean, nullable=True)
    # Sizing
    contracts_suggested = Column(Integer, nullable=True)
    notional_usd = Column(Float, nullable=True)
    # Was it acted on?
    acted_on = Column(Boolean, default=False)
    trade_id = Column(Integer, nullable=True)            # FK-like link to trades.id
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Signal {self.id} | {self.coin} | {self.direction} | conf={self.confidence:.1%}>"


# ====================================================================
# PYDANTIC SCHEMAS
# ====================================================================

# ── Trade ──
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


