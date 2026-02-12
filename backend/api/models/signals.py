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
    
# ── Signal ──
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