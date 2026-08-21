from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from models.base import Base


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

    # -- the decision, decomposed ------------------------------------------
    # These replace the classification columns above, which no longer describe
    # what the model produces. `confidence` and `raw_probability` were a
    # probability and an AUC; the model regresses net return, so the honest
    # record is the forecast broken into its parts and the cost it has to clear.
    # The old columns stay nullable and unwritten rather than being dropped,
    # because historical rows still hold real values from the previous system.
    expected_net_bps = Column(Float, nullable=True)
    expected_price_bps = Column(Float, nullable=True)
    expected_carry_bps = Column(Float, nullable=True)
    cost_bps = Column(Float, nullable=True)
    sigma_bps = Column(Float, nullable=True)
    edge_to_risk = Column(Float, nullable=True)
    # Share of the expected edge that is carry rather than direction. A book
    # whose edge is mostly carry is a different strategy with different risks.
    carry_share = Column(Float, nullable=True)
    participation = Column(Float, nullable=True)
    # Which promoted model produced this. Without it, a signal cannot be
    # attributed after a retrain, and calibration is measured across two models.
    model_version = Column(String, nullable=True)

    def __repr__(self):
        return f"<Signal {self.id} | {self.coin} | {self.direction} | conf={self.confidence:.1%}>"


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
    passed_gates: bool = True
    gate_failure_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    expected_net_bps: Optional[float] = None
    expected_price_bps: Optional[float] = None
    expected_carry_bps: Optional[float] = None
    cost_bps: Optional[float] = None
    sigma_bps: Optional[float] = None
    edge_to_risk: Optional[float] = None
    carry_share: Optional[float] = None
    participation: Optional[float] = None
    model_version: Optional[str] = None

    class Config:
        from_attributes = True
