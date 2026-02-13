from sqlalchemy import Column, DateTime, Float, Integer
from sqlalchemy.sql import func

from models.base import Base


class Wallet(Base):
    __tablename__ = "wallet"

    id = Column(Integer, primary_key=True)
    balance = Column(Float, default=10000.0)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
