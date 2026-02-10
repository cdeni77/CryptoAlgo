from sqlalchemy.orm import Session
from models.trade import Trade, TradeStatus
from typing import List, Optional

def get_trade(db: Session, trade_id: int) -> Optional[Trade]:
    """Get a single trade by ID"""
    return db.query(Trade).filter(Trade.id == trade_id).first()

def get_all_trades(db: Session, skip: int = 0, limit: int = 100) -> List[Trade]:
    """Get all trades (paginated)"""
    return db.query(Trade).offset(skip).limit(limit).all()

def get_open_trades(db: Session) -> List[Trade]:
    """Get currently open trades"""
    return db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()

def get_closed_trades(db: Session) -> List[Trade]:
    """Get all closed trades"""
    return db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).all()

def get_trades_by_coin(db: Session, coin: str) -> List[Trade]:
    """Get trades for a specific coin"""
    return db.query(Trade).filter(Trade.coin == coin).all()

def get_recent_trades(db: Session, limit: int = 20) -> List[Trade]:
    """Get most recent trades (ordered by open time descending)"""
    return db.query(Trade).order_by(Trade.datetime_open.desc()).limit(limit).all()