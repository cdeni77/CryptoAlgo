from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from models.trade import Signal


def get_recent_signals(db: Session, limit: int = 50) -> List[Signal]:
    """Most recent signals across all coins."""
    return (
        db.query(Signal)
        .order_by(desc(Signal.timestamp))
        .limit(limit)
        .all()
    )


def get_signals_by_coin(
    db: Session,
    coin: str,
    limit: int = 100,
    hours: Optional[int] = None,
) -> List[Signal]:
    """Signals for a specific coin, optionally limited to last N hours."""
    q = db.query(Signal).filter(Signal.coin == coin)
    if hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        q = q.filter(Signal.timestamp >= cutoff)
    return q.order_by(desc(Signal.timestamp)).limit(limit).all()


def get_signal_by_id(db: Session, signal_id: int) -> Optional[Signal]:
    return db.query(Signal).filter(Signal.id == signal_id).first()