from typing import List

from sqlalchemy import desc
from sqlalchemy.orm import Session

from models.signals import Signal


def get_recent_signals(db: Session, limit: int = 50) -> List[Signal]:
    """Most recent signals across all coins."""
    return (
        db.query(Signal)
        .order_by(desc(Signal.timestamp))
        .limit(limit)
        .all()
    )