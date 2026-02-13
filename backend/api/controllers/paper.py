from typing import List

from sqlalchemy import desc
from sqlalchemy.orm import Session

from models.trade import PaperEquityCurve, PaperFill, PaperOrder, PaperPosition


def get_recent_paper_orders(db: Session, limit: int = 100) -> List[PaperOrder]:
    return db.query(PaperOrder).order_by(desc(PaperOrder.created_at)).limit(limit).all()


def get_recent_paper_fills(db: Session, limit: int = 100) -> List[PaperFill]:
    return db.query(PaperFill).order_by(desc(PaperFill.created_at)).limit(limit).all()


def get_open_paper_positions(db: Session) -> List[PaperPosition]:
    return (
        db.query(PaperPosition)
        .filter(PaperPosition.is_open.is_(True))
        .order_by(desc(PaperPosition.updated_at))
        .all()
    )


def get_equity_curve(db: Session, limit: int = 500) -> List[PaperEquityCurve]:
    return db.query(PaperEquityCurve).order_by(desc(PaperEquityCurve.timestamp)).limit(limit).all()
