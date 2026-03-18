from typing import Any, Dict, List

from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session

from models.trade import PaperEquityCurve, PaperFill, PaperPosition

INITIAL_EQUITY = 10_000.0


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


def get_paper_summary(db: Session) -> Dict[str, Any]:
    """Aggregate paper trading performance stats."""
    latest_equity_row = (
        db.query(PaperEquityCurve).order_by(desc(PaperEquityCurve.timestamp)).first()
    )
    latest_equity = latest_equity_row.equity if latest_equity_row else INITIAL_EQUITY
    realized_pnl = latest_equity_row.realized_pnl if latest_equity_row else 0.0
    unrealized_pnl = latest_equity_row.unrealized_pnl if latest_equity_row else 0.0
    open_positions = latest_equity_row.open_positions if latest_equity_row else 0

    paper_return_pct = (latest_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    fill_count = db.query(func.count(PaperFill.id)).scalar() or 0
    total_fees = db.query(func.sum(PaperFill.fee)).scalar() or 0.0
    total_notional = db.query(func.sum(PaperFill.notional)).scalar() or 0.0

    # Max drawdown from chronological equity curve
    equity_rows = (
        db.query(PaperEquityCurve.equity)
        .order_by(asc(PaperEquityCurve.timestamp))
        .all()
    )
    max_drawdown_pct = 0.0
    if equity_rows:
        peak = INITIAL_EQUITY
        for (eq,) in equity_rows:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

    # Win rate from closed positions
    closed_positions = db.query(PaperPosition).filter(PaperPosition.is_open.is_(False)).all()
    wins = [p for p in closed_positions if (p.realized_pnl or 0) > 0]
    win_rate = len(wins) / len(closed_positions) * 100 if closed_positions else None

    return {
        "initial_equity": INITIAL_EQUITY,
        "latest_equity": round(latest_equity, 2),
        "paper_return_pct": round(paper_return_pct, 2),
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "open_positions": open_positions,
        "fill_count": fill_count,
        "total_fees": round(total_fees, 4),
        "total_notional": round(total_notional, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "closed_positions": len(closed_positions),
    }
