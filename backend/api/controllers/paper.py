from typing import Any, Dict, List

from sqlalchemy import asc, desc, func
from sqlalchemy.orm import Session

from models.trade import PaperEquityCurve, PaperFill, PaperPosition

INITIAL_EQUITY = 100_000.0


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


def _compute_true_equity(db: Session) -> Dict[str, float]:
    """Compute correct equity/cash from fill and position history.

    The paper engine resets in-memory state on restart, which can corrupt
    the equity curve.  This derives the ground-truth values from the
    transactional records instead.

    cash = INITIAL - sum(open fill fees) + sum(closed position net realized pnl)
    Each PaperFill.fee is an open fee; each closed PaperPosition.realized_pnl
    is already net of the close fee.
    """
    total_open_fees = db.query(func.sum(PaperFill.fee)).scalar() or 0.0
    total_realized = (
        db.query(func.sum(PaperPosition.realized_pnl))
        .filter(PaperPosition.is_open.is_(False))
        .scalar()
    ) or 0.0
    unrealized_pnl = (
        db.query(func.sum(PaperPosition.unrealized_pnl))
        .filter(PaperPosition.is_open.is_(True))
        .scalar()
    ) or 0.0
    open_count = (
        db.query(func.count(PaperPosition.id))
        .filter(PaperPosition.is_open.is_(True))
        .scalar()
    ) or 0
    cash = float(INITIAL_EQUITY) - float(total_open_fees) + float(total_realized)
    return {
        "cash_balance": cash,
        "realized_pnl": float(total_realized),
        "unrealized_pnl": float(unrealized_pnl),
        "equity": cash + float(unrealized_pnl),
        "open_positions": int(open_count),
    }


def _has_activity(db: Session) -> bool:
    """Has this account traded at all?

    Without this the endpoint reported a complete, confident account on an empty
    database: `equity` and `cash_balance` were the hardcoded `INITIAL_EQUITY`,
    `total_return_pct` was derived from it, and `max_drawdown_pct` was seeded 0.0
    and stayed there. A never-started account was indistinguishable from one that
    had traded and broken even at zero drawdown. `win_rate` already returned null
    in the same payload, which is the pattern the rest should follow.
    """
    return bool(
        db.query(func.count(PaperFill.id)).scalar()
        or db.query(func.count(PaperPosition.id)).scalar()
        or db.query(func.count(PaperEquityCurve.id)).scalar()
    )


def get_paper_summary(db: Session) -> Dict[str, Any]:
    """Aggregate paper trading performance stats."""
    if not _has_activity(db):
        return {
            "initial_equity": INITIAL_EQUITY,
            "equity": None,
            "cash_balance": None,
            "realized_pnl": None,
            "unrealized_pnl": None,
            "total_return_pct": None,
            "max_drawdown_pct": None,
            "win_rate": None,
            "sharpe_ratio": None,
            "profit_factor": None,
            "open_positions": 0,
            "closed_positions": 0,
            "fill_count": 0,
            "total_fees": None,
            "total_notional": None,
            "unavailable_reason": (
                "no paper fills, positions or equity points; the account has not "
                "started trading"
            ),
        }

    true_state = _compute_true_equity(db)
    latest_equity = true_state["equity"]
    realized_pnl = true_state["realized_pnl"]
    unrealized_pnl = true_state["unrealized_pnl"]
    open_positions = true_state["open_positions"]

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
    # None rather than 0.0 with no curve: "we never observed a drawdown" and
    # "the drawdown was zero" are different claims.
    max_drawdown_pct = None
    if equity_rows:
        max_drawdown_pct = 0.0
        peak = INITIAL_EQUITY
        for (eq,) in equity_rows:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

    # Win rate from closed positions. Counted in SQL: this endpoint is polled,
    # and the closed book only grows, so materialising every closed position as
    # an ORM object to count two integers made the cost of the poll a function
    # of the account's whole history.
    closed_count = (
        db.query(func.count(PaperPosition.id))
        .filter(PaperPosition.is_open.is_(False))
        .scalar()
        or 0
    )
    win_count = (
        db.query(func.count(PaperPosition.id))
        .filter(PaperPosition.is_open.is_(False), PaperPosition.realized_pnl > 0)
        .scalar()
        or 0
    )
    win_rate = win_count / closed_count * 100 if closed_count else None

    cash_balance = true_state["cash_balance"]
    return {
        "initial_equity": INITIAL_EQUITY,
        # canonical field names expected by frontend PaperSummary type
        "equity": round(latest_equity, 2),
        "cash_balance": round(cash_balance, 2),
        "total_return_pct": round(paper_return_pct, 2),
        "realized_pnl": round(realized_pnl, 4),
        "unrealized_pnl": round(unrealized_pnl, 4),
        "open_positions": open_positions,
        "fill_count": fill_count,
        "total_fees": round(total_fees, 4),
        "total_notional": round(total_notional, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
        # win_rate as 0-1 fraction (frontend multiplies by 100 for display)
        "win_rate": round(win_rate / 100.0, 4) if win_rate is not None else None,
        "closed_positions": closed_count,
        "sharpe_ratio": None,
        "profit_factor": None,
    }
