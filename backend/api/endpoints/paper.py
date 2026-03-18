from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from controllers.paper import (
    get_equity_curve,
    get_open_paper_positions,
    get_paper_summary,
    get_recent_paper_fills,
)
from database import get_db
from models.trade import (
    PaperEquityCurveResponse,
    PaperFillResponse,
    PaperPositionResponse,
)

router = APIRouter(prefix="/paper", tags=["paper-trading"])


@router.get("/summary")
def paper_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Aggregate paper trading stats: return %, drawdown, win rate, fill count."""
    return get_paper_summary(db)


@router.get("/fills", response_model=List[PaperFillResponse])
def list_paper_fills(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    return get_recent_paper_fills(db, limit=limit)


@router.get("/positions", response_model=List[PaperPositionResponse])
def list_paper_positions(db: Session = Depends(get_db)):
    return get_open_paper_positions(db)


@router.get("/equity", response_model=List[PaperEquityCurveResponse])
def list_equity_curve(limit: int = Query(500, ge=1, le=2000), db: Session = Depends(get_db)):
    return get_equity_curve(db, limit=limit)
