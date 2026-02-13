from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from models.signals import SignalResponse
from controllers.signals import get_recent_signals, get_signals_by_coin, get_signal_by_id
from database import get_db

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/", response_model=List[SignalResponse])
def list_signals(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get most recent signals across all coins."""
    return get_recent_signals(db, limit=limit)


@router.get("/coin/{coin}", response_model=List[SignalResponse])
def signals_for_coin(
    coin: str,
    limit: int = Query(100, ge=1, le=500),
    hours: Optional[int] = Query(None, ge=1, le=720, description="Filter to last N hours"),
    db: Session = Depends(get_db),
):
    """Get signals for a specific coin."""
    return get_signals_by_coin(db, coin, limit=limit, hours=hours)


@router.get("/{signal_id}", response_model=SignalResponse)
def get_single_signal(signal_id: int, db: Session = Depends(get_db)):
    sig = get_signal_by_id(db, signal_id)
    if not sig:
        raise HTTPException(status_code=404, detail="Signal not found")
    return sig