from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List

from models.signals import SignalResponse
from controllers.signals import get_recent_signals
from database import get_db

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/", response_model=List[SignalResponse])
def list_signals(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Get most recent signals across all coins."""
    return get_recent_signals(db, limit=limit)