"""Read-only routes for the dashboard.

Left unauthenticated on purpose: they serve a local dashboard, expose no
credentials, and gating them would break the frontend's polling for no gain the
origin policy does not already provide. The mutating surface — there is one
route that starts a process — is gated in `endpoints/research.py`.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from controllers import serving
from database import get_db

router = APIRouter(tags=['serving'])


@router.get('/account')
def get_account(db: Session = Depends(get_db)):
    return serving.account_state(db)


@router.get('/account/equity')
def get_equity(days: int = Query(30, ge=1, le=365), db: Session = Depends(get_db)):
    return {'days': days, 'points': serving.equity_curve(db, days=days)}


@router.get('/live')
def get_live(db: Session = Depends(get_db)):
    """The current barrier state per symbol, and the account beside it."""
    return {
        'windows': serving.live_windows(db),
        'account': serving.account_state(db),
        'open_positions': serving.positions(db, open_only=True),
    }


@router.get('/predictions')
def get_predictions(
    limit: int = Query(100, ge=1, le=1000),
    traded_only: bool = False,
    db: Session = Depends(get_db),
):
    return {'predictions': serving.recent_predictions(
        db, limit=limit, traded_only=traded_only)}


@router.get('/funnel')
def get_funnel(days: int = Query(7, ge=1, le=365), db: Session = Depends(get_db)):
    """Why the system declined. Expected to be dominated by `edge_below_gate`."""
    return {'days': days, 'stages': serving.funnel(db, days=days)}


@router.get('/positions')
def get_positions(
    open_only: bool = False,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    return {'positions': serving.positions(db, open_only=open_only, limit=limit)}


@router.get('/model')
def get_model(db: Session = Depends(get_db)):
    return serving.model_state(db)


@router.get('/model/history')
def get_model_history(limit: int = Query(50, ge=1, le=200),
                      db: Session = Depends(get_db)):
    return {'attempts': serving.model_history(db, limit=limit)}


@router.get('/model/calibration')
def get_calibration(version: Optional[str] = None, db: Session = Depends(get_db)):
    return serving.calibration(db, version=version)
