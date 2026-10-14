"""Read-only routes for the dashboard.

Unauthenticated on purpose: they serve a local dashboard, expose no
credentials, and mutate nothing. The CORS policy limits callers.
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


@router.get('/account/venue')
def get_venue_account(db: Session = Depends(get_db)):
    """The account as the venue's ledger reports it, with ours beside it.

    The venue's ledger is the account of record; `/account` is our own
    arithmetic. `pnl_gap` is the disagreement. Returns `available: false` with
    a reason when no venue ledger is stored, as for a paper account.
    """
    return serving.venue_account_state(db)


@router.get('/account/venue/equity')
def get_venue_equity(days: int = Query(30, ge=1, le=365),
                     db: Session = Depends(get_db)):
    """The account chart: realised P&L per settlement, venue cash beside it.

    Built from settlement P&L, not balance differences, because a deposit and a
    profit look the same in the balance.
    """
    return serving.venue_equity_curve(db, days=days)


@router.get('/venue/settlements')
def get_venue_settlements(days: Optional[int] = Query(None, ge=1, le=3650),
                          limit: int = Query(200, ge=1, le=1000),
                          db: Session = Depends(get_db)):
    """Every settled market as the venue paid it: cost, revenue, fee, P&L."""
    return {'settlements': serving.venue_settlements(db, days=days, limit=limit)}


@router.get('/live')
def get_live(db: Session = Depends(get_db)):
    """The current barrier state per symbol, and the account beside it."""
    return {
        'windows': serving.live_windows(db),
        'account': serving.account_state(db),
        'open_positions': serving.positions(db, open_only=True),
        'tickets': serving.tickets(db, open_only=True, limit=20),
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


@router.get('/prices/{symbol}')
def get_prices(symbol: str, minutes: int = Query(240, ge=15, le=2880),
               db: Session = Depends(get_db)):
    """One-minute bars and the strike of each window they span."""
    return {
        'symbol': symbol, 'minutes': minutes,
        'bars': serving.minute_prices(db, symbol, minutes=minutes),
        'strikes': [s for s in serving.window_strikes(db, minutes=minutes)
                    if s['symbol'] == symbol],
    }


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
