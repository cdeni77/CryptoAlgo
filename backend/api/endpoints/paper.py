import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from controllers.paper import (
    get_equity_curve,
    get_open_paper_positions,
    get_paper_summary,
    get_recent_paper_fills,
)
from database import get_db
from models.signals import Signal
from models.trade import (
    ModelRun,
    PaperEngineConfig,
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


@router.get("/config")
def paper_config(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the paper engine's runtime config (active coins, tier map) written on startup."""
    row = db.query(PaperEngineConfig).filter(PaperEngineConfig.id == 1).first()
    if not row:
        return {"active_coins": [], "tier_map": {}, "updated_at": None}
    return {
        "active_coins": row.active_coins or [],
        "tier_map": row.tier_map or {},
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/model-status")
def model_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Per-coin ML model status derived from recent signals and model run history."""
    # Active coins from paper engine config
    config_row = db.query(PaperEngineConfig).filter(PaperEngineConfig.id == 1).first()
    active_coins = [c.upper() for c in (config_row.active_coins or [])] if config_row else []

    # Most recent signal per coin over the last 48h
    since = datetime.now(timezone.utc) - timedelta(hours=48)
    recent_signals = (
        db.query(Signal)
        .filter(Signal.timestamp >= since)
        .order_by(Signal.timestamp.desc())
        .all()
    )
    latest_by_coin: Dict[str, Any] = {}
    for sig in recent_signals:
        coin = sig.coin.upper()
        if coin not in latest_by_coin:
            latest_by_coin[coin] = sig

    now = datetime.now(timezone.utc)
    coins = []
    for coin in active_coins:
        sig = latest_by_coin.get(coin)
        if sig is None:
            # No signal in the last 48h — most likely AUC rejection prevented writing
            coins.append({
                "coin": coin,
                "last_signal_at": None,
                "model_auc": None,
                "gate_failure_reason": None,
                "passed_gates": False,
                "status": "auc_rejected",
                "hours_since_signal": None,
            })
        else:
            ts = sig.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            hours_since = (now - ts).total_seconds() / 3600
            if hours_since > 4:
                status = "stale"
            elif sig.passed_gates:
                status = "active"
            else:
                status = "gate_rejected"
            coins.append({
                "coin": coin,
                "last_signal_at": ts.isoformat(),
                "model_auc": sig.model_auc,
                "gate_failure_reason": sig.gate_failure_reason,
                "passed_gates": bool(sig.passed_gates),
                "status": status,
                "hours_since_signal": round(hours_since, 1),
            })

    # Latest model run (any status) and latest successful one
    last_run = db.query(ModelRun).order_by(ModelRun.run_started_at.desc()).first()
    last_success = (
        db.query(ModelRun)
        .filter(ModelRun.status == "success")
        .order_by(ModelRun.run_started_at.desc())
        .first()
    )

    retrain_every_days = int(os.getenv("RETRAIN_EVERY_DAYS", "7"))
    next_retrain_at: Optional[str] = None
    if last_success and last_success.run_started_at:
        ts = last_success.run_started_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        next_retrain_at = (ts + timedelta(days=retrain_every_days)).isoformat()

    last_retrain = None
    if last_run:
        started = last_run.run_started_at
        finished = last_run.run_finished_at
        if started and started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished and finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
        last_retrain = {
            "started_at": started.isoformat() if started else None,
            "finished_at": finished.isoformat() if finished else None,
            "status": last_run.status,
            "symbols_trained": last_run.symbols_trained,
            "symbols_total": last_run.symbols_total,
            "version": last_run.artifacts_version,
            "error": last_run.error,
        }

    return {
        "coins": coins,
        "last_retrain": last_retrain,
        "next_retrain_at": next_retrain_at,
        "retrain_every_days": retrain_every_days,
    }
