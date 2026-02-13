from fastapi import APIRouter, Query

from models.ops import OpsActionResponse, OpsLogsResponse, OpsStatusResponse
from services.ops_service import ops_service


router = APIRouter(prefix="/ops", tags=["ops"])


@router.post("/pipeline/start", response_model=OpsActionResponse)
def start_pipeline():
    pid = ops_service.start_pipeline()
    return OpsActionResponse(
        action="pipeline_start",
        status="ok",
        detail="Pipeline started (or already running)",
        pid=pid,
    )


@router.post("/pipeline/stop", response_model=OpsActionResponse)
def stop_pipeline():
    stopped = ops_service.stop_pipeline()
    return OpsActionResponse(
        action="pipeline_stop",
        status="ok",
        detail="Pipeline stopped" if stopped else "Pipeline was not running",
    )


@router.post("/retrain", response_model=OpsActionResponse)
def retrain(
    train_window_days: int = Query(90, ge=7, le=365),
    retrain_every_days: int = Query(7, ge=1, le=90),
    debug: bool = Query(False),
):
    pid = ops_service.retrain(
        train_window_days=train_window_days,
        retrain_every_days=retrain_every_days,
        debug=debug,
    )
    return OpsActionResponse(
        action="retrain",
        status="ok",
        detail="Retraining started (or already running)",
        pid=pid,
    )


@router.post("/parallel-launch", response_model=OpsActionResponse)
def parallel_launch(
    trials: int = Query(200, ge=1, le=5000),
    jobs: int = Query(16, ge=1, le=64),
    coins: str = Query("BTC,ETH,SOL,XRP,DOGE"),
    plateau_patience: int = Query(80, ge=1, le=10000),
    plateau_min_delta: float = Query(0.02, ge=0, le=2),
    plateau_warmup: int = Query(40, ge=1, le=5000),
):
    pid = ops_service.launch_parallel(
        trials=trials,
        jobs=jobs,
        coins=coins,
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        plateau_warmup=plateau_warmup,
    )
    return OpsActionResponse(
        action="parallel_launch",
        status="ok",
        detail="Parallel optimization started (or already running)",
        pid=pid,
    )


@router.post("/train-scratch", response_model=OpsActionResponse)
def train_scratch(
    backfill_days: int = Query(30, ge=1, le=365),
    include_oi: bool = Query(True),
    debug: bool = Query(False),
    threshold: float = Query(0.74, ge=0, le=1),
    min_auc: float = Query(0.54, ge=0, le=1),
    leverage: int = Query(4, ge=1, le=100),
    exclude_symbols: str = Query("BIP,DOP"),
):
    pid = ops_service.train_from_scratch(
        backfill_days=backfill_days,
        include_oi=include_oi,
        debug=debug,
        threshold=threshold,
        min_auc=min_auc,
        leverage=leverage,
        exclude_symbols=exclude_symbols,
    )
    return OpsActionResponse(
        action="train_scratch",
        status="ok",
        detail="Scratch training started (or already running)",
        pid=pid,
    )


@router.get("/status", response_model=OpsStatusResponse)
def status():
    return ops_service.get_status()


@router.get("/logs", response_model=OpsLogsResponse)
def logs(limit: int = Query(200, ge=1, le=2000)):
    return OpsLogsResponse(entries=ops_service.get_logs(limit=limit))
