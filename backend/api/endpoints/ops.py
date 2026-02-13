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
def retrain():
    pid = ops_service.retrain()
    return OpsActionResponse(
        action="retrain",
        status="ok",
        detail="Retraining started (or already running)",
        pid=pid,
    )


@router.get("/status", response_model=OpsStatusResponse)
def status():
    return ops_service.get_status()


@router.get("/logs", response_model=OpsLogsResponse)
def logs(limit: int = Query(200, ge=1, le=2000)):
    return OpsLogsResponse(entries=ops_service.get_logs(limit=limit))
