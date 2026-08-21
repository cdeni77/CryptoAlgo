from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from controllers.research import (
    get_research_coin,
    get_research_features,
    get_research_runs,
    get_research_summary,
    launch_research_job,
    list_research_scripts,
    get_research_job_logs,
    list_research_jobs,
)
from database import get_db
from security import require_token, validate_job_args
from models.research import (
    ResearchCoinDetailResponse,
    ResearchFeaturesResponse,
    ResearchJobLaunchRequest,
    ResearchJobLaunchResponse,
    ResearchJobLogResponse,
    ResearchRunResponse,
    ResearchScriptListResponse,
    ResearchSummaryResponse,
)

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/summary", response_model=ResearchSummaryResponse)
def summary(db: Session = Depends(get_db)):
    return get_research_summary(db)


@router.get("/coins/{coin}", response_model=ResearchCoinDetailResponse)
def coin_detail(coin: str, db: Session = Depends(get_db)):
    return get_research_coin(db, coin)


@router.get("/runs", response_model=List[ResearchRunResponse])
def runs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    return get_research_runs(db, limit=limit)


@router.get("/features/{coin}", response_model=ResearchFeaturesResponse)
def features(coin: str, db: Session = Depends(get_db)):
    return get_research_features(db, coin)


@router.get("/scripts", response_model=ResearchScriptListResponse)
def scripts():
    try:
        return ResearchScriptListResponse(scripts=list_research_scripts())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{pid}/logs", response_model=ResearchJobLogResponse)
def job_logs(pid: int, lines: int = Query(200, ge=1, le=2000)):
    try:
        return get_research_job_logs(pid=pid, lines=lines)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs", response_model=List[ResearchJobLaunchResponse])
def jobs(limit: int = Query(25, ge=1, le=200)):
    return list_research_jobs(limit=limit)


@router.post(
    "/launch/{job}",
    response_model=ResearchJobLaunchResponse,
    dependencies=[Depends(require_token)],
)
def launch(job: str, request: ResearchJobLaunchRequest):
    """Start a research script.

    The only endpoint that runs a process, in a container that holds the exchange
    API keys, so it carries both controls: `require_token` decides who may launch,
    `validate_job_args` decides what they may pass. Authentication alone is not
    enough — an authenticated caller still should not be able to hand a script an
    arbitrary filesystem path.
    """
    try:
        return launch_research_job(job=job, args=validate_job_args(request.args))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
