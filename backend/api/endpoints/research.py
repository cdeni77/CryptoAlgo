from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from controllers.research import (
    get_research_coin,
    get_research_features,
    get_research_runs,
    get_research_summary,
)
from database import get_db
from models.research import (
    ResearchCoinDetailResponse,
    ResearchFeaturesResponse,
    ResearchRunResponse,
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
