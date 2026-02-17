from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ResearchSummaryKpis(BaseModel):
    holdout_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    precision_at_threshold: Optional[float] = None
    win_rate_realized: float
    acted_signal_rate: float
    drift_delta: float
    robustness_gate: bool


class CoinHealthRow(BaseModel):
    coin: str
    holdout_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    precision_at_threshold: Optional[float] = None
    win_rate_realized: float
    acted_signal_rate: float
    drift_delta: float
    robustness_gate: bool
    optimization_freshness_hours: Optional[float] = None
    last_optimized_at: Optional[datetime] = None
    health: str


class ResearchSummaryResponse(BaseModel):
    generated_at: datetime
    kpis: ResearchSummaryKpis
    coins: List[CoinHealthRow]


class ResearchCoinDetailResponse(BaseModel):
    generated_at: datetime
    coin: CoinHealthRow


class ResearchRunResponse(BaseModel):
    id: str
    coin: str
    run_type: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: int
    holdout_auc: Optional[float] = None
    robustness_gate: bool


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class SignalDistributionItem(BaseModel):
    label: str
    value: int


class ResearchFeaturesResponse(BaseModel):
    coin: str
    generated_at: datetime
    feature_importance: List[FeatureImportanceItem]
    signal_distribution: List[SignalDistributionItem]


class ResearchJobLaunchRequest(BaseModel):
    args: List[str] = Field(default_factory=list)


class ResearchJobLaunchResponse(BaseModel):
    job: str
    module: str
    pid: int
    command: List[str]
    cwd: str
    log_path: str
    launched_at: datetime


class ResearchScriptInfo(BaseModel):
    name: str
    module: str
    default_args: List[str] = Field(default_factory=list)


class ResearchScriptListResponse(BaseModel):
    scripts: List[ResearchScriptInfo]


class ResearchJobLogResponse(BaseModel):
    pid: int
    running: bool
    command: List[str]
    launched_at: datetime
    log_path: str
    logs: List[str]
