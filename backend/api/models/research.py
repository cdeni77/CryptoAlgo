"""Response shapes for the research surface.

Rewritten because the previous shapes described a classifier that no longer
exists. `holdout_auc` was read from `signals.model_auc`, which the new signal
writer leaves null — the model regresses net return, and AUC is not defined for
it. Worse, `pr_auc` was computed as `holdout_auc - 0.06` and
`precision_at_threshold` as `holdout_auc - 0.04`: the same number twice with
different constants subtracted, presented as three independent metrics. And
`drift_delta` was `realised_win_rate - holdout_auc * 100`, which subtracts an AUC
from a percentage.

What replaces them is the one comparison that matters for a net-return model:
what `decide()` said it expected, against what the trade actually earned. The
model claims an edge in basis points before every trade, and that claim is
checkable. A model whose realised net runs consistently below its forecast is
mispriced, and the size of the gap is how mispriced.

Every field is nullable and null means "not measured". Nothing here substitutes a
plausible value for a missing one.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Health is derived from measurements, never from a stored grade. The old
# `readiness_tier` came from `optimization_results/*_validation.json`, an artifact
# of a deleted pipeline, so it read "UNKNOWN" for everything.
Health = Literal["healthy", "watch", "at_risk", "unknown"]


class EdgeCalibration(BaseModel):
    """Forecast against outcome, in basis points of notional.

    `delta_bps` is realised minus expected. Persistently negative means the model
    is overstating its edge, which over-sizes every position that clears the
    conviction floor — the failure mode that costs money quietly.
    """

    expected_net_bps: Optional[float] = None
    realised_net_bps: Optional[float] = None
    delta_bps: Optional[float] = None
    sample: int = 0


class CoinHealthRow(BaseModel):
    coin: str

    # Signals — what the model proposed.
    signals_total: int = 0
    # `signals_total` counts a capped window, not the universe. Named plainly
    # so a saturated count is not read as a total.
    signals_window: Optional[int] = None
    signals_truncated: bool = False
    signals_passed_gates: int = 0
    gate_pass_rate: Optional[float] = None
    top_gate_reason: Optional[str] = None
    last_signal_at: Optional[datetime] = None

    # The forecast, averaged over signals that cleared the gates.
    expected_net_bps: Optional[float] = None
    expected_carry_share: Optional[float] = None
    mean_cost_bps: Optional[float] = None

    # Outcomes — what actually happened.
    trades_closed: int = 0
    win_rate_realized: Optional[float] = None
    net_pnl: Optional[float] = None
    realised_net_bps: Optional[float] = None

    calibration: EdgeCalibration = Field(default_factory=EdgeCalibration)
    health: Health = "unknown"
    health_reason: Optional[str] = None


class ResearchSummaryKpis(BaseModel):
    """Universe-wide totals, plus the live model's identity.

    Aggregated over instruments rather than averaged where averaging would
    mislead: a win rate is a ratio, so summing the numerators and denominators is
    correct and averaging the per-coin rates is not.
    """

    signals_total: int = 0
    # `signals_total` counts a capped window, not the universe. Named plainly
    # so a saturated count is not read as a total.
    signals_window: Optional[int] = None
    signals_truncated: bool = False
    signals_passed_gates: int = 0
    gate_pass_rate: Optional[float] = None
    trades_closed: int = 0
    win_rate_realized: Optional[float] = None
    net_pnl: Optional[float] = None

    expected_net_bps: Optional[float] = None
    realised_net_bps: Optional[float] = None
    calibration_delta_bps: Optional[float] = None
    expected_carry_share: Optional[float] = None

    model_version: Optional[str] = None
    model_promoted: bool = False
    model_forced: bool = False
    model_age_hours: Optional[float] = None
    gates_failed: List[str] = Field(default_factory=list)
    kill_switch_status: str = "unknown"
    trials_to_date: int = 0
    health: Health = "unknown"


class ResearchSummaryResponse(BaseModel):
    generated_at: datetime
    kpis: ResearchSummaryKpis
    coins: List[CoinHealthRow]


class ResearchRunResponse(BaseModel):
    """A real retrain attempt, from `model_runs` and the promotion ledger.

    The previous version invented three runs per signal — a "train", an
    "optimize" and a "validate" — with fabricated start times, fabricated
    durations of 12, 20 and 8 minutes, and a status hardcoded to "success". None
    of them had happened.
    """

    id: str
    run_type: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    artifacts_version: Optional[str] = None
    symbols_trained: int = 0
    symbols_total: int = 0
    retrain_window_days: Optional[int] = None
    promoted: Optional[bool] = None
    forced: bool = False
    failed_gates: List[str] = Field(default_factory=list)
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    error: Optional[str] = None


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
    # Set when the importances could not be read, so the client can say why
    # instead of rendering an empty chart that looks like a zero result.
    importance_unavailable_reason: Optional[str] = None


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
    launch_metadata: dict = Field(default_factory=dict)


class ResearchScriptListResponse(BaseModel):
    scripts: List[ResearchScriptInfo]


class ResearchJobLogResponse(BaseModel):
    pid: int
    running: bool
    command: List[str]
    launched_at: datetime
    log_path: str
    logs: List[str]
