"""Response shapes for the model provenance and promotion-gate surfaces.

These replace the `readiness_tier` / `robustness_gate` scheme, which read
`optimization_results/*_validation.json` — artifacts of a pipeline that no longer
exists, so every field fell back to "UNKNOWN" and a position scale of zero. A
dashboard that reports UNKNOWN for everything is not telling you less than it
could; it is telling you nothing while looking like it is telling you something.

The promotion ledger already holds the real answer. Nothing here recomputes a
verdict: it serves what `core/promotion.py` recorded when it decided.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class GateResult(BaseModel):
    """One promotion criterion, its measured value, and whether it passed."""

    name: str
    value: Optional[float] = None
    threshold: float
    comparison: str  # 'min' — at least; 'max' — at most
    passed: bool
    note: Optional[str] = None


class ModelProvenance(BaseModel):
    """What a model was trained on. Enough to tell a stale model from a fresh one."""

    version: Optional[str] = None
    feature_set_hash: Optional[str] = None
    n_features: Optional[int] = None
    heads: List[str] = Field(default_factory=list)
    uses_symbol_identity: bool = False
    horizon_bars: Optional[int] = None
    cost_config_version: Optional[str] = None
    trained_at: Optional[str] = None
    data_as_of: Optional[str] = None
    train_rows: Optional[int] = None
    # Overlapping labels are not independent observations, so this is the number
    # that should be used for any significance claim — not train_rows.
    effective_observations: Optional[float] = None
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)


class BacktestSummary(BaseModel):
    """The out-of-sample result, decomposed.

    Price, funding and fees are kept apart because they diagnose different
    problems: positive gross price PnL with a negative net is a cost problem, and
    no amount of retraining fixes it.
    """

    trades: Optional[int] = None
    net_pnl: Optional[float] = None
    price_pnl: Optional[float] = None
    funding_pnl: Optional[float] = None
    fees: Optional[float] = None
    carry_contribution: Optional[float] = None
    return_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    liquidations: Optional[int] = None
    max_entry_participation: Optional[float] = None
    max_exit_participation: Optional[float] = None


class PathDistributionSummary(BaseModel):
    """A distribution rather than a point estimate — the whole point of the stack."""

    n: Optional[int] = None
    median: Optional[float] = None
    mean: Optional[float] = None
    p05: Optional[float] = None
    p95: Optional[float] = None
    positive_fraction: Optional[float] = None


class SimulationSummary(BaseModel):
    bootstrap_sharpe: Optional[PathDistributionSummary] = None
    bootstrap_max_drawdown: Optional[PathDistributionSummary] = None
    probability_positive: Optional[float] = None
    risk_of_ruin: Optional[float] = None
    block_length: Optional[float] = None
    per_period_sharpe: Optional[PathDistributionSummary] = None
    synthetic_sharpe: Optional[PathDistributionSummary] = None
    stressed_worst_sharpe: Optional[float] = None
    parameter_plateau: Optional[float] = None


class PromotionRecordResponse(BaseModel):
    """One candidate evaluation, promoted or not.

    Rejections are served alongside successes on purpose: the count of attempts
    is what the deflated Sharpe ratio discounts by, so hiding the failures would
    make every survivor look better than the evidence supports.
    """

    version: str
    created_at: Optional[str] = None
    promoted: bool = False
    forced: bool = False
    force_reason: Optional[str] = None
    is_live: bool = False
    failed_gates: List[str] = Field(default_factory=list)
    gates: List[GateResult] = Field(default_factory=list)
    provenance: ModelProvenance = Field(default_factory=ModelProvenance)
    backtest: BacktestSummary = Field(default_factory=BacktestSummary)
    simulation: SimulationSummary = Field(default_factory=SimulationSummary)
    error: Optional[str] = None


class KillSwitchStatus(BaseModel):
    """Realised paper results on the *live* model.

    Distinct from the gates, which measure a candidate before it trades. A model
    can clear every gate and still decay, because the market it was fitted to
    stopped existing. `insufficient_samples` is reported but is not a quarantine:
    "not enough to judge yet" is a different finding from "this is broken".
    """

    status: str = 'unknown'
    version: Optional[str] = None
    evaluated_at: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    trades: Optional[int] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    drawdown: Optional[float] = None
    expectancy: Optional[float] = None
    trades_per_week: Optional[float] = None
    window_days: Optional[int] = None


class LiveModelResponse(BaseModel):
    """Everything about what is trading right now, or why nothing is."""

    generated_at: datetime
    has_model: bool
    artifact_path: Optional[str] = None
    artifact_modified_at: Optional[str] = None
    trials_to_date: int = 0
    live: Optional[PromotionRecordResponse] = None
    kill_switch: KillSwitchStatus = Field(default_factory=KillSwitchStatus)
    # Present when an artifact exists with no ledger entry — a model installed by
    # hand, outside the gates. Worth surfacing rather than rendering as normal.
    unrecorded_artifact: bool = False


class PromotionHistoryResponse(BaseModel):
    generated_at: datetime
    trials_to_date: int
    live_version: Optional[str] = None
    records: List[PromotionRecordResponse] = Field(default_factory=list)


class FeatureImportanceEntry(BaseModel):
    feature: str
    importance: float
    head: str


class FeatureImportanceResponse(BaseModel):
    """Real gains from the trained boosters, or an empty list and a reason.

    The previous implementation returned a hardcoded table — `momentum_24h: 0.26`
    and five more — whenever the artifact it wanted was missing, which it always
    was. Invented explainability is worse than none: it looks exactly like the
    real thing.
    """

    generated_at: datetime
    version: Optional[str] = None
    # Which measure the numbers are: LightGBM's sklearn attribute is split count,
    # the Booster API can be asked for gain. Both used to be served under one
    # field with the docstring claiming gain.
    importance_kind: Optional[str] = None
    features: List[FeatureImportanceEntry] = Field(default_factory=list)
    unavailable_reason: Optional[str] = None
