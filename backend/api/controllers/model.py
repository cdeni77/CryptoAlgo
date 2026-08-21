"""Serve the promotion ledger and the live model's provenance.

This reads what `core/promotion.py` wrote. It does not recompute a verdict, and
it does not invent one when the data is absent — a missing measurement is served
as null with a reason, so the dashboard can say "not measured" instead of
rendering a plausible number nobody produced.

The trader writes the ledger from its own container; the API reads it through the
`TRADER_DIR` mount. Reading JSON rather than importing the trader package is
deliberate: the API should not need LightGBM installed to answer "what is live",
and the one place it does need the model itself (feature importance) degrades to
an explicit "unavailable" if the import fails.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from models.model import (
    BacktestSummary,
    FeatureImportanceEntry,
    FeatureImportanceResponse,
    GateResult,
    KillSwitchStatus,
    LiveModelResponse,
    ModelProvenance,
    PathDistributionSummary,
    PromotionHistoryResponse,
    PromotionRecordResponse,
    SimulationSummary,
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = 'forecast.joblib'
PROMOTIONS_DIRNAME = 'promotions'
CURRENT_FILENAME = 'current.json'

# Cap on how much of the ledger one response carries. The ledger grows without
# bound by design; a dashboard does not need all of it.
MAX_HISTORY = 200

TOP_FEATURES = 25


def _trader_dir() -> Path:
    return Path(os.getenv('TRADER_DIR', '/trader'))


def _models_dir() -> Path:
    configured = os.getenv('MODELS_DIR')
    if configured:
        return Path(configured)
    return _trader_dir() / 'models'


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    """Parse a JSON file, or None.

    An absent file is a normal state — nothing promoted yet — and warns nothing.
    A file that exists but will not parse is a real problem and says so, once per
    call rather than being swallowed.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('unreadable %s: %s', path, exc)
        return None


def _iso(value: Any) -> Optional[str]:
    return str(value) if value is not None else None


def _distribution(payload: Any) -> Optional[PathDistributionSummary]:
    if not isinstance(payload, dict):
        return None
    return PathDistributionSummary(
        n=payload.get('n'),
        median=payload.get('median'),
        mean=payload.get('mean'),
        p05=payload.get('p05'),
        p95=payload.get('p95'),
        positive_fraction=payload.get('positive_fraction'),
    )


def _simulation(payload: Any) -> SimulationSummary:
    if not isinstance(payload, dict):
        return SimulationSummary()
    bootstrap = payload.get('bootstrap') or {}
    stress = payload.get('stress') or {}
    surface = payload.get('surface') or {}
    return SimulationSummary(
        bootstrap_sharpe=_distribution(bootstrap.get('sharpe')),
        bootstrap_max_drawdown=_distribution(bootstrap.get('max_drawdown')),
        probability_positive=bootstrap.get('probability_positive'),
        risk_of_ruin=bootstrap.get('risk_of_ruin'),
        block_length=bootstrap.get('block_length'),
        per_period_sharpe=_distribution(payload.get('per_period')),
        synthetic_sharpe=_distribution(payload.get('synthetic')),
        stressed_worst_sharpe=stress.get('worst'),
        parameter_plateau=surface.get('retention'),
    )


def _provenance(payload: Any, version: Optional[str]) -> ModelProvenance:
    payload = payload if isinstance(payload, dict) else {}
    return ModelProvenance(
        version=version,
        feature_set_hash=payload.get('feature_set_hash'),
        n_features=payload.get('n_features'),
        heads=list(payload.get('heads') or []),
        uses_symbol_identity=bool(payload.get('uses_symbol_identity')),
        horizon_bars=payload.get('horizon_bars'),
        cost_config_version=payload.get('cost_config_version'),
        trained_at=_iso(payload.get('trained_at')),
        data_as_of=_iso(payload.get('data_as_of')),
        train_rows=payload.get('train_rows'),
        effective_observations=payload.get('effective_observations'),
        train_start=_iso(payload.get('train_start')),
        train_end=_iso(payload.get('train_end')),
        symbols=list(payload.get('symbols') or []),
    )


def _backtest(payload: Any) -> BacktestSummary:
    payload = payload if isinstance(payload, dict) else {}
    return BacktestSummary(
        trades=payload.get('trades'),
        net_pnl=payload.get('net_pnl'),
        price_pnl=payload.get('price_pnl'),
        funding_pnl=payload.get('funding_pnl'),
        fees=payload.get('fees'),
        carry_contribution=payload.get('carry_contribution'),
        return_pct=payload.get('return_pct'),
        sharpe=payload.get('sharpe'),
        max_drawdown=payload.get('max_drawdown'),
        win_rate=payload.get('win_rate'),
        liquidations=payload.get('liquidations'),
        max_entry_participation=payload.get('max_entry_participation'),
        max_exit_participation=payload.get('max_exit_participation'),
    )


def _as_record(payload: dict[str, Any], *, is_live: bool = False) -> PromotionRecordResponse:
    gates = [
        GateResult(
            name=g.get('name', '?'),
            value=g.get('value'),
            # None, not 0.0. A missing threshold rendered as `>= 0.00`,
            # indistinguishable from a real gate that happens to sit at zero.
            threshold=g.get('threshold'),
            comparison=g.get('comparison'),
            passed=bool(g.get('passed')),
            note=g.get('note'),
        )
        for g in (payload.get('gates') or [])
        if isinstance(g, dict)
    ]
    version = payload.get('version', 'unknown')
    return PromotionRecordResponse(
        version=version,
        created_at=_iso(payload.get('created_at')),
        promoted=bool(payload.get('promoted')),
        forced=bool(payload.get('forced')),
        force_reason=payload.get('force_reason'),
        is_live=is_live,
        failed_gates=[g.name for g in gates if not g.passed],
        gates=gates,
        provenance=_provenance(payload.get('provenance'), version),
        backtest=_backtest(payload.get('backtest')),
        simulation=_simulation(payload.get('simulation')),
        error=payload.get('error'),
    )


def _ledger_files(models_dir: Path) -> list[Path]:
    directory = models_dir / PROMOTIONS_DIRNAME
    if not directory.exists():
        return []
    return sorted(
        (p for p in directory.glob('*.json') if p.name != CURRENT_FILENAME),
        reverse=True,
    )


def _current(models_dir: Path) -> Optional[dict[str, Any]]:
    """The live pointer, or None when nothing has been promoted."""
    return _read_json(models_dir / PROMOTIONS_DIRNAME / CURRENT_FILENAME)


def _kill_switch() -> KillSwitchStatus:
    """The orchestrator's verdict on the live model, from its state file."""
    state_path = Path(
        os.getenv('ORCHESTRATOR_STATE_FILE', str(_trader_dir() / 'data/orchestrator_state.json'))
    )
    state = _read_json(state_path) if state_path.exists() else None
    record = (state or {}).get('paper_monitoring')
    if not isinstance(record, dict):
        return KillSwitchStatus(status='not_evaluated')

    kpis = record.get('kpis') or {}
    return KillSwitchStatus(
        status=record.get('status', 'unknown'),
        version=record.get('version'),
        evaluated_at=_iso(record.get('evaluated_at')),
        reasons=list(record.get('reasons') or []),
        trades=kpis.get('trades'),
        win_rate=kpis.get('win_rate'),
        profit_factor=kpis.get('profit_factor'),
        drawdown=kpis.get('drawdown'),
        expectancy=kpis.get('expectancy'),
        trades_per_week=kpis.get('trades_per_week'),
        window_days=kpis.get('window_days'),
    )


def get_live_model() -> LiveModelResponse:
    """What is trading right now, or why nothing is."""
    models_dir = _models_dir()
    artifact = models_dir / MODEL_FILENAME
    has_model = artifact.exists()
    current = _current(models_dir)

    modified = None
    if has_model:
        modified = datetime.fromtimestamp(
            artifact.stat().st_mtime, tz=timezone.utc
        ).isoformat()

    return LiveModelResponse(
        generated_at=datetime.now(timezone.utc),
        has_model=has_model,
        artifact_path=str(artifact) if has_model else None,
        artifact_modified_at=modified,
        trials_to_date=max(len(_ledger_files(models_dir)), 1 if current else 0),
        live=_as_record(current, is_live=True) if current else None,
        kill_switch=_kill_switch(),
        # An artifact with no ledger entry was installed outside the gates.
        unrecorded_artifact=has_model and current is None,
    )


def get_promotion_history(limit: int = 50) -> PromotionHistoryResponse:
    """Every evaluation, newest first, rejections included."""
    models_dir = _models_dir()
    current = _current(models_dir)
    live_version = (current or {}).get('version')
    files = _ledger_files(models_dir)

    records: list[PromotionRecordResponse] = []
    for path in files[: min(limit, MAX_HISTORY)]:
        payload = _read_json(path)
        if not payload:
            continue
        records.append(
            _as_record(payload, is_live=payload.get('version') == live_version)
        )

    return PromotionHistoryResponse(
        generated_at=datetime.now(timezone.utc),
        trials_to_date=max(len(files), 1 if current else 0),
        live_version=live_version,
        records=records,
    )


def get_feature_importance(head: str = 'price') -> FeatureImportanceResponse:
    """Real split gains from the trained booster, or an empty list and a reason.

    Loading the artifact requires the trader package and LightGBM, which the API
    container may not have. That failure is reported rather than papered over: the
    previous implementation substituted a hardcoded table of six plausible-looking
    feature names whenever the file it wanted was missing — which it always was,
    because it pointed at an artifact the deleted pipeline used to write.
    """
    generated_at = datetime.now(timezone.utc)
    models_dir = _models_dir()
    artifact = models_dir / MODEL_FILENAME
    version = (_current(models_dir) or {}).get('version')

    if not artifact.exists():
        return FeatureImportanceResponse(
            generated_at=generated_at, version=version,
            unavailable_reason=(
                f'no promoted model at {artifact}. Evaluate one with '
                f'"python -m scripts.promote".'
            ),
        )

    try:
        import sys

        trader = str(_trader_dir())
        if trader not in sys.path:
            sys.path.insert(0, trader)
        from core.model import ForecastModel  # type: ignore[import-not-found]

        model = ForecastModel.load(artifact)
    except Exception as exc:  # noqa: BLE001 - the reason is the useful output
        logger.warning('cannot load %s: %s', artifact, exc)
        return FeatureImportanceResponse(
            generated_at=generated_at, version=version,
            unavailable_reason=f'cannot load the model artifact: {exc}',
        )

    booster = model.heads.get(head)
    if booster is None:
        available = ', '.join(sorted(model.heads)) or 'none'
        return FeatureImportanceResponse(
            generated_at=generated_at, version=version,
            unavailable_reason=f'no "{head}" head on this model (heads: {available})',
        )

    # The heads are LightGBM's sklearn estimators (`LGBMRegressor`), whose
    # attributes are `feature_importances_` and `feature_name_`. The Booster API
    # — `feature_importance(importance_type=...)` — is a different object, and
    # calling it here produced "no attribute 'feature_importance'" on every
    # request. Both are handled, because either can end up in a saved artifact.
    # `feature_importances_` on LightGBM's sklearn wrapper defaults to SPLIT
    # COUNT, while the Booster API's `feature_importance` can be asked for gain.
    # Both used to land in a list named `gains` and be served under one field, so
    # the endpoint's documented "split gains" was true of one branch only. Prefer
    # the inner Booster, which can be asked for gain explicitly, and record which
    # measure was actually used.
    importance_kind = 'gain'
    try:
        if hasattr(booster, 'booster_'):
            inner = booster.booster_
            gains = list(inner.feature_importance(importance_type='gain'))
            names = list(inner.feature_name())
        elif hasattr(booster, 'feature_importance'):
            gains = list(booster.feature_importance(importance_type='gain'))
            names = list(booster.feature_name())
        else:
            # Only the sklearn attribute is available, which is split count.
            importance_kind = 'split_count'
            gains = list(booster.feature_importances_)
            names = list(
                getattr(booster, 'feature_name_', None)
                or getattr(booster, 'feature_names_in_', None)
                or [f'f{i}' for i in range(len(gains))]
            )
    except Exception as exc:  # noqa: BLE001
        return FeatureImportanceResponse(
            generated_at=generated_at, version=version,
            unavailable_reason=f'head "{head}" exposes no importances: {exc}',
        )

    ranked_all = sorted(zip(names, gains), key=lambda pair: pair[1], reverse=True)
    # Normalise over what is actually returned, not over every feature: dividing
    # by the full sum while serving the top 25 meant the list never summed to one,
    # which is what the endpoint documents.
    kept = [pair for pair in ranked_all if pair[1] > 0][:TOP_FEATURES]
    total = float(sum(value for _, value in kept)) or 1.0
    return FeatureImportanceResponse(
        generated_at=generated_at,
        version=version,
        importance_kind=importance_kind,
        features=[
            FeatureImportanceEntry(feature=name, importance=float(gain) / total, head=head)
            for name, gain in kept
        ],
    )
