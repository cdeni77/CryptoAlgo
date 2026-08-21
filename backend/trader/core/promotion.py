"""Promotion: the only route a model takes from training to live.

The previous system's promotion rule was "did any model finish training", with a
paper-trading win-rate check bolted on afterwards. That is backwards. Win rate is
a lagging indicator measured on the model already trading; by the time it moves,
the money is spent. The decision has to happen before the model is live, on
evidence the model has never seen.

So a candidate here is trained, walk-forward backtested, resampled, stressed, and
measured against `core.metrics.DEFAULT_GATES`. Only a candidate that clears every
gate is promoted, and the whole evaluation is written to a ledger entry that lives
next to the artifact:

    models/forecast.joblib               the promoted model
    models/promotions/<version>.json     one entry per evaluation, kept forever
    models/promotions/current.json       which version is live, and why

Keeping rejected candidates is the point of the ledger. A directory containing
only successes cannot tell you how many configurations were tried, and the number
of attempts is exactly what the deflated Sharpe ratio needs to discount by. The
ledger is also what the API serves to the provenance and gates screens: the
frontend does not recompute any of this, it reads what the promotion recorded.

`--force` exists because a human may have a reason the gates cannot see, but it
records the override in the ledger rather than bypassing it, so a forced model is
visibly forced for as long as it is live.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from core.config import Config
from core.dataset import Dataset
from core.metrics import Gate, evaluate_gates, gate_report, sharpe_ratio, summarise_paths
from core.model import ForecastModel, train_forecast_model
from core.simulation import SimulationReport, bootstrap_trades, cost_stress, synthetic_panel

logger = logging.getLogger(__name__)

PROMOTIONS_DIRNAME = 'promotions'
CURRENT_FILENAME = 'current.json'
MODEL_FILENAME = 'forecast.joblib'
STAGING_DIRNAME = '.staging'

# Resamples for the trade bootstrap. Two thousand is enough for a stable p05 on a
# few hundred trades and cheap enough to run on every candidate.
BOOTSTRAP_RESAMPLES = 2_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_version() -> str:
    """A sortable, collision-free version stamp.

    The timestamp prefix dominates the sort, so sorting by name still orders by
    time. The suffix is what makes it safe: a search campaign evaluates many
    candidates per second, and a bare second-resolution stamp had them
    overwriting each other's ledger entries — which silently shrinks the trial
    count that the deflated Sharpe ratio discounts by.
    """
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{stamp}-{uuid4().hex[:6]}'


def _record_path(directory: Path, version: str) -> Path:
    """Where a version's record lives, never colliding with a different one.

    `new_version` makes collisions essentially impossible, but a caller may pass
    its own version string, and an overwritten record is a lost trial rather than
    a visible error. So a clash disambiguates instead of clobbering.
    """
    path = directory / f'{version}.json'
    if not path.exists():
        return path
    try:
        existing = json.loads(path.read_text()).get('version')
    except (json.JSONDecodeError, OSError):
        existing = None
    if existing == version:
        return path  # a rewrite of the same record, which is fine
    for n in range(2, 1_000):
        candidate = directory / f'{version}-{n}.json'
        if not candidate.exists():
            logger.warning('version %s already recorded; writing %s', version, candidate.name)
            return candidate
    raise RuntimeError(f'cannot find a free record path for version {version}')


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass
class PromotionRecord:
    """One candidate evaluation, whether or not it was promoted.

    This is the unit the ledger stores and the API serves. It carries the whole
    case for or against the model — provenance, out-of-sample result, simulation
    distributions, and every gate with its measured value — so nothing
    downstream has to re-derive a verdict and reach a different one.
    """

    version: str
    created_at: str = field(default_factory=_now)
    promoted: bool = False
    forced: bool = False
    force_reason: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    simulation: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Optional[float]] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def failed_gates(self) -> list[str]:
        return [g['name'] for g in self.gates if not g.get('passed')]

    def as_dict(self) -> dict[str, Any]:
        return {
            'version': self.version,
            'created_at': self.created_at,
            'promoted': self.promoted,
            'forced': self.forced,
            'force_reason': self.force_reason,
            'failed_gates': self.failed_gates,
            'provenance': self.provenance,
            'backtest': self.backtest,
            'simulation': self.simulation,
            'measurements': self.measurements,
            'gates': self.gates,
            'dataset': self.dataset,
            'error': self.error,
        }

    def __str__(self) -> str:
        if self.error:
            return f'{self.version}: failed to evaluate ({self.error})'
        verdict = 'PROMOTED' if self.promoted else f'BLOCKED by {len(self.failed_gates)} gate(s)'
        if self.forced:
            verdict += ' (FORCED)'
        return f'{self.version}: {verdict}'


def _gate_rows(gates: list[Gate]) -> list[dict[str, Any]]:
    return [{
        'name': g.name,
        'value': g.value,
        'threshold': g.threshold,
        'comparison': g.comparison,
        'passed': g.passed,
        'note': getattr(g, 'note', '') or None,
    } for g in gates]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_candidate(
    dataset: Dataset,
    config: Config,
    *,
    version: Optional[str] = None,
    n_periods: int = 6,
    initial_equity: float = 100_000.0,
    spread_bps: float = 4.0,
    synthetic_paths: int = 20,
    full: bool = True,
    data_as_of: Optional[str] = None,
) -> tuple[Optional[ForecastModel], PromotionRecord]:
    """Train a candidate and build the whole case for or against it.

    Returns the trained model and its record. The model is returned even when the
    gates block it, because a blocked candidate is still worth inspecting — but
    `promote` will refuse to install it, which is the separation that matters.

    Set `full=False` to skip synthetic panels and cost stress. That is for a fast
    development loop only: both are gated, and a skipped gate fails, so a
    `full=False` evaluation can never promote.
    """
    from core.backtest import walk_forward_backtest

    record = PromotionRecord(version=version or new_version())
    record.dataset = dataset.summary()

    model = train_forecast_model(
        dataset.features, dataset.targets, config=config, data_as_of=data_as_of,
    )
    if model is None:
        record.error = 'not enough resolved rows to train'
        return None, record
    record.provenance = model.provenance()

    # The out-of-sample result. Walk-forward is not an option here: backtesting a
    # model over its own training window measures memorisation, and on driftless
    # random walks that read as a t-statistic of +7.
    result, generated = walk_forward_backtest(
        dataset.features, dataset.targets,
        bars_by_symbol=dataset.bars, funding_by_symbol=dataset.funding,
        config=config, profiles=dataset.profiles,
        n_periods=n_periods, initial_equity=initial_equity, spread_bps=spread_bps,
    )
    record.backtest = {
        **result.summary(),
        'periods': [[str(a), str(b)] for a, b in generated.periods],
        'forecasts': generated.summary(),
    }

    report = SimulationReport(
        oos_trades=result.n_trades,
        max_exit_participation=result.max_exit_participation,
    )

    if result.trades:
        returns = result.trades_frame()['net_return'].to_numpy()
        report.bootstrap = bootstrap_trades(returns, n_resamples=BOOTSTRAP_RESAMPLES)

        # Each walk-forward period is an independent out-of-sample stretch, so the
        # spread of their Sharpes stands in for the CPCV path distribution.
        period_sharpes = []
        for start, end in generated.periods:
            window = result.equity_curve.loc[
                (result.equity_curve.index >= start) & (result.equity_curve.index <= end)
            ]
            if len(window) > 2:
                period_sharpes.append(sharpe_ratio(window.pct_change().dropna()))
        if period_sharpes:
            report.cpcv = summarise_paths(period_sharpes)

        if full:
            def run_with(candidate_config: Config) -> float:
                outcome, _ = walk_forward_backtest(
                    dataset.features, dataset.targets,
                    bars_by_symbol=dataset.bars, funding_by_symbol=dataset.funding,
                    config=candidate_config, profiles=dataset.profiles,
                    n_periods=n_periods, initial_equity=initial_equity,
                    spread_bps=spread_bps,
                )
                return outcome.sharpe

            report.stress = cost_stress(run_with, config)

            synthetic_sharpes = []
            for seed in range(synthetic_paths):
                outcome, _ = walk_forward_backtest(
                    dataset.features, dataset.targets,
                    bars_by_symbol=synthetic_panel(dataset.bars, seed=seed),
                    funding_by_symbol=dataset.funding,
                    config=config, profiles=dataset.profiles,
                    n_periods=n_periods, initial_equity=initial_equity,
                    spread_bps=spread_bps,
                )
                synthetic_sharpes.append(outcome.sharpe)
            report.synthetic = summarise_paths(synthetic_sharpes)

    record.simulation = report.as_dict()
    record.measurements = report.measurements()
    promoted, gates = evaluate_gates(record.measurements)
    record.gates = _gate_rows(gates)
    record.promoted = bool(promoted)
    return model, record


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def promotions_dir(models_dir: Path) -> Path:
    return Path(models_dir) / PROMOTIONS_DIRNAME


def write_record(record: PromotionRecord, models_dir: Path) -> Path:
    """Append the record to the ledger. Rejections are kept too.

    A ledger of successes only cannot answer "how many configurations were
    tried", and that count is the trials figure the deflated Sharpe ratio
    discounts by. Throwing the failures away makes the surviving number look
    better than the evidence supports.
    """
    directory = promotions_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = _record_path(directory, record.version)
    _write_json_atomically(path, record.as_dict())
    return path


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


def load_records(models_dir: Path, *, limit: Optional[int] = None) -> list[PromotionRecord]:
    """Every evaluation, newest first. Malformed entries are skipped, not fatal."""
    directory = promotions_dir(models_dir)
    if not directory.exists():
        return []

    paths = sorted(
        (p for p in directory.glob('*.json') if p.name != CURRENT_FILENAME),
        reverse=True,
    )
    records: list[PromotionRecord] = []
    for path in paths[:limit] if limit else paths:
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('skipping unreadable promotion record %s: %s', path, exc)
            continue
        payload.pop('failed_gates', None)  # derived, not stored state
        records.append(PromotionRecord(**payload))
    return records


def current_record(models_dir: Path) -> Optional[PromotionRecord]:
    """Which version is live, according to the pointer written at promotion."""
    path = promotions_dir(models_dir) / CURRENT_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    payload.pop('failed_gates', None)
    return PromotionRecord(**payload)


def trials_to_date(models_dir: Path) -> int:
    """How many candidates have been evaluated, for the deflated Sharpe.

    At least one, because the model in front of you is itself a trial.
    """
    return max(len(load_records(models_dir)), 1)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote(
    model: ForecastModel,
    record: PromotionRecord,
    *,
    models_dir: Path,
    force: bool = False,
    force_reason: Optional[str] = None,
) -> tuple[bool, PromotionRecord]:
    """Install the model, but only if its gates passed or a human overrode them.

    The install goes through a staging directory and an atomic rename, so a crash
    mid-write cannot leave a half-written artifact being scored against live
    prices. A refused candidate still gets a ledger entry — that is how the trial
    count stays honest.
    """
    models_dir = Path(models_dir)

    if not record.promoted and force:
        if not force_reason:
            raise ValueError('forcing a blocked promotion requires a reason')
        record.forced = True
        record.force_reason = force_reason
        record.promoted = True
        logger.warning(
            'forcing promotion of %s past %d failed gate(s) (%s): %s',
            record.version, len(record.failed_gates),
            ', '.join(record.failed_gates), force_reason,
        )

    if not record.promoted:
        write_record(record, models_dir)
        logger.error(
            'refusing to promote %s: %s failed',
            record.version, ', '.join(record.failed_gates) or 'evaluation',
        )
        return False, record

    staging = models_dir / STAGING_DIRNAME / record.version
    staging.mkdir(parents=True, exist_ok=True)
    try:
        staged = model.save(staging / MODEL_FILENAME)
        models_dir.mkdir(parents=True, exist_ok=True)
        os.replace(staged, models_dir / MODEL_FILENAME)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    write_record(record, models_dir)
    directory = promotions_dir(models_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(directory / CURRENT_FILENAME, record.as_dict())

    logger.info('promoted %s to %s', record.version, models_dir / MODEL_FILENAME)
    return True, record


def report(record: PromotionRecord) -> str:
    """The human-readable verdict, gates and failures first."""
    if record.error:
        return f'{record.version}: {record.error}'
    gates = [
        Gate(g['name'], g['value'], g['threshold'], g['comparison'], g['passed'],
             g.get('note') or '')
        for g in record.gates
    ]
    lines = [gate_report(gates)]
    if record.forced:
        lines.append(f'FORCED: {record.force_reason}')
    return '\n'.join(lines)
