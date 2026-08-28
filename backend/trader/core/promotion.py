"""Promotion is the gate, and rejections are kept.

A candidate is trained, walk-forward evaluated, gated, and only then installed.
The staging directory is renamed into place atomically, so `models/forecast.joblib`
is either the old model or the new one and never a half-written file that the
live path loads at the wrong moment.

**Rejections stay on disk, and that is not sentiment.** The trial count is what
a deflated Sharpe discounts by, and a project that deletes its failures cannot
compute its own multiple-testing correction. `models/promotions/` is the ledger:
every attempt, its gates, and — if forced — the written reason.

**`--force` needs a reason and records it.** There is one genuinely good argument
for overriding a gate here: a model can be right on a high-conviction tail while
its average forecast is not, and the gates read averages. That argument is real,
and it is also exactly the argument that kept a losing perp system alive for
months. So it is available, it is written down, and it travels with the artifact.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.metrics import (
    DEFAULT_GATES, EvaluationReport, Gate, evaluate_gates, gate_report, gates_passed,
)
from core.model import ForecastModel

logger = logging.getLogger(__name__)

_TRADER_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.getenv('MODELS_ROOT') or _TRADER_ROOT / 'models')
LIVE_MODEL = 'forecast.joblib'
STAGING = '.staging'
LEDGER = 'promotions'


def version_stamp(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%SZ')


def _unique_version(ledger: Path, now: Optional[datetime] = None) -> str:
    """A version no ledger entry already uses.

    The stamp has second resolution, so two candidates evaluated in the same
    second collided and the second overwrote the first — which silently lost an
    attempt from the ledger. That matters more than it sounds: the ledger *is*
    the trial count, and a trial count that undercounts makes every
    multiple-testing correction computed from it too generous. Found by a test
    that promoted three candidates in a loop and got one row back.
    """
    base = version_stamp(now)
    if not (ledger / f'{base}.json').exists():
        return base
    for suffix in range(1, 1000):
        candidate = f'{base}-{suffix:03d}'
        if not (ledger / f'{candidate}.json').exists():
            return candidate
    raise RuntimeError(f'a thousand attempts already share the stamp {base}')


@dataclass
class PromotionAttempt:
    """One evaluated candidate, whatever the verdict."""

    version: str
    created_at: str
    gates: list[Gate]
    forced: bool = False
    force_reason: Optional[str] = None
    installed: bool = False
    model_provenance: dict = field(default_factory=dict)
    report_provenance: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return gates_passed(self.gates)

    def payload(self) -> dict:
        return {
            'version': self.version,
            'created_at': self.created_at,
            'passed': self.passed,
            'installed': self.installed,
            'forced': self.forced,
            'force_reason': self.force_reason,
            'gates': [
                {'name': g.name, 'value': g.value, 'threshold': g.threshold,
                 'direction': g.direction, 'passed': g.passed}
                for g in self.gates
            ],
            'failed_gates': [g.name for g in self.gates if not g.passed],
            'model': self.model_provenance,
            'report': self.report_provenance,
        }

    def summary(self) -> str:
        verdict = ('installed' if self.installed else
                   'blocked' if not self.passed else 'passed but not installed')
        forced = f' (forced: {self.force_reason})' if self.forced else ''
        return f'{self.version}: {verdict}{forced}\n' + gate_report(self.gates)


def report_provenance(report: EvaluationReport) -> dict:
    return {
        'folds': report.folds_total,
        'windows_evaluated': report.total_windows,
        'log_loss_skill': report.mean_skill,
        'log_loss_skill_se': report.skill_standard_error,
        'log_loss_skill_t': report.skill_t,
        'folds_positive': report.folds_positive,
        'sign_agreement_p': report.sign_agreement_p_value,
        'max_calibration_error': report.max_ece,
        'mean_residual_scale': report.mean_residual_scale,
        'max_control_gain_share': report.max_control_gain_share,
        'gate_values': report.gate_values(),
        'config': report.config_provenance,
        'notes': report.notes,
    }


def evaluate_candidate(
    model: ForecastModel,
    report: EvaluationReport,
    *,
    gates: Optional[dict[str, tuple[float, str]]] = None,
    version: Optional[str] = None,
    extra: Optional[dict[str, float]] = None,
) -> PromotionAttempt:
    """Score a candidate without touching the filesystem.

    `extra` is for measurements the report cannot produce — the market
    comparison, which needs live-recorded quotes. Omitting it leaves those gates
    NaN, and NaN fails, so a caller that forgets cannot promote by accident.
    """
    return PromotionAttempt(
        version=version or version_stamp(),
        created_at=datetime.now(timezone.utc).isoformat(),
        gates=evaluate_gates(report, gates or DEFAULT_GATES, extra=extra),
        model_provenance=model.provenance(),
        report_provenance=report_provenance(report),
    )


def promote(
    model: ForecastModel,
    report: EvaluationReport,
    *,
    root: Optional[Path] = None,
    gates: Optional[dict[str, tuple[float, str]]] = None,
    force: bool = False,
    force_reason: Optional[str] = None,
    trades: Optional[pd.DataFrame] = None,
    extra: Optional[dict[str, float]] = None,
) -> PromotionAttempt:
    """Gate, then install atomically. Records the attempt either way."""
    if force and not force_reason:
        raise ValueError(
            'a forced promotion needs a written reason. The one good argument for '
            'overriding these gates — skill on a high-conviction tail that the '
            'average forecast does not show — is also the argument that kept a '
            'losing system alive, so it has to be stated and stored.'
        )
    # An artifact that cannot score is not a candidate, and --force must not
    # reach past this. Every other gate asks whether the model is GOOD; this one
    # asks whether `scripts/live.py` can load it at all. It was missing, and a
    # refit fitted without its scoring bundle installed cleanly, printed a full
    # gate table, and then failed on every live cycle instead.
    if not getattr(model, 'deployable', True):
        raise ValueError(
            'this candidate carries no scoring bundle, so it cannot score a '
            'window it has never seen and the live loop will refuse it on every '
            'cycle. Installing it would replace a working artifact with one that '
            'cannot trade. This is not a gate --force may override.')
    root = Path(root) if root else MODELS_ROOT
    ledger = root / LEDGER
    ledger.mkdir(parents=True, exist_ok=True)
    attempt = evaluate_candidate(model, report, gates=gates, extra=extra,
                                 version=_unique_version(ledger))
    attempt.forced = bool(force)
    attempt.force_reason = force_reason

    if attempt.passed or force:
        staging = root / STAGING / attempt.version
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        model.save(staging / LIVE_MODEL)
        (staging / 'promotion.json').write_text(
            json.dumps(attempt.payload(), indent=2, default=str))
        if trades is not None and not trades.empty:
            trades.to_parquet(staging / 'trades.parquet', index=False)

        # Stage the whole directory, then move one file into place. The live
        # path opens `forecast.joblib` by name, so a partially written file at
        # that path is a loaded model with a truncated booster — which fails
        # somewhere else entirely, hours later.
        destination = root / LIVE_MODEL
        temporary = root / f'.{LIVE_MODEL}.incoming'
        shutil.copy2(staging / LIVE_MODEL, temporary)
        os.replace(temporary, destination)
        provenance = staging / f'{Path(LIVE_MODEL).stem}.provenance.json'
        if provenance.exists():
            shutil.copy2(provenance, root / provenance.name)
        attempt.installed = True
        logger.info('installed %s -> %s', attempt.version, destination)
    else:
        logger.warning('blocked %s: %s', attempt.version,
                       ', '.join(g.name for g in attempt.gates if not g.passed))

    (ledger / f'{attempt.version}.json').write_text(
        json.dumps(attempt.payload(), indent=2, default=str))
    return attempt


def load_live(root: Optional[Path] = None, config=None) -> Optional[ForecastModel]:
    """The promoted artifact, verified against `config` when one is given.

    `scripts/live.py` never read `config_provenance`, so a model promoted under
    one set of economics traded under whatever the current defaults happened to
    be — worth up to the whole `min_edge_pp` gate in probability terms, silently.
    """
    path = (Path(root) if root else MODELS_ROOT) / LIVE_MODEL
    if not path.exists():
        return None
    return ForecastModel.load(path, config)


def history(root: Optional[Path] = None) -> pd.DataFrame:
    """Every attempt, newest first. What has been tried, and why not.

    The trial count is the point: a deflated Sharpe needs it, and so does anyone
    reading a passing result and wondering how many candidates it took.
    """
    ledger = (Path(root) if root else MODELS_ROOT) / LEDGER
    if not ledger.exists():
        return pd.DataFrame()
    rows = []
    for path in sorted(ledger.glob('*.json'), reverse=True):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning('unreadable ledger entry %s', path)
            continue
        report = payload.get('report', {})
        rows.append({
            'version': payload.get('version'),
            'created_at': payload.get('created_at'),
            'passed': payload.get('passed'),
            'installed': payload.get('installed'),
            'forced': payload.get('forced'),
            'failed_gates': ', '.join(payload.get('failed_gates', [])),
            'log_loss_skill': report.get('log_loss_skill'),
            'folds_positive': report.get('folds_positive'),
            'windows_evaluated': report.get('windows_evaluated'),
            'force_reason': payload.get('force_reason'),
        })
    return pd.DataFrame(rows)


def trial_count(root: Optional[Path] = None) -> int:
    """How many candidates have been evaluated. The multiple-testing denominator."""
    frame = history(root)
    return 0 if frame.empty else len(frame)
