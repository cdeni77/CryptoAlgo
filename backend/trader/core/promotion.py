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
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.metrics import (
    DEFAULT_GATES, EvaluationReport, Gate, deflated_sharpe_ratio, evaluate_gates,
    gate_report, gate_spec, gate_spec_diff, gate_spec_hash, gates_passed,
    load_gate_lock,
)
from core.model import ForecastModel

logger = logging.getLogger(__name__)

_TRADER_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.getenv('MODELS_ROOT') or _TRADER_ROOT / 'models')
LIVE_MODEL = 'forecast.joblib'
STAGING = '.staging'
LEDGER = 'promotions'
# One line per research run, beside the ledger: the trial count the ledger
# alone undercounts, because evaluations, ablations and dry runs never reach it.
TRIALS = 'trials.jsonl'
# Which sealed holdouts have been read, when and why.
HOLDOUT_MARKER = 'holdout_unsealed.jsonl'


def models_root(root: Optional[Path] = None) -> Path:
    """`root`, else the module's MODELS_ROOT read at call time (tests patch it)."""
    return Path(root) if root else Path(MODELS_ROOT)


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
    gate_spec_hash: Optional[str] = None
    # Set when promotion refused to run the gates at all (a changed gate spec).
    refused: Optional[str] = None
    gate_change: Optional[dict] = None
    trial_id: Optional[str] = None
    trial_count: Optional[int] = None
    deflated_sharpe: dict = field(default_factory=dict)
    holdout: Optional[dict] = None

    @property
    def passed(self) -> bool:
        return self.refused is None and gates_passed(self.gates)

    def payload(self) -> dict:
        return {
            'version': self.version,
            'created_at': self.created_at,
            'passed': self.passed,
            'installed': self.installed,
            'forced': self.forced,
            'force_reason': self.force_reason,
            'refused': self.refused,
            'gate_spec_hash': self.gate_spec_hash,
            'gate_change': self.gate_change,
            'trial_id': self.trial_id,
            'trial_count': self.trial_count,
            'deflated_sharpe': self.deflated_sharpe,
            'holdout': self.holdout,
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
        verdict = ('refused' if self.refused else 'installed' if self.installed else
                   'blocked' if not self.passed else 'passed but not installed')
        forced = f' (forced: {self.force_reason})' if self.forced else ''
        why = f'\n{self.refused}' if self.refused else ''
        return f'{self.version}: {verdict}{forced}{why}\n' + gate_report(self.gates)


def _json_float(value) -> Optional[float]:
    """NaN and inf are not JSON. `json.dumps` emits bare `NaN`, which Python
    reads back and every strict parser rejects — including the dashboard's."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def reliability_rows(report: EvaluationReport) -> list[dict]:
    """The deployed fold's reliability table, as plain rows.

    **The LAST fold**: trained on the most history, so the one whose model is
    installed. Same choice `scripts/promote.publish_to_serving` makes, and they
    must not diverge.

    **This belongs in the ledger because the ledger is the record of account.**
    It was not, and the Calibration page's only other source was a best-effort
    mirror into Postgres at promotion time. On 2026-09-10 a retrain ran on a
    host with no `DATABASE_URL`; the mirror logged one line and returned, and
    the reliability table for an installed artifact was gone — every other
    number in the attempt survived on disk, that one did not. A record of
    account with a hole in it is not one.
    """
    folds = list(getattr(report, 'folds', None) or ())
    table = getattr(folds[-1], 'reliability_table', None) if folds else None
    if table is None:
        return []
    return [
        {'bin_low': _json_float(row.bin_low), 'bin_high': _json_float(row.bin_high),
         'predicted': _json_float(row.predicted),
         'observed': _json_float(row.observed), 'count': int(row.count)}
        for row in table.frame().itertuples() if int(row.count) > 0
    ]


def report_provenance(report: EvaluationReport) -> dict:
    unweighted = getattr(report, 'unweighted_skill', None)
    geometry = dict(getattr(report, 'fold_geometry', None) or {})
    return {
        'reliability': reliability_rows(report),
        # The fold count actually evaluated; `config.n_folds` is only a cap
        # under the count scheme and is ignored by the calendar one.
        'folds': report.folds_total,
        'fold_geometry': geometry,
        'windows_evaluated': report.total_windows,
        'log_loss_skill': report.mean_skill,
        'log_loss_skill_se': report.skill_standard_error,
        'log_loss_skill_t': report.skill_t,
        'log_loss_skill_p': getattr(report, 'skill_p_value', None),
        'effective_folds': getattr(report, 'effective_folds', None),
        'skill_weighting': 'windows',
        'log_loss_skill_unweighted': getattr(unweighted, 'mean', None),
        'log_loss_skill_se_unweighted': getattr(unweighted, 'standard_error', None),
        'log_loss_skill_t_unweighted': getattr(unweighted, 't', None),
        'sharpe_moments': dict(getattr(report, 'sharpe_moments', None) or {}),
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
        gate_spec_hash=gate_spec_hash(gates),
    )


def gate_lock_status(gates: Optional[dict[str, tuple[float, str]]] = None,
                     lock_path: Optional[Path] = None) -> dict:
    """The running gate spec against the committed lock.

    The lock lives in the code (`core/gate_spec.lock.json`), not under the
    models volume, and a test pins it to `GATE_SPEC`. What this catches is a
    gate edited on a host that runs the mounted source — the shape of every
    threshold change in this ledger's history, each made while that gate was
    the one blocking.
    """
    lock = load_gate_lock(lock_path)
    current = gate_spec(gates)
    current_hash = gate_spec_hash(gates)
    locked_hash = lock.get('gate_spec_hash')
    return {'matches': current_hash == locked_hash,
            'current_hash': current_hash, 'locked_hash': locked_hash,
            'diff': gate_spec_diff(lock.get('gates') or [], current)}


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
    accept_gate_change: Optional[str] = None,
    lock_path: Optional[Path] = None,
    trial_id: Optional[str] = None,
    holdout: Optional[dict] = None,
) -> PromotionAttempt:
    """Gate, then install atomically. Records the attempt either way.

    Refuses outright — no install, `--force` or not — when the gate spec
    differs from the committed lock, unless `accept_gate_change` gives a reason;
    the reason and the spec diff then travel with the ledger entry.
    """
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
    root = models_root(root)
    ledger = root / LEDGER
    ledger.mkdir(parents=True, exist_ok=True)
    attempt = evaluate_candidate(model, report, gates=gates, extra=extra,
                                 version=_unique_version(ledger))
    attempt.forced = bool(force)
    attempt.force_reason = force_reason
    attempt.holdout = holdout
    attempt.trial_id = trial_id or new_trial_id()
    attempt.trial_count = trial_count(root, including=attempt.trial_id)
    attempt.deflated_sharpe = deflated_sharpe_for(
        report, root=root, trial_id=attempt.trial_id)

    status = gate_lock_status(gates, lock_path)
    if not status['matches']:
        change = {'locked_hash': status['locked_hash'],
                  'current_hash': status['current_hash'], 'diff': status['diff']}
        if accept_gate_change:
            attempt.gate_change = {**change, 'reason': accept_gate_change}
            logger.warning('gate spec differs from the lock; accepted: %s',
                           accept_gate_change)
        else:
            attempt.gate_change = change
            attempt.refused = refusal_text(status)

    if attempt.refused:
        logger.error('refused %s: %s', attempt.version, attempt.refused)
    elif attempt.passed or force:
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


def installed_version(root: Optional[Path] = None) -> Optional[str]:
    """The ledger version of the installed artifact, by content.

    `promote` stages each candidate under `.staging/<version>/`, so the
    installed file is byte-identical to one staged copy. None when there is no
    installed file or no staged copy matches.
    """
    import hashlib

    root = models_root(root)
    path = root / LIVE_MODEL
    staging = root / STAGING
    if not path.exists() or not staging.is_dir():
        return None
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    for candidate in sorted(staging.iterdir(), reverse=True):
        staged = candidate / LIVE_MODEL
        if staged.exists() and hashlib.md5(staged.read_bytes()).hexdigest() == digest:
            return candidate.name
    return None


def ledger_entry(version: str, root: Optional[Path] = None) -> Optional[dict]:
    """The ledger payload of one attempt, or None."""
    try:
        return json.loads((models_root(root) / LEDGER / f'{version}.json').read_text())
    except (OSError, ValueError):
        return None


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
            'refused': payload.get('refused'),
            'failed_gates': ', '.join(payload.get('failed_gates', [])),
            'log_loss_skill': report.get('log_loss_skill'),
            'folds_positive': report.get('folds_positive'),
            'windows_evaluated': report.get('windows_evaluated'),
            'force_reason': payload.get('force_reason'),
            'trial_id': payload.get('trial_id'),
            'gate_spec_hash': payload.get('gate_spec_hash'),
        })
    return pd.DataFrame(rows)


def refusal_text(status: dict) -> str:
    diff = status.get('diff') or {}
    parts = [f'{k}: {", ".join(r["name"] for r in v)}' for k, v in
             (('added', diff.get('added')), ('removed', diff.get('removed'))) if v]
    if diff.get('changed'):
        parts.append('changed: ' + ', '.join(diff['changed']))
    return (f'the gate spec ({str(status.get("current_hash"))[:12]}) differs from '
            f'the committed lock ({str(status.get("locked_hash"))[:12]}); '
            + ('; '.join(parts) or 'no readable lock')
            + '. Pass --accept-gate-change "<reason>" to promote under it.')


def record_refusal(*, root: Optional[Path] = None, refused: str,
                   gate_change: Optional[dict] = None, trial_id: Optional[str] = None,
                   holdout: Optional[dict] = None,
                   gates: Optional[dict[str, tuple[float, str]]] = None) -> PromotionAttempt:
    """A ledger entry for a promotion refused before anything was evaluated."""
    root = models_root(root)
    ledger = root / LEDGER
    ledger.mkdir(parents=True, exist_ok=True)
    trial_id = trial_id or new_trial_id()
    attempt = PromotionAttempt(
        version=_unique_version(ledger),
        created_at=datetime.now(timezone.utc).isoformat(), gates=[],
        gate_spec_hash=gate_spec_hash(gates), refused=refused,
        gate_change=gate_change, trial_id=trial_id, holdout=holdout,
        trial_count=trial_count(root, including=trial_id))
    (ledger / f'{attempt.version}.json').write_text(
        json.dumps(attempt.payload(), indent=2, default=str))
    return attempt


# ---- the trial log -------------------------------------------------------

def new_trial_id() -> str:
    import uuid
    return uuid.uuid4().hex


def git_commit() -> Optional[str]:
    """HEAD of the checkout running this, or `GIT_COMMIT`, or None."""
    import subprocess

    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_TRADER_ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return os.getenv('GIT_COMMIT') or None


def _clean(value):
    """Strict JSON: NaN and inf become None, numpy scalars become Python ones."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)) or hasattr(value, 'item'):
        try:
            number = value.item() if hasattr(value, 'item') else value
        except (TypeError, ValueError):
            return str(value)
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return number
    return str(value)


def trial_metrics(report: Optional[EvaluationReport] = None,
                  extra: Optional[dict] = None) -> dict:
    """The numbers a trial is remembered by, whichever of them exist."""
    out: dict = {}
    if report is not None:
        values = report.gate_values()
        unweighted = report.unweighted_skill
        out.update({
            'log_loss_skill': report.mean_skill,
            'log_loss_skill_se': report.skill_standard_error,
            'log_loss_skill_p': report.skill_p_value,
            'log_loss_skill_unweighted': unweighted.mean,
            'log_loss_skill_se_unweighted': unweighted.standard_error,
            'folds': report.folds_total,
            'windows_evaluated': report.total_windows,
            'realised_edge_pp': values.get('realised_edge_pp'),
            'sharpe': values.get('sharpe'),
            'trades': values.get('trades'),
            'total_return': values.get('total_return'),
            'sharpe_moments': dict(report.sharpe_moments or {}),
        })
    for key, value in (extra or {}).items():
        out[key] = value
    return out


def record_trial(
    script: str,
    *,
    root: Optional[Path] = None,
    trial_id: Optional[str] = None,
    argv: Optional[Sequence[str]] = None,
    config: Optional[Config] = None,
    groups: Optional[Sequence[str]] = None,
    data_span: Optional[dict] = None,
    metrics: Optional[dict] = None,
    holdout: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Append one line to `trials.jsonl`. Never raises: a run whose result is
    computed must not die on its own bookkeeping, but the failure is logged."""
    import sys

    record = {
        'trial_id': trial_id or new_trial_id(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'script': script,
        'argv': list(sys.argv[1:] if argv is None else argv),
        'git_commit': git_commit(),
        'config_hash': config.provenance_hash() if config is not None else None,
        'config': config.provenance() if config is not None else None,
        'groups': list(groups) if groups else None,
        'data_span': data_span,
        'metrics': metrics or {},
        'gate_spec_hash': gate_spec_hash(),
        'holdout': holdout,
        **(extra or {}),
    }
    record = _clean(record)
    try:
        path = models_root(root) / TRIALS
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
    except OSError as exc:
        logger.error('could not append to the trial log (%s); this run is NOT '
                     'counted as a trial', exc)
    return record


def read_trials(root: Optional[Path] = None) -> list[dict]:
    path = models_root(root) / TRIALS
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning('unreadable trial log line in %s', path)
    return rows


def distinct_trials(root: Optional[Path] = None) -> dict[str, dict]:
    """trial_id -> {'sharpe': ...}, from the trial log and the ledger together.

    A promotion writes both, under one trial id, and counts once. A ledger entry
    from before the trial log carries no id and counts as its own trial.
    """
    out: dict[str, dict] = {}
    skipped: set[str] = set()
    for row in read_trials(root):
        key = row.get('trial_id') or f"log:{row.get('timestamp')}"
        # A run refused before it evaluated anything tested no hypothesis.
        if row.get('evaluated') is False:
            skipped.add(key)
            continue
        sharpe = (row.get('metrics') or {}).get('sharpe')
        out.setdefault(key, {'sharpe': sharpe, 'source': 'trials'})
        if out[key].get('sharpe') is None and sharpe is not None:
            out[key]['sharpe'] = sharpe
    ledger = models_root(root) / LEDGER
    for path in sorted(ledger.glob('*.json')) if ledger.exists() else ():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        key = payload.get('trial_id') or f'ledger:{payload.get("version") or path.stem}'
        if key in skipped or (payload.get('refused') and not payload.get('gates')):
            continue
        sharpe = ((payload.get('report') or {}).get('gate_values') or {}).get('sharpe')
        if key not in out:
            out[key] = {'sharpe': sharpe, 'source': 'ledger'}
        elif out[key].get('sharpe') is None and sharpe is not None:
            out[key]['sharpe'] = sharpe
    return out


def trial_count(root: Optional[Path] = None, *, including: Optional[str] = None) -> int:
    """Distinct trials: the trial log and the ledger, deduplicated by trial id.
    The multiple-testing denominator. `including` adds a run not yet written."""
    trials = distinct_trials(root)
    return len(trials) + (1 if including and including not in trials else 0)


def deflated_sharpe_for(report: EvaluationReport, *, root: Optional[Path] = None,
                        trial_id: Optional[str] = None) -> dict:
    """The candidate's Sharpe deflated by every other trial's. Never raises."""
    try:
        sharpe = (report.gate_values() or {}).get('sharpe')
        moments = dict(getattr(report, 'sharpe_moments', None) or {})
        others = {k: v for k, v in distinct_trials(root).items() if k != trial_id}
        result = deflated_sharpe_ratio(
            sharpe, n_obs=moments.get('n_obs', 0), skew=moments.get('skew'),
            kurtosis=moments.get('kurtosis'),
            trial_sharpes=[v.get('sharpe') for v in others.values()] + [sharpe],
            n_trials=len(others) + 1)
        return _clean(result)
    except Exception as exc:                          # noqa: BLE001 - a diagnostic
        logger.warning('deflated Sharpe could not be computed (%s)', exc)
        return {}


# ---- the sealed holdout --------------------------------------------------

class HoldoutError(RuntimeError):
    """A sealed holdout was asked for in a way that is not allowed."""


def holdout_unseals(root: Optional[Path] = None) -> list[dict]:
    path = models_root(root) / HOLDOUT_MARKER
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def unseal_holdout(holdout_start: str, reason: str, *, force: bool = False,
                   script: str = '', argv: Optional[Sequence[str]] = None,
                   root: Optional[Path] = None) -> dict:
    """Record that `holdout_start` is being read. Once per date unless forced.

    A holdout that has been looked at is a validation set: the second look is
    selected on the first. The refusal is the default; `force` is recorded.
    """
    import sys

    if not reason or not str(reason).strip():
        raise HoldoutError('unsealing the holdout needs a written reason')
    earlier = [r for r in holdout_unseals(root)
               if str(r.get('holdout_start')) == str(holdout_start)]
    if earlier and not force:
        first = earlier[0]
        raise HoldoutError(
            f'the holdout from {holdout_start} was already unsealed at '
            f'{first.get("unsealed_at")} ({first.get("reason")!r}). A second look '
            f'is not out of sample; --force-unseal records that you took one anyway.')
    record = {'holdout_start': str(holdout_start),
              'unsealed_at': datetime.now(timezone.utc).isoformat(),
              'reason': str(reason), 'script': script,
              'argv': list(sys.argv[1:] if argv is None else argv),
              'forced': bool(earlier), 'previous_unseals': len(earlier)}
    path = models_root(root) / HOLDOUT_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    return record
