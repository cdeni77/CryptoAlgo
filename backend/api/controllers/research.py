"""Research metrics and the script runner.

The metrics half was rewritten because it described a classifier that no longer
exists. Its inputs were `signals.model_auc` — which the new signal writer leaves
null, because AUC is undefined for a regression on net return — and
`optimization_results/*_validation.json`, an artifact of a deleted pipeline. Every
tier read "UNKNOWN", every AUC-derived figure was a constant subtracted from a
null, and `drift_delta` subtracted an AUC from a win-rate percentage.

What it reports now is the comparison the model can actually be held to: the edge
`decide()` claimed in basis points before each trade, against what the trade
earned. That claim is checkable, and the gap between claim and outcome is the
number worth watching — a model that overstates its edge over-sizes every
position that clears the conviction floor.

Nothing here substitutes a value for a missing measurement. A null means "not
measured", and the dashboard says so.
"""

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections import Counter
from typing import Any, List, Optional, Sequence

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from models.research import (
    CoinHealthRow,
    EdgeCalibration,
    FeatureImportanceItem,
    ResearchCoinDetailResponse,
    ResearchFeaturesResponse,
    ResearchRunResponse,
    ResearchSummaryKpis,
    ResearchSummaryResponse,
    SignalDistributionItem,
)
from models.signals import Signal
from models.trade import ModelRun, PaperPosition, Trade

DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

# How many recent signals a per-coin view averages over. Long enough for the mean
# forecast to be stable, short enough that a retrain three weeks ago does not
# dominate the current model's numbers.
SIGNAL_WINDOW = 500

# Calibration needs a sample before it means anything. Below this, the delta is
# one or two trades' noise and reporting it invites acting on it.
MIN_CALIBRATION_SAMPLE = 10

# A realised net edge this far below the forecast is a mispriced model, not
# variance: it over-sizes every position that clears the conviction floor.
CALIBRATION_AT_RISK_BPS = -20.0
CALIBRATION_WATCH_BPS = -8.0


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _basis_points(pnl: Optional[float], notional: Optional[float]) -> Optional[float]:
    if pnl is None or not notional:
        return None
    return float(pnl) / float(notional) * 10_000


# ---------------------------------------------------------------------------
# Per-instrument health
# ---------------------------------------------------------------------------


def _closed_paper_positions(db: Session, coin: Optional[str] = None) -> list[PaperPosition]:
    """Closed paper positions, which carry the notional a return needs.

    `trades` holds live/manual trades and has no notional column, so it cannot
    express a return in basis points. Paper positions can, and paper is what the
    model is actually being evaluated on.
    """
    query = db.query(PaperPosition).filter(PaperPosition.is_open.is_(False))
    if coin:
        query = query.filter(PaperPosition.coin == coin)
    return query.all()


def _calibration(signals: Sequence[Signal], closed: Sequence[PaperPosition]) -> EdgeCalibration:
    """Forecast against outcome, both in basis points of notional.

    Matched at the aggregate rather than trade by trade: a signal does not carry
    the id of the position it produced, so pairing them individually would mean
    guessing. Means over a common window answer the question that matters — is
    the model's stated edge systematically too large — without that guess.
    """
    acted = [s for s in signals if s.passed_gates and s.expected_net_bps is not None]
    expected = _mean([s.expected_net_bps for s in acted])

    realised_values = [
        _basis_points(p.realized_pnl, p.notional) for p in closed if p.notional
    ]
    realised = _mean(realised_values)
    sample = min(len(acted), len([v for v in realised_values if v is not None]))

    delta = None
    if expected is not None and realised is not None and sample >= MIN_CALIBRATION_SAMPLE:
        delta = realised - expected

    return EdgeCalibration(
        expected_net_bps=expected,
        realised_net_bps=realised,
        delta_bps=delta,
        sample=sample,
    )


def _health(calibration: EdgeCalibration, trades_closed: int, signals_total: int) -> tuple[str, Optional[str]]:
    """A grade derived from measurements, with the reason attached.

    "unknown" is a real answer and the correct one until there is enough to
    judge. The previous implementation always produced a grade, which meant a
    brand-new install displayed "at_risk" for every instrument on no evidence.
    """
    if signals_total == 0:
        return "unknown", "no signals yet"
    if calibration.sample < MIN_CALIBRATION_SAMPLE:
        return "unknown", (
            f"{calibration.sample} matched observations, need "
            f"{MIN_CALIBRATION_SAMPLE} before calibration means anything"
        )

    delta = calibration.delta_bps
    if delta is None:
        return "unknown", "no calibration measurement"
    if delta <= CALIBRATION_AT_RISK_BPS:
        return "at_risk", (
            f"realised net runs {abs(delta):.0f}bp below forecast — the model is "
            f"overstating its edge, which over-sizes every position it clears"
        )
    if delta <= CALIBRATION_WATCH_BPS:
        return "watch", f"realised net {abs(delta):.0f}bp below forecast"
    if trades_closed == 0:
        return "watch", "signals but no closed trades yet"
    return "healthy", f"realised net within {abs(delta):.0f}bp of forecast"


def _coin_row(db: Session, coin: str) -> CoinHealthRow:
    signals = (
        db.query(Signal)
        .filter(Signal.coin == coin)
        .order_by(desc(Signal.timestamp))
        .limit(SIGNAL_WINDOW)
        .all()
    )
    passed = [s for s in signals if s.passed_gates]
    blocked = [s.gate_failure_reason for s in signals if not s.passed_gates and s.gate_failure_reason]

    closed = _closed_paper_positions(db, coin)
    wins = len([p for p in closed if (p.realized_pnl or 0.0) > 0])
    net_pnl = sum(float(p.realized_pnl or 0.0) for p in closed) if closed else None

    calibration = _calibration(signals, closed)
    health, reason = _health(calibration, len(closed), len(signals))

    return CoinHealthRow(
        coin=coin,
        signals_total=len(signals),
        signals_passed_gates=len(passed),
        gate_pass_rate=(len(passed) / len(signals)) if signals else None,
        top_gate_reason=Counter(blocked).most_common(1)[0][0] if blocked else None,
        last_signal_at=signals[0].timestamp if signals else None,
        expected_net_bps=_mean([s.expected_net_bps for s in passed]),
        expected_carry_share=_mean([s.carry_share for s in passed]),
        mean_cost_bps=_mean([s.cost_bps for s in signals]),
        trades_closed=len(closed),
        win_rate_realized=(wins / len(closed)) if closed else None,
        net_pnl=net_pnl,
        realised_net_bps=calibration.realised_net_bps,
        calibration=calibration,
        health=health,
        health_reason=reason,
    )


def _coins(db: Session) -> list[str]:
    seen = [c[0] for c in db.query(Signal.coin).distinct().all() if c[0]]
    return sorted(set(DEFAULT_COINS) | set(seen))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def get_research_summary(db: Session) -> ResearchSummaryResponse:
    """Universe totals plus the live model's identity and gate verdict."""
    rows = [_coin_row(db, coin) for coin in _coins(db)]

    signals_total = sum(r.signals_total for r in rows)
    signals_passed = sum(r.signals_passed_gates for r in rows)
    trades_closed = sum(r.trades_closed for r in rows)

    # Ratios are pooled, not averaged. Averaging per-coin win rates weights a
    # coin with two trades the same as one with two hundred.
    all_closed = _closed_paper_positions(db)
    wins = len([p for p in all_closed if (p.realized_pnl or 0.0) > 0])
    all_signals = (
        db.query(Signal).order_by(desc(Signal.timestamp)).limit(SIGNAL_WINDOW * 4).all()
    )
    calibration = _calibration(all_signals, all_closed)

    from controllers.model import get_live_model

    live = get_live_model()
    record = live.live
    provenance = record.provenance if record else None

    age_hours = None
    if provenance and provenance.trained_at:
        try:
            trained = datetime.fromisoformat(str(provenance.trained_at))
            if trained.tzinfo is None:
                trained = trained.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - trained).total_seconds() / 3600
        except ValueError:
            age_hours = None

    health, _ = _health(calibration, trades_closed, signals_total)
    if live.kill_switch.status == "quarantined":
        health = "at_risk"
    elif not live.has_model:
        health = "unknown"

    kpis = ResearchSummaryKpis(
        signals_total=signals_total,
        signals_passed_gates=signals_passed,
        gate_pass_rate=(signals_passed / signals_total) if signals_total else None,
        trades_closed=trades_closed,
        win_rate_realized=(wins / len(all_closed)) if all_closed else None,
        net_pnl=sum(float(p.realized_pnl or 0.0) for p in all_closed) if all_closed else None,
        expected_net_bps=calibration.expected_net_bps,
        realised_net_bps=calibration.realised_net_bps,
        calibration_delta_bps=calibration.delta_bps,
        expected_carry_share=_mean(
            [s.carry_share for s in all_signals if s.passed_gates]
        ),
        model_version=record.version if record else None,
        model_promoted=bool(record and record.promoted),
        model_forced=bool(record and record.forced),
        model_age_hours=age_hours,
        gates_failed=record.failed_gates if record else [],
        kill_switch_status=live.kill_switch.status,
        trials_to_date=live.trials_to_date,
        health=health,
    )

    return ResearchSummaryResponse(
        generated_at=datetime.now(timezone.utc), kpis=kpis, coins=rows
    )


def get_research_coin(db: Session, coin: str) -> ResearchCoinDetailResponse:
    return ResearchCoinDetailResponse(
        generated_at=datetime.now(timezone.utc), coin=_coin_row(db, coin.upper())
    )


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


def get_research_runs(db: Session, limit: int = 50) -> List[ResearchRunResponse]:
    """Real retrain attempts, from `model_runs` joined to the promotion ledger.

    The previous version invented three runs per signal — "train", "optimize" and
    "validate" — with start times derived by subtracting twelve minutes from the
    signal timestamp, hardcoded durations, and a status of "success" for all of
    them. Nothing in that timeline had happened.
    """
    from controllers.model import get_promotion_history

    ledger = {r.version: r for r in get_promotion_history(limit=200).records}

    runs = (
        db.query(ModelRun).order_by(desc(ModelRun.run_started_at)).limit(limit).all()
    )
    out: List[ResearchRunResponse] = []
    for run in runs:
        duration = None
        if run.run_finished_at and run.run_started_at:
            duration = int((run.run_finished_at - run.run_started_at).total_seconds())

        record = ledger.get(run.artifacts_version) if run.artifacts_version else None
        out.append(
            ResearchRunResponse(
                id=f"run-{run.id}",
                run_type="retrain",
                status=run.status,
                started_at=run.run_started_at,
                finished_at=run.run_finished_at,
                duration_seconds=duration,
                artifacts_version=run.artifacts_version,
                symbols_trained=run.symbols_trained or 0,
                symbols_total=run.symbols_total or 0,
                retrain_window_days=run.retrain_window_days,
                promoted=record.promoted if record else None,
                forced=bool(record and record.forced),
                failed_gates=record.failed_gates if record else [],
                sharpe=record.backtest.sharpe if record else None,
                trades=record.backtest.trades if record else None,
                error=run.error,
            )
        )

    # Ledger entries with no `model_runs` row: a candidate evaluated by hand
    # rather than by the orchestrator. Worth showing — it still counts as a trial.
    seen = {r.artifacts_version for r in runs if r.artifacts_version}
    for version, record in list(ledger.items())[: max(0, limit - len(out))]:
        if version in seen or not record.created_at:
            continue
        try:
            started = datetime.fromisoformat(str(record.created_at))
        except ValueError:
            continue
        out.append(
            ResearchRunResponse(
                id=f"ledger-{version}",
                run_type="evaluation",
                status="success" if record.promoted else "blocked",
                started_at=started,
                finished_at=started,
                artifacts_version=version,
                promoted=record.promoted,
                forced=record.forced,
                failed_gates=record.failed_gates,
                sharpe=record.backtest.sharpe,
                trades=record.backtest.trades,
                error=record.error,
            )
        )

    # Sorted on a timezone-normalised key. The two sources disagree: `model_runs`
    # timestamps come back naive from SQLite, while the ledger's are parsed from
    # ISO strings that carry an offset — and Python refuses to compare the two,
    # which took the whole endpoint down with a 500.
    return sorted(out, key=lambda r: _as_utc(r.started_at), reverse=True)[:limit]


def _as_utc(value: datetime) -> datetime:
    """Assume UTC for a naive timestamp, so mixed sources can be ordered."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def get_research_features(db: Session, coin: str) -> ResearchFeaturesResponse:
    """Signal distribution for one instrument, plus real feature importances.

    Importances come from the promoted model's booster and are per-*model*, not
    per-coin: the model is a pooled panel with one feature set across the
    universe, so the same ranking is correct for every instrument. The per-coin
    part of this response is the signal distribution.

    They used to be read from `pruned_features_<coin>.json`, an artifact of a
    deleted pipeline, and the fallback when it was missing — which it always was
    — was a hardcoded table of six plausible names with plausible weights. That
    is the worst failure available to an explainability panel: it renders exactly
    like the real thing, so nobody notices it is fiction.
    """
    coin = coin.upper()
    recent = (
        db.query(Signal)
        .filter(Signal.coin == coin)
        .order_by(desc(Signal.timestamp))
        .limit(200)
        .all()
    )

    from controllers.model import get_feature_importance

    importance = get_feature_importance()

    long_count = len([s for s in recent if s.direction == "long"])
    short_count = len([s for s in recent if s.direction == "short"])
    passed = len([s for s in recent if s.passed_gates])

    return ResearchFeaturesResponse(
        coin=coin,
        generated_at=datetime.now(timezone.utc),
        feature_importance=[
            FeatureImportanceItem(feature=e.feature, importance=e.importance)
            for e in importance.features[:10]
        ],
        importance_unavailable_reason=importance.unavailable_reason,
        signal_distribution=[
            SignalDistributionItem(label="Long", value=long_count),
            SignalDistributionItem(label="Short", value=short_count),
            SignalDistributionItem(label="Passed gates", value=passed),
            SignalDistributionItem(label="Blocked", value=len(recent) - passed),
        ],
    )


SCRIPT_PACKAGE = "scripts"
RUNNER_LOG_DIR = "logs/script_runner"
_JOB_REGISTRY: dict[int, dict[str, Any]] = {}


def _discover_script_modules(trader_dir: Path) -> dict[str, str]:
    scripts_dir = trader_dir / SCRIPT_PACKAGE
    if not scripts_dir.exists():
        return {}

    modules: dict[str, str] = {}
    for file in scripts_dir.glob("*.py"):
        if file.name.startswith("_") or file.name == "__init__.py":
            continue
        script_name = file.stem
        modules[script_name] = f"{SCRIPT_PACKAGE}.{script_name}"
    return modules


def _safe_literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _script_default_args(script_path: Path) -> List[str]:
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return []

    defaults: List[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        option_strings: List[str] = []
        for arg in node.args:
            value = _safe_literal(arg)
            if isinstance(value, str) and value.startswith("--"):
                option_strings.append(value)
        if not option_strings:
            continue

        default_value = None
        action_value = None
        for kw in node.keywords:
            if kw.arg == "default":
                default_value = _safe_literal(kw.value)
            elif kw.arg == "action":
                action_value = _safe_literal(kw.value)

        option = option_strings[0]
        if action_value in {"store_true", "store_false"}:
            if isinstance(default_value, bool) and default_value is True and action_value == "store_false":
                defaults.append(option)
            if isinstance(default_value, bool) and default_value is False and action_value == "store_true":
                continue
            continue

        if default_value in (None, False, ""):
            continue

        if isinstance(default_value, (list, tuple)):
            for item in default_value:
                defaults.extend([option, str(item)])
            continue

        if isinstance(default_value, bool):
            if default_value:
                defaults.append(option)
            continue

        defaults.extend([option, str(default_value)])

    return defaults




def _script_launch_metadata(script_path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return {}

    metadata: dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        option_strings: List[str] = []
        for arg in node.args:
            value = _safe_literal(arg)
            if isinstance(value, str) and value.startswith("--"):
                option_strings.append(value)
        if "--preset" not in option_strings:
            continue

        choices = None
        default_value = None
        for kw in node.keywords:
            if kw.arg == "choices":
                choices = _safe_literal(kw.value)
            elif kw.arg == "default":
                default_value = _safe_literal(kw.value)

        if isinstance(choices, (list, tuple)):
            metadata["preset_choices"] = [str(choice) for choice in choices]
        if isinstance(default_value, str):
            metadata["preset_default"] = default_value
        break

    return metadata

def list_research_scripts() -> List[dict[str, Any]]:
    trader_dir = Path(os.getenv("TRADER_DIR", "/trader"))
    if not trader_dir.exists():
        raise FileNotFoundError(f"TRADER_DIR does not exist: {trader_dir}")

    script_modules = _discover_script_modules(trader_dir)
    scripts = []
    for script_name in sorted(script_modules.keys()):
        script_path = trader_dir / SCRIPT_PACKAGE / f"{script_name}.py"
        scripts.append(
            {
                "name": script_name,
                "module": script_modules[script_name],
                "default_args": _script_default_args(script_path),
                "launch_metadata": _script_launch_metadata(script_path),
            }
        )
    return scripts


def list_research_jobs(limit: int = 25):
    from models.research import ResearchJobLaunchResponse

    jobs: List[ResearchJobLaunchResponse] = []
    ordered_jobs = sorted(_JOB_REGISTRY.values(), key=lambda job: job["launched_at"], reverse=True)
    for job in ordered_jobs[:max(1, limit)]:
        jobs.append(
            ResearchJobLaunchResponse(
                job=job["job"],
                module=job["module"],
                pid=job["pid"],
                command=job["command"],
                cwd=job["cwd"],
                log_path=job["log_path"],
                launched_at=job["launched_at"],
            )
        )
    return jobs


def launch_research_job(job: str, args: List[str] | None = None):
    trader_dir = Path(os.getenv("TRADER_DIR", "/trader"))
    if not trader_dir.exists():
        raise FileNotFoundError(f"TRADER_DIR does not exist: {trader_dir}")

    job_key = job.strip().lower()
    script_modules = _discover_script_modules(trader_dir)
    module = script_modules.get(job_key)
    if module is None:
        allowed = ", ".join(sorted(script_modules.keys()))
        raise ValueError(f"Unknown research job '{job}'. Allowed jobs: {allowed}")

    safe_args = [a for a in (args or []) if a and a.strip()]
    # Use unbuffered Python so script prints stream into the log file immediately
    # (especially important for long-running jobs launched from the frontend).
    command = [sys.executable, "-u", "-m", module, *safe_args]

    logs_dir = trader_dir / RUNNER_LOG_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    launched_at = datetime.now(timezone.utc)
    log_file = logs_dir / f"{job_key}_{launched_at.strftime('%Y%m%d_%H%M%S')}.log"

    # Line buffering helps ensure launcher preamble entries are written promptly.
    log_handle = log_file.open("a", encoding="utf-8", buffering=1)
    log_handle.write(f"# Launched at {launched_at.isoformat()}\n")
    log_handle.write(f"# Command: {shlex.join(command)}\n\n")
    log_handle.flush()

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        command,
        cwd=trader_dir,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    log_handle.close()

    _JOB_REGISTRY[process.pid] = {
        "job": job_key,
        "module": module,
        "pid": process.pid,
        "command": command,
        "cwd": str(trader_dir),
        "launched_at": launched_at,
        "log_path": str(log_file),
    }

    from models.research import ResearchJobLaunchResponse

    return ResearchJobLaunchResponse(
        job=job_key,
        module=module,
        pid=process.pid,
        command=command,
        cwd=str(trader_dir),
        log_path=str(log_file),
        launched_at=launched_at,
    )


def get_research_job_logs(pid: int, lines: int = 200):
    if pid not in _JOB_REGISTRY:
        raise ValueError(f"No launched job found for pid {pid}")

    job = _JOB_REGISTRY[pid]
    log_path = Path(job["log_path"])
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found for pid {pid}: {log_path}")

    raw_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail_lines = raw_lines[-max(1, lines):]

    running = True
    try:
        os.kill(pid, 0)
    except OSError:
        running = False

    from models.research import ResearchJobLogResponse

    return ResearchJobLogResponse(
        pid=pid,
        running=running,
        command=job["command"],
        launched_at=job["launched_at"],
        log_path=str(log_path),
        logs=tail_lines,
    )
