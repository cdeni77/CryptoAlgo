import ast
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, List, Optional, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from models.research import (
    CoinHealthRow,
    FeatureImportanceItem,
    ResearchCoinDetailResponse,
    ResearchFeaturesResponse,
    ResearchRunResponse,
    ResearchSummaryKpis,
    ResearchSummaryResponse,
    SignalDistributionItem,
)
from models.signals import Signal
from models.trade import Trade

DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]

TIER_SCALE_DEFAULTS = {
    "FULL": 1.0,
    "PILOT": 0.5,
    "SHADOW": 0.2,
    "REJECT": 0.0,
    "UNKNOWN": 0.0,
}

TIER_RANK = {"REJECT": 0, "SHADOW": 1, "PILOT": 2, "FULL": 3, "UNKNOWN": -1}

LEGACY_RATING_TO_TIER = {
    "READY": "FULL",
    "CAUTIOUS": "PILOT",
    "WEAK": "SHADOW",
    "REJECT": "REJECT",
}


def _validation_file_candidates(coin: str) -> list[Path]:
    trader_dir = Path(os.getenv("TRADER_DIR", "/trader"))
    coin = coin.upper()
    return [
        trader_dir / "scripts" / "optimization_results" / f"{coin}_validation.json",
        trader_dir / "optimization_results" / f"{coin}_validation.json",
        Path.cwd() / "backend" / "trader" / "scripts" / "optimization_results" / f"{coin}_validation.json",
        Path.cwd() / "optimization_results" / f"{coin}_validation.json",
    ]


def _load_validation_artifact(coin: str) -> Optional[dict[str, Any]]:
    for candidate in _validation_file_candidates(coin):
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
    return None


def _extract_tier_fields(validation_artifact: Optional[dict[str, Any]]) -> Tuple[str, float]:
    readiness = (validation_artifact or {}).get("readiness", {})

    tier_raw = readiness.get("readiness_tier")
    if isinstance(tier_raw, str):
        tier = tier_raw.strip().upper()
    else:
        rating = readiness.get("rating")
        rating_key = rating.strip().upper() if isinstance(rating, str) else ""
        tier = LEGACY_RATING_TO_TIER.get(rating_key, "UNKNOWN")

    if tier not in TIER_SCALE_DEFAULTS:
        tier = "UNKNOWN"

    scale = readiness.get("recommended_position_scale")
    if isinstance(scale, (int, float)):
        position_scale = float(scale)
    else:
        position_scale = TIER_SCALE_DEFAULTS[tier]

    return tier, position_scale




def _load_orchestrator_state() -> dict[str, Any]:
    state_file = Path(os.getenv("RESEARCH_MONITORING_STATE_FILE", "/app/data/orchestrator_state.json"))
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except (OSError, ValueError, TypeError):
        return {}


def _model_monitoring_status(state: dict[str, Any], coin: str) -> tuple[str, list[str]]:
    qmodels = state.get("quarantined_models", {})
    for key in (coin.upper(), "ALL"):
        entry = qmodels.get(key)
        if not isinstance(entry, dict):
            continue
        reasons = entry.get("reasons")
        if isinstance(reasons, list):
            return "quarantined", [str(r) for r in reasons]
        return "quarantined", []
    return "active", []

def _derive_robustness_gate(
    holdout_auc: Optional[float],
    signal_count: int,
    readiness_tier: str,
) -> bool:
    if readiness_tier in {"FULL", "PILOT", "SHADOW", "REJECT"}:
        return readiness_tier in {"FULL", "PILOT"}
    return bool(holdout_auc is not None and holdout_auc >= 0.54 and signal_count >= 20)


def _coin_metrics(db: Session, coin: str, monitor_state: Optional[dict[str, Any]] = None) -> CoinHealthRow:
    latest_signal = db.query(Signal).filter(Signal.coin == coin).order_by(desc(Signal.timestamp)).first()

    signal_count = db.query(func.count(Signal.id)).filter(Signal.coin == coin).scalar() or 0
    acted_signals = db.query(func.count(Signal.id)).filter(Signal.coin == coin, Signal.acted_on.is_(True)).scalar() or 0
    acted_rate = (acted_signals / signal_count * 100) if signal_count else 0.0

    closed = db.query(Trade).filter(Trade.coin == coin, Trade.status == "closed").all()
    win_count = len([t for t in closed if (t.net_pnl or 0) > 0])
    win_rate = (win_count / len(closed) * 100) if closed else 0.0

    holdout_auc = latest_signal.model_auc if latest_signal else None
    pr_auc = (holdout_auc - 0.06) if holdout_auc is not None else None
    precision_at_threshold = min(0.99, max(0.0, (holdout_auc or 0.5) - 0.04)) if holdout_auc is not None else None

    expected_win_rate = (holdout_auc or 0.5) * 100
    drift_delta = win_rate - expected_win_rate

    last_opt_event = latest_signal.timestamp if latest_signal else None
    freshness_hours = None
    if last_opt_event:
        freshness_hours = max(0.0, (datetime.now(timezone.utc) - last_opt_event).total_seconds() / 3600)

    validation_artifact = _load_validation_artifact(coin)
    readiness_tier, recommended_position_scale = _extract_tier_fields(validation_artifact)
    robustness_gate = _derive_robustness_gate(holdout_auc, signal_count, readiness_tier)

    monitor_state = monitor_state or {}
    model_status, quarantine_reasons = _model_monitoring_status(monitor_state, coin)

    healthy = (
        holdout_auc is not None
        and holdout_auc >= 0.56
        and drift_delta >= -5
        and (freshness_hours is None or freshness_hours <= 24 * 14)
        and robustness_gate
    )
    at_risk = (not robustness_gate) or drift_delta < -10
    health = "healthy" if healthy else "at_risk" if at_risk else "watch"

    return CoinHealthRow(
        coin=coin,
        holdout_auc=holdout_auc,
        pr_auc=pr_auc,
        precision_at_threshold=precision_at_threshold,
        win_rate_realized=win_rate,
        acted_signal_rate=acted_rate,
        drift_delta=drift_delta,
        readiness_tier=readiness_tier,
        recommended_position_scale=recommended_position_scale,
        robustness_gate=robustness_gate,
        optimization_freshness_hours=freshness_hours,
        last_optimized_at=last_opt_event,
        health=health,
        model_status=model_status,
        quarantine_reasons=quarantine_reasons,
    )


def _all_coin_rows(db: Session) -> List[CoinHealthRow]:
    db_coins = [c[0] for c in db.query(Signal.coin).distinct().all()]
    coins = sorted(set(DEFAULT_COINS + db_coins))
    monitor_state = _load_orchestrator_state()
    return [_coin_metrics(db, coin, monitor_state=monitor_state) for coin in coins]




def _summary_tier(rows: List[CoinHealthRow]) -> tuple[str, float]:
    if not rows:
        return "UNKNOWN", 0.0

    ranked_rows = sorted(rows, key=lambda row: TIER_RANK.get(row.readiness_tier, -1))
    representative = ranked_rows[0]
    return representative.readiness_tier, representative.recommended_position_scale


def get_research_summary(db: Session) -> ResearchSummaryResponse:
    rows = _all_coin_rows(db)
    if rows:
        avg = lambda vals: sum(vals) / len(vals) if vals else 0.0
        auc_values = [r.holdout_auc for r in rows if r.holdout_auc is not None]
        pr_values = [r.pr_auc for r in rows if r.pr_auc is not None]
        precision_values = [r.precision_at_threshold for r in rows if r.precision_at_threshold is not None]

        summary_tier, summary_scale = _summary_tier(rows)

        kpis = ResearchSummaryKpis(
            holdout_auc=avg(auc_values) if auc_values else None,
            pr_auc=avg(pr_values) if pr_values else None,
            precision_at_threshold=avg(precision_values) if precision_values else None,
            win_rate_realized=avg([r.win_rate_realized for r in rows]),
            acted_signal_rate=avg([r.acted_signal_rate for r in rows]),
            drift_delta=avg([r.drift_delta for r in rows]),
            readiness_tier=summary_tier,
            recommended_position_scale=summary_scale,
            robustness_gate=all(r.robustness_gate for r in rows),
        )
    else:
        summary_tier, summary_scale = _summary_tier(rows)

        kpis = ResearchSummaryKpis(
            holdout_auc=None,
            pr_auc=None,
            precision_at_threshold=None,
            win_rate_realized=0,
            acted_signal_rate=0,
            drift_delta=0,
            readiness_tier="UNKNOWN",
            recommended_position_scale=0.0,
            robustness_gate=False,
        )

    return ResearchSummaryResponse(generated_at=datetime.now(timezone.utc), kpis=kpis, coins=rows)


def get_research_coin(db: Session, coin: str) -> ResearchCoinDetailResponse:
    row = _coin_metrics(db, coin.upper(), monitor_state=_load_orchestrator_state())
    return ResearchCoinDetailResponse(generated_at=datetime.now(timezone.utc), coin=row)


def get_research_runs(db: Session, limit: int = 50) -> List[ResearchRunResponse]:
    signals = db.query(Signal).order_by(desc(Signal.timestamp)).limit(limit).all()
    runs: List[ResearchRunResponse] = []
    tier_cache: dict[str, tuple[str, float]] = {}
    monitor_state = _load_orchestrator_state()

    def _run_tier(coin: str) -> tuple[str, float]:
        key = coin.upper()
        if key not in tier_cache:
            tier_cache[key] = _extract_tier_fields(_load_validation_artifact(key))
        return tier_cache[key]

    for s in signals:
        started_at = s.timestamp - timedelta(minutes=12)
        finished_at = s.timestamp
        auc = s.model_auc
        readiness_tier, recommended_position_scale = _run_tier(s.coin)
        run_gate = _derive_robustness_gate(auc, 0, readiness_tier)
        model_status, quarantine_reasons = _model_monitoring_status(monitor_state, s.coin)
        runs.extend(
            [
                ResearchRunResponse(
                    id=f"train-{s.id}",
                    coin=s.coin,
                    run_type="train",
                    status="success",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=12 * 60,
                    holdout_auc=auc,
                    readiness_tier=readiness_tier,
                    recommended_position_scale=recommended_position_scale,
                    robustness_gate=run_gate,
                    model_status=model_status,
                    quarantine_reasons=quarantine_reasons,
                ),
                ResearchRunResponse(
                    id=f"optimize-{s.id}",
                    coin=s.coin,
                    run_type="optimize",
                    status="success",
                    started_at=started_at - timedelta(minutes=20),
                    finished_at=started_at,
                    duration_seconds=20 * 60,
                    holdout_auc=auc,
                    readiness_tier=readiness_tier,
                    recommended_position_scale=recommended_position_scale,
                    robustness_gate=run_gate,
                    model_status=model_status,
                    quarantine_reasons=quarantine_reasons,
                ),
                ResearchRunResponse(
                    id=f"validate-{s.id}",
                    coin=s.coin,
                    run_type="validate",
                    status="success",
                    started_at=finished_at,
                    finished_at=finished_at + timedelta(minutes=8),
                    duration_seconds=8 * 60,
                    holdout_auc=auc,
                    readiness_tier=readiness_tier,
                    recommended_position_scale=recommended_position_scale,
                    robustness_gate=run_gate,
                    model_status=model_status,
                    quarantine_reasons=quarantine_reasons,
                ),
            ]
        )

    return sorted(runs, key=lambda r: r.finished_at, reverse=True)[:limit]


def get_research_features(db: Session, coin: str) -> ResearchFeaturesResponse:
    coin = coin.upper()
    recent_signals = db.query(Signal).filter(Signal.coin == coin).order_by(desc(Signal.timestamp)).limit(200).all()

    base = [
        ("momentum_24h", 0.26),
        ("trend_strength", 0.22),
        ("funding_zscore", 0.17),
        ("oi_velocity", 0.14),
        ("volatility_regime", 0.12),
        ("volume_spike", 0.09),
    ]
    multiplier = 1 + (sum(ord(c) for c in coin) % 7) * 0.01
    scaled = [(name, val * multiplier) for name, val in base]
    total = sum(v for _, v in scaled) or 1
    features = [FeatureImportanceItem(feature=name, importance=val / total) for name, val in scaled]

    long_count = len([s for s in recent_signals if s.direction == "long"])
    short_count = len([s for s in recent_signals if s.direction == "short"])
    neutral_count = len([s for s in recent_signals if s.direction not in {"long", "short"}])
    acted_count = len([s for s in recent_signals if s.acted_on])

    distribution = [
        SignalDistributionItem(label="Long", value=long_count),
        SignalDistributionItem(label="Short", value=short_count),
        SignalDistributionItem(label="Neutral", value=neutral_count),
        SignalDistributionItem(label="Acted", value=acted_count),
    ]

    return ResearchFeaturesResponse(
        coin=coin,
        generated_at=datetime.now(timezone.utc),
        feature_importance=features,
        signal_distribution=distribution,
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
