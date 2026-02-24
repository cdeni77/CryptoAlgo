#!/usr/bin/env python3
"""Live orchestrator for data pipeline, signal generation, and scheduled retraining."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from core.coin_profiles import MODELS_DIR
from core.pg_writer import PgWriter
from scripts.train_model import Config, load_data, retrain_models

LOGGER = logging.getLogger("live_orchestrator")
STOP_REQUESTED = False
STATE_FILE = Path(os.getenv("ORCHESTRATOR_STATE_FILE", "./data/orchestrator_state.json"))


@dataclass
class KillSwitchThresholds:
    min_win_rate: float
    min_profit_factor: float
    max_drawdown: float
    min_trades_per_week: float
    max_negative_expectancy_streak: int
    min_samples: int


def _monitoring_thresholds() -> KillSwitchThresholds:
    return KillSwitchThresholds(
        min_win_rate=float(os.getenv("PAPER_MONITOR_MIN_WIN_RATE", "0.42")),
        min_profit_factor=float(os.getenv("PAPER_MONITOR_MIN_PROFIT_FACTOR", "0.9")),
        max_drawdown=float(os.getenv("PAPER_MONITOR_MAX_DRAWDOWN", "0.12")),
        min_trades_per_week=float(os.getenv("PAPER_MONITOR_MIN_TRADES_PER_WEEK", "2.0")),
        max_negative_expectancy_streak=int(os.getenv("PAPER_MONITOR_NEG_EXPECTANCY_STREAK", "2")),
        min_samples=int(os.getenv("PAPER_MONITOR_MIN_SAMPLES", "8")),
    )


def _handle_signal(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info("Received signal %s, shutting down after current step.", signum)


def _setup_logging(log_level: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def _run_step(name: str, command: List[str]) -> None:
    LOGGER.info("Starting step: %s", name)
    LOGGER.info("Command: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    LOGGER.info("Completed step: %s", name)


def _build_train_model_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [sys.executable, "-m", "scripts.train_model", "--signals"]

    threshold = os.getenv("SIGNAL_THRESHOLD")
    min_auc = os.getenv("MIN_AUC")
    leverage = os.getenv("LEVERAGE")
    exclude = os.getenv("EXCLUDE_SYMBOLS")

    if threshold:
        cmd.extend(["--threshold", threshold])
    if min_auc:
        cmd.extend(["--min-auc", min_auc])
    if leverage:
        cmd.extend(["--leverage", leverage])
    if exclude:
        cmd.extend(["--exclude", exclude])
    if args.debug:
        cmd.append("--debug")

    return cmd


def _run_cycle(backfill_days: int, include_oi: bool, db_path: str, train_cmd: List[str]) -> None:
    run_pipeline_cmd = [
        sys.executable,
        "-m", "scripts.run_pipeline",
        "--backfill-only",
        "--backfill-days",
        str(backfill_days),
        "--db-path",
        db_path,
    ]
    if include_oi:
        run_pipeline_cmd.append("--include-oi")

    _run_step("run_pipeline backfill", run_pipeline_cmd)
    _run_step("compute_features", [sys.executable, "-m", "scripts.compute_features"])
    _run_step("train_model signals", train_cmd)


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _paper_kpis(writer: PgWriter, lookback_days: int = 14) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    closed_positions = writer.get_closed_paper_positions_since(since)
    equity_points = writer.get_paper_equity_curve_since(since)

    pnl_values = [_as_float(p.realized_pnl) for p in closed_positions]
    trades = len(pnl_values)
    wins = len([p for p in pnl_values if p > 0])
    losses = [p for p in pnl_values if p < 0]
    gross_profit = sum(p for p in pnl_values if p > 0)
    gross_loss = abs(sum(losses))
    win_rate = (wins / trades) if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    expectancy = (sum(pnl_values) / trades) if trades else 0.0
    trades_per_week = trades * (7 / lookback_days)

    drawdown = 0.0
    if equity_points:
        peak = _as_float(equity_points[0].equity)
        for point in equity_points:
            equity = _as_float(point.equity)
            peak = max(peak, equity)
            if peak > 0:
                drawdown = max(drawdown, (peak - equity) / peak)

    return {
        "window_days": lookback_days,
        "trades": trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "drawdown": drawdown,
        "expectancy": expectancy,
        "trades_per_week": trades_per_week,
    }


def _quarantine_reasons(kpis: dict[str, Any], thresholds: KillSwitchThresholds, state: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if kpis["trades"] < thresholds.min_samples:
        reasons.append(
            f"insufficient_samples:{kpis['trades']}<{thresholds.min_samples}"
        )
        return reasons

    if kpis["win_rate"] < thresholds.min_win_rate:
        reasons.append(f"win_rate_collapse:{kpis['win_rate']:.3f}<{thresholds.min_win_rate:.3f}")
    if kpis["profit_factor"] < thresholds.min_profit_factor:
        reasons.append(f"profit_factor_breach:{kpis['profit_factor']:.3f}<{thresholds.min_profit_factor:.3f}")
    if kpis["drawdown"] > thresholds.max_drawdown:
        reasons.append(f"drawdown_breach:{kpis['drawdown']:.3f}>{thresholds.max_drawdown:.3f}")
    if kpis["trades_per_week"] < thresholds.min_trades_per_week:
        reasons.append(f"low_trade_velocity:{kpis['trades_per_week']:.2f}<{thresholds.min_trades_per_week:.2f}")

    neg_streak = int(state.get("negative_expectancy_streak", 0))
    if kpis["expectancy"] < 0:
        neg_streak += 1
    else:
        neg_streak = 0
    state["negative_expectancy_streak"] = neg_streak
    if neg_streak >= thresholds.max_negative_expectancy_streak:
        reasons.append(
            f"sustained_negative_expectancy:{neg_streak}>={thresholds.max_negative_expectancy_streak}"
        )

    return reasons


def _evaluate_paper_monitoring(version: str, writer: PgWriter | None) -> tuple[bool, dict[str, Any], list[str]]:
    if writer is None:
        return False, {}, []

    state = _load_state()
    thresholds = _monitoring_thresholds()
    kpis = _paper_kpis(writer, lookback_days=int(os.getenv("PAPER_MONITOR_LOOKBACK_DAYS", "14")))
    reasons = _quarantine_reasons(kpis, thresholds, state)
    quarantined = any(not r.startswith("insufficient_samples") for r in reasons)

    monitoring_record = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "kpis": kpis,
        "thresholds": thresholds.__dict__,
        "reasons": reasons,
        "status": "quarantined" if quarantined else "active",
    }
    state["paper_monitoring"] = monitoring_record

    quarantined_models = state.setdefault("quarantined_models", {})
    if quarantined:
        quarantined_models["ALL"] = {
            "version": version,
            "status": "quarantined",
            "reasons": reasons,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    elif "ALL" in quarantined_models:
        quarantined_models.pop("ALL", None)

    _save_state(state)
    return quarantined, monitoring_record, reasons


def _promote_models(staging_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for staged in staging_dir.glob("*.joblib"):
        os.replace(staged, target_dir / staged.name)


def _run_retrain(train_window_days: int, retrain_every_days: int, writer: PgWriter | None) -> bool:
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%dT%H%M%SZ")
    staged_dir = MODELS_DIR / ".staging" / version
    staged_dir.mkdir(parents=True, exist_ok=True)

    config = Config(retrain_frequency_days=retrain_every_days)
    data = load_data()
    run_id = None
    if writer:
        run_id = writer.create_model_run(
            retrain_window_days=train_window_days,
            symbols_total=len(data),
            artifacts_version=version,
        )

    try:
        result = retrain_models(data, config, target_dir=staged_dir, train_window_days=train_window_days)
        if result.get("symbols_trained", 0) == 0:
            raise RuntimeError("No models trained successfully; refusing promotion")

        quarantined, monitoring_record, quarantine_reasons = _evaluate_paper_monitoring(version, writer)
        result["paper_monitoring"] = monitoring_record
        if quarantined:
            LOGGER.error(
                "Kill-switch triggered for version %s; quarantining models. Reasons: %s",
                version,
                ", ".join(quarantine_reasons),
            )
            raise RuntimeError(
                "Model version quarantined by paper monitor: " + ", ".join(quarantine_reasons)
            )

        _promote_models(staged_dir, MODELS_DIR)

        if writer and run_id:
            writer.complete_model_run(
                run_id=run_id,
                success=True,
                symbols_trained=result["symbols_trained"],
                metrics=result,
            )

        state = _load_state()
        state["last_successful_retrain_at"] = now.isoformat()
        state["last_artifacts_version"] = version
        _save_state(state)
        LOGGER.info(
            "Retrain success. Promoted %s models (version=%s).",
            result["symbols_trained"],
            version,
        )
        return True
    except Exception as exc:
        shutil.rmtree(staged_dir, ignore_errors=True)
        if writer and run_id:
            writer.complete_model_run(
                run_id=run_id,
                success=False,
                symbols_trained=0,
                error=str(exc),
            )
        LOGGER.exception("Retrain failed. Keeping previous models active.")
        return False


def _retrain_due(retrain_every_days: int) -> bool:
    state = _load_state()
    last_success = state.get("last_successful_retrain_at")
    if not last_success:
        return True

    try:
        last_dt = datetime.fromisoformat(last_success)
    except ValueError:
        return True

    return datetime.now(timezone.utc) >= (last_dt + timedelta(days=retrain_every_days))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trader orchestrator")
    parser.add_argument("--backfill-days", type=int, default=int(os.getenv("INITIAL_BACKFILL_DAYS", "30")))
    parser.add_argument("--cycle-interval-seconds", type=int, default=int(os.getenv("CYCLE_INTERVAL_SECONDS", "3600")))
    parser.add_argument("--incremental-backfill-hours", type=int, default=int(os.getenv("INCREMENTAL_BACKFILL_HOURS", "6")))
    parser.add_argument("--db-path", type=str, default=os.getenv("TRADER_DB_PATH", "/app/data/trading.db"))
    parser.add_argument("--retrain-every-days", type=int, default=int(os.getenv("RETRAIN_EVERY_DAYS", "7")))
    parser.add_argument("--train-window-days", type=int, default=int(os.getenv("TRAIN_WINDOW_DAYS", "90")))
    parser.add_argument("--retrain-only", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--include-oi", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", type=str, default=os.getenv("ORCHESTRATOR_LOG_FILE", "/app/logs/live_orchestrator.log"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level, Path(args.log_file))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    writer = PgWriter() if os.environ.get("DATABASE_URL") else None

    if args.retrain_only:
        return 0 if _run_retrain(args.train_window_days, args.retrain_every_days, writer) else 1

    incremental_days = max(1, math.ceil(args.incremental_backfill_hours / 24))
    train_cmd = _build_train_model_cmd(args)

    LOGGER.info("Live orchestrator starting.")

    try:
        _run_cycle(args.backfill_days, args.include_oi, args.db_path, train_cmd)

        cycle_num = 1
        while not STOP_REQUESTED:
            if _retrain_due(args.retrain_every_days):
                _run_retrain(args.train_window_days, args.retrain_every_days, writer)

            if args.run_once:
                break

            LOGGER.info("Sleeping for %s seconds before next cycle.", args.cycle_interval_seconds)
            time.sleep(args.cycle_interval_seconds)
            if STOP_REQUESTED:
                break

            cycle_num += 1
            LOGGER.info("Starting recurring cycle #%s", cycle_num)
            _run_cycle(incremental_days, args.include_oi, args.db_path, train_cmd)

    except Exception as exc:
        LOGGER.exception("Orchestrator failed: %s", exc)
        return 1

    LOGGER.info("Orchestrator stopped cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
