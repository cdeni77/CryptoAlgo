#!/usr/bin/env python3
"""Orchestrate live trading pipeline steps on startup + hourly cycles."""

from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

LOGGER = logging.getLogger("live_orchestrator")
STOP_REQUESTED = False


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
    cmd = [sys.executable, "train_model.py", "--signals"]

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
        "run_pipeline.py",
        "--backfill-only",
        "--backfill-days",
        str(backfill_days),
        "--db-path",
        db_path,
    ]
    if include_oi:
        run_pipeline_cmd.append("--include-oi")

    _run_step("run_pipeline backfill", run_pipeline_cmd)
    _run_step("compute_features", [sys.executable, "compute_features.py"])
    _run_step("train_model signals", train_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live trader orchestrator")
    parser.add_argument("--backfill-days", type=int, default=int(os.getenv("INITIAL_BACKFILL_DAYS", "30")))
    parser.add_argument("--cycle-interval-seconds", type=int, default=int(os.getenv("CYCLE_INTERVAL_SECONDS", "3600")))
    parser.add_argument("--incremental-backfill-hours", type=int, default=int(os.getenv("INCREMENTAL_BACKFILL_HOURS", "6")))
    parser.add_argument("--db-path", type=str, default=os.getenv("TRADER_DB_PATH", "/app/data/trading.db"))
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

    incremental_days = max(1, math.ceil(args.incremental_backfill_hours / 24))
    train_cmd = _build_train_model_cmd(args)

    LOGGER.info("Live orchestrator starting.")
    LOGGER.info(
        "Initial backfill=%s days | recurring interval=%s sec | incremental window=%s hour(s) (~%s day(s))",
        args.backfill_days,
        args.cycle_interval_seconds,
        args.incremental_backfill_hours,
        incremental_days,
    )

    try:
        _run_cycle(args.backfill_days, args.include_oi, args.db_path, train_cmd)

        cycle_num = 1
        while not STOP_REQUESTED:
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
