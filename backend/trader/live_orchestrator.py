#!/usr/bin/env python3
"""Live orchestrator with scheduled retraining and atomic model promotion."""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from coin_profiles import MODELS_DIR
from pg_writer import PgWriter
from train_model import Config, load_data, retrain_models

STATE_FILE = Path("./data/orchestrator_state.json")


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


def _promote_models(staging_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for staged in staging_dir.glob("*.joblib"):
        active = target_dir / staged.name
        os.replace(staged, active)


def _run_retrain(config: Config, train_window_days: int, writer: PgWriter | None) -> bool:
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%dT%H%M%SZ")
    staged_dir = MODELS_DIR / ".staging" / version
    staged_dir.mkdir(parents=True, exist_ok=True)

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
        if result["symbols_trained"] == 0:
            raise RuntimeError("No models trained successfully; refusing promotion")

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
        print(f"✅ Retrain success. Promoted {result['symbols_trained']} models (version={version}).")
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
        print(f"❌ Retrain failed. Keeping previous models active. Error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Scheduler for periodic model retraining")
    parser.add_argument("--retrain-every-days", type=int, default=7)
    parser.add_argument("--train-window-days", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()

    config = Config(retrain_frequency_days=args.retrain_every_days)
    writer = None
    if os.environ.get("DATABASE_URL"):
        writer = PgWriter()

    while True:
        state = _load_state()
        last_success = state.get("last_successful_retrain_at")
        now = datetime.now(timezone.utc)

        due = True
        if last_success:
            last_dt = datetime.fromisoformat(last_success)
            due = now >= (last_dt + timedelta(days=args.retrain_every_days))

        if due:
            _run_retrain(config, train_window_days=args.train_window_days, writer=writer)
        else:
            print("ℹ️ Retrain not due yet.")

        if args.run_once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
