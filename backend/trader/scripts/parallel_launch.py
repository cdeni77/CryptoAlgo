#!/usr/bin/env python3
"""Parallel optimization launcher using direct function calls (no subprocess arg fan-out)."""

import argparse
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from scripts.optimize import PREFIX_FOR_COIN, load_data, optimize_coin_multiseed

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


@dataclass
class OptimizationConfig:
    trials: int = 300
    jobs: int = 1
    seeds: list[int] = field(default_factory=lambda: [42, 1337])
    n_cv_folds: int = 5
    top_candidates: int = 10
    holdout_mode: str = "multi_slice"
    holdout_days: int = 90
    plateau_patience: int = 120
    plateau_min_delta: float = 0.012
    plateau_warmup: int = 60
    plateau_min_completed: int = 0
    min_total_trades: int = 8
    holdout_min_trades: int = 8
    holdout_min_sharpe: float = 0.0
    holdout_min_return: float = -0.05
    target_trades_per_week: float = 1.0
    require_holdout_pass: bool = False
    gate_mode: str = "initial_paper_qualification"


def _optimize_single(args: tuple[dict[str, Any], str, OptimizationConfig, str]) -> tuple[str, dict[str, Any] | None]:
    all_data, coin, config, run_id = args
    result = optimize_coin_multiseed(
        all_data,
        coin_prefix=PREFIX_FOR_COIN.get(coin, coin),
        coin_name=coin,
        n_trials=config.trials,
        n_jobs=config.jobs,
        sampler_seeds=config.seeds,
        n_cv_folds=config.n_cv_folds,
        holdout_days=config.holdout_days,
        holdout_mode=config.holdout_mode,
        holdout_candidates=config.top_candidates,
        plateau_patience=config.plateau_patience,
        plateau_min_delta=config.plateau_min_delta,
        plateau_warmup=config.plateau_warmup,
        plateau_min_completed=config.plateau_min_completed,
        min_total_trades=config.min_total_trades,
        holdout_min_trades=config.holdout_min_trades,
        holdout_min_sharpe=config.holdout_min_sharpe,
        holdout_min_return=config.holdout_min_return,
        target_trades_per_week=config.target_trades_per_week,
        require_holdout_pass=config.require_holdout_pass,
        gate_mode=config.gate_mode,
        study_suffix=run_id,
    )
    return coin, result


def run_optimization(coins: list[str], config: OptimizationConfig, workers: int) -> dict[str, dict[str, Any] | None]:
    all_data = load_data(days=2200)
    run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    work_items = [(all_data, coin, config, run_id) for coin in coins]

    if workers <= 1 or len(work_items) == 1:
        outputs = [_optimize_single(item) for item in work_items]
    else:
        with mp.Pool(processes=min(workers, len(work_items))) as pool:
            outputs = pool.map(_optimize_single, work_items)

    return {coin: result for coin, result in outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct parallel launcher for optimization")
    parser.add_argument("--coins", type=str, default=",".join(COINS))
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--sampler-seeds", type=str, default="42,1337")
    parser.add_argument("--n-cv-folds", type=int, default=5)
    parser.add_argument("--holdout-candidates", type=int, default=10)
    parser.add_argument("--holdout-mode", type=str, default="multi_slice")
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--plateau-patience", type=int, default=120)
    parser.add_argument("--plateau-min-delta", type=float, default=0.012)
    parser.add_argument("--plateau-warmup", type=int, default=60)
    parser.add_argument("--plateau-min-completed", type=int, default=0)
    parser.add_argument("--min-total-trades", type=int, default=8)
    parser.add_argument("--holdout-min-trades", type=int, default=8)
    parser.add_argument("--holdout-min-sharpe", type=float, default=0.0)
    parser.add_argument("--holdout-min-return", type=float, default=-0.05)
    parser.add_argument("--target-trades-per-week", type=float, default=1.0)
    parser.add_argument("--require-holdout-pass", action="store_true")
    parser.add_argument("--gate-mode", type=str, default="initial_paper_qualification")
    args = parser.parse_args()

    selected_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    seeds = [int(item.strip()) for item in args.sampler_seeds.split(",") if item.strip()]

    config = OptimizationConfig(
        trials=args.trials,
        jobs=args.jobs,
        seeds=seeds,
        n_cv_folds=args.n_cv_folds,
        top_candidates=args.holdout_candidates,
        holdout_mode=args.holdout_mode,
        holdout_days=args.holdout_days,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        plateau_warmup=args.plateau_warmup,
        plateau_min_completed=args.plateau_min_completed,
        min_total_trades=args.min_total_trades,
        holdout_min_trades=args.holdout_min_trades,
        holdout_min_sharpe=args.holdout_min_sharpe,
        holdout_min_return=args.holdout_min_return,
        target_trades_per_week=args.target_trades_per_week,
        require_holdout_pass=args.require_holdout_pass,
        gate_mode=args.gate_mode,
    )

    results = run_optimization(selected_coins, config, workers=max(1, int(args.workers)))

    print("\n=== Optimization Summary ===")
    for coin in selected_coins:
        result = results.get(coin)
        if not result:
            print(f"{coin}: FAILED")
            continue
        score = result.get("optim_score")
        holdout = result.get("holdout_metrics", {})
        print(
            f"{coin}: score={score:.4f} "
            f"holdout_sr={float(holdout.get('holdout_sharpe', 0.0)):.3f} "
            f"trades={int(holdout.get('holdout_trades', 0))}"
        )


if __name__ == "__main__":
    main()
