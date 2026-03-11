#!/usr/bin/env python3
"""Parallel optimization launcher using direct function calls (no subprocess arg fan-out)."""

import argparse
import json
import multiprocessing as mp
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.optimize import PREFIX_FOR_COIN, apply_runtime_preset, load_data, optimize_coin_multiseed

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
MANIFEST_DIR = Path("optimization_results/manifests")


def _parse_sampler_seeds(raw: str) -> list[int]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return [int(item) for item in values]


def resolve_workers(worker_arg: str | int, num_coins: int, cpu_count: int | None = None) -> int:
    selected = max(1, int(num_coins or 1))
    detected_cpu = int(cpu_count or mp.cpu_count() or 1)
    if str(worker_arg).lower() == "auto":
        resolved = min(selected, max(1, detected_cpu // 4))
    else:
        resolved = max(1, int(worker_arg))
    return min(selected, resolved)


def _build_coin_study_suffix(base_suffix: str, coin: str, run_id: str) -> str:
    coin_key = str(coin).upper()
    if base_suffix:
        return f"{base_suffix}_{coin_key}_{run_id}"
    return f"{run_id}_{coin_key}"


def build_run_manifest(
    script_dir,
    args,
    target_coins: list[str],
    run_id: str,
    workers: int | None = None,
    cpu_count: int | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for a parallel optimization run."""
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    sampler_seeds_raw = getattr(args, "sampler_seeds", "")
    seeds = _parse_sampler_seeds(sampler_seeds_raw)
    study_suffix = str(getattr(args, "study_suffix", "") or "")
    resolved_cpu_count = int(cpu_count or mp.cpu_count() or 1)
    resolved_workers = int(workers or resolve_workers(getattr(args, "workers", "auto"), len(target_coins), resolved_cpu_count))
    return {
        "run_id": str(run_id),
        "generated_at": generated_at,
        "script_dir": str(script_dir),
        "cpu_count": resolved_cpu_count,
        "workers": resolved_workers,
        "jobs": int(getattr(args, "jobs", 1)),
        "preset": getattr(args, "preset", None),
        "pruned_only": bool(getattr(args, "pruned_only", False)),
        "target_coins": [str(coin).upper() for coin in (target_coins or [])],
        "study_suffix_strategy": {
            "provided_suffix": study_suffix,
            "mode": "user_suffix_plus_coin_plus_run" if study_suffix else "run_plus_coin",
            "examples": {
                coin: _build_coin_study_suffix(study_suffix, coin, run_id)
                for coin in (target_coins or [])
            },
        },
        "seed_policy": {
            "sampler_seeds_raw": sampler_seeds_raw,
            "sampler_seeds": seeds,
        },
        "optimizer_flags": {
            "holdout_mode": getattr(args, "holdout_mode", None),
            "holdout_days": getattr(args, "holdout_days", None),
            "holdout_candidates": getattr(args, "holdout_candidates", None),
            "require_holdout_pass": bool(getattr(args, "require_holdout_pass", False)),
            "gate_mode": getattr(args, "gate_mode", None),
            "min_psr": getattr(args, "min_psr", None),
            "min_psr_cv": getattr(args, "min_psr_cv", None),
            "min_psr_holdout": getattr(args, "min_psr_holdout", None),
            "min_dsr": getattr(args, "min_dsr", None),
            "proxy_fidelity_candidates": getattr(args, "proxy_fidelity_candidates", None),
            "proxy_fidelity_eval_days": getattr(args, "proxy_fidelity_eval_days", None),
            "target_trades_per_week": getattr(args, "target_trades_per_week", None),
            "target_trades_per_year": getattr(args, "target_trades_per_year", None),
        },
    }


def save_run_manifest(manifest: dict[str, Any], manifest_dir: Path = MANIFEST_DIR) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"parallel_launch_{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


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
    target_trades_per_year: float | None = None
    require_holdout_pass: bool = False
    gate_mode: str = "initial_paper_qualification"
    study_suffix: str = ""
    resume_study: bool = False
    min_psr: float = 0.55
    min_psr_cv: float | None = None
    min_psr_holdout: float | None = None
    min_dsr: float | None = None
    seed_stability_min_pass_rate: float = 0.67
    seed_stability_max_param_dispersion: float = 0.60
    seed_stability_max_oos_sharpe_dispersion: float = 0.35
    cv_mode: str = "walk_forward"
    purge_days: int | None = None
    purge_bars: int | None = None
    embargo_days: int | None = None
    embargo_bars: int | None = None
    embargo_frac: float = 0.0
    pruned_only: bool = True
    preset_name: str = "paper_ready"
    cost_config_path: str | None = None
    proxy_fidelity_candidates: int = 0
    proxy_fidelity_eval_days: int = 0


def _resolve_coin_prefix(coin: str) -> str:
    """Return the canonical CDE prefix for a configured coin."""
    coin_key = coin.upper()
    if coin_key not in PREFIX_FOR_COIN:
        raise ValueError(f"Unsupported coin '{coin}' for CDE prefix mapping")
    return PREFIX_FOR_COIN[coin_key]


def _optimize_single(args: tuple[dict[str, Any], str, OptimizationConfig, str]) -> dict[str, Any]:
    all_data, coin, config, run_id = args
    coin_study_suffix = _build_coin_study_suffix(config.study_suffix, coin, run_id)
    try:
        coin_prefix = _resolve_coin_prefix(coin)
        result = optimize_coin_multiseed(
            all_data,
            coin_prefix=coin_prefix,
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
            target_trades_per_year=config.target_trades_per_year,
            require_holdout_pass=config.require_holdout_pass,
            gate_mode=config.gate_mode,
            study_suffix=coin_study_suffix,
            resume_study=config.resume_study,
            min_psr=config.min_psr,
            min_psr_cv=config.min_psr_cv,
            min_psr_holdout=config.min_psr_holdout,
            min_dsr=config.min_dsr,
            seed_stability_min_pass_rate=config.seed_stability_min_pass_rate,
            seed_stability_max_param_dispersion=config.seed_stability_max_param_dispersion,
            seed_stability_max_oos_sharpe_dispersion=config.seed_stability_max_oos_sharpe_dispersion,
            cv_mode=config.cv_mode,
            purge_days=config.purge_days,
            purge_bars=config.purge_bars,
            embargo_days=config.embargo_days,
            embargo_bars=config.embargo_bars,
            embargo_frac=config.embargo_frac,
            pruned_only=config.pruned_only,
            preset_name=config.preset_name,
            cost_config_path=config.cost_config_path,
            proxy_fidelity_candidates=config.proxy_fidelity_candidates,
            proxy_fidelity_eval_days=config.proxy_fidelity_eval_days,
        )
        return {
            "coin": coin,
            "success": bool(result),
            "result": result,
            "error": None,
            "study_suffix": coin_study_suffix,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "coin": coin,
            "success": False,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "study_suffix": coin_study_suffix,
        }


def run_optimization(coins: list[str], config: OptimizationConfig, workers: int) -> dict[str, dict[str, Any]]:
    all_data = load_data()
    run_id = datetime.utcnow().strftime("run_%Y%m%dT%H%M%S%fZ")
    work_items = [(all_data, coin, config, run_id) for coin in coins]

    if workers <= 1 or len(work_items) == 1:
        outputs = [_optimize_single(item) for item in work_items]
    else:
        with mp.Pool(processes=min(workers, len(work_items))) as pool:
            outputs = pool.map(_optimize_single, work_items)

    return {
        "run_id": run_id,
        "results": {str(item["coin"]): item for item in outputs},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct parallel launcher for optimization")
    parser.add_argument("--coins", type=str, default=",".join(COINS))
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--workers", type=str, default="auto", help="Worker count or 'auto'")
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
    parser.add_argument("--target-trades-per-year", type=float, default=None)
    parser.add_argument("--require-holdout-pass", action="store_true")
    parser.add_argument("--gate-mode", type=str, default="initial_paper_qualification")
    parser.add_argument("--preset", type=str, default="paper_ready")
    parser.add_argument("--pruned-only", action="store_true")
    parser.add_argument("--allow-unpruned", action="store_false", dest="pruned_only")
    parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-psr", type=float, default=0.55)
    parser.add_argument("--min-psr-cv", type=float, default=None)
    parser.add_argument("--min-psr-holdout", type=float, default=None)
    parser.add_argument("--min-dsr", type=float, default=None)
    parser.add_argument("--seed-stability-min-pass-rate", type=float, default=0.67)
    parser.add_argument("--seed-stability-max-param-dispersion", type=float, default=0.60)
    parser.add_argument("--seed-stability-max-oos-sharpe-dispersion", type=float, default=0.35)
    parser.add_argument("--cv-mode", type=str, default="walk_forward")
    parser.add_argument("--purge-days", type=int, default=None)
    parser.add_argument("--purge-bars", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=None)
    parser.add_argument("--embargo-bars", type=int, default=None)
    parser.add_argument("--embargo-frac", type=float, default=0.0)
    parser.add_argument("--cost-config-path", type=str, default=None)
    parser.add_argument("--proxy-fidelity-candidates", type=int, default=0)
    parser.add_argument("--proxy-fidelity-eval-days", type=int, default=0)
    parser.set_defaults(pruned_only=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    return apply_runtime_preset(args)


def main() -> None:
    args = parse_args()
    selected_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    seeds = [int(item.strip()) for item in args.sampler_seeds.split(",") if item.strip()]
    cpu_count = int(mp.cpu_count() or 1)
    resolved_workers = resolve_workers(args.workers, len(selected_coins), cpu_count=cpu_count)

    print("\n=== Parallel Optimization Launch ===")
    print(f"CPU count detected: {cpu_count}")
    print(f"Selected coins: {', '.join(selected_coins)}")
    print(f"Workers: {resolved_workers}")
    print(f"Jobs per coin optimization: {args.jobs}")
    print("Concurrency model: coin-level parallelism with single-job inner optimization.")

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
        target_trades_per_year=args.target_trades_per_year,
        require_holdout_pass=args.require_holdout_pass,
        gate_mode=args.gate_mode,
        study_suffix=args.study_suffix,
        resume_study=args.resume,
        min_psr=args.min_psr,
        min_psr_cv=args.min_psr_cv,
        min_psr_holdout=args.min_psr_holdout,
        min_dsr=args.min_dsr,
        seed_stability_min_pass_rate=args.seed_stability_min_pass_rate,
        seed_stability_max_param_dispersion=args.seed_stability_max_param_dispersion,
        seed_stability_max_oos_sharpe_dispersion=args.seed_stability_max_oos_sharpe_dispersion,
        cv_mode=args.cv_mode,
        purge_days=args.purge_days,
        purge_bars=args.purge_bars,
        embargo_days=args.embargo_days,
        embargo_bars=args.embargo_bars,
        embargo_frac=args.embargo_frac,
        pruned_only=args.pruned_only,
        preset_name=args.preset,
        cost_config_path=args.cost_config_path,
        proxy_fidelity_candidates=args.proxy_fidelity_candidates,
        proxy_fidelity_eval_days=args.proxy_fidelity_eval_days,
    )

    run_outputs = run_optimization(selected_coins, config, workers=resolved_workers)
    run_id = run_outputs["run_id"]
    per_coin = run_outputs["results"]

    manifest = build_run_manifest(Path(__file__).resolve().parent, args, selected_coins, run_id, resolved_workers, cpu_count)
    manifest["config"] = asdict(config)
    manifest_path = save_run_manifest(manifest)

    print("\n=== Optimization Summary ===")
    print(f"Manifest: {manifest_path}")
    for coin in selected_coins:
        payload = per_coin.get(coin)
        if not payload or not payload.get("success"):
            err = payload.get("error") if payload else "missing result payload"
            print(f"{coin}: FAILED | error={err}")
            continue
        result = payload.get("result") or {}
        score = float(result.get("optim_score", 0.0) or 0.0)
        holdout = result.get("holdout_metrics", {}) or {}
        holdout_sr = float(holdout.get("holdout_sharpe", 0.0) or 0.0)
        holdout_trades = int(holdout.get("holdout_trades", 0) or 0)
        result_path = result.get("result_json_path") or result.get("result_path") or "n/a"
        print(
            f"{coin}: SUCCESS | score={score:.4f} holdout_sr={holdout_sr:.3f} "
            f"holdout_trades={holdout_trades} result={result_path}"
        )


if __name__ == "__main__":
    main()
