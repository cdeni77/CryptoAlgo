from argparse import Namespace
import inspect
import json

from scripts import optimize
from scripts import parallel_launch


def test_parse_args_supports_new_flags() -> None:
    args = parallel_launch.parse_args(
        [
            "--coins",
            "BTC,ETH",
            "--workers",
            "auto",
            "--parallel-mode",
            "coin-seed",
            "--preset",
            "none",
            "--pruned-only",
            "--study-suffix",
            "batch42",
            "--min-psr",
            "0.6",
            "--min-psr-cv",
            "0.61",
            "--min-psr-holdout",
            "0.7",
            "--min-dsr",
            "0.5",
            "--proxy-fidelity-candidates",
            "4",
            "--proxy-fidelity-eval-days",
            "28",
        ]
    )

    assert args.parallel_mode == "coin-seed"
    assert args.preset == "none"
    assert args.pruned_only is True
    assert args.study_suffix == "batch42"
    assert args.min_psr == 0.6
    assert args.min_psr_cv == 0.61
    assert args.min_psr_holdout == 0.7
    assert args.min_dsr == 0.5
    assert args.proxy_fidelity_candidates == 4
    assert args.proxy_fidelity_eval_days == 28


def test_resolve_workers_auto_and_clamp() -> None:
    assert parallel_launch.resolve_workers("auto", task_count=15, cpu_count=20) == 5
    assert parallel_launch.resolve_workers("13", task_count=3, cpu_count=20) == 3


def test_unique_study_suffixes() -> None:
    run_id = "run_20260101T000000Z"
    btc = parallel_launch._build_coin_study_suffix("user", "BTC", run_id)
    eth = parallel_launch._build_coin_study_suffix("user", "ETH", run_id)
    btc_seed = parallel_launch._build_coin_seed_study_suffix("user", "BTC", 42, run_id)
    eth_seed = parallel_launch._build_coin_seed_study_suffix("user", "ETH", 42, run_id)
    assert btc != eth
    assert btc_seed != eth_seed
    assert btc == "user_BTC_run_20260101T000000Z"
    assert btc_seed == "user_BTC_seed42_run_20260101T000000Z"


def test_optimize_single_forwards_flags(monkeypatch) -> None:
    captured = {}

    def _fake_optimize_coin_multiseed(_all_data, **kwargs):
        captured.update(kwargs)
        return {"optim_score": 1.0, "holdout_metrics": {"holdout_sharpe": 0.5, "holdout_trades": 11}}

    monkeypatch.setattr(parallel_launch, "optimize_coin_multiseed", _fake_optimize_coin_multiseed)

    config = parallel_launch.OptimizationConfig(
        study_suffix="suite",
        pruned_only=True,
        preset_name="paper_ready",
        min_psr=0.6,
        min_psr_cv=0.55,
        min_psr_holdout=0.52,
        min_dsr=0.4,
        proxy_fidelity_candidates=3,
        proxy_fidelity_eval_days=21,
    )

    payload = parallel_launch._optimize_single(({"x": 1}, "BTC", config, "run1"))
    assert payload["success"] is True
    assert captured["study_suffix"] == "suite_BTC_run1"
    assert captured["pruned_only"] is True


def test_seed_worker_calls_optimize_coin_with_seed_args(monkeypatch) -> None:
    captured = {}

    def _fake_optimize_coin(_all_data, _coin_prefix, _coin_name, **kwargs):
        captured.update(kwargs)
        return {"optim_score": 1.0, "holdout_metrics": {"holdout_sharpe": 0.4, "holdout_trades": 12}}

    monkeypatch.setattr(parallel_launch, "optimize_coin", _fake_optimize_coin)
    config = parallel_launch.OptimizationConfig(study_suffix="suite", jobs=1)

    payload = parallel_launch._optimize_seed_worker(({"x": 1}, "BTC", 1337, config, "run1"))
    assert payload["success"] is True
    assert payload["seed"] == 1337
    assert payload["study_suffix"] == "suite_BTC_seed1337_run1"
    assert captured["sampler_seed"] == 1337
    assert captured["n_jobs"] == 1


def test_optimize_coin_signature_supports_parallel_forwarded_proxy_args() -> None:
    signature = inspect.signature(optimize.optimize_coin)
    assert "proxy_fidelity_candidates" in signature.parameters
    assert "proxy_fidelity_eval_days" in signature.parameters


def test_worker_failure_isolation(monkeypatch) -> None:
    def _fake_optimize_coin(_all_data, _coin_prefix, _coin_name, **kwargs):
        if kwargs.get("sampler_seed") == 2:
            raise RuntimeError("boom")
        return {"optim_score": 0.2, "holdout_metrics": {"holdout_sharpe": 0.1, "holdout_trades": 2}}

    monkeypatch.setattr(parallel_launch, "optimize_coin", _fake_optimize_coin)

    config = parallel_launch.OptimizationConfig()
    ok = parallel_launch._optimize_seed_worker(({"x": 1}, "BTC", 1, config, "run2"))
    bad = parallel_launch._optimize_seed_worker(({"x": 1}, "BTC", 2, config, "run2"))

    assert ok["success"] is True
    assert bad["success"] is False
    assert "RuntimeError" in bad["error"]


def test_parallel_optimize_forwarded_kwargs_are_supported_downstream() -> None:
    config_to_optimize = {
        "trials": "n_trials",
        "jobs": "n_jobs",
        "n_cv_folds": "n_cv_folds",
        "holdout_mode": "holdout_mode",
        "holdout_days": "holdout_days",
        "plateau_patience": "plateau_patience",
        "plateau_min_delta": "plateau_min_delta",
        "plateau_warmup": "plateau_warmup",
        "plateau_min_completed": "plateau_min_completed",
        "min_total_trades": "min_total_trades",
        "holdout_min_trades": "holdout_min_trades",
        "holdout_min_sharpe": "holdout_min_sharpe",
        "holdout_min_return": "holdout_min_return",
        "target_trades_per_week": "target_trades_per_week",
        "target_trades_per_year": "target_trades_per_year",
        "require_holdout_pass": "require_holdout_pass",
        "gate_mode": "gate_mode",
        "resume_study": "resume_study",
        "min_psr": "min_psr",
        "min_psr_cv": "min_psr_cv",
        "min_psr_holdout": "min_psr_holdout",
        "min_dsr": "min_dsr",
        "seed_stability_min_pass_rate": "seed_stability_min_pass_rate",
        "seed_stability_max_param_dispersion": "seed_stability_max_param_dispersion",
        "seed_stability_max_oos_sharpe_dispersion": "seed_stability_max_oos_sharpe_dispersion",
        "cv_mode": "cv_mode",
        "purge_days": "purge_days",
        "purge_bars": "purge_bars",
        "embargo_days": "embargo_days",
        "embargo_bars": "embargo_bars",
        "embargo_frac": "embargo_frac",
        "pruned_only": "pruned_only",
        "preset_name": "preset_name",
        "cost_config_path": "cost_config_path",
        "proxy_fidelity_candidates": "proxy_fidelity_candidates",
        "proxy_fidelity_eval_days": "proxy_fidelity_eval_days",
    }
    optimize_params = set(inspect.signature(optimize.optimize_coin).parameters.keys())
    for expected_param in config_to_optimize.values():
        assert expected_param in optimize_params


def test_seed_payload_aggregation_uses_shared_helper(monkeypatch) -> None:
    captured = {}

    def _fake_aggregate(**kwargs):
        captured.update(kwargs)
        return {"optim_score": 1.2, "holdout_metrics": {"holdout_sharpe": 0.6, "holdout_trades": 12}}

    monkeypatch.setattr(parallel_launch, "aggregate_multiseed_results", _fake_aggregate)
    config = parallel_launch.OptimizationConfig()
    payloads = [
        {"coin": "BTC", "seed": 1, "success": True, "result": {"optim_score": 1.0}, "error": None, "study_suffix": "s1"},
        {"coin": "BTC", "seed": 2, "success": False, "result": None, "error": "bad", "study_suffix": "s2"},
        {"coin": "BTC", "seed": 3, "success": True, "result": {"optim_score": 0.9}, "error": None, "study_suffix": "s3"},
    ]

    out = parallel_launch._aggregate_seed_payloads("BTC", [1, 2, 3], payloads, config)
    assert out["success"] is True
    assert out["failed_seeds"] == [2]
    assert captured["failed_seeds"] == [2]
    assert captured["emit_artifacts"] is True


def test_seed_payload_aggregation_fails_only_when_no_seed_results() -> None:
    config = parallel_launch.OptimizationConfig()
    payloads = [
        {"coin": "BTC", "seed": 1, "success": False, "result": None, "error": "a", "study_suffix": "s1"},
        {"coin": "BTC", "seed": 2, "success": False, "result": None, "error": "b", "study_suffix": "s2"},
    ]
    out = parallel_launch._aggregate_seed_payloads("BTC", [1, 2], payloads, config)
    assert out["success"] is False
    assert out["failed_seeds"] == [1, 2]


def test_manifest_writing_coin_seed_mode(tmp_path) -> None:
    args = Namespace(
        sampler_seeds="42,1337,7",
        parallel_mode="coin-seed",
        preset="paper_ready",
        pruned_only=True,
        jobs=1,
        holdout_mode="multi_slice",
        holdout_days=90,
        holdout_candidates=3,
        require_holdout_pass=True,
        gate_mode="initial_paper_qualification",
        min_psr=0.55,
        min_psr_cv=0.56,
        min_psr_holdout=0.57,
        min_dsr=0.4,
        proxy_fidelity_candidates=2,
        proxy_fidelity_eval_days=14,
        target_trades_per_week=1.0,
        target_trades_per_year=None,
        study_suffix="nightly",
        workers="auto",
    )
    manifest = parallel_launch.build_run_manifest("/tmp", args, ["BTC", "ETH"], "run42", workers=6, cpu_count=20, total_tasks=6)
    path = parallel_launch.save_run_manifest(manifest, manifest_dir=tmp_path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "run42"
    assert saved["workers"] == 6
    assert saved["cpu_count"] == 20
    assert saved["parallel_mode"] == "coin-seed"
    assert saved["total_tasks"] == 6
    assert saved["aggregation_serial_after_tasks"] is True
    assert saved["study_suffix_strategy"]["examples"]["BTC"][0] == "nightly_BTC_seed42_run42"


def test_run_optimization_coin_seed_aggregates_per_coin(monkeypatch) -> None:
    monkeypatch.setattr(parallel_launch, "load_data", lambda: {"d": 1})

    def _fake_worker(item):
        _all_data, coin, seed, _config, _run_id = item
        if coin == "ETH" and seed == 2:
            return {"coin": coin, "seed": seed, "success": False, "result": None, "error": "boom", "study_suffix": "x"}
        return {"coin": coin, "seed": seed, "success": True, "result": {"optim_score": float(seed)}, "error": None, "study_suffix": "x"}

    monkeypatch.setattr(parallel_launch, "_optimize_seed_worker", _fake_worker)
    monkeypatch.setattr(
        parallel_launch,
        "aggregate_multiseed_results",
        lambda **kwargs: {"optim_score": max(r["optim_score"] for r in kwargs["run_results"]), "holdout_metrics": {"holdout_sharpe": 0.2, "holdout_trades": 8}},
    )

    out = parallel_launch.run_optimization(["BTC", "ETH"], parallel_launch.OptimizationConfig(seeds=[1, 2]), workers=1, parallel_mode="coin-seed")
    assert out["total_tasks"] == 4
    assert out["results"]["BTC"]["success"] is True
    assert out["results"]["ETH"]["success"] is True
    assert out["results"]["ETH"]["failed_seeds"] == [2]


def test_summary_mixed_success_failure(monkeypatch, capsys, tmp_path) -> None:
    def _fake_run(_coins, _config, workers, parallel_mode):
        assert workers == 4
        assert parallel_mode == "coin-seed"
        return {
            "run_id": "run_summary",
            "results": {
                "BTC": {
                    "coin": "BTC",
                    "success": True,
                    "error": None,
                    "successful_seeds": [42],
                    "failed_seeds": [1337],
                    "result": {
                        "optim_score": 0.9,
                        "holdout_metrics": {"holdout_sharpe": 1.2, "holdout_trades": 20},
                        "result_json_path": "optimization_results/BTC.json",
                    },
                },
                "ETH": {
                    "coin": "ETH",
                    "success": False,
                    "error": "RuntimeError: bad",
                    "successful_seeds": [],
                    "failed_seeds": [42, 1337],
                    "result": None,
                },
            },
        }

    monkeypatch.setattr(parallel_launch, "run_optimization", _fake_run)
    monkeypatch.setattr(parallel_launch.mp, "cpu_count", lambda: 20)
    monkeypatch.setattr(parallel_launch, "save_run_manifest", lambda manifest, manifest_dir=parallel_launch.MANIFEST_DIR: tmp_path / "m.json")

    args = ["--coins", "BTC,ETH", "--workers", "auto", "--jobs", "1", "--parallel-mode", "coin-seed"]
    monkeypatch.setattr("sys.argv", ["parallel_launch.py", *args])
    parallel_launch.main()

    out = capsys.readouterr().out
    assert "BTC: SUCCESS | success_seeds=1 failed_seeds=1" in out
    assert "ETH: FAILED | success_seeds=0 failed_seeds=2" in out
