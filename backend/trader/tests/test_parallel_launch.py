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
    assert parallel_launch.resolve_workers("auto", num_coins=5, cpu_count=20) == 5
    assert parallel_launch.resolve_workers("7", num_coins=3, cpu_count=20) == 3


def test_unique_study_suffix_per_coin() -> None:
    run_id = "run_20260101T000000Z"
    btc = parallel_launch._build_coin_study_suffix("user", "BTC", run_id)
    eth = parallel_launch._build_coin_study_suffix("user", "ETH", run_id)
    assert btc != eth
    assert btc == "user_BTC_run_20260101T000000Z"


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
    assert captured["preset_name"] == "paper_ready"
    assert captured["min_psr"] == 0.6
    assert captured["min_psr_cv"] == 0.55
    assert captured["min_psr_holdout"] == 0.52
    assert captured["min_dsr"] == 0.4
    assert captured["proxy_fidelity_candidates"] == 3
    assert captured["proxy_fidelity_eval_days"] == 21




def test_optimize_coin_signature_supports_parallel_forwarded_proxy_args() -> None:
    signature = inspect.signature(optimize.optimize_coin)
    assert "proxy_fidelity_candidates" in signature.parameters
    assert "proxy_fidelity_eval_days" in signature.parameters

def test_worker_failure_isolation(monkeypatch) -> None:
    def _fake_optimize_coin_multiseed(_all_data, **kwargs):
        if kwargs.get("coin_name") == "ETH":
            raise RuntimeError("boom")
        return {"optim_score": 0.2, "holdout_metrics": {"holdout_sharpe": 0.1, "holdout_trades": 2}}

    monkeypatch.setattr(parallel_launch, "optimize_coin_multiseed", _fake_optimize_coin_multiseed)

    config = parallel_launch.OptimizationConfig()
    btc = parallel_launch._optimize_single(({"x": 1}, "BTC", config, "run2"))
    eth = parallel_launch._optimize_single(({"x": 1}, "ETH", config, "run2"))

    assert btc["success"] is True
    assert eth["success"] is False
    assert "RuntimeError" in eth["error"]




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

def test_manifest_writing(tmp_path) -> None:
    args = Namespace(
        sampler_seeds="42,1337",
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
    )
    manifest = parallel_launch.build_run_manifest("/tmp", args, ["BTC", "ETH"], "run42", workers=2, cpu_count=20)
    path = parallel_launch.save_run_manifest(manifest, manifest_dir=tmp_path)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["run_id"] == "run42"
    assert saved["workers"] == 2
    assert saved["cpu_count"] == 20
    assert saved["preset"] == "paper_ready"
    assert saved["pruned_only"] is True
    assert saved["study_suffix_strategy"]["examples"]["BTC"] == "nightly_BTC_run42"


def test_summary_mixed_success_failure(monkeypatch, capsys, tmp_path) -> None:
    def _fake_run(_coins, _config, workers):
        assert workers == 2
        return {
            "run_id": "run_summary",
            "results": {
                "BTC": {
                    "coin": "BTC",
                    "success": True,
                    "error": None,
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
                    "result": None,
                },
            },
        }

    monkeypatch.setattr(parallel_launch, "run_optimization", _fake_run)
    monkeypatch.setattr(parallel_launch.mp, "cpu_count", lambda: 20)
    monkeypatch.setattr(parallel_launch, "save_run_manifest", lambda manifest, manifest_dir=parallel_launch.MANIFEST_DIR: tmp_path / "m.json")

    args = ["--coins", "BTC,ETH", "--workers", "auto", "--jobs", "1"]
    monkeypatch.setattr("sys.argv", ["parallel_launch.py", *args])
    parallel_launch.main()

    out = capsys.readouterr().out
    assert "BTC: SUCCESS" in out
    assert "ETH: FAILED" in out
