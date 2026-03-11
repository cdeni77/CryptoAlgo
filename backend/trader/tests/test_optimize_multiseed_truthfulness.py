from copy import deepcopy

import scripts.optimize as optimize


def test_multiseed_flags_consensus_as_not_revalidated_and_keeps_selected_seed_metrics(monkeypatch):
    seed_results = {
        1: {
            "coin": "BTC",
            "optim_score": 1.2,
            "params": {"signal_threshold": 0.71, "max_depth": 7},
            "holdout_metrics": {"holdout_sharpe": 0.33, "holdout_trades": 20, "holdout_return": 0.04},
            "deployment_blocked": False,
            "selection_meta": {},
        },
        2: {
            "coin": "BTC",
            "optim_score": 1.0,
            "params": {"signal_threshold": 0.83, "max_depth": 5},
            "holdout_metrics": {"holdout_sharpe": 0.31, "holdout_trades": 22, "holdout_return": 0.03},
            "deployment_blocked": False,
            "selection_meta": {},
        },
    }

    monkeypatch.setattr(
        optimize,
        "optimize_coin",
        lambda _all_data, _coin_prefix, _coin_name, sampler_seed, **_kwargs: deepcopy(seed_results[sampler_seed]),
    )
    monkeypatch.setattr(optimize, "_persist_result_json", lambda *_args, **_kwargs: None)

    result = optimize.optimize_coin_multiseed({}, "BIP", "BTC", sampler_seeds=[1, 2])

    assert result is not None
    assert result["evaluated_params_source"] == "selected_seed"
    assert result["consensus_revalidated"] is False
    assert result["params"] == seed_results[1]["params"]
    assert result["holdout_metrics"] == seed_results[1]["holdout_metrics"]

    assert result["consensus_params"] != result["params"]
    assert result["consensus_params"] == {"signal_threshold": 0.77, "max_depth": 6}
