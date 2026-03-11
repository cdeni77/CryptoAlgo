from copy import deepcopy

import scripts.optimize as optimize


def test_multiseed_keeps_selected_seed_params_and_metrics(monkeypatch):
    seed_results = {
        101: {
            "coin": "BTC",
            "optim_score": 1.2,
            "params": {"signal_threshold": 0.71, "max_depth": 8},
            "holdout_metrics": {"holdout_sharpe": 0.41, "holdout_trades": 20, "holdout_return": 0.06},
            "deployment_blocked": False,
            "selection_meta": {},
            "research_confidence_tier": "PAPER_QUALIFIED",
            "gate_profile": {"mode": "initial_paper_qualification"},
            "meta": {"study_name": "s1"},
            "run_id": "run-1",
        },
        202: {
            "coin": "BTC",
            "optim_score": 0.9,
            "params": {"signal_threshold": 0.88, "max_depth": 4},
            "holdout_metrics": {"holdout_sharpe": 0.15, "holdout_trades": 16, "holdout_return": 0.01},
            "deployment_blocked": False,
            "selection_meta": {},
            "research_confidence_tier": "PAPER_QUALIFIED",
            "gate_profile": {"mode": "initial_paper_qualification"},
            "meta": {"study_name": "s2"},
            "run_id": "run-2",
        },
    }

    def fake_optimize_coin(_all_data, _coin_prefix, _coin_name, sampler_seed, **_kwargs):
        return deepcopy(seed_results[sampler_seed])

    monkeypatch.setattr(optimize, "optimize_coin", fake_optimize_coin)
    monkeypatch.setattr(optimize, "_persist_result_json", lambda *_args, **_kwargs: None)
    artifact_payloads = []
    monkeypatch.setattr(
        optimize,
        "_persist_paper_candidate_json",
        lambda _coin, payload: artifact_payloads.append(payload) or "artifact.json",
    )

    result = optimize.optimize_coin_multiseed(
        all_data={},
        coin_prefix="BIP",
        coin_name="BTC",
        sampler_seeds=[101, 202],
    )

    assert result is not None
    assert result["params"] == seed_results[101]["params"]
    assert result["holdout_metrics"] == seed_results[101]["holdout_metrics"]
    assert result["consensus_params"] != result["params"]
    assert result["consensus_params"] == {"signal_threshold": 0.795, "max_depth": 6}
    assert result["consensus_revalidated"] is False
    assert result["evaluated_params_source"] == "selected_seed"

    assert len(artifact_payloads) == 1
    artifact = artifact_payloads[0]
    assert artifact["evaluated_params"] == result["params"]
    assert artifact["holdout_metrics"] == result["holdout_metrics"]


def test_multiseed_records_consensus_metadata_when_no_consensus(monkeypatch):
    seed_results = {
        11: {
            "coin": "BTC",
            "optim_score": 1.0,
            "params": {},
            "holdout_metrics": {"holdout_sharpe": 0.2, "holdout_trades": 18, "holdout_return": 0.02},
            "deployment_blocked": False,
            "selection_meta": {},
        },
        22: {
            "coin": "BTC",
            "optim_score": 0.5,
            "params": {},
            "holdout_metrics": {"holdout_sharpe": 0.1, "holdout_trades": 15, "holdout_return": 0.01},
            "deployment_blocked": False,
            "selection_meta": {},
        },
    }

    def fake_optimize_coin(_all_data, _coin_prefix, _coin_name, sampler_seed, **_kwargs):
        return deepcopy(seed_results[sampler_seed])

    monkeypatch.setattr(optimize, "optimize_coin", fake_optimize_coin)
    monkeypatch.setattr(optimize, "_persist_result_json", lambda *_args, **_kwargs: None)

    try:
        optimize.optimize_coin_multiseed(
            all_data={},
            coin_prefix="BIP",
            coin_name="BTC",
            sampler_seeds=[11, 22],
        )
    except ValueError as exc:
        assert "params missing" in str(exc)
    else:
        raise AssertionError("Expected guardrail to reject empty selected-seed params")


def test_multiseed_artifact_requires_pass_not_blocked_and_min_tier(monkeypatch):
    seed_results = {
        1: {
            "coin": "BTC",
            "optim_score": 2.0,
            "params": {"signal_threshold": 0.72},
            "holdout_metrics": {"holdout_sharpe": 0.4, "holdout_trades": 2, "holdout_return": 0.05},
            "deployment_blocked": False,
            "selection_meta": {},
            "research_confidence_tier": "SCREENED",
        },
        2: {
            "coin": "BTC",
            "optim_score": 1.0,
            "params": {"signal_threshold": 0.70},
            "holdout_metrics": {"holdout_sharpe": 0.2, "holdout_trades": 2, "holdout_return": 0.01},
            "deployment_blocked": False,
            "selection_meta": {},
            "research_confidence_tier": "SCREENED",
        },
    }

    monkeypatch.setattr(
        optimize,
        "optimize_coin",
        lambda _all_data, _coin_prefix, _coin_name, sampler_seed, **_kwargs: deepcopy(seed_results[sampler_seed]),
    )
    monkeypatch.setattr(optimize, "_persist_result_json", lambda *_args, **_kwargs: None)
    artifact_payloads = []
    monkeypatch.setattr(
        optimize,
        "_persist_paper_candidate_json",
        lambda _coin, payload: artifact_payloads.append(payload) or "artifact.json",
    )

    result = optimize.optimize_coin_multiseed({}, "BIP", "BTC", sampler_seeds=[1, 2])

    assert result is not None
    assert artifact_payloads == []
