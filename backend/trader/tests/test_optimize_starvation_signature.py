from scripts.optimize import _derive_starvation_signature


def test_starvation_signature_prefers_momentum_when_momentum_gates_dominate() -> None:
    gate_summary = {
        "gate_rates": {
            "momentum_magnitude": 0.40,
            "momentum_dir_agreement": 0.30,
            "vol_regime_low": 0.10,
            "primary_threshold": 0.05,
            "ensemble_agreement": 0.05,
        }
    }

    sig = _derive_starvation_signature(gate_summary)
    assert sig["label"] == "momentum_starved"
    assert sig["shares"]["momentum_starved"] > sig["shares"]["vol_starved"]


def test_starvation_signature_handles_missing_gate_data() -> None:
    sig = _derive_starvation_signature({"gate_rates": {}})
    assert sig["label"] == "insufficient_signal"
    assert sig["shares"] == {}
