import json
from pathlib import Path

from scripts.optimize import aggregate_cumulative_trial_counts
from scripts.validate_robustness import resolve_dsr_trial_count


def test_trial_ledger_aggregation_and_dsr_scope(tmp_path: Path) -> None:
    ledger = tmp_path / "trial_ledger.jsonl"
    entries = [
        {
            "coin": "BTC",
            "preset": "paper_ready",
            "run_id": "btc-1",
            "timestamp": "2026-01-01T00:00:00",
            "completed_trials": 50,
        },
        {
            "coin": "ETH",
            "preset": "paper_ready",
            "run_id": "eth-1",
            "timestamp": "2026-01-01T01:00:00",
            "completed_trials": 30,
        },
        {
            "coin": "BTC",
            "preset": "quick",
            "run_id": "btc-2",
            "timestamp": "2026-01-02T01:00:00",
            "completed_trials": 20,
        },
    ]

    with open(ledger, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    counts = aggregate_cumulative_trial_counts(ledger)
    assert counts["coin_totals"]["BTC"] == 70
    assert counts["coin_totals"]["ETH"] == 30
    assert counts["global_total"] == 100
    assert counts["ledger_timestamp"] == "2026-01-02T01:00:00"

    coin_scope = resolve_dsr_trial_count("BTC", scope="coin", ledger_path=ledger)
    global_scope = resolve_dsr_trial_count("BTC", scope="global", ledger_path=ledger)

    assert coin_scope["n_trials_used"] == 70
    assert coin_scope["scope"] == "coin"
    assert global_scope["n_trials_used"] == 100
    assert global_scope["scope"] == "global"
