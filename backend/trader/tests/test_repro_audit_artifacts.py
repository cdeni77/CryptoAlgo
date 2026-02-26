import json
from argparse import Namespace
from pathlib import Path

from core.reason_codes import ALL_REASON_CODES, ReasonCode
from scripts.optimize import EventLogger
from scripts.parallel_launch import build_run_manifest


def test_manifest_schema_smoke_required_keys() -> None:
    args = Namespace(
        preset="paper_ready",
        sampler_seeds="42,1337",
        holdout_mode="multi_slice",
        require_holdout_pass=True,
        holdout_candidates=3,
        gate_mode="initial_paper_qualification",
        min_psr=0.55,
        min_psr_cv=None,
        min_psr_holdout=None,
        min_dsr=None,
        proxy_fidelity_candidates=3,
        proxy_fidelity_eval_days=90,
    )
    manifest = build_run_manifest(
        script_dir=Path("backend/trader/scripts"),
        args=args,
        target_coins=["BTC", "ETH"],
        run_id="run-123",
    )
    for key in (
        "run_id",
        "generated_at",
        "preset",
        "target_coins",
        "seed_policy",
        "optimizer_flags",
    ):
        assert key in manifest


def test_structured_reject_log_serialization(tmp_path) -> None:
    path = tmp_path / "rejects.jsonl"
    logger = EventLogger(path)
    logger.emit(
        {
            "event_type": "reject",
            "coin": "BTC",
            "trial_number": 7,
            "stage": "fold_eval",
            "reason_code": ReasonCode.TOO_FEW_TRADES,
            "metrics": {"n_trades": 1, "threshold": 4},
        }
    )

    row = json.loads(path.read_text().strip())
    assert row["coin"] == "BTC"
    assert row["trial_number"] == 7
    assert row["reason_code"] == ReasonCode.TOO_FEW_TRADES
    assert "timestamp" in row


def test_reason_code_import_smoke() -> None:
    assert ReasonCode.TOO_FEW_TRADES in ReasonCode
    assert ReasonCode.TOO_FEW_TRADES.value in ALL_REASON_CODES
