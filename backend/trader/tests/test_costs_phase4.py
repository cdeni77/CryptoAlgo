from pathlib import Path

from core.costs import compute_cost_breakdown, load_exchange_cost_assumptions
from scripts.optimize import _build_cost_config
from scripts.train_model import Config, calculate_pnl_exact


def _exchange_config_path(filename: str) -> Path:
    for base in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
        candidate = base / "configs" / "exchange" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate exchange config '{filename}' from {__file__}")


def test_cost_breakdown_components_are_explicit() -> None:
    breakdown = compute_cost_breakdown(
        entry_notional=10_000.0,
        exit_notional=10_200.0,
        n_contracts=20,
        fee_pct_per_side=0.001,
        min_fee_per_contract=0.20,
        slippage_bps_per_side=2.0,
        impact_bps_per_contract=0.01,
        impact_max_bps_per_side=5.0,
        apply_slippage=True,
        apply_impact=True,
    )
    assert breakdown.total_cost_dollars > 0
    assert breakdown.pct_fee_component >= 0
    assert breakdown.min_fee_component >= 0
    assert breakdown.slippage_component > 0
    assert breakdown.impact_component > 0


def test_exchange_cost_config_load_and_metadata() -> None:
    cfg_path = _exchange_config_path("binance_perps_v202602.json")
    assumptions = load_exchange_cost_assumptions(cfg_path)
    config, metadata = _build_cost_config(assumptions)

    assert metadata["version"] == "binance_perps_v202602"
    assert metadata["cost_config_id"] == "binance_perps_v202602"
    assert metadata["source_path"].endswith("binance_perps_v202602.json")
    assert metadata["applied"]["impact"] is True
    assert config.cost_config_version == "binance_perps_v202602"
    assert config.apply_impact is True


def test_coinbase_retail_config_uses_bps_execution_fee_path() -> None:
    cfg_path = _exchange_config_path("coinbase_us_perps_retail_v202602.json")
    assumptions = load_exchange_cost_assumptions(cfg_path)
    _, metadata = _build_cost_config(assumptions)

    breakdown = compute_cost_breakdown(
        entry_notional=1000.0,
        exit_notional=1000.0,
        n_contracts=1,
        fee_pct_per_side=assumptions.effective_fee_pct_per_side(),
        min_fee_per_contract=assumptions.effective_min_fee_per_contract("BIP"),
        slippage_bps_per_side=0.0,
        impact_bps_per_contract=0.0,
        impact_max_bps_per_side=0.0,
        apply_slippage=False,
        apply_impact=False,
    )

    assert assumptions.effective_fee_pct_per_side() == 0.001
    assert assumptions.effective_min_fee_per_contract("BIP") == 0.0
    assert breakdown.pct_fee_component == 2.0
    assert metadata["execution_fee_mode"] == "bps"
    assert metadata["exchange_fee_mode"] == "per_contract_usd"
    assert metadata["funding_interval_hours"] == 1
    assert metadata["assumption_profile"] == "retail"
    assert metadata["observed_ui_fee_bps"] == 10.0


def test_coinbase_cde_config_uses_symbol_specific_per_contract_fees() -> None:
    cfg_path = _exchange_config_path("coinbase_us_perps_cde_v202602.json")
    assumptions = load_exchange_cost_assumptions(cfg_path)
    _, metadata = _build_cost_config(assumptions)

    assert assumptions.effective_fee_pct_per_side() == 0.0
    assert assumptions.effective_min_fee_per_contract("BIP") == 0.75
    assert assumptions.effective_min_fee_per_contract("DOP") == 0.10
    assert assumptions.funding.interval_hours == 1
    assert assumptions.funding.method == "coinbase_us_perps_hourly"
    assert metadata["assumption_profile"] == "cde"
    assert metadata["participant_type_assumption"] == "non_professional"
    assert metadata["execution_type_assumption"] == "electronic"


def test_smoke_gross_and_net_metrics_populate() -> None:
    config = Config()
    net_pnl, raw_pnl, *_ = calculate_pnl_exact(
        entry_price=100.0,
        exit_price=102.0,
        direction=1,
        accum_funding=0.0,
        n_contracts=10,
        symbol="BTC",
        config=config,
    )
    assert isinstance(raw_pnl, float)
    assert isinstance(net_pnl, float)
    assert raw_pnl != 0.0
    assert net_pnl != raw_pnl
