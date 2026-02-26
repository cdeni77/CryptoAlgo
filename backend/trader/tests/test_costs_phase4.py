from pathlib import Path

from core.costs import compute_cost_breakdown, load_exchange_cost_assumptions
from scripts.optimize import _build_cost_config
from scripts.train_model import Config, calculate_pnl_exact


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
    cfg_path = Path(__file__).resolve().parents[3] / "configs/exchange/binance_perps_v202602.json"
    assumptions = load_exchange_cost_assumptions(cfg_path)
    config, metadata = _build_cost_config(assumptions)

    assert metadata["version"] == "binance_perps_v202602"
    assert metadata["source_path"].endswith("binance_perps_v202602.json")
    assert metadata["applied"]["impact"] is True
    assert config.cost_config_version == "binance_perps_v202602"
    assert config.apply_impact is True


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
