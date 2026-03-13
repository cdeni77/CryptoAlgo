from __future__ import annotations

from dataclasses import dataclass

from core.costs import compute_cost_breakdown
from core.trading_costs import get_contract_spec


@dataclass(frozen=True)
class ExecutionPnlBreakdown:
    net_pnl_pct: float
    raw_pnl_pct: float
    fee_pnl_pct: float
    fee_pct_component_pct: float
    min_fee_component_pct: float
    slippage_component_pct: float
    funding_pnl_pct: float
    pnl_dollars: float
    notional: float
    total_fees_dollars: float
    pct_fee_component_dollars: float
    min_fee_component_dollars: float
    slippage_component_dollars: float


def compute_trade_execution_pnl(
    *,
    entry_price: float,
    exit_price: float,
    direction: int,
    accum_funding: float,
    n_contracts: int,
    symbol: str,
    fee_pct_per_side: float,
    min_fee_per_contract: float,
    slippage_bps: float,
    apply_funding: bool,
    apply_slippage: bool,
    apply_impact: bool,
    impact_bps_per_contract: float,
    impact_max_bps_per_side: float,
) -> ExecutionPnlBreakdown:
    spec = get_contract_spec(symbol)
    notional_per_contract = float(spec['units']) * float(entry_price)
    total_notional = float(n_contracts) * notional_per_contract
    if total_notional <= 0:
        return ExecutionPnlBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    raw_pnl_pct = ((float(exit_price) - float(entry_price)) / float(entry_price)) * int(direction)
    raw_pnl_dollars = total_notional * raw_pnl_pct

    entry_notional = float(n_contracts) * float(spec['units']) * float(entry_price)
    exit_notional = float(n_contracts) * float(spec['units']) * float(exit_price)
    cost_breakdown = compute_cost_breakdown(
        entry_notional=entry_notional,
        exit_notional=exit_notional,
        n_contracts=int(n_contracts),
        fee_pct_per_side=float(fee_pct_per_side),
        min_fee_per_contract=float(min_fee_per_contract),
        slippage_bps_per_side=float(slippage_bps),
        impact_bps_per_contract=float(impact_bps_per_contract),
        impact_max_bps_per_side=float(impact_max_bps_per_side),
        apply_slippage=bool(apply_slippage),
        apply_impact=bool(apply_impact),
    )

    total_fee_dollars = float(cost_breakdown.total_cost_dollars)
    funding_dollars = (float(accum_funding) * total_notional) if apply_funding else 0.0
    net_pnl_dollars = raw_pnl_dollars - total_fee_dollars + funding_dollars

    fee_pct_component = float(cost_breakdown.pct_fee_component)
    min_fee_component = float(cost_breakdown.min_fee_component)
    slippage_component = float(cost_breakdown.slippage_component + cost_breakdown.impact_component)
    fee_pnl_pct = -(total_fee_dollars / total_notional)

    return ExecutionPnlBreakdown(
        net_pnl_pct=float(net_pnl_dollars / total_notional),
        raw_pnl_pct=float(raw_pnl_pct),
        fee_pnl_pct=float(fee_pnl_pct),
        fee_pct_component_pct=float(-(fee_pct_component / total_notional)),
        min_fee_component_pct=float(-(min_fee_component / total_notional)),
        slippage_component_pct=float(-(slippage_component / total_notional)),
        funding_pnl_pct=float(funding_dollars / total_notional),
        pnl_dollars=float(net_pnl_dollars),
        notional=float(total_notional),
        total_fees_dollars=float(total_fee_dollars),
        pct_fee_component_dollars=float(fee_pct_component),
        min_fee_component_dollars=float(min_fee_component),
        slippage_component_dollars=float(slippage_component),
    )

