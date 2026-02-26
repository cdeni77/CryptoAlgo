from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeeAssumptions:
    maker_bps: float = 10.0
    taker_bps: float = 10.0
    min_fee_per_contract: float = 0.20
    use_taker: bool = True

    @property
    def fee_pct_per_side(self) -> float:
        bps = self.taker_bps if self.use_taker else self.maker_bps
        return float(max(bps, 0.0) / 10000.0)


@dataclass(frozen=True)
class SlippageAssumptions:
    enabled: bool = True
    bps_per_side: float = 2.0


@dataclass(frozen=True)
class ImpactAssumptions:
    enabled: bool = False
    bps_per_contract: float = 0.0
    max_bps_per_side: float = 10.0


@dataclass(frozen=True)
class FundingAssumptions:
    enabled: bool = True
    interval_hours: int = 1


@dataclass(frozen=True)
class ExchangeCostAssumptions:
    version: str
    exchange: str
    market: str
    fees: FeeAssumptions
    slippage: SlippageAssumptions
    impact: ImpactAssumptions
    funding: FundingAssumptions
    source_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source_path: str | None = None) -> "ExchangeCostAssumptions":
        fees = payload.get("fees", {})
        slippage = payload.get("slippage", {})
        impact = payload.get("impact", {})
        funding = payload.get("funding", {})
        return cls(
            version=str(payload.get("version", "legacy_default")),
            exchange=str(payload.get("exchange", "unknown")),
            market=str(payload.get("market", "perps")),
            fees=FeeAssumptions(
                maker_bps=float(fees.get("maker_bps", 10.0)),
                taker_bps=float(fees.get("taker_bps", 10.0)),
                min_fee_per_contract=float(fees.get("min_fee_per_contract", 0.20)),
                use_taker=bool(fees.get("use_taker", True)),
            ),
            slippage=SlippageAssumptions(
                enabled=bool(slippage.get("enabled", True)),
                bps_per_side=float(slippage.get("bps_per_side", 2.0)),
            ),
            impact=ImpactAssumptions(
                enabled=bool(impact.get("enabled", False)),
                bps_per_contract=float(impact.get("bps_per_contract", 0.0)),
                max_bps_per_side=float(impact.get("max_bps_per_side", 10.0)),
            ),
            funding=FundingAssumptions(
                enabled=bool(funding.get("enabled", True)),
                interval_hours=int(funding.get("interval_hours", 1)),
            ),
            source_path=source_path,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source_path": self.source_path,
            "exchange": self.exchange,
            "market": self.market,
            "applied": {
                "funding": bool(self.funding.enabled),
                "slippage": bool(self.slippage.enabled),
                "impact": bool(self.impact.enabled),
            },
        }


@dataclass(frozen=True)
class CostBreakdown:
    total_cost_dollars: float
    pct_fee_component: float
    min_fee_component: float
    slippage_component: float
    impact_component: float


def load_exchange_cost_assumptions(path: str | Path) -> ExchangeCostAssumptions:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return ExchangeCostAssumptions.from_dict(payload, source_path=str(p))


def compute_cost_breakdown(
    *,
    entry_notional: float,
    exit_notional: float,
    n_contracts: int,
    fee_pct_per_side: float,
    min_fee_per_contract: float,
    slippage_bps_per_side: float,
    impact_bps_per_contract: float,
    impact_max_bps_per_side: float,
    apply_slippage: bool,
    apply_impact: bool,
) -> CostBreakdown:
    entry_pct_fee = entry_notional * max(fee_pct_per_side, 0.0)
    exit_pct_fee = exit_notional * max(fee_pct_per_side, 0.0)
    entry_min_fee = max(n_contracts, 0) * max(min_fee_per_contract, 0.0)
    exit_min_fee = max(n_contracts, 0) * max(min_fee_per_contract, 0.0)

    entry_fee = max(entry_pct_fee, entry_min_fee)
    exit_fee = max(exit_pct_fee, exit_min_fee)
    pct_component = min(entry_pct_fee, entry_fee) + min(exit_pct_fee, exit_fee)
    min_component = max(entry_fee - entry_pct_fee, 0.0) + max(exit_fee - exit_pct_fee, 0.0)

    slip_component = 0.0
    if apply_slippage:
        slip_component = (entry_notional + exit_notional) * max(slippage_bps_per_side, 0.0) / 10000.0

    impact_component = 0.0
    if apply_impact:
        side_bps = min(max(impact_bps_per_contract, 0.0) * max(n_contracts, 0), max(impact_max_bps_per_side, 0.0))
        impact_component = (entry_notional + exit_notional) * side_bps / 10000.0

    total = entry_fee + exit_fee + slip_component + impact_component
    return CostBreakdown(
        total_cost_dollars=float(total),
        pct_fee_component=float(pct_component),
        min_fee_component=float(min_component),
        slippage_component=float(slip_component),
        impact_component=float(impact_component),
    )

