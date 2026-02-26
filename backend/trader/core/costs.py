from __future__ import annotations

import json
import re
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
class RetailExecutionFeeAssumptions:
    enabled: bool = False
    mode: str = "bps"
    taker_fee_bps: float = 10.0
    maker_fee_bps: float = 10.0
    use_taker: bool = True

    @property
    def fee_pct_per_side(self) -> float:
        if not self.enabled:
            return 0.0
        bps = self.taker_fee_bps if self.use_taker else self.maker_fee_bps
        return float(max(bps, 0.0) / 10000.0)


@dataclass(frozen=True)
class ExchangeFeeAssumptions:
    enabled: bool = False
    mode: str = "per_contract_usd"
    per_contract_usd: float = 0.0
    symbol_overrides: dict[str, float] | None = None
    participant_type_assumption: str | None = None
    execution_type_assumption: str | None = None

    def per_contract_for_symbol(self, symbol: str | None = None) -> float:
        if not self.enabled:
            return 0.0
        if symbol and self.symbol_overrides:
            return float(self.symbol_overrides.get(symbol.upper(), self.per_contract_usd))
        return float(self.per_contract_usd)


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
    method: str = "default"


@dataclass(frozen=True)
class ExchangeCostAssumptions:
    version: str
    exchange: str
    market: str
    fees: FeeAssumptions
    retail_execution_fee: RetailExecutionFeeAssumptions
    exchange_fee: ExchangeFeeAssumptions
    slippage: SlippageAssumptions
    impact: ImpactAssumptions
    funding: FundingAssumptions
    execution_fee_mode: str = "bps"
    exchange_fee_mode: str = "per_contract_usd"
    assumption_profile: str = "legacy"
    contract_sizes: dict[str, float] | None = None
    observed_ui_fee_bps: float | None = None
    observed_ui_fee_source: str | None = None
    source_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source_path: str | None = None) -> "ExchangeCostAssumptions":
        fees = payload.get("fees", {})
        retail_execution_fee = payload.get("retail_execution_fee", {})
        exchange_fee = payload.get("exchange_fee", {})
        slippage = payload.get("slippage", {})
        impact = payload.get("impact", {})
        funding = payload.get("funding", {})

        default_maker_bps = float(fees.get("maker_bps", 10.0))
        default_taker_bps = float(fees.get("taker_bps", 10.0))
        default_use_taker = bool(fees.get("use_taker", True))

        funding_interval_hours = int(funding.get("funding_interval_hours", funding.get("interval_hours", 1)))

        symbol_overrides_payload = exchange_fee.get("symbol_overrides", {}) or {}
        symbol_overrides = {
            str(k).upper(): float(v)
            for k, v in symbol_overrides_payload.items()
            if isinstance(k, str)
        }

        return cls(
            version=str(payload.get("version", "legacy_default")),
            exchange=str(payload.get("exchange", "unknown")),
            market=str(payload.get("market", "perps")),
            fees=FeeAssumptions(
                maker_bps=default_maker_bps,
                taker_bps=default_taker_bps,
                min_fee_per_contract=float(fees.get("min_fee_per_contract", 0.20)),
                use_taker=default_use_taker,
            ),
            retail_execution_fee=RetailExecutionFeeAssumptions(
                enabled=bool(retail_execution_fee.get("enabled", False)),
                mode=str(retail_execution_fee.get("mode", "bps")),
                taker_fee_bps=float(retail_execution_fee.get("taker_fee_bps", default_taker_bps)),
                maker_fee_bps=float(retail_execution_fee.get("maker_fee_bps", default_maker_bps)),
                use_taker=bool(retail_execution_fee.get("use_taker", default_use_taker)),
            ),
            exchange_fee=ExchangeFeeAssumptions(
                enabled=bool(exchange_fee.get("enabled", False)),
                mode=str(exchange_fee.get("mode", "per_contract_usd")),
                per_contract_usd=float(exchange_fee.get("per_contract_usd", 0.0)),
                symbol_overrides=symbol_overrides,
                participant_type_assumption=exchange_fee.get("participant_type_assumption"),
                execution_type_assumption=exchange_fee.get("execution_type_assumption"),
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
                interval_hours=funding_interval_hours,
                method=str(funding.get("method", "default")),
            ),
            execution_fee_mode=str(payload.get("execution_fee_mode", "bps")),
            exchange_fee_mode=str(payload.get("exchange_fee_mode", "per_contract_usd")),
            assumption_profile=str(payload.get("assumption_profile", "legacy")),
            contract_sizes={str(k).upper(): float(v) for k, v in (payload.get("contract_sizes", {}) or {}).items()},
            observed_ui_fee_bps=float(payload["observed_ui_fee_bps"]) if payload.get("observed_ui_fee_bps") is not None else None,
            observed_ui_fee_source=payload.get("observed_ui_fee_source"),
            source_path=source_path,
        )

    def effective_fee_pct_per_side(self) -> float:
        if self.retail_execution_fee.enabled and self.retail_execution_fee.mode == "bps":
            return self.retail_execution_fee.fee_pct_per_side
        return self.fees.fee_pct_per_side

    def effective_min_fee_per_contract(self, symbol: str | None = None) -> float:
        if self.exchange_fee.enabled and self.exchange_fee.mode == "per_contract_usd":
            return self.exchange_fee.per_contract_for_symbol(symbol)
        return float(self.fees.min_fee_per_contract)

    def to_metadata(self) -> dict[str, Any]:
        cost_config_id = Path(self.source_path).stem if self.source_path else self.version
        parts = cost_config_id.split("_") if cost_config_id else []
        parsed_exchange = parts[0] if len(parts) > 0 else None
        venue = parts[1] if len(parts) > 1 else None
        version_tag = next((token for token in parts if re.match(r"^v\d+", token)), None)
        return {
            "version": self.version,
            "cost_config_id": cost_config_id,
            "config_id": cost_config_id,
            "source_path": self.source_path,
            "exchange": parsed_exchange or self.exchange,
            "market": self.market,
            "venue": venue,
            "version_tag": version_tag,
            "execution_fee_mode": self.execution_fee_mode,
            "exchange_fee_mode": self.exchange_fee_mode,
            "funding_interval_hours": int(self.funding.interval_hours),
            "assumption_profile": self.assumption_profile,
            "observed_ui_fee_bps": self.observed_ui_fee_bps,
            "observed_ui_fee_source": self.observed_ui_fee_source,
            "participant_type_assumption": self.exchange_fee.participant_type_assumption,
            "execution_type_assumption": self.exchange_fee.execution_type_assumption,
            "applied": {
                "funding": bool(self.funding.enabled),
                "slippage": bool(self.slippage.enabled),
                "impact": bool(self.impact.enabled),
                "retail_execution_fee": bool(self.retail_execution_fee.enabled),
                "exchange_fee": bool(self.exchange_fee.enabled),
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
