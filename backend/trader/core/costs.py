"""Execution cost model: contract specs, exchange fee assumptions, and trade PnL.

This is the single source of truth for anything that converts a position into
dollars. It absorbs what used to be split across `core/costs.py`,
`core/trading_costs.py` and `core/execution_sim.py`, plus the fee/sizing helpers
that lived in `scripts/train_model.py`.

Three layers, in dependency order:

1. Contract specs      — how many base units one contract represents.
2. Cost assumptions    — fees/slippage/impact/funding, loaded from configs/exchange/*.json.
3. Computation         — round-trip costs, trade PnL, and position sizing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.profiles import CoinProfile


# ---------------------------------------------------------------------------
# 1. Contract specs
# ---------------------------------------------------------------------------

# Base units per contract, keyed by the underlying ticker. Coinbase CDE sizes,
# user-verified. Only `units` is ever needed to price a position — the old table
# also carried per-symbol `fee_pct`/`min_fee_usd` columns that nothing read,
# because fees come from the exchange cost assumptions below.
CONTRACT_UNITS: dict[str, float] = {
    'BTC': 0.01,
    'ETH': 0.10,
    'XRP': 500,
    'SOL': 5,
    'DOGE': 5000,
    'AVAX': 10,
    'ADA': 1000,
    'LINK': 50,
    'LTC': 5,
    'NEAR': 500,
    'SUI': 500,
    'BCH': 1,
    'XLM': 5000,
    'DOT': 100,
    '1000SHIB': 10000,
    '1000PEPE': 100000,
}

# Coinbase CDE product codes -> underlying ticker.
CDE_CODE_TO_BASE: dict[str, str] = {
    'BIP': 'BTC',
    'ETP': 'ETH',
    'XPP': 'XRP',
    'SLP': 'SOL',
    'DOP': 'DOGE',
    'AVP': 'AVAX',
    'ADP': 'ADA',
    'LNP': 'LINK',
    'LCP': 'LTC',
    'NER': 'NEAR',
    'SUP': 'SUI',
    'BCP': 'BCH',
    'XLP': 'XLM',
    'POP': 'DOT',
    'SHP': '1000SHIB',
    'PEP': '1000PEPE',
}

# Short aliases people actually type for the 1000x meme contracts.
_BASE_ALIASES: dict[str, str] = {
    'SHIB': '1000SHIB',
    'PEPE': '1000PEPE',
}

DEFAULT_UNITS = 1.0


@dataclass(frozen=True)
class ContractSpec:
    """Resolved contract sizing for a symbol."""

    symbol: str
    base: str
    units: float

    @property
    def is_default(self) -> bool:
        return self.base == 'UNKNOWN'

    def notional(self, n_contracts: float, price: float) -> float:
        return float(n_contracts) * self.units * float(price)


def _resolve_base(symbol: str) -> Optional[str]:
    """Map any symbol spelling to an underlying ticker, or None if unknown."""
    token = symbol.upper().strip()

    for candidate in (token, token.split('-')[0]):
        if candidate in CDE_CODE_TO_BASE:
            return CDE_CODE_TO_BASE[candidate]
        if candidate in _BASE_ALIASES:
            return _BASE_ALIASES[candidate]
        if candidate in CONTRACT_UNITS:
            return candidate

    # Fall back to a substring scan for decorated symbols like
    # "BTC-PERP-20DEC30-CDE". Longest key first so '1000SHIB' beats 'SHIB'
    # and the result never depends on dict insertion order.
    keys = list(CDE_CODE_TO_BASE) + list(_BASE_ALIASES) + list(CONTRACT_UNITS)
    for key in sorted(keys, key=len, reverse=True):
        if key in token:
            return CDE_CODE_TO_BASE.get(key) or _BASE_ALIASES.get(key) or key

    return None


def get_contract_spec(symbol: str) -> ContractSpec:
    """Resolve a symbol (CDE code, ticker, or decorated product id) to its spec."""
    base = _resolve_base(symbol)
    if base is None:
        return ContractSpec(symbol=symbol, base='UNKNOWN', units=DEFAULT_UNITS)
    return ContractSpec(symbol=symbol, base=base, units=float(CONTRACT_UNITS[base]))


# ---------------------------------------------------------------------------
# 2. Exchange cost assumptions (configs/exchange/*.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeeAssumptions:
    maker_bps: float = 10.0
    taker_bps: float = 10.0
    min_fee_per_contract: float = 0.0
    use_taker: bool = True

    @property
    def fee_pct_per_side(self) -> float:
        bps = self.taker_bps if self.use_taker else self.maker_bps
        return float(max(bps, 0.0) / 10_000.0)


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
        return float(max(bps, 0.0) / 10_000.0)


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
        retail = payload.get("retail_execution_fee", {})
        exch = payload.get("exchange_fee", {})
        slippage = payload.get("slippage", {})
        impact = payload.get("impact", {})
        funding = payload.get("funding", {})

        maker_bps = float(fees.get("maker_bps", 10.0))
        taker_bps = float(fees.get("taker_bps", 10.0))
        use_taker = bool(fees.get("use_taker", True))

        return cls(
            version=str(payload.get("version", "legacy_default")),
            exchange=str(payload.get("exchange", "unknown")),
            market=str(payload.get("market", "perps")),
            fees=FeeAssumptions(
                maker_bps=maker_bps,
                taker_bps=taker_bps,
                min_fee_per_contract=float(fees.get("min_fee_per_contract", 0.20)),
                use_taker=use_taker,
            ),
            retail_execution_fee=RetailExecutionFeeAssumptions(
                enabled=bool(retail.get("enabled", False)),
                mode=str(retail.get("mode", "bps")),
                taker_fee_bps=float(retail.get("taker_fee_bps", taker_bps)),
                maker_fee_bps=float(retail.get("maker_fee_bps", maker_bps)),
                use_taker=bool(retail.get("use_taker", use_taker)),
            ),
            exchange_fee=ExchangeFeeAssumptions(
                enabled=bool(exch.get("enabled", False)),
                mode=str(exch.get("mode", "per_contract_usd")),
                per_contract_usd=float(exch.get("per_contract_usd", 0.0)),
                symbol_overrides={
                    str(k).upper(): float(v)
                    for k, v in (exch.get("symbol_overrides") or {}).items()
                    if isinstance(k, str)
                },
                participant_type_assumption=exch.get("participant_type_assumption"),
                execution_type_assumption=exch.get("execution_type_assumption"),
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
                interval_hours=int(funding.get("funding_interval_hours", funding.get("interval_hours", 1))),
                method=str(funding.get("method", "default")),
            ),
            execution_fee_mode=str(payload.get("execution_fee_mode", "bps")),
            exchange_fee_mode=str(payload.get("exchange_fee_mode", "per_contract_usd")),
            assumption_profile=str(payload.get("assumption_profile", "legacy")),
            contract_sizes={
                str(k).upper(): float(v) for k, v in (payload.get("contract_sizes") or {}).items()
            },
            observed_ui_fee_bps=(
                float(payload["observed_ui_fee_bps"])
                if payload.get("observed_ui_fee_bps") is not None
                else None
            ),
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
        return {
            "version": self.version,
            "cost_config_id": cost_config_id,
            "config_id": cost_config_id,
            "source_path": self.source_path,
            "exchange": (parts[0] if parts else None) or self.exchange,
            "market": self.market,
            "venue": parts[1] if len(parts) > 1 else None,
            "version_tag": next((t for t in parts if re.match(r"^v\d+", t)), None),
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


def load_exchange_cost_assumptions(path: str | Path) -> ExchangeCostAssumptions:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return ExchangeCostAssumptions.from_dict(json.load(f), source_path=str(p))


# ---------------------------------------------------------------------------
# 3. Computation
# ---------------------------------------------------------------------------


class CostParams(Protocol):
    """The subset of Config the cost functions read."""

    fee_pct_per_side: float
    min_fee_per_contract: float
    slippage_bps: float
    impact_bps_per_contract: float
    impact_max_bps_per_side: float
    apply_slippage: bool
    apply_impact: bool
    apply_funding: bool
    leverage: float


@dataclass(frozen=True)
class CostBreakdown:
    total_cost_dollars: float
    pct_fee_component: float
    min_fee_component: float
    slippage_component: float
    impact_component: float

    @property
    def execution_component(self) -> float:
        """Slippage and impact together — how they are reported downstream."""
        return self.slippage_component + self.impact_component


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
    """Round-trip execution cost in dollars, decomposed by source.

    Per side the fee is `max(pct_of_notional, per_contract_floor)`, matching how
    Coinbase CDE bills. The components are reported so a run can tell whether it
    is being eaten by the percentage fee or by the per-contract floor on small
    notionals.
    """
    entry_pct_fee = entry_notional * max(fee_pct_per_side, 0.0)
    exit_pct_fee = exit_notional * max(fee_pct_per_side, 0.0)
    floor = max(n_contracts, 0) * max(min_fee_per_contract, 0.0)

    entry_fee = max(entry_pct_fee, floor)
    exit_fee = max(exit_pct_fee, floor)
    pct_component = min(entry_pct_fee, entry_fee) + min(exit_pct_fee, exit_fee)
    min_component = max(entry_fee - entry_pct_fee, 0.0) + max(exit_fee - exit_pct_fee, 0.0)

    round_trip_notional = entry_notional + exit_notional

    slip_component = 0.0
    if apply_slippage:
        slip_component = round_trip_notional * max(slippage_bps_per_side, 0.0) / 10_000.0

    impact_component = 0.0
    if apply_impact:
        side_bps = min(
            max(impact_bps_per_contract, 0.0) * max(n_contracts, 0),
            max(impact_max_bps_per_side, 0.0),
        )
        impact_component = round_trip_notional * side_bps / 10_000.0

    return CostBreakdown(
        total_cost_dollars=float(entry_fee + exit_fee + slip_component + impact_component),
        pct_fee_component=float(pct_component),
        min_fee_component=float(min_component),
        slippage_component=float(slip_component),
        impact_component=float(impact_component),
    )


@dataclass(frozen=True)
class TradePnL:
    """Realized PnL for one round trip, in both percent-of-notional and dollars."""

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

    @classmethod
    def empty(cls) -> "TradePnL":
        return cls(*([0.0] * 13))


def entry_fee(n_contracts: int, price: float, symbol: str, params: CostParams) -> float:
    """One-side fee in dollars: percentage of notional, floored per contract."""
    spec = get_contract_spec(symbol)
    pct_fee = spec.notional(n_contracts, price) * params.fee_pct_per_side
    floor = n_contracts * params.min_fee_per_contract
    return float(max(pct_fee, floor))


def round_trip_costs(
    n_contracts: int,
    entry_price: float,
    exit_price: float,
    symbol: str,
    params: CostParams,
) -> CostBreakdown:
    """Round-trip cost for a position, using a Config-shaped params object."""
    spec = get_contract_spec(symbol)
    return compute_cost_breakdown(
        entry_notional=spec.notional(n_contracts, entry_price),
        exit_notional=spec.notional(n_contracts, exit_price),
        n_contracts=int(n_contracts),
        fee_pct_per_side=params.fee_pct_per_side,
        min_fee_per_contract=params.min_fee_per_contract,
        slippage_bps_per_side=params.slippage_bps,
        impact_bps_per_contract=params.impact_bps_per_contract,
        impact_max_bps_per_side=params.impact_max_bps_per_side,
        apply_slippage=params.apply_slippage,
        apply_impact=params.apply_impact,
    )


def compute_trade_pnl(
    *,
    entry_price: float,
    exit_price: float,
    direction: int,
    n_contracts: int,
    symbol: str,
    params: CostParams,
    accum_funding: float = 0.0,
) -> TradePnL:
    """Net PnL for a closed round trip, after fees, slippage, impact and funding."""
    spec = get_contract_spec(symbol)
    notional = spec.notional(n_contracts, entry_price)
    if notional <= 0:
        return TradePnL.empty()

    raw_pnl_pct = ((float(exit_price) - float(entry_price)) / float(entry_price)) * int(direction)
    costs = round_trip_costs(n_contracts, entry_price, exit_price, symbol, params)

    fees = costs.total_cost_dollars
    funding_dollars = (float(accum_funding) * notional) if params.apply_funding else 0.0
    net_dollars = notional * raw_pnl_pct - fees + funding_dollars

    return TradePnL(
        net_pnl_pct=float(net_dollars / notional),
        raw_pnl_pct=float(raw_pnl_pct),
        fee_pnl_pct=float(-fees / notional),
        fee_pct_component_pct=float(-costs.pct_fee_component / notional),
        min_fee_component_pct=float(-costs.min_fee_component / notional),
        slippage_component_pct=float(-costs.execution_component / notional),
        funding_pnl_pct=float(funding_dollars / notional),
        pnl_dollars=float(net_dollars),
        notional=float(notional),
        total_fees_dollars=float(fees),
        pct_fee_component_dollars=float(costs.pct_fee_component),
        min_fee_component_dollars=float(costs.min_fee_component),
        slippage_component_dollars=float(costs.execution_component),
    )


# Position sizing -----------------------------------------------------------

VOL_SCALE_MIN = 0.3
VOL_SCALE_MAX = 1.5
KELLY_FRACTION = 0.5          # half-Kelly
KELLY_POS_SIZE_CAP = 1.5      # cap half-Kelly at 1.5x the profile's base size


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Kelly fraction for a binary payoff. <= 0 means no edge."""
    if win_rate <= 0 or payoff_ratio <= 0:
        return 0.0
    return (payoff_ratio * win_rate - (1.0 - win_rate)) / payoff_ratio


def size_position(
    *,
    equity: float,
    price: float,
    symbol: str,
    params: CostParams,
    profile: Optional["CoinProfile"] = None,
    position_size: float | None = None,
    vol_sizing_target: float | None = None,
    vol_24h: float = 0.0,
) -> int:
    """Contracts to trade, from equity, leverage, volatility and Kelly stats.

    Returns 0 when the position rounds below one contract, or when the profile
    carries calibrated Kelly stats that say there is no edge.
    """
    spec = get_contract_spec(symbol)
    notional_per_contract = spec.units * float(price)
    if notional_per_contract <= 0:
        return 0

    pos_size = position_size if position_size is not None else getattr(profile, 'position_size', 0.15)
    vol_target = (
        vol_sizing_target if vol_sizing_target is not None
        else getattr(profile, 'vol_sizing_target', 0.025)
    )

    # A calibrated Kelly edge overrides the static position size. A non-positive
    # Kelly fraction is a hard skip: the profile's own stats say don't trade.
    if profile is not None and getattr(profile, 'kelly_win_rate', 0.0) > 0 and getattr(profile, 'kelly_payoff_ratio', 0.0) > 0:
        f = kelly_fraction(profile.kelly_win_rate, profile.kelly_payoff_ratio)
        if f <= 0:
            return 0
        pos_size = min(f * KELLY_FRACTION, pos_size * KELLY_POS_SIZE_CAP)

    # Scale down into high volatility, up into low, within bounds.
    if vol_24h > 0 and vol_target > 0:
        pos_size *= min(max(vol_target / vol_24h, VOL_SCALE_MIN), VOL_SCALE_MAX)

    target_notional = float(equity) * pos_size * params.leverage
    return max(int(target_notional / notional_per_contract), 0)
