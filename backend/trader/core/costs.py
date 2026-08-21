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
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Protocol

logger = logging.getLogger(__name__)

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
    'HYPE': 10,
    'ONDO': 1000,
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
    'HYP': 'HYPE',
    'OND': 'ONDO',
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


def resolve_base(symbol: str) -> Optional[str]:
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


_resolve_base = resolve_base  # historical name


# CME-style futures month codes, which CDE follows. Not a lookup anyone should
# reinvent inline: getting one letter wrong resolves to a contract that exists
# and is the wrong month.
FUTURES_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z',
}

_CDE_PRODUCT = re.compile(
    r'^(?P<code>[A-Z0-9]+)-(?P<day>\d{1,2})(?P<month>[A-Z]{3})(?P<year>\d{2})-CDE$',
    re.IGNORECASE,
)
_MONTH_NAMES = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12,
}


# The Coinbase *spot* product quoting each traded underlying. Spot is the
# reference venue this account can actually reach — the offshore perp venues are
# 451 from a US IP — and it is the market the perp's index is built from, so its
# basis is what drives funding.
#
# Explicit rather than derived: the meme contracts are keyed `1000PEPE` and
# `1000SHIB` here because that is how their contract units are quoted, while
# spot is plain `PEPE-USD` / `SHIB-USD`. A `.replace('1000', '')` would work
# today and break on the first asset with 1000 in its real ticker.
SPOT_PRODUCTS: dict[str, str] = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
    'XRP': 'XRP-USD',
    'DOGE': 'DOGE-USD',
    'AVAX': 'AVAX-USD',
    'ADA': 'ADA-USD',
    'LINK': 'LINK-USD',
    'LTC': 'LTC-USD',
    'BCH': 'BCH-USD',
    'DOT': 'DOT-USD',
    'NEAR': 'NEAR-USD',
    'SUI': 'SUI-USD',
    'XLM': 'XLM-USD',
    '1000PEPE': 'PEPE-USD',
    '1000SHIB': 'SHIB-USD',
    'HYPE': 'HYPE-USD',
    'ONDO': 'ONDO-USD',
}


def spot_product(symbol: str) -> Optional[str]:
    """The Coinbase spot product for whatever spelling names an instrument.

    Takes a CDE product id, a contract code or a bare base — anything
    `resolve_base` understands — and returns the spot product id, or None for an
    underlying with no spot listing.
    """
    base = resolve_base(symbol)
    return SPOT_PRODUCTS.get(base) if base else None


def spot_universe(symbols: Iterable[str]) -> list[str]:
    """Spot products for a set of instruments, deduplicated and ordered.

    This exists so the spot scrape is never hand-typed. Written by hand once, it
    listed nine products against the sixteen the trader models — the API and
    frontend serve nine, and that list got mistaken for the traded universe — so
    seven instruments would have silently had no cross-venue features.
    """
    out: list[str] = []
    for symbol in symbols:
        product = spot_product(symbol)
        if product and product not in out:
            out.append(product)
    return out


def psf_symbol(product_id: str) -> Optional[str]:
    """Convert a CDE product id to the Perp Style Futures symbol.

    `BIP-20DEC30-CDE` -> `BIPZ30`. The two name the same instrument, but the
    Coinbase Derivatives historical-funding endpoint keys on the PSF form while
    the Advanced Trade product and candle endpoints key on the long form — so
    the scraper needs both spellings for one contract.

    Returns None for anything that is not a decorated CDE product id, including
    the bare codes and `*-PERP` spellings, because there is nothing to derive an
    expiry from.
    """
    match = _CDE_PRODUCT.match((product_id or '').strip().upper())
    if match is None:
        return None
    month = _MONTH_NAMES.get(match.group('month').upper())
    if month is None:
        return None
    return f"{match.group('code')}{FUTURES_MONTH_CODES[month]}{match.group('year')}"


def contract_size_disagreements(
    declared: dict[str, float] | None,
) -> dict[str, tuple[float, float]]:
    """Where a venue schedule's contract sizes differ from `CONTRACT_UNITS`.

    Returns {base: (schedule_size, contract_units_size)}.

    `ExchangeCostAssumptions.contract_sizes` is parsed out of every venue file
    and read by nothing — `get_contract_spec` always uses `CONTRACT_UNITS` — so
    a schedule that disagreed had no way to say so. Three instruments do
    disagree in the shipped CDE file (AVAX 2x, LINK 5x, LTC 5x), and contract
    size multiplies into notional, fees, margin, liquidation price and PnL.
    Which side is right is a question for Coinbase's published specs; this
    function exists so the question gets asked.
    """
    out: dict[str, tuple[float, float]] = {}
    for key, size in (declared or {}).items():
        base = resolve_base(key)
        if base is None or base not in CONTRACT_UNITS:
            continue
        ours = float(CONTRACT_UNITS[base])
        if ours != float(size):
            out[base] = (float(size), ours)
    return out


def get_contract_spec(symbol: str) -> ContractSpec:
    """Resolve a symbol (CDE code, ticker, or decorated product id) to its spec."""
    base = resolve_base(symbol)
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
        assumptions = ExchangeCostAssumptions.from_dict(json.load(f), source_path=str(p))

    disagreements = contract_size_disagreements(assumptions.contract_sizes)
    if disagreements:
        logger.warning(
            "%s declares contract sizes that disagree with core.costs "
            "CONTRACT_UNITS, which is what actually sizes every position: %s. "
            "Contract size multiplies into notional, fees, margin, liquidation "
            "price and PnL — check Coinbase's published specs and fix whichever "
            "is wrong",
            p.name,
            ", ".join(
                f"{base} schedule={theirs:g} used={ours:g} ({ours / theirs:.3g}x)"
                for base, (theirs, ours) in sorted(disagreements.items())
            ),
        )
    return assumptions


# ---------------------------------------------------------------------------
# 3. Computation
# ---------------------------------------------------------------------------


class CostParams(Protocol):
    """The subset of Config the cost functions read."""

    fee_pct_per_side: float
    min_fee_per_contract: float
    # Optional per-symbol overrides of the floor. Coinbase CDE bills $0.75 per
    # contract on BTC and ETH but $0.10 on everything else, so a single scalar
    # silently charges every instrument the most expensive rate.
    min_fee_per_contract_by_symbol: dict[str, float]
    slippage_bps: float
    impact_bps_per_contract: float
    impact_max_bps_per_side: float
    apply_slippage: bool
    apply_impact: bool
    apply_funding: bool
    leverage: float


def fee_floor(symbol: str, params: CostParams) -> float:
    """Per-contract commission for `symbol`, falling back to the flat default.

    The venue's schedule is keyed by a mix of CDE product codes and plain
    tickers, so try the symbol as given, then its prefix, then the underlying
    ticker, then any product code for that underlying.
    """
    overrides = getattr(params, 'min_fee_per_contract_by_symbol', None) or {}
    if not overrides:
        return float(params.min_fee_per_contract)

    token = symbol.upper().strip()
    base = resolve_base(token)
    candidates = [token, token.split('-')[0]]
    if base:
        candidates.append(base)
        candidates.extend(code for code, mapped in CDE_CODE_TO_BASE.items() if mapped == base)

    for candidate in candidates:
        if candidate in overrides:
            return float(overrides[candidate])
    return float(params.min_fee_per_contract)


def symbols_missing_fee_schedule(
    symbols: Iterable[str],
    params: CostParams,
) -> list[str]:
    """Symbols with no explicit entry in the loaded per-contract fee schedule.

    Those fall back to the flat default, which for Coinbase CDE is the expensive
    BTC/ETH rate. That errs toward understating profitability rather than
    overstating it, but it is a data gap and callers should say so rather than
    let it pass silently. Resolve it against the venue's published schedule.
    """
    overrides = getattr(params, 'min_fee_per_contract_by_symbol', None) or {}
    if not overrides:
        return []
    default = float(params.min_fee_per_contract)
    return sorted(
        {s for s in symbols if fee_floor(s, params) == default and s.upper() not in overrides}
    )


# ---------------------------------------------------------------------------
# Deliberately absent: trade PnL and position sizing
# ---------------------------------------------------------------------------
#
# This module used to carry `CostBreakdown`, `compute_cost_breakdown`,
# `round_trip_costs`, `entry_fee`, `TradePnL`, `compute_trade_pnl`,
# `kelly_fraction` and `size_position` — roughly 215 lines that nothing outside
# the module ever called. What actually runs is:
#
#   round-trip cost   ->  core/targets.py:round_trip_cost (and the per-bar series)
#   entry fee         ->  core/execution.py:entry_cost
#   Kelly sizing      ->  core/execution.py:fractional_kelly + size_from_forecast
#
# CLAUDE.md calls this module the single source of truth for money, and for the
# parts that remain — contract specs, fee assumptions, the fee floor — it is.
# Two implementations of the parts that were dead was worse than one, because the
# dead copy read as authoritative.
