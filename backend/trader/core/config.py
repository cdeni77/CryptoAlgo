"""Run configuration: one dataclass, resolved from defaults, env and CLI.

Precedence for any tunable is always the same, and lives in `Config.resolve`:

    CLI flag  >  per-coin profile  >  Config default

`cli_overrides` records which fields the user actually passed on the command
line, so a flag can beat a profile while an untouched default cannot — that is
what `resolve` consults, and it is why `dataclasses.replace` alone is not enough
to override a per-coin value.

Flags live in `scripts/_common.py`, next to the code that reads them. A
declarative `CLI_PARAMS`/`ENV_PARAMS` layer used to be declared here and was
called by nothing; see the note at the end of this module.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING

from core.costs import load_exchange_cost_assumptions

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.profiles import CoinProfile


CALIBRATION_STRATEGIES = ('platt', 'isotonic', 'beta')
FILTER_MODES = ('hard', 'soft', 'off')
TRADE_FREQ_BUCKETS = ('conservative', 'balanced', 'aggressive')


# Where a venue fee schedule can be found. Searched in order, because the same
# code runs from a repo checkout (cwd anywhere) and from the container image
# (/app), and a single computed relative path got this wrong in the container:
# `configs/` used to live above the build context, so it was never copied into
# the image and every containerised run silently priced contracts at the
# hardcoded 10bp/side.
COST_CONFIG_SEARCH_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parent.parent / 'configs' / 'exchange',
    Path('/app/configs/exchange'),
    Path('configs/exchange'),
)

DEFAULT_COST_CONFIG_NAME = 'coinbase_us_perps_cde_v202602.json'


def find_cost_config(name: str = DEFAULT_COST_CONFIG_NAME) -> Optional[Path]:
    """Locate a fee schedule by name, or return None.

    An absolute or already-valid relative path is used as given; a bare filename
    is looked up in the search paths. Returning None rather than raising is
    deliberate: the caller decides whether a missing schedule is fatal, and it
    should say so loudly rather than fall through to a default nobody chose.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        return candidate if candidate.exists() else None
    for directory in COST_CONFIG_SEARCH_PATHS:
        found = directory / candidate.name
        if found.exists():
            return found
    return None


@dataclass
class Config:
    """Global run settings. Per-coin values live in `core.profiles.CoinProfile`."""

    # --- Walk-forward windows ---

    # --- Entry filters (profiles override per coin) ---
    signal_threshold: float = 0.80
    # Classification-era: a probability threshold plus this margin. The forecast
    # is a return now, so the decision path uses `min_edge_over_cost` instead —
    # reusing this as a return threshold demanded 200bp of expected net, which no
    # hourly forecast will ever clear.
    # Expected net return must exceed the round-trip cost by this multiple again.
    # Expressed relative to cost rather than as an absolute, because cost ranges
    # from ~5bp on the group-B contracts to ~54bp on ETH: an absolute floor would
    # be trivially met on one and unreachable on the other. At 0.5, DOGE needs
    # ~2.5bp of forecast edge and ETH needs ~27bp.
    min_edge_over_cost: float = 0.5
    # Minimum forecast-to-risk ratio before a position is worth taking: a
    # forecast smaller than a fraction of its own uncertainty is noise, whatever
    # its sign. Lived as a module constant in `core/signal.py`, which made it the
    # one gate threshold no caller could sweep — so a sensitivity run had to
    # monkeypatch it. Every other threshold here is a field; this is now one too.
    min_edge_to_risk: float = 0.05
    min_momentum_magnitude: float = 0.07
    max_ensemble_std: float = 0.12
    min_directional_agreement: float = 0.67
    meta_probability_threshold: float = 0.57

    # --- Regime filter ---
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06

    # --- Directional macro filter policies ---

    # --- Exits ---
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    max_hold_hours: int = 96
    cooldown_hours: float = 24.0

    # --- Risk / sizing ---
    max_positions: int = 5
    position_size: float = 0.15
    # Float, not int: `--leverage 1.5` is a reasonable thing for an operator to
    # want, and `execution.py` already casts to float at both use sites.
    leverage: float = 4.0
    vol_sizing_target: float = 0.025
    min_equity: float = 1000.0

    # --- Execution costs (see core.costs; satisfies its CostParams protocol) ---
    # 0.10% of notional plus $0.12 per contract, which is what Coinbase's own
    # order ticket charges — measured, not assumed, on three contracts spanning
    # 3.2x in notional (see `core.costs.per_contract_fee`). The two are added,
    # not maxed. These defaults used to be percentage-only, so an unconfigured
    # run was systematically cheap by the commission's share of notional: 1.5bp
    # a side on a $782 BIP contract, 5bp on a $242 ETP one.
    fee_pct_per_side: float = 0.0010
    per_contract_fee_usd: float = 0.12
    # Per-symbol overrides of the per-contract commission. Empty means the
    # scalar above applies everywhere, which is what the app was measured doing;
    # a venue that bills by instrument group fills this in.
    per_contract_fee_by_symbol: dict[str, float] = field(default_factory=dict)
    slippage_bps: float = 2.0
    # Half-spread crossed on entry and exit. Threaded through the backtest as an
    # argument before, which meant `cost_stress`'s "3x slippage" scenario had
    # nothing to multiply — `core/execution.py` reads this, never `slippage_bps`.
    spread_bps: float = 4.0
    apply_funding: bool = True
    apply_slippage: bool = True
    apply_impact: bool = False
    impact_bps_per_contract: float = 0.0
    impact_max_bps_per_side: float = 10.0
    # Which schedule was loaded, for provenance. Set by
    # `with_cost_assumptions`; reported alongside the version.
    cost_config_path: Optional[str] = None
    cost_config_version: str = 'legacy_default'

    # --- Labels ---
    label_forward_hours: int = 24
    label_vol_target: float = 1.8

    # --- Model / validation ---
    min_val_auc: float = 0.54
    recency_half_life_days: float = 50.0

    # --- Portfolio ---
    max_portfolio_correlation: float = 0.75
    correlation_lookback_hours: int = 72
    excluded_symbols: Optional[list[str]] = None

    # --- Features ---

    # --- Strategy selection ---
    strategy_family: str = 'momentum_trend'
    trade_freq_bucket: str = 'balanced'

    # --- Labelling ---
    # Directional consensus needed before a bar is labelled at all: 2 requires
    # all three momentum components to agree, 1 accepts two of three.
    direction_score_threshold: int = 2

    # --- Family-specific knobs (profiles override per coin) ---
    # Every value a profile can override needs a default here, or `resolve`
    # cannot complete the CLI > profile > default chain for it. The last four
    # belong to the funding_carry, squeeze_breakout and oi_divergence families,
    # which is why they were missing: those families were unreachable.
    pullback_depth_threshold: float = 0.020
    rebound_confirmation_threshold: float = 0.004
    trend_strength_min: float = 0.002
    pullback_lookback: int = 24
    breakout_lookback: int = 48
    breakout_buffer: float = 0.003
    expansion_confirm_threshold: float = 0.004
    funding_z_threshold: float = 2.5
    squeeze_pct_threshold: float = 0.20
    liq_threshold: float = 0.30
    oi_z_threshold: float = 1.0

    # Fields the user explicitly set on the command line.
    cli_overrides: frozenset[str] = frozenset()

    # -- Resolution ---------------------------------------------------------

    def resolve(
        self,
        name: str,
        profile: Optional["CoinProfile"] = None,
        mode: str = 'direct',
    ) -> Any:
        """Effective value of `name` under CLI > profile > default precedence.

        `mode` makes clamping explicit at the call site: 'ceiling' caps the
        result at the Config default, 'floor' raises it to the default,
        'direct' leaves it alone. Clamping applies to numbers only.
        """
        default = _default_of(type(self), name)

        if name in self.cli_overrides:
            value = getattr(self, name)
        elif profile is not None and hasattr(profile, name):
            value = getattr(profile, name)
        else:
            value = default

        if isinstance(default, str) or isinstance(value, str):
            return str(value or default)

        if isinstance(value, bool):
            return bool(value)

        value = float(value)
        if mode == 'ceiling':
            return min(value, float(default))
        if mode == 'floor':
            return max(value, float(default))
        return value

    def resolve_int(self, name: str, profile: Optional["CoinProfile"] = None, mode: str = 'direct') -> int:
        return int(self.resolve(name, profile, mode))

    def label_horizon_hours(self, profile: Optional["CoinProfile"] = None) -> int:
        """Prefer the execution hold horizon; fall back to the label horizon.

        Labels must span at least as long as a position can stay open, or the
        model is trained on an outcome the backtest never waits for.
        """
        max_hold = self.resolve_int('max_hold_hours', profile)
        if max_hold > 0:
            return max_hold
        return self.resolve_int('label_forward_hours', profile)


    def with_cost_assumptions(self, path: str | Path) -> "Config":
        """Return a copy with fees/slippage/impact taken from an exchange config.

        Loads `configs/exchange/*.json` (see `core.costs`). Without this the
        run uses the hardcoded Coinbase CDE defaults above.
        """
        a = load_exchange_cost_assumptions(path)
        return replace(
            self,
            fee_pct_per_side=a.effective_fee_pct_per_side(),
            per_contract_fee_usd=a.effective_per_contract_fee(),
            # Only when the exchange fee is actually enabled. `per_contract_fee`
            # prefers this dict whenever it is non-empty, which bypassed
            # `effective_per_contract_fee`'s own `enabled` check — so loading the
            # retail schedule (10bp/side, exchange_fee disabled) still charged
            # the CDE per-contract commission and produced an identical
            # round-trip to CDE from a completely different fee model.
            per_contract_fee_by_symbol=(
                dict(a.exchange_fee.symbol_overrides or {})
                if a.exchange_fee.enabled else {}
            ),
            slippage_bps=a.slippage.bps_per_side,
            apply_slippage=a.slippage.enabled,
            apply_impact=a.impact.enabled,
            impact_bps_per_contract=a.impact.bps_per_contract,
            impact_max_bps_per_side=a.impact.max_bps_per_side,
            apply_funding=a.funding.enabled,
            cost_config_path=str(path),
            cost_config_version=a.version,
        )





def _default_of(cls: type, name: str) -> Any:
    """Dataclass default for `name`, resolving default_factory."""
    for f in fields(cls):
        if f.name != name:
            continue
        if f.default is not MISSING:
            return f.default
        if f.default_factory is not MISSING:      # type: ignore[misc]
            return f.default_factory()             # type: ignore[misc]
        break
    raise AttributeError(f"{cls.__name__} has no field {name!r}")






# Env vars that, when set, outrank per-coin profiles.
_ENV_HARD_OVERRIDES = frozenset({'signal_threshold', 'min_val_auc'})


# ---------------------------------------------------------------------------
# Deliberately absent: a declarative CLI/env layer
# ---------------------------------------------------------------------------
#
# `CliParam`, `CLI_PARAMS`, `ENV_PARAMS`, `Config.add_cli_args`, `from_env`,
# `from_args` and `build_parser` used to live here. Nothing called any of them:
# every script builds a bare `Config()` and `scripts/_common.py` declares the
# flags it needs by hand. So the declarative layer was documentation that looked
# like wiring — 22 flags and 5 environment variables that parsed, stored, and
# reached nothing, including `LEVERAGE` in docker-compose.yml, which an operator
# could lower while the book kept trading at 4x.
#
# The surface that exists is `scripts/_common.py:add_data_arguments`. Add a flag
# there, where it is visibly connected to something.
