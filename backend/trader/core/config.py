"""Run configuration: one dataclass, resolved from defaults, env and CLI.

Precedence for any tunable is always the same, and lives in `Config.resolve`:

    CLI flag  >  per-coin profile  >  Config default

`cli_overrides` records which fields the user actually passed on the command
line, so a flag can beat a profile while an untouched default cannot. The
overridable fields are declared once in `CLI_PARAMS` and the argparse wiring is
generated from it, rather than being restated in three places per flag.
"""

from __future__ import annotations

import argparse
import os
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
    retrain_frequency_days: int = 7
    min_train_samples: int = 400
    train_embargo_hours: int = 24
    oos_eval_days: int = 60
    val_fraction: float = 0.20

    # --- Entry filters (profiles override per coin) ---
    signal_threshold: float = 0.80
    # Classification-era: a probability threshold plus this margin. The forecast
    # is a return now, so the decision path uses `min_edge_over_cost` instead —
    # reusing this as a return threshold demanded 200bp of expected net, which no
    # hourly forecast will ever clear.
    min_signal_edge: float = 0.02
    # Expected net return must exceed the round-trip cost by this multiple again.
    # Expressed relative to cost rather than as an absolute, because cost ranges
    # from ~5bp on the group-B contracts to ~54bp on ETH: an absolute floor would
    # be trivially met on one and unreachable on the other. At 0.5, DOGE needs
    # ~2.5bp of forecast edge and ETH needs ~27bp.
    min_edge_over_cost: float = 0.5
    min_momentum_magnitude: float = 0.07
    momentum_score_threshold: float = 1.0
    momentum_strict_mode: bool = False
    min_funding_z: float = 0.0
    max_ensemble_std: float = 0.12
    min_directional_agreement: float = 0.67
    disagreement_confidence_cap: float = 0.86
    meta_probability_threshold: float = 0.57

    # --- Regime filter ---
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06

    # --- Directional macro filter policies ---
    trend_filter_mode: str = 'off'
    funding_filter_mode: str = 'soft'

    # --- Exits ---
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    max_hold_hours: int = 96
    breakeven_trigger: float = 999.0
    trailing_active: bool = False
    trailing_mult: float = 999.0
    cooldown_hours: float = 24.0

    # --- Risk / sizing ---
    max_positions: int = 5
    position_size: float = 0.15
    leverage: int = 4
    vol_sizing_target: float = 0.025
    min_equity: float = 1000.0
    max_weekly_equity_growth: float = 0.03

    # --- Execution costs (see core.costs; satisfies its CostParams protocol) ---
    fee_pct_per_side: float = 0.0010
    min_fee_per_contract: float = 0.0
    # Per-symbol overrides of the per-contract floor. Empty means the scalar
    # above applies everywhere; a loaded venue config fills this in.
    min_fee_per_contract_by_symbol: dict[str, float] = field(default_factory=dict)
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
    cost_config_path: Optional[str] = None
    cost_config_version: str = 'legacy_default'

    # --- Labels ---
    label_forward_hours: int = 24
    label_vol_target: float = 1.8

    # --- Model / validation ---
    min_val_auc: float = 0.54
    calibration_strategy: str = 'platt'
    recency_half_life_days: float = 50.0
    max_n_estimators_optimize: int = 0

    # --- Portfolio ---
    max_portfolio_correlation: float = 0.75
    correlation_lookback_hours: int = 72
    excluded_symbols: Optional[list[str]] = None

    # --- Features ---
    enforce_pruned_features: bool = False

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

    def embargo_hours(self, profile: Optional["CoinProfile"] = None) -> int:
        """Purge window between train and test. Never shorter than a label."""
        return max(self.train_embargo_hours, self.label_horizon_hours(profile), 1)

    # -- Construction -------------------------------------------------------

    def with_cost_assumptions(self, path: str | Path) -> "Config":
        """Return a copy with fees/slippage/impact taken from an exchange config.

        Loads `configs/exchange/*.json` (see `core.costs`). Without this the
        run uses the hardcoded Coinbase CDE defaults above.
        """
        a = load_exchange_cost_assumptions(path)
        return replace(
            self,
            fee_pct_per_side=a.effective_fee_pct_per_side(),
            min_fee_per_contract=a.effective_min_fee_per_contract(),
            min_fee_per_contract_by_symbol=dict(a.exchange_fee.symbol_overrides or {}),
            slippage_bps=a.slippage.bps_per_side,
            apply_slippage=a.slippage.enabled,
            apply_impact=a.impact.enabled,
            impact_bps_per_contract=a.impact.bps_per_contract,
            impact_max_bps_per_side=a.impact.max_bps_per_side,
            apply_funding=a.funding.enabled,
            cost_config_path=str(path),
            cost_config_version=a.version,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add a flag per entry in CLI_PARAMS.

        Overridable params default to None so we can tell "user passed the
        default value" from "user passed nothing" — that distinction is what
        lets a profile win over an untouched default.
        """
        for p in CLI_PARAMS:
            kwargs: dict[str, Any] = {'help': p.help}
            if p.choices:
                kwargs['choices'] = list(p.choices)
            if p.kind is bool:
                kwargs['action'] = 'store_true'
            else:
                kwargs['type'] = p.kind
                kwargs['default'] = None if p.overridable else _default_of(cls, p.field)
            parser.add_argument(p.flag, dest=p.field, **kwargs)
        parser.add_argument('--exclude', type=str, default='',
                            help='Comma-separated symbols to skip')
        parser.add_argument('--cost-config', dest='cost_config_path', type=str, default=None,
                            help='Path to a configs/exchange/*.json cost assumption file')
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Build a Config from parsed CLI args, recording explicit overrides."""
        values: dict[str, Any] = {}
        overrides: set[str] = set()

        for p in CLI_PARAMS:
            given = getattr(args, p.field, None)
            if given is None:
                continue
            if p.kind is bool and not given:
                continue        # store_true left unset
            values[p.field] = given
            if p.overridable:
                overrides.add(p.field)

        excluded = getattr(args, 'exclude', '') or ''
        parsed = [s.strip() for s in excluded.split(',') if s.strip()]
        if parsed:
            values['excluded_symbols'] = parsed

        config = cls(**values, cli_overrides=frozenset(overrides))

        cost_path = getattr(args, 'cost_config_path', None)
        if cost_path:
            config = config.with_cost_assumptions(cost_path)
        return config

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Build a Config from the environment (the container's contract).

        Only env vars that are actually set take effect; everything else keeps
        its dataclass default. Explicit keyword `overrides` beat the env.
        """
        values: dict[str, Any] = {}
        cli_overrides: set[str] = set()

        for env_name, field_name, cast in ENV_PARAMS:
            raw = os.getenv(env_name)
            if raw is None or raw == '':
                continue
            try:
                values[field_name] = cast(raw)
            except (TypeError, ValueError):
                continue
            # An operator setting SIGNAL_THRESHOLD or MIN_AUC means it, the same
            # way a CLI flag does — it must beat the per-coin profile.
            if field_name in _ENV_HARD_OVERRIDES:
                cli_overrides.add(field_name)

        excluded = os.getenv('EXCLUDE_SYMBOLS')
        if excluded:
            values['excluded_symbols'] = [s.strip() for s in excluded.split(',') if s.strip()]

        values.update(overrides)
        config = cls(**values, cli_overrides=frozenset(cli_overrides))

        cost_path = os.getenv('COST_CONFIG_PATH')
        if cost_path and Path(cost_path).exists():
            config = config.with_cost_assumptions(cost_path)
        return config


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


@dataclass(frozen=True)
class CliParam:
    """One CLI flag bound to a Config field.

    `overridable=True` means an explicit value outranks the per-coin profile;
    those flags default to None so "unset" is distinguishable from "set to the
    default value".
    """

    flag: str
    field: str
    kind: Callable[[str], Any]
    help: str
    overridable: bool = False
    choices: Sequence[str] | None = None


CLI_PARAMS: tuple[CliParam, ...] = (
    # Flags that outrank per-coin profiles when explicitly passed.
    CliParam('--threshold', 'signal_threshold', float, 'Primary probability threshold', overridable=True),
    CliParam('--min-auc', 'min_val_auc', float, 'Minimum validation AUC to accept a model', overridable=True),
    CliParam('--momentum', 'min_momentum_magnitude', float, 'Minimum |72h return| to enter', overridable=True),
    CliParam('--min-directional-agreement', 'min_directional_agreement', float,
             'Minimum fraction of ensemble members agreeing with direction', overridable=True),
    CliParam('--max-ensemble-std', 'max_ensemble_std', float,
             'Maximum std across ensemble probabilities', overridable=True),
    CliParam('--meta-threshold', 'meta_probability_threshold', float,
             'Secondary (meta) model probability threshold', overridable=True),
    # Plain global settings.
    CliParam('--leverage', 'leverage', int, 'Account leverage'),
    CliParam('--tp', 'vol_mult_tp', float, 'Take-profit volatility multiple'),
    CliParam('--sl', 'vol_mult_sl', float, 'Stop-loss volatility multiple'),
    CliParam('--hold', 'max_hold_hours', int, 'Maximum hold in hours'),
    CliParam('--cooldown', 'cooldown_hours', float, 'Hours to wait after an exit'),
    CliParam('--min-edge', 'min_signal_edge', float, 'Require prob >= threshold + edge'),
    CliParam('--min-train-samples', 'min_train_samples', int, 'Minimum training rows per fold'),
    CliParam('--momentum-score-threshold', 'momentum_score_threshold', float,
             'Directional consensus score needed to trade'),
    CliParam('--recency-half-life-days', 'recency_half_life_days', float,
             'Half-life for recency sample weighting'),
    CliParam('--disagreement-confidence-cap', 'disagreement_confidence_cap', float,
             'Cap confidence when the ensemble is not unanimous'),
    CliParam('--calibration', 'calibration_strategy', str, 'Probability calibration method',
             choices=CALIBRATION_STRATEGIES),
    CliParam('--trend-filter-mode', 'trend_filter_mode', str, 'SMA200 trend filter policy',
             choices=FILTER_MODES),
    CliParam('--funding-filter-mode', 'funding_filter_mode', str, 'Funding z-score filter policy',
             choices=FILTER_MODES),
    CliParam('--trade-freq-bucket', 'trade_freq_bucket', str, 'Trade frequency bucket',
             choices=TRADE_FREQ_BUCKETS),
    # Boolean switches.
    CliParam('--strict-momentum-consensus', 'momentum_strict_mode', bool,
             'Reject entries when any momentum component disagrees'),
    CliParam('--pruned-only', 'enforce_pruned_features', bool,
             'Use only the persisted pruned feature list per coin'),
)

# Env var -> Config field. This is the container's contract (docker-compose.yml).
ENV_PARAMS: tuple[tuple[str, str, Callable[[str], Any]], ...] = (
    ('SIGNAL_THRESHOLD', 'signal_threshold', float),
    ('MIN_AUC', 'min_val_auc', float),
    ('LEVERAGE', 'leverage', int),
    ('RETRAIN_EVERY_DAYS', 'retrain_frequency_days', int),
    ('PRUNED_ONLY', 'enforce_pruned_features', lambda v: v.strip().lower() in ('1', 'true', 'yes', 'on')),
)

# Env vars that, when set, outrank per-coin profiles.
_ENV_HARD_OVERRIDES = frozenset({'signal_threshold', 'min_val_auc'})


def build_parser(description: str) -> argparse.ArgumentParser:
    """An argparse parser preloaded with every Config flag."""
    parser = argparse.ArgumentParser(description=description)
    return Config.add_cli_args(parser)
