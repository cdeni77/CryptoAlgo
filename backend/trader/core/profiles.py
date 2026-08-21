"""Per-coin trading profiles.

A profile is the per-coin half of the configuration; `core.config.Config` is the
global half, and `Config.resolve` decides which wins (see its docstring).

Coins are described by a **feature archetype** plus the handful of tuned values
that actually differ. The 16 coins here resolve to 5 archetypes:

    mean_reversion        support/resistance, z-scores, RSI extremes
    momentum_breakout     acceleration, efficiency, breakout strength
    meme                  FOMO/panic, pump-dump, autocorrelation decay
    trend_persistence     trend spread, impulse, pullback depth
    compression_breakout  compression ratio, whipsaw, breakout distance

The last two are declared as templates (`{coin}_trend_spread_12h`) because the
features are the same shape per coin, computed on that coin's own data. That is
why ETH and LINK, or XRP/ADA/XLM, share one declaration instead of each
carrying a hand-copied list.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_VERSION = "v11-redundancy-pruned"
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
PRUNED_FEATURES_DIR = Path(os.getenv('PRUNED_FEATURES_DIR', 'data/features'))


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# Shared by every coin.
BASE_FEATURES: tuple[str, ...] = (
    # Momentum
    'return_1h', 'return_12h', 'return_24h', 'return_168h',
    'rsi_14', 'rsi_6',
    'range_position_24h', 'range_position_72h',
    'bb_position_20',
    'ma_distance_24h',
    # Volatility
    'volatility_1h', 'volatility_4h', 'volatility_24h',
    'volume_ratio_1h', 'volume_ratio_24h',
    'parkinson_vol_24h',
    # Microstructure / distribution
    'body_to_range', 'close_to_high', 'close_to_low', 'atr_pct_24h',
    'buy_volume_ratio_24h', 'volume_zscore_24h',
    'ret_skew_72h', 'ret_kurt_72h',
    # Funding
    'funding_rate_bps', 'funding_rate_zscore',
    'cumulative_funding_24h', 'cumulative_funding_72h',
    # Open interest
    'oi_change_4h', 'oi_change_24h',
    # Regime
    'trend_sma20_50', 'vol_regime_ratio', 'trend_strength_24h',
    # Cost-aware execution hurdle
    'fee_hurdle_pct', 'breakout_vs_cost', 'expected_cost_to_vol_ratio',
)

# Appended for every coin except BTC, where "relative to BTC" is degenerate.
BTC_RELATIVE_FEATURES: tuple[str, ...] = (
    'btc_rel_return_4h', 'btc_rel_return_24h', 'btc_rel_return_72h',
    'btc_corr_24h', 'btc_corr_72h', 'btc_beta_24h', 'btc_beta_72h',
)


@dataclass(frozen=True)
class FeatureArchetype:
    """A named feature set shared by coins that behave alike.

    `shared` names features computed identically for every coin using this
    archetype. `templates` are `str.format` patterns taking `coin=` (lowercased
    profile name) for features whose column name embeds the coin.
    """

    name: str
    shared: tuple[str, ...] = ()
    templates: tuple[str, ...] = ()

    def features_for(self, coin: str, include_btc_relative: bool = True) -> list[str]:
        cols = list(self.shared)
        cols += [t.format(coin=coin.lower()) for t in self.templates]
        if include_btc_relative:
            cols += list(BTC_RELATIVE_FEATURES)
        return cols


MEAN_REVERSION = FeatureArchetype(
    name='mean_reversion',
    shared=(
        # Support/resistance proximity
        'at_max_10d', 'at_min_10d', 'dist_from_max_10d', 'dist_from_min_10d',
        'at_max_20d', 'at_min_20d', 'dist_from_max_20d', 'dist_from_min_20d',
        # Z-scores
        'zscore_24h', 'zscore_48h', 'zscore_72h', 'zscore_168h',
        # RSI extremes (contrarian)
        'rsi_6_oversold', 'rsi_6_overbought', 'rsi_14_oversold', 'rsi_14_overbought',
        'rsi_28_oversold', 'rsi_28_overbought',
        # Volume climax + Bollinger squeeze
        'volume_climax', 'bb_squeeze',
        # Consecutive direction
        'consecutive_up', 'consecutive_down',
    ),
)

MOMENTUM_BREAKOUT = FeatureArchetype(
    name='momentum_breakout',
    shared=(
        'momentum_accel_6h', 'momentum_accel_12h', 'momentum_accel_24h', 'momentum_accel_48h',
        'efficiency_ratio_24h', 'efficiency_ratio_72h',
        'breakout_strength_24h', 'breakout_strength_72h', 'breakout_strength_168h',
        'vol_surge_persistence',
        'vol_term_structure',
        'ret_autocorr_lag1', 'ret_autocorr_lag2', 'ret_autocorr_lag4',
        'range_expansion',
    ),
)

MEME = FeatureArchetype(
    name='meme',
    shared=(
        'fomo_score', 'panic_score',
        'pump_dump_signal',
        'extreme_move_freq_24h', 'extreme_move_freq_72h',
        'vol_asymmetry',
        'autocorr_1h', 'autocorr_6h', 'autocorr_12h', 'autocorr_24h',
        'vwap_distance_24h',
        'hype_cycle_position', 'consecutive_big_moves',
    ),
)

TREND_PERSISTENCE = FeatureArchetype(
    name='trend_persistence',
    templates=(
        '{coin}_trend_spread_12h', '{coin}_trend_spread_24h',
        '{coin}_trend_spread_72h', '{coin}_trend_spread_168h',
        '{coin}_impulse_12h', '{coin}_impulse_24h', '{coin}_impulse_48h',
        '{coin}_volume_support', '{coin}_pullback_depth_72h', '{coin}_breakout_pressure',
    ),
)

COMPRESSION_BREAKOUT = FeatureArchetype(
    name='compression_breakout',
    templates=(
        '{coin}_compression_ratio', '{coin}_breakout_distance', '{coin}_whipsaw_score',
        '{coin}_body_efficiency', '{coin}_volume_breakout_confirm', '{coin}_reversal_pressure',
    ),
)

ARCHETYPES: Dict[str, FeatureArchetype] = {
    a.name: a for a in (MEAN_REVERSION, MOMENTUM_BREAKOUT, MEME, TREND_PERSISTENCE, COMPRESSION_BREAKOUT)
}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoinProfile:
    """Per-coin trading configuration.

    Field names are a contract: `Config.resolve` looks them up by name, so a
    profile field only overrides a global setting when the two names match.
    """

    # Identity
    name: str
    prefixes: tuple[str, ...]
    archetype: str = 'momentum_breakout'
    extra_features: tuple[str, ...] = ()      # explicit additions beyond the archetype

    # Signal thresholds
    signal_threshold: float = 0.58
    min_val_auc: float = 0.50
    max_ensemble_std: float = 0.18
    min_directional_agreement: float = 0.55
    meta_probability_threshold: float = 0.50

    # Labeling
    label_forward_hours: int = 24
    label_vol_target: float = 1.8
    min_momentum_magnitude: float = 0.02
    # 2 = all three momentum components must agree; 1 = any two of three.
    # Use 1 for noisy, high-vol coins so sideways markets still produce labels.
    direction_score_threshold: int = 2

    # Exits
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    max_hold_hours: int = 96
    cooldown_hours: float = 12.0

    # Regime filter
    min_vol_24h: float = 0.004
    max_vol_24h: float = 0.09

    # Sizing
    position_size: float = 0.15
    vol_sizing_target: float = 0.025

    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_samples: int = 20

    # Strategy family selection
    strategy_family: str = 'momentum_trend'
    trade_freq_bucket: str = 'balanced'

    # Family-specific knobs
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

    # Kelly sizing calibration, populated from backtest results.
    # kelly_win_rate == 0 means uncalibrated: sizing falls back to vol-scaled fixed fraction.
    kelly_win_rate: float = 0.0
    kelly_payoff_ratio: float = 0.0

    # -- Features -----------------------------------------------------------

    @property
    def include_btc_relative(self) -> bool:
        """BTC-relative features are meaningless for BTC itself."""
        return self.name != 'BTC'

    @property
    def archetype_features(self) -> list[str]:
        return ARCHETYPES[self.archetype].features_for(
            self.name, include_btc_relative=self.include_btc_relative
        )

    @property
    def feature_columns(self) -> list[str]:
        """Full feature list: base + archetype + explicit extras."""
        return list(BASE_FEATURES) + self.archetype_features + list(self.extra_features)

    def load_pruned_features(self, features_dir: Optional[Path] = None) -> Optional[list[str]]:
        """Load this coin's persisted pruned feature list, if one exists."""
        path = (features_dir or PRUNED_FEATURES_DIR) / f"pruned_features_{self.name.lower()}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load pruned features for %s: %s", self.name, exc)
            return None
        selected = payload.get('selected_features')
        if isinstance(selected, list) and selected:
            return [f for f in selected if isinstance(f, str)]
        return None

    def resolve_feature_columns(
        self,
        use_pruned_features: bool = False,
        features_dir: Optional[Path] = None,
        strict_pruned: bool = False,
    ) -> list[str]:
        """Feature list for training, optionally restricted to pruned features.

        With `strict_pruned`, a missing pruned artifact yields an empty list so
        the caller skips the coin rather than silently training on everything.
        """
        if use_pruned_features:
            pruned = self.load_pruned_features(features_dir=features_dir)
            if pruned:
                return pruned
            if strict_pruned:
                return []
        return self.feature_columns

    def with_overrides(self, **overrides: Any) -> "CoinProfile":
        """Copy of this profile with fields replaced (used by paper overrides)."""
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"Unknown CoinProfile fields: {sorted(unknown)}")
        return replace(self, **overrides)


def _p(name: str, prefixes: Sequence[str], archetype: str, **tuned: Any) -> CoinProfile:
    """Profile constructor: identity, archetype, and only the tuned deltas."""
    return CoinProfile(name=name, prefixes=tuple(prefixes), archetype=archetype, **tuned)


# Values below are the surviving output of the search campaigns — each one is a
# verified backtest result, not a guess. Anything not listed uses the dataclass
# default above.
COIN_PROFILES: Dict[str, CoinProfile] = {
    'ETH': _p('ETH', ('ETP', 'ETH'), 'trend_persistence',
              signal_threshold=0.51, min_val_auc=0.48, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, min_momentum_magnitude=0.025,
              vol_mult_tp=5.0, vol_mult_sl=3.5, max_hold_hours=72, cooldown_hours=24.0,
              min_vol_24h=0.0034, max_vol_24h=0.069, position_size=0.12,
              strategy_family='mean_reversion',
              kelly_win_rate=0.508, kelly_payoff_ratio=1.383),

    'BTC': _p('BTC', ('BIP', 'BTC'), 'mean_reversion',
              signal_threshold=0.53, min_val_auc=0.48, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_forward_hours=36,
              vol_mult_tp=4.5, vol_mult_sl=5.0, max_hold_hours=60, cooldown_hours=72,
              min_vol_24h=0.0021, max_vol_24h=0.084, position_size=0.12,
              strategy_family='breakout'),

    'XRP': _p('XRP', ('XPP', 'XRP'), 'compression_breakout',
              signal_threshold=0.53, min_directional_agreement=0.48,
              meta_probability_threshold=0.46, min_momentum_magnitude=0.0125,
              vol_mult_tp=4.5, max_hold_hours=108, cooldown_hours=16.0,
              min_vol_24h=0.001, max_vol_24h=0.10),

    'SOL': _p('SOL', ('SLP', 'SOL'), 'momentum_breakout',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_vol_target=1.6,
              min_momentum_magnitude=0.03, direction_score_threshold=1,
              vol_mult_tp=5.0, vol_mult_sl=3.5, cooldown_hours=36.0,
              min_vol_24h=0.0008, max_vol_24h=0.12, position_size=0.12,
              kelly_win_rate=0.505, kelly_payoff_ratio=1.169),

    'DOGE': _p('DOGE', ('DOP', 'DOGE'), 'meme',
               signal_threshold=0.52, min_directional_agreement=0.50,
               meta_probability_threshold=0.48, label_forward_hours=12,
               label_vol_target=1.4, min_momentum_magnitude=0.01,
               direction_score_threshold=1, vol_mult_tp=5.0, vol_mult_sl=3.5,
               max_hold_hours=72, cooldown_hours=6.0,
               min_vol_24h=0.0007, max_vol_24h=0.14,
               position_size=0.08, vol_sizing_target=0.02,
               n_estimators=80, min_child_samples=25,
               strategy_family='btc_lead',
               kelly_win_rate=0.496, kelly_payoff_ratio=1.451),

    'AVAX': _p('AVAX', ('AVP', 'AVAX'), 'momentum_breakout',
               signal_threshold=0.53, min_directional_agreement=0.52,
               label_vol_target=1.6, min_momentum_magnitude=0.015,
               vol_mult_tp=4.5, cooldown_hours=8.0,
               min_vol_24h=0.0008, max_vol_24h=0.14, position_size=0.10,
               strategy_family='breakout',
               kelly_win_rate=0.493, kelly_payoff_ratio=1.542),

    'ADA': _p('ADA', ('ADP', 'ADA'), 'compression_breakout',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_vol_target=1.6,
              min_momentum_magnitude=0.018, vol_mult_tp=5.0, vol_mult_sl=3.5,
              cooldown_hours=36.0, min_vol_24h=0.0008, max_vol_24h=0.12,
              position_size=0.10, strategy_family='breakout',
              kelly_win_rate=0.473, kelly_payoff_ratio=1.267),

    'LINK': _p('LINK', ('LNP', 'LINK'), 'trend_persistence',
               signal_threshold=0.53, min_val_auc=0.48, min_directional_agreement=0.50,
               meta_probability_threshold=0.48, label_forward_hours=12,
               min_momentum_magnitude=0.03, vol_mult_tp=5.0, vol_mult_sl=3.5,
               max_hold_hours=72, cooldown_hours=24.0,
               min_vol_24h=0.0004, max_vol_24h=0.094, position_size=0.12,
               strategy_family='btc_lead',
               kelly_win_rate=0.733, kelly_payoff_ratio=0.534),

    'LTC': _p('LTC', ('LCP', 'LTC'), 'mean_reversion',
              signal_threshold=0.53, min_directional_agreement=0.52,
              label_forward_hours=36, min_momentum_magnitude=0.012,
              vol_mult_tp=4.0, max_hold_hours=72, cooldown_hours=36.0,
              min_vol_24h=0.0004, max_vol_24h=0.10,
              position_size=0.10, vol_sizing_target=0.02,
              n_estimators=150, max_depth=4, min_child_samples=30),

    'NEAR': _p('NEAR', ('NER', 'NEAR'), 'momentum_breakout',
               signal_threshold=0.53, min_directional_agreement=0.50,
               meta_probability_threshold=0.48, label_vol_target=1.6,
               min_momentum_magnitude=0.03, vol_mult_tp=4.5, cooldown_hours=24.0,
               min_vol_24h=0.0008, max_vol_24h=0.14, position_size=0.10),

    'SUI': _p('SUI', ('SUP', 'SUI'), 'momentum_breakout',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_vol_target=1.6,
              min_momentum_magnitude=0.03, direction_score_threshold=1,
              vol_mult_tp=4.5, cooldown_hours=24.0,
              min_vol_24h=0.0008, max_vol_24h=0.16, position_size=0.10),

    'BCH': _p('BCH', ('BCP', 'BCH'), 'mean_reversion',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_forward_hours=36,
              vol_mult_tp=4.5, max_hold_hours=72, cooldown_hours=48.0,
              min_vol_24h=0.0004, max_vol_24h=0.10, position_size=0.10,
              strategy_family='breakout'),

    'XLM': _p('XLM', ('XLP', 'XLM'), 'compression_breakout',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_vol_target=1.6,
              min_momentum_magnitude=0.018, vol_mult_tp=4.5, cooldown_hours=18.0,
              min_vol_24h=0.0006, max_vol_24h=0.12, position_size=0.10,
              strategy_family='breakout'),

    'DOT': _p('DOT', ('POP', 'DOT'), 'mean_reversion',
              signal_threshold=0.53, min_directional_agreement=0.50,
              meta_probability_threshold=0.48, label_forward_hours=36,
              min_momentum_magnitude=0.065, vol_mult_tp=5.0, vol_mult_sl=3.5,
              max_hold_hours=72, cooldown_hours=48.0,
              min_vol_24h=0.0005, max_vol_24h=0.12, position_size=0.10,
              strategy_family='mean_reversion',
              kelly_win_rate=0.692, kelly_payoff_ratio=1.400),

    'SHIB': _p('SHIB', ('SHP', 'SHIB'), 'meme',
               signal_threshold=0.52, min_directional_agreement=0.50,
               meta_probability_threshold=0.48, label_forward_hours=12,
               label_vol_target=1.4, min_momentum_magnitude=0.05,
               direction_score_threshold=1, vol_mult_tp=5.0, vol_mult_sl=3.5,
               max_hold_hours=72, cooldown_hours=24.0,
               min_vol_24h=0.0007, max_vol_24h=0.16,
               position_size=0.08, vol_sizing_target=0.02,
               n_estimators=80, min_child_samples=25,
               strategy_family='mean_reversion',
               kelly_win_rate=0.600, kelly_payoff_ratio=1.400),

    'PEPE': _p('PEPE', ('PEP', 'PEPE'), 'meme',
               signal_threshold=0.52, min_directional_agreement=0.50,
               meta_probability_threshold=0.48, label_forward_hours=12,
               label_vol_target=1.4, min_momentum_magnitude=0.065,
               direction_score_threshold=1, vol_mult_tp=5.0, vol_mult_sl=3.5,
               max_hold_hours=72, cooldown_hours=18.0,
               min_vol_24h=0.0007, max_vol_24h=0.18,
               position_size=0.07, vol_sizing_target=0.02,
               n_estimators=80, min_child_samples=25,
               kelly_win_rate=0.400, kelly_payoff_ratio=1.500),
}

DEFAULT_PROFILE = 'ETH'


def get_coin_profile(symbol: str) -> CoinProfile:
    """Resolve a symbol to its profile by prefix, falling back to ETH."""
    prefix = symbol.split('-')[0].upper()
    for profile in COIN_PROFILES.values():
        if prefix in profile.prefixes:
            return profile
    logger.warning("No profile for %r, using %s defaults", symbol, DEFAULT_PROFILE)
    return COIN_PROFILES[DEFAULT_PROFILE]


def all_feature_columns() -> list[str]:
    """Every feature name any profile can ask for — what the builder must produce."""
    seen: dict[str, None] = {}
    for profile in COIN_PROFILES.values():
        for col in profile.feature_columns:
            seen.setdefault(col, None)
    return list(seen)
