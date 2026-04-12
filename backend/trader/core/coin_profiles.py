"""
coin_profiles.py — Per-coin trading profiles for v8 system.

Each coin gets:
  1. A tailored feature list (base + coin-specific)
  2. Tuned signal thresholds, exit parameters, regime filters
  3. ML hyperparameters (n_estimators, max_depth, etc.)

All coins use the same momentum strategy — differentiation is in
threshold tuning, exit structure, and extra features.

Usage in train_model.py:
    from core.coin_profiles import get_coin_profile, COIN_PROFILES
    profile = get_coin_profile('BIP-20DEC30-CDE')
"""

import os
import json
import hashlib
import joblib
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)
FEATURE_SCHEMA_VERSION = "v11-redundancy-pruned"

# Directory for persisted models
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PRUNED_FEATURES_DIR = Path(os.getenv('PRUNED_FEATURES_DIR', 'data/features'))


# BASE FEATURES (shared across all coins)
BASE_FEATURES = [
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
    # Market microstructure / distribution
    'body_to_range', 'close_to_high', 'close_to_low', 'atr_pct_24h',
    'buy_volume_ratio_24h', 'volume_zscore_24h',
    'ret_skew_72h', 'ret_kurt_72h',
    # Funding
    'funding_rate_bps', 'funding_rate_zscore',
    'cumulative_funding_24h', 'cumulative_funding_72h',
    # OI
    'oi_change_4h', 'oi_change_24h',
    # Regime
    'trend_sma20_50', 'vol_regime_ratio', 'trend_strength_24h',
    # Cost-aware execution hurdle
    'fee_hurdle_pct', 'breakout_vs_cost', 'expected_cost_to_vol_ratio',
]

# COIN-SPECIFIC EXTRA FEATURES
BTC_EXTRA_FEATURES = [
    # Mean-reversion: MIN/MAX proximity
    'at_max_10d', 'at_min_10d', 'dist_from_max_10d', 'dist_from_min_10d',
    'at_max_20d', 'at_min_20d', 'dist_from_max_20d', 'dist_from_min_20d',
    # Z-scores
    'zscore_24h', 'zscore_48h', 'zscore_72h', 'zscore_168h',
    # RSI extremes (contrarian)
    'rsi_6_oversold', 'rsi_6_overbought', 'rsi_14_oversold', 'rsi_14_overbought',
    'rsi_28_oversold', 'rsi_28_overbought',
    # Volume climax + BB squeeze
    'volume_climax', 'bb_squeeze',
    # Consecutive direction
    'consecutive_up', 'consecutive_down',
]

SOL_EXTRA_FEATURES = [
    # Momentum acceleration
    'momentum_accel_6h', 'momentum_accel_12h', 'momentum_accel_24h', 'momentum_accel_48h',
    # Efficiency ratio
    'efficiency_ratio_24h', 'efficiency_ratio_72h',
    # Breakout strength
    'breakout_strength_24h', 'breakout_strength_72h', 'breakout_strength_168h',
    # Volume surge
    'vol_surge_persistence',
    # Volatility term structure
    'vol_term_structure',
    # Return autocorrelation
    'ret_autocorr_lag1', 'ret_autocorr_lag2', 'ret_autocorr_lag4',
    # Range expansion
    'range_expansion',
]

DOGE_EXTRA_FEATURES = [
    # Retail FOMO/panic
    'fomo_score', 'panic_score',
    # Pump-and-dump
    'pump_dump_signal',
    # Extreme moves
    'extreme_move_freq_24h', 'extreme_move_freq_72h',
    # Volatility asymmetry
    'vol_asymmetry',
    # Autocorrelation decay
    'autocorr_1h', 'autocorr_6h', 'autocorr_12h', 'autocorr_24h',
    # VWAP
    'vwap_distance_24h',
    # Hype cycle + consecutive big moves
    'hype_cycle_position', 'consecutive_big_moves',
]


ETH_EXTRA_FEATURES = [
    'eth_trend_spread_12h', 'eth_trend_spread_24h', 'eth_trend_spread_72h', 'eth_trend_spread_168h',
    'eth_impulse_12h', 'eth_impulse_24h', 'eth_impulse_48h',
    'eth_volume_support', 'eth_pullback_depth_72h', 'eth_breakout_pressure',
]

XRP_EXTRA_FEATURES = [
    'xrp_compression_ratio', 'xrp_breakout_distance', 'xrp_whipsaw_score',
    'xrp_body_efficiency', 'xrp_volume_breakout_confirm', 'xrp_reversal_pressure',
]

BTC_RELATIVE_FEATURES = [
    'btc_rel_return_4h', 'btc_rel_return_24h', 'btc_rel_return_72h',
    'btc_corr_24h', 'btc_corr_72h', 'btc_beta_24h', 'btc_beta_72h',
]

ETH_EXTRA_FEATURES += BTC_RELATIVE_FEATURES
SOL_EXTRA_FEATURES += BTC_RELATIVE_FEATURES
XRP_EXTRA_FEATURES += BTC_RELATIVE_FEATURES
DOGE_EXTRA_FEATURES += BTC_RELATIVE_FEATURES

# ── New altcoins ──────────────────────────────────────────────────────────────

# AVAX reuses SOL momentum-breakout feature set (same generic names, different data)
AVAX_EXTRA_FEATURES = list(SOL_EXTRA_FEATURES)  # already includes BTC_RELATIVE_FEATURES

# ADA: compression-breakout pattern similar to XRP
ADA_EXTRA_FEATURES = [
    'ada_compression_ratio', 'ada_breakout_distance', 'ada_whipsaw_score',
    'ada_body_efficiency', 'ada_volume_breakout_confirm', 'ada_reversal_pressure',
] + BTC_RELATIVE_FEATURES

# LINK: trend-persistence features similar to ETH
LINK_EXTRA_FEATURES = [
    'link_trend_spread_12h', 'link_trend_spread_24h',
    'link_trend_spread_72h', 'link_trend_spread_168h',
    'link_impulse_12h', 'link_impulse_24h', 'link_impulse_48h',
    'link_volume_support', 'link_pullback_depth_72h', 'link_breakout_pressure',
] + BTC_RELATIVE_FEATURES

# LTC: BTC mean-reversion pattern (store-of-value narrative, halving cycles)
LTC_EXTRA_FEATURES = list(BTC_EXTRA_FEATURES) + BTC_RELATIVE_FEATURES

# ── New 20DEC30-CDE coins ──────────────────────────────────────────────────────

# NEAR: Layer-1 ecosystem momentum — reuse SOL feature set
NEAR_EXTRA_FEATURES = list(SOL_EXTRA_FEATURES)  # already includes BTC_RELATIVE_FEATURES

# SUI: Layer-1 parallel execution — reuse SOL feature set
SUI_EXTRA_FEATURES = list(SOL_EXTRA_FEATURES)   # already includes BTC_RELATIVE_FEATURES

# BCH: BTC-lite — reuse LTC feature set (same BTC mean-reversion + BTC relative)
BCH_EXTRA_FEATURES = list(LTC_EXTRA_FEATURES)

# XLM: compression-breakout like XRP
XLM_EXTRA_FEATURES = [
    'xlm_compression_ratio', 'xlm_breakout_distance', 'xlm_whipsaw_score',
    'xlm_body_efficiency', 'xlm_volume_breakout_confirm', 'xlm_reversal_pressure',
] + BTC_RELATIVE_FEATURES

# DOT: parachain ecosystem, BTC-like support/resistance — reuse BTC mean-reversion
DOT_EXTRA_FEATURES = list(BTC_EXTRA_FEATURES) + BTC_RELATIVE_FEATURES

# SHIB (1000SHIB): meme coin sentiment — reuse DOGE feature set
SHIB_EXTRA_FEATURES = list(DOGE_EXTRA_FEATURES)  # already includes BTC_RELATIVE_FEATURES

# PEPE (1000PEPE): meme coin — reuse DOGE feature set
PEPE_EXTRA_FEATURES = list(DOGE_EXTRA_FEATURES)  # already includes BTC_RELATIVE_FEATURES


@dataclass
class CoinProfile:
    """Per-coin trading configuration."""
    # Identity
    name: str                               # e.g. "BTC", "SOL", "DOGE"
    prefixes: List[str]                     # symbol prefixes to match
    
    # Features
    extra_features: List[str] = field(default_factory=list)
    
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
    # Direction filter strictness: 2 = all 3 momentum signals must agree (default),
    # 1 = any 2-of-3 agree. Use 1 for noisy/high-vol coins (DOGE, SOL) to generate
    # more labeled bars in sideways markets without sacrificing directional bias.
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
    
    # Model config
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_samples: int = 20

    # Family-specific knobs (kept broad so all families can coexist)
    pullback_depth_threshold: float = 0.020
    rebound_confirmation_threshold: float = 0.004
    trend_strength_min: float = 0.002
    pullback_lookback: int = 24
    breakout_lookback: int = 48
    breakout_buffer: float = 0.003
    expansion_confirm_threshold: float = 0.004
    # funding_carry
    funding_z_threshold: float = 2.5
    # squeeze_breakout
    squeeze_pct_threshold: float = 0.20
    # oi_divergence
    liq_threshold: float = 0.30
    oi_z_threshold: float = 1.0

    # Categorical strategy knobs
    strategy_family: str = 'momentum_trend'
    trade_freq_bucket: str = 'balanced'

    # Kelly Criterion sizing calibration (populated from backtest results)
    # kelly_win_rate = 0.0 means not calibrated; sizing falls back to vol-scaled fixed fraction
    kelly_win_rate: float = 0.0
    kelly_payoff_ratio: float = 0.0   # avg_win_net_pnl / avg_loss_net_pnl from backtest trades

    def load_pruned_features(self, features_dir: Optional[Path] = None) -> Optional[List[str]]:
        """Load persisted pruned feature list for this coin if available."""
        artifact_dir = features_dir or PRUNED_FEATURES_DIR
        path = artifact_dir / f"pruned_features_{self.name.lower()}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            features = payload.get('selected_features')
            if isinstance(features, list) and features:
                return [f for f in features if isinstance(f, str)]
        except Exception as exc:
            logger.warning(f"Failed to load pruned features for {self.name}: {exc}")
        return None

    def resolve_feature_columns(
        self,
        use_pruned_features: bool = False,
        features_dir: Optional[Path] = None,
        strict_pruned: bool = False,
    ) -> List[str]:
        """Resolve feature list, optionally preferring pruned artifact features."""
        if use_pruned_features:
            pruned = self.load_pruned_features(features_dir=features_dir)
            if pruned:
                return pruned
            if strict_pruned:
                return []
        return BASE_FEATURES + self.extra_features
    
    @property
    def feature_columns(self) -> List[str]:
        """Full feature list = base + coin-specific extras."""
        return self.resolve_feature_columns(use_pruned_features=False)


COIN_PROFILES: Dict[str, CoinProfile] = {
    # ── ETH: comprehensive_search v2 2026-04-05: mean_reversion/24h/0.025, tp=5.0/sl=3.5
    #         Verified SR=+0.601, WR=50.8%, 27.2/yr ──
    'ETH': CoinProfile(
        name='ETH',
        prefixes=['ETP', 'ETH'],
        extra_features=ETH_EXTRA_FEATURES,
        signal_threshold=0.51,
        min_val_auc=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        label_forward_hours=24,
        min_momentum_magnitude=0.025,   # comprehensive_search v2 winner: mean_reversion/24h/0.025
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        cooldown_hours=24.0,            # comprehensive_search v2 winner: 24h, 27.2/yr
        min_vol_24h=0.0034,
        max_vol_24h=0.069,
        position_size=0.12,
        vol_sizing_target=0.025,
        strategy_family='mean_reversion',  # comprehensive_search v2 winner
        trade_freq_bucket='balanced',
        kelly_win_rate=0.508,           # from verified backtest WR=50.8%
        kelly_payoff_ratio=1.383,       # approx (tp/sl) × (1-WR)/WR = 1.429 × 0.968
    ),

    # ── XRP: baseline for optimizer search ──
    # iter6: raised thresholds to match priors floors — prevents overtrading in full backtest
    'XRP': CoinProfile(
        name='XRP',
        prefixes=['XPP', 'XRP'],
        extra_features=XRP_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        vol_mult_tp=4.5,
        vol_mult_sl=3.0,
        max_hold_hours=108,
        min_momentum_magnitude=0.0125,  # iter8: lowered → priors midpoint (0.005,0.020); blocked momentum_trend+vol_overlay, try breakout/mean_reversion
        min_directional_agreement=0.48, # iter8: priors lower bound
        meta_probability_threshold=0.46, # iter8: priors lower bound
        cooldown_hours=16.0,
        min_vol_24h=0.001,
        max_vol_24h=0.10,
        strategy_family='momentum_trend',
        trade_freq_bucket='balanced',
    ),

    # ── BTC: optimizer 2026-03-31: breakout/72h/0.020 → PROMOTION_READY (holdout SR=-0.046) ──
    'BTC': CoinProfile(
        name='BTC',
        prefixes=['BIP', 'BTC'],
        extra_features=BTC_EXTRA_FEATURES,
        signal_threshold=0.53,          # optimizer
        min_val_auc=0.48,
        label_forward_hours=36,
        label_vol_target=1.8,
        min_momentum_magnitude=0.020,   # screen: mean_reversion/72h/0.020, kept by optimizer
        vol_mult_tp=4.5,                # optimizer
        vol_mult_sl=5.0,                # optimizer
        max_hold_hours=60,              # optimizer
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        cooldown_hours=72,              # screen winner: 72h cooldown
        min_vol_24h=0.0021,
        max_vol_24h=0.084,
        position_size=0.12,
        vol_sizing_target=0.025,
        strategy_family='breakout',     # optimizer switched from mean_reversion
        trade_freq_bucket='balanced',
    ),

    # ── SOL: comprehensive_search 2026-04-01: momentum_trend/36h/0.030, tp=5.0/sl=3.5
    #         Verified SR=+0.323, WR=50.5%, 21.2/yr, PF=1.193, PnL=+$27,336 ──
    'SOL': CoinProfile(
        name='SOL',
        prefixes=['SLP', 'SOL'],
        extra_features=SOL_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.030,   # comprehensive_search winner: momentum_trend/36h/0.030
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=96,
        cooldown_hours=36.0,            # comprehensive_search winner: 36h, 21.2/yr
        min_vol_24h=0.0008,
        max_vol_24h=0.12,
        position_size=0.12,
        vol_sizing_target=0.025,
        strategy_family='momentum_trend',
        trade_freq_bucket='balanced',
        direction_score_threshold=1,     # SOL is noisy — allow 2-of-3 momentum signal
        kelly_win_rate=0.505,           # from verified backtest WR=50.5%
        kelly_payoff_ratio=1.169,       # PF=1.193 × (1-0.505)/0.505
    ),

    # ── DOGE: comprehensive_search v2 2026-04-05: btc_lead/6h/0.010, tp=5.0/sl=3.5
    #         Verified SR=+0.314, WR=49.6%, 30.1/yr ──
    'DOGE': CoinProfile(
        name='DOGE',
        prefixes=['DOP', 'DOGE'],
        extra_features=DOGE_EXTRA_FEATURES,
        signal_threshold=0.52,
        min_val_auc=0.50,
        label_forward_hours=12,
        label_vol_target=1.4,
        min_momentum_magnitude=0.010,   # comprehensive_search v2 winner: btc_lead/6h/0.010
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        cooldown_hours=6.0,             # comprehensive_search v2 winner: 6h, 30.1/yr
        min_vol_24h=0.0007,
        max_vol_24h=0.14,
        position_size=0.08,
        vol_sizing_target=0.020,
        n_estimators=80,
        max_depth=3,
        min_child_samples=25,
        strategy_family='btc_lead',     # BTC lead-lag catch-up
        trade_freq_bucket='balanced',
        direction_score_threshold=1,     # DOGE is noisy/memecoin — allow 2-of-3 momentum
        kelly_win_rate=0.496,           # from verified backtest WR=49.6%
        kelly_payoff_ratio=1.451,       # approx (tp/sl) × (1-WR)/WR = 1.429 × 1.016
    ),

    # ── AVAX: comprehensive_search v2 2026-04-05: breakout/8h/0.015, tp=4.5/sl=3.0
    #         Verified SR=+0.658, WR=49.3%, 29.4/yr ──
    'AVAX': CoinProfile(
        name='AVAX',
        prefixes=['AVP', 'AVAX'],
        extra_features=AVAX_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.015,   # comprehensive_search v2 winner: breakout/8h/0.015
        min_directional_agreement=0.52,
        meta_probability_threshold=0.50,
        vol_mult_tp=4.5,                # tp > sl: tp=4.5, sl=3.0
        vol_mult_sl=3.0,
        max_hold_hours=96,
        cooldown_hours=8.0,             # comprehensive_search v2 winner: 8h, 29.4/yr
        min_vol_24h=0.0008,
        max_vol_24h=0.14,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='breakout',
        trade_freq_bucket='balanced',
        kelly_win_rate=0.493,           # from verified backtest WR=49.3%
        kelly_payoff_ratio=1.542,       # approx (tp/sl) × (1-WR)/WR = 1.500 × 1.028
    ),

    # ── ADA: comprehensive_search 2026-04-01: breakout/36h/0.018, tp=5.0/sl=3.5
    #         Verified SR=+0.498, WR=47.3%, 28.1/yr, PF=1.137, PnL=+$22,138 ──
    'ADA': CoinProfile(
        name='ADA',
        prefixes=['ADP', 'ADA'],
        extra_features=ADA_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.018,   # comprehensive_search winner: breakout/36h/0.018
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=96,
        cooldown_hours=36.0,            # comprehensive_search winner: 36h, 28.1/yr
        min_vol_24h=0.0008,
        max_vol_24h=0.12,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='breakout',
        trade_freq_bucket='balanced',
        kelly_win_rate=0.473,           # from verified backtest WR=47.3%
        kelly_payoff_ratio=1.267,       # PF=1.137 × (1-0.473)/0.473
    ),

    # ── LINK: comprehensive_search 2026-04-03: btc_lead/24h/0.030, tp=5.0/sl=3.5
    #         Verified SR=+1.568, WR=73.3%, 23.9/yr, PnL=+$14,870 ──
    'LINK': CoinProfile(
        name='LINK',
        prefixes=['LNP', 'LINK'],
        extra_features=LINK_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.48,
        label_forward_hours=12,
        label_vol_target=1.8,
        min_momentum_magnitude=0.030,   # comprehensive_search winner: btc_lead/24h/0.030
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        cooldown_hours=24.0,            # comprehensive_search winner: 24h, 23.9/yr
        min_vol_24h=0.0004,
        max_vol_24h=0.094,
        position_size=0.12,
        vol_sizing_target=0.025,
        strategy_family='btc_lead',     # comprehensive_search winner
        trade_freq_bucket='balanced',
        kelly_win_rate=0.733,           # from verified backtest WR=73.3%
        kelly_payoff_ratio=0.534,       # approx (tp/sl) × (1-WR)/WR = 1.429 × 0.364
    ),

    # ── LTC: accurate screen 2026-03-31: momentum_trend/36h/0.012 → Sharpe=0.761, 20.6/yr ──
    'LTC': CoinProfile(
        name='LTC',
        prefixes=['LCP', 'LTC'],
        extra_features=LTC_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=36,
        label_vol_target=1.8,
        min_momentum_magnitude=0.012,   # accurate screen winner: momentum_trend/36h/0.012
        min_directional_agreement=0.52,
        meta_probability_threshold=0.50,
        vol_mult_tp=4.0,
        vol_mult_sl=3.0,
        max_hold_hours=72,
        cooldown_hours=36.0,            # accurate screen winner: 36h cooldown, 20.6/yr
        min_vol_24h=0.0004,
        max_vol_24h=0.10,
        n_estimators=150,
        max_depth=4,
        min_child_samples=30,
        position_size=0.10,
        vol_sizing_target=0.020,
        strategy_family='momentum_trend',  # accurate screen winner
        trade_freq_bucket='balanced',
    ),

    # ── New 20DEC30-CDE coins — baseline profiles pending strategy search ────

    # NEAR: Layer-1 ecosystem, high BTC correlation, fast momentum
    'NEAR': CoinProfile(
        name='NEAR',
        prefixes=['NER', 'NEAR'],
        extra_features=NEAR_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.030,
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=4.5,
        vol_mult_sl=3.0,
        max_hold_hours=96,
        cooldown_hours=24.0,
        min_vol_24h=0.0008,
        max_vol_24h=0.14,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='momentum_trend',
        trade_freq_bucket='balanced',
    ),

    # SUI: Layer-1 parallel execution, high-beta ecosystem like SOL
    'SUI': CoinProfile(
        name='SUI',
        prefixes=['SUP', 'SUI'],
        extra_features=SUI_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.030,
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=4.5,
        vol_mult_sl=3.0,
        max_hold_hours=96,
        cooldown_hours=24.0,
        min_vol_24h=0.0008,
        max_vol_24h=0.16,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='momentum_trend',
        trade_freq_bucket='balanced',
        direction_score_threshold=1,    # high-vol Layer-1 — allow 2-of-3 momentum
    ),

    # BCH: BTC-lite, halving cycles, mean-reversion at support/resistance
    'BCH': CoinProfile(
        name='BCH',
        prefixes=['BCP', 'BCH'],
        extra_features=BCH_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=36,
        label_vol_target=1.8,
        min_momentum_magnitude=0.020,
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=4.5,
        vol_mult_sl=3.0,
        max_hold_hours=72,
        cooldown_hours=48.0,
        min_vol_24h=0.0004,
        max_vol_24h=0.10,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='breakout',
        trade_freq_bucket='balanced',
    ),

    # XLM: compression-breakout like XRP, low volatility
    'XLM': CoinProfile(
        name='XLM',
        prefixes=['XLP', 'XLM'],
        extra_features=XLM_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=24,
        label_vol_target=1.6,
        min_momentum_magnitude=0.018,
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=4.5,
        vol_mult_sl=3.0,
        max_hold_hours=96,
        cooldown_hours=18.0,
        min_vol_24h=0.0006,
        max_vol_24h=0.12,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='breakout',
        trade_freq_bucket='balanced',
    ),

    # ── DOT: comprehensive_search 2026-04-03: mean_reversion/48h/0.065, tp=5.0/sl=3.5
    #         Verified SR=+1.510, WR=69.2%, ~19-25/yr ──
    'DOT': CoinProfile(
        name='DOT',
        prefixes=['POP', 'DOT'],
        extra_features=DOT_EXTRA_FEATURES,
        signal_threshold=0.53,
        min_val_auc=0.50,
        label_forward_hours=36,
        label_vol_target=1.8,
        min_momentum_magnitude=0.065,   # comprehensive_search winner: mean_reversion/48h/0.065
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        cooldown_hours=48.0,            # comprehensive_search winner: 48h cooldown
        min_vol_24h=0.0005,
        max_vol_24h=0.12,
        position_size=0.10,
        vol_sizing_target=0.025,
        strategy_family='mean_reversion',  # comprehensive_search winner
        trade_freq_bucket='balanced',
        kelly_win_rate=0.692,           # from verified backtest WR=69.2%
        kelly_payoff_ratio=1.400,       # approx tp/sl ratio (5.0/3.5); recalibrate after 90d live
    ),

    # ── SHIB: comprehensive_search 2026-04-03 round 2: mean_reversion/24h/0.050, tp=5.0/sl=3.5
    #         Verified SR=+1.242, WR=60.0%, 29.9/yr, PnL=+$6,644 ──
    'SHIB': CoinProfile(
        name='SHIB',
        prefixes=['SHP', 'SHIB'],
        extra_features=SHIB_EXTRA_FEATURES,
        signal_threshold=0.52,
        min_val_auc=0.50,
        label_forward_hours=12,
        label_vol_target=1.4,
        min_momentum_magnitude=0.050,   # comprehensive_search winner: mean_reversion/24h/0.050
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        cooldown_hours=24.0,            # comprehensive_search winner: 24h cooldown, 29.9/yr
        min_vol_24h=0.0007,
        max_vol_24h=0.16,
        position_size=0.08,
        vol_sizing_target=0.020,
        n_estimators=80,
        max_depth=3,
        min_child_samples=25,
        strategy_family='mean_reversion',  # comprehensive_search winner (switched from btc_lead)
        trade_freq_bucket='balanced',
        direction_score_threshold=1,    # noisy/memecoin — allow 2-of-3 momentum
        kelly_win_rate=0.600,           # from verified backtest WR=60.0%
        kelly_payoff_ratio=1.400,       # approx tp/sl ratio; recalibrate after 90d live
    ),

    # ── PEPE: comprehensive_search 2026-04-03: momentum_trend/18h/0.065, tp=5.0/sl=3.5
    #         Verified SR=+0.371, WR=40.0%, ~22-37/yr  ⚠️ only ~2.3yr data — monitor closely ──
    'PEPE': CoinProfile(
        name='PEPE',
        prefixes=['PEP', 'PEPE'],
        extra_features=PEPE_EXTRA_FEATURES,
        signal_threshold=0.52,
        min_val_auc=0.50,
        label_forward_hours=12,
        label_vol_target=1.4,
        min_momentum_magnitude=0.065,   # comprehensive_search winner: momentum_trend/18h/0.065
        min_directional_agreement=0.50,
        meta_probability_threshold=0.48,
        vol_mult_tp=5.0,                # tp > sl: tp=5.0, sl=3.5
        vol_mult_sl=3.5,
        max_hold_hours=72,
        cooldown_hours=18.0,            # comprehensive_search winner: 18h cooldown
        min_vol_24h=0.0007,
        max_vol_24h=0.18,
        position_size=0.07,             # reduced: marginal SR, limited data history
        vol_sizing_target=0.020,
        n_estimators=80,
        max_depth=3,
        min_child_samples=25,
        strategy_family='momentum_trend',  # comprehensive_search winner (switched from btc_lead)
        trade_freq_bucket='balanced',
        direction_score_threshold=1,    # noisy/memecoin — allow 2-of-3 momentum
        kelly_win_rate=0.400,           # from verified backtest WR=40.0%
        kelly_payoff_ratio=1.500,       # estimated; recalibrate after 90d live
    ),
}


def get_coin_profile(symbol: str) -> CoinProfile:
    """Look up profile by symbol prefix. Falls back to ETH profile."""
    prefix = symbol.split('-')[0].upper()
    for profile in COIN_PROFILES.values():
        if prefix in profile.prefixes:
            return profile
    # Fallback: use ETH/XRP momentum config
    logger.warning(f"No profile for '{symbol}', using ETH defaults")
    return COIN_PROFILES['ETH']


def save_model(
    symbol: str,
    model: Any,
    scaler: Any,
    calibrator: Any,
    feature_columns: List[str],
    auc: float,
    profile_name: str,
    extra_meta: Optional[Dict] = None,
    target_dir: Optional[Path] = None,
    secondary_model: Any = None,
    secondary_scaler: Any = None,
    secondary_calibrator: Any = None,
) -> Path:
    """
    Save a trained model + metadata to disk.
    
    Saves: {MODELS_DIR}/{symbol_clean}.joblib
    Contains: dict with model, scaler, calibrator, columns, auc, profile, meta
    """
    # Clean symbol for filename
    symbol_clean = symbol.replace('/', '_').replace('-', '_')
    models_dir = target_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{symbol_clean}.joblib"
    feature_set_hash = hashlib.sha256(
        json.dumps(sorted(feature_columns), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    
    payload = {
        'model': model,
        'scaler': scaler,
        'calibrator': calibrator,
        'secondary_model': secondary_model,
        'secondary_scaler': secondary_scaler,
        'secondary_calibrator': secondary_calibrator,
        'feature_columns': feature_columns,
        'auc': auc,
        'profile_name': profile_name,
        'symbol': symbol,
    }
    payload['meta'] = {
        'feature_set_hash': feature_set_hash,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'model_version': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        **(extra_meta or {}),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"💾 Saved model for {symbol} → {path} (AUC={auc:.3f})")
    return path


def load_model(symbol: str) -> Optional[Dict]:
    """Load a persisted model. Returns None if not found."""
    symbol_clean = symbol.replace('/', '_').replace('-', '_')
    path = MODELS_DIR / f"{symbol_clean}.joblib"
    if not path.exists():
        return None
    payload = joblib.load(path)
    logger.info(f"📂 Loaded model for {symbol} from {path} (AUC={payload.get('auc', '?')})")
    return payload


def list_saved_models() -> List[str]:
    """Return list of symbols with saved models."""
    return [p.stem for p in MODELS_DIR.glob("*.joblib")]
