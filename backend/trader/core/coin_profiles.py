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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Directory for persisted models
MODELS_DIR = Path(os.getenv('MODELS_DIR', 'models'))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# BASE FEATURES (shared across all coins)
BASE_FEATURES = [
    # Momentum
    'return_1h', 'return_4h', 'return_12h', 'return_24h', 'return_48h', 'return_168h',
    'rsi_14', 'rsi_6',
    'range_position_24h', 'range_position_72h',
    'bb_position_20',
    'ma_distance_24h', 'ma_distance_168h',
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
    'oi_change_1h', 'oi_change_4h', 'oi_change_24h',
    # Regime
    'trend_sma20_50', 'vol_regime_ratio', 'trend_strength_24h',
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


@dataclass
class CoinProfile:
    """Per-coin trading configuration."""
    # Identity
    name: str                               # e.g. "BTC", "SOL", "DOGE"
    prefixes: List[str]                     # symbol prefixes to match
    
    # Features
    extra_features: List[str] = field(default_factory=list)
    
    # Signal thresholds
    signal_threshold: float = 0.80
    min_val_auc: float = 0.54
    
    # Labeling
    label_forward_hours: int = 24
    label_vol_target: float = 1.8
    min_momentum_magnitude: float = 0.07
    
    # Exits
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    max_hold_hours: int = 96
    cooldown_hours: float = 24.0
    
    # Regime filter
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06
    
    # Sizing
    position_size: float = 0.15
    vol_sizing_target: float = 0.025
    
    # Model config
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    min_child_samples: int = 20
    
    @property
    def feature_columns(self) -> List[str]:
        """Full feature list = base + coin-specific extras."""
        return BASE_FEATURES + self.extra_features


COIN_PROFILES: Dict[str, CoinProfile] = {
    # ── ETH: momentum (strong baseline, slightly stricter filters) ──
    'ETH': CoinProfile(
        name='ETH',
        prefixes=['ETP', 'ETH'],
        extra_features=[],
        signal_threshold=0.82,
        vol_mult_tp=5.8,
        vol_mult_sl=3.0,
        max_hold_hours=72,
        min_momentum_magnitude=0.08,
    ),
    
    # ── XRP: momentum (raise bar to avoid fee-heavy chop) ──
    'XRP': CoinProfile(
        name='XRP',
        prefixes=['XPP', 'XRP'],
        extra_features=[],
        signal_threshold=0.84,
        vol_mult_tp=6.0,
        vol_mult_sl=3.0,
        max_hold_hours=108,
        min_momentum_magnitude=0.08,
    ),
    
    # ── BTC: strict momentum (mean-reversion failed at 25% WR) ──
    # v7 momentum was never tested on BTC (it was excluded).
    # BTC is macro-driven — hourly RSI extremes don't predict bounces.
    # Use same momentum approach as ETH but with very strict filters:
    # high threshold, high AUC bar, small position, long cooldown.
    'BTC': CoinProfile(
        name='BTC',
        prefixes=['BIP', 'BTC'],
        extra_features=BTC_EXTRA_FEATURES,
        signal_threshold=0.88,          # Extra strict bar to avoid fee-churn
        min_val_auc=0.56,               # Require strong model
        label_forward_hours=36,         # Longer horizon for BTC trend persistence
        label_vol_target=1.8,           # Standard barriers
        min_momentum_magnitude=0.12,    # Require only strongest trends to clear fees
        vol_mult_tp=7.0,               # Push for larger winners
        vol_mult_sl=2.5,               # Tighter risk control
        max_hold_hours=72,
        cooldown_hours=48.0,            # Longer cooldown — reduce overtrading
        min_vol_24h=0.006,             # BTC has lower vol than alts
        max_vol_24h=0.045,
        position_size=0.10,            # Smaller position — less confident edge
        vol_sizing_target=0.020,
        n_estimators=150,
        max_depth=4,
        min_child_samples=30,
    ),
    
    # ── SOL: momentum with tuned exits for higher vol ──
    # SOL momentum works (41% WR) but fees eat the edge.
    # Problem: TP=5.5x is too wide for SOL — price spikes then reverses.
    # Solution: tighter TP to capture breakouts, wider SL for SOL's chop.
    'SOL': CoinProfile(
        name='SOL',
        prefixes=['SLP', 'SOL'],
        extra_features=SOL_EXTRA_FEATURES,
        signal_threshold=0.84,          # Raise confidence floor to reduce churn
        min_val_auc=0.54,
        label_forward_hours=24,         # Slightly longer horizon to capture full moves
        label_vol_target=1.6,           # Tighter barriers
        min_momentum_magnitude=0.08,    # Filter weak breakouts
        vol_mult_tp=6.0,               # Require larger move vs fees
        vol_mult_sl=3.5,               # Wider SL — avoid stop-hunting in SOL chop
        max_hold_hours=96,              # Let winners work when trend extends
        cooldown_hours=36.0,
        min_vol_24h=0.010,             # SOL has higher base vol
        max_vol_24h=0.08,
        position_size=0.12,
        vol_sizing_target=0.025,
    ),
    
    # ── DOGE: strict momentum (mean-reversion produced 0 trades) ──
    # DOGE trends hard when it trends (Elon tweets, retail FOMO).
    # MR failed because hourly RSI rarely hits 25/75 on DOGE.
    # Use momentum with strict filters and small position sizes.
    'DOGE': CoinProfile(
        name='DOGE',
        prefixes=['DOP', 'DOGE'],
        extra_features=DOGE_EXTRA_FEATURES,
        signal_threshold=0.86,           # Very high bar — reduce noisy entries
        min_val_auc=0.55,
        label_forward_hours=12,          # Short horizon — DOGE moves fast
        label_vol_target=1.4,            # Tight barriers — high vol
        min_momentum_magnitude=0.12,     # Require strong momentum — filter out noise
        vol_mult_tp=5.5,                # Larger TP target to beat fees
        vol_mult_sl=3.0,                # Avoid noise stop-outs
        max_hold_hours=72,               # Give trends time to materialize
        cooldown_hours=36.0,
        min_vol_24h=0.012,              # DOGE always volatile
        max_vol_24h=0.10,               # Allow high vol
        position_size=0.08,             # Small positions — high risk
        vol_sizing_target=0.020,
        n_estimators=80,                # Fewer trees — don't overfit noise
        max_depth=3,
        min_child_samples=25,
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
        'feature_columns': feature_columns,
        'auc': auc,
        'profile_name': profile_name,
        'symbol': symbol,
    }
    payload['meta'] = {
        'feature_set_hash': feature_set_hash,
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
