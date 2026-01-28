"""
Feature Engineering Module - Enhanced for Phase 1

Key enhancements:
1. Proper funding rate feature integration
2. Funding rate z-score (primary signal)
3. Cumulative funding (carry cost/benefit)
4. Funding persistence (long/short imbalance proxy)

All features are computed with strict point-in-time constraints.
"""

import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Point-in-Time Normalization
# =============================================================================

def normalize_point_in_time(
    series: pd.Series,
    lookback: int = 168,
    min_periods: int = 24
) -> pd.Series:
    """
    Normalize using only data available at each point in time.
    
    CRITICAL: Prevents lookahead bias.
    """
    rolling_mean = series.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=lookback, min_periods=min_periods).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return (series - rolling_mean) / rolling_std


def winsorize_point_in_time(
    series: pd.Series,
    lookback: int = 168,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.Series:
    """Winsorize using rolling quantiles."""
    rolling_lower = series.rolling(window=lookback, min_periods=24).quantile(lower_pct)
    rolling_upper = series.rolling(window=lookback, min_periods=24).quantile(upper_pct)
    return series.clip(lower=rolling_lower, upper=rolling_upper)


# =============================================================================
# Price Features
# =============================================================================

class PriceFeatures:
    """Price-derived features."""
    
    DEFAULT_LOOKBACKS = [1, 4, 12, 24, 48, 168]
    
    @classmethod
    def compute(cls, df: pd.DataFrame, lookbacks: Optional[List[int]] = None) -> pd.DataFrame:
        """Compute price features."""
        lookbacks = lookbacks or cls.DEFAULT_LOOKBACKS
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            features[f'return_{lb}h'] = df['close'].pct_change(lb)
            features[f'log_return_{lb}h'] = np.log(df['close'] / df['close'].shift(lb))
            features[f'volatility_{lb}h'] = df['close'].pct_change().rolling(lb).std()
            
            ma = df['close'].rolling(lb).mean()
            features[f'ma_distance_{lb}h'] = (df['close'] - ma) / ma
            
            rolling_high = df['high'].rolling(lb).max()
            rolling_low = df['low'].rolling(lb).min()
            range_size = rolling_high - rolling_low
            features[f'range_position_{lb}h'] = (df['close'] - rolling_low) / range_size.replace(0, np.nan)
        
        # RSI
        features['rsi_14'] = cls._compute_rsi(df['close'], 14)
        features['rsi_7'] = cls._compute_rsi(df['close'], 7)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = ma_20 + 2 * std_20
        features['bb_lower'] = ma_20 - 2 * std_20
        features['bb_position'] = (df['close'] - ma_20) / (2 * std_20)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / ma_20
        
        # Trend strength
        features['trend_strength'] = cls._compute_trend_strength(df, 14)
        
        return features
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _compute_trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ADX-like trend strength."""
        high, low, close = df['high'], df['low'], df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        return dx.rolling(period).mean()


# =============================================================================
# Volume Features
# =============================================================================

class VolumeVolatilityFeatures:
    """Volume and volatility features."""
    
    DEFAULT_LOOKBACKS = [1, 4, 12, 24, 48]
    
    @classmethod
    def compute(cls, df: pd.DataFrame, lookbacks: Optional[List[int]] = None) -> pd.DataFrame:
        lookbacks = lookbacks or cls.DEFAULT_LOOKBACKS
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            avg_volume = df['volume'].rolling(lb).mean()
            features[f'volume_ratio_{lb}h'] = df['volume'] / avg_volume.replace(0, np.nan)
        
        # Parkinson volatility
        features['parkinson_vol_24h'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['high'] / df['low']) ** 2).rolling(24).mean()
        )
        
        # Volume-price correlation
        features['volume_price_corr_24h'] = df['close'].pct_change().rolling(24).corr(
            df['volume'].pct_change()
        )
        
        return features


# =============================================================================
# FUNDING RATE FEATURES - KEY FOR PHASE 1
# =============================================================================

class FundingFeatures:
    """
    Funding rate features - PRIMARY SIGNAL SOURCE.
    
    These features capture:
    1. Funding rate extremes (mean reversion opportunity)
    2. Cumulative funding costs (carry)
    3. Market positioning (long/short imbalance)
    
    Key insight from design.md:
    - Extreme funding rates (|z| > 2) tend to revert
    - This creates profitable opportunities for contrarian positions
    """
    
    @classmethod
    def compute(
        cls,
        ohlcv_df: pd.DataFrame,
        funding_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute funding rate features.
        
        Args:
            ohlcv_df: OHLCV data (hourly)
            funding_df: Funding rate data (8-hourly from exchanges)
        
        Returns:
            DataFrame of funding features aligned to OHLCV index
        """
        features = pd.DataFrame(index=ohlcv_df.index)
        
        if funding_df is None or funding_df.empty:
            logger.warning("No funding data - returning empty features")
            return features
        
        # Resample funding to hourly using forward fill
        # (Funding rate announced every 8h, but applies to next period)
        funding_hourly = funding_df['rate'].resample('1h').ffill()
        funding_aligned = funding_hourly.reindex(ohlcv_df.index, method='ffill')
        
        # 1. Raw funding rate (in basis points for readability)
        features['funding_rate'] = funding_aligned
        features['funding_rate_bps'] = funding_aligned * 10000
        
        # 2. Funding rate moving averages
        # Note: 24 hours = 3 funding periods (8h each)
        features['funding_rate_ma_24h'] = funding_aligned.rolling(24).mean()
        features['funding_rate_ma_72h'] = funding_aligned.rolling(72).mean()  # ~9 funding periods
        features['funding_rate_ma_168h'] = funding_aligned.rolling(168).mean()  # 1 week
        
        # 3. FUNDING RATE Z-SCORE - KEY SIGNAL
        # This is the primary signal from design.md Section 5.1.1
        # Extreme funding (|z| > 2) suggests mean reversion opportunity
        features['funding_rate_zscore'] = normalize_point_in_time(
            funding_aligned, 
            lookback=168,  # 1 week baseline
            min_periods=72  # Need 3 days minimum
        )
        
        # 4. Funding rate momentum (is funding increasing or decreasing?)
        features['funding_rate_change_24h'] = funding_aligned.diff(24)
        features['funding_rate_change_72h'] = funding_aligned.diff(72)
        
        # 5. Cumulative funding (carry cost/benefit)
        # This is how much you'd pay/receive holding a position
        # For 8h funding data resampled to hourly, we sum over periods
        features['cumulative_funding_24h'] = funding_aligned.rolling(24).sum()
        features['cumulative_funding_72h'] = funding_aligned.rolling(72).sum()
        features['cumulative_funding_168h'] = funding_aligned.rolling(168).sum()
        
        # 6. Annualized funding rate (for comparison with other yields)
        # Assuming 3 funding periods per day
        features['funding_rate_annualized'] = funding_aligned * 3 * 365
        
        # 7. Funding persistence (long/short imbalance proxy)
        # What fraction of recent funding periods were positive?
        # High persistence = market consistently bullish (longs pay shorts)
        features['funding_persistence_24h'] = funding_aligned.rolling(24).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        features['funding_persistence_72h'] = funding_aligned.rolling(72).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # 8. Funding rate volatility (how stable is the funding?)
        features['funding_rate_vol_24h'] = funding_aligned.rolling(24).std()
        features['funding_rate_vol_72h'] = funding_aligned.rolling(72).std()
        
        # 9. Extreme funding flags (for filtering)
        features['funding_extreme_positive'] = (features['funding_rate_zscore'] > 2.0).astype(int)
        features['funding_extreme_negative'] = (features['funding_rate_zscore'] < -2.0).astype(int)
        
        # 10. Funding rate percentile (rolling)
        features['funding_rate_percentile'] = funding_aligned.rolling(168).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        return features
    
    @classmethod
    def compute_funding_carry_signal(
        cls,
        funding_zscore: pd.Series,
        threshold: float = 1.5,
    ) -> pd.Series:
        """
        Generate funding carry/arbitrage signal.
        
        Signal interpretation:
        - +1: Funding is extremely positive (go SHORT to receive funding)
        - -1: Funding is extremely negative (go LONG to receive funding)
        -  0: Funding is neutral (no funding-based signal)
        
        This is a CONTRARIAN signal - we bet against the crowd.
        """
        signal = pd.Series(0, index=funding_zscore.index)
        
        # High positive funding -> Short (receive funding from longs)
        signal[funding_zscore > threshold] = -1
        
        # High negative funding -> Long (receive funding from shorts)
        signal[funding_zscore < -threshold] = 1
        
        return signal


# =============================================================================
# Regime Features
# =============================================================================

class RegimeFeatures:
    """Market regime detection."""
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # Trend detection
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['trend_sma20_50'] = (sma_20 > sma_50).astype(int)
        
        # Volatility regime
        vol_short = df['close'].pct_change().rolling(24).std()
        vol_long = df['close'].pct_change().rolling(168).std()
        features['vol_regime_ratio'] = vol_short / vol_long.replace(0, np.nan)
        
        # Momentum
        features['momentum_24h_positive'] = (df['close'].pct_change(24) > 0).astype(int)
        features['momentum_168h_positive'] = (df['close'].pct_change(168) > 0).astype(int)
        
        # Drawdown from high
        rolling_max = df['close'].rolling(168).max()
        features['drawdown_from_high'] = (df['close'] - rolling_max) / rolling_max
        
        return features


# =============================================================================
# Cross-Asset Features
# =============================================================================

class CrossAssetFeatures:
    """Cross-asset relationship features."""
    
    @classmethod
    def compute(
        cls,
        dfs: Dict[str, pd.DataFrame],
        reference: str = 'BTC-PERP',
        lookback: int = 168
    ) -> Dict[str, pd.DataFrame]:
        if reference not in dfs:
            return {}
        
        ref_df = dfs[reference]
        ref_returns = ref_df['close'].pct_change()
        
        features = {}
        
        for symbol, df in dfs.items():
            symbol_features = pd.DataFrame(index=df.index)
            
            if symbol == reference:
                symbol_features['is_reference'] = 1
                features[symbol] = symbol_features
                continue
            
            asset_returns = df['close'].pct_change()
            aligned_ref, aligned_asset = ref_returns.align(asset_returns, join='inner')
            
            # Beta
            rolling_cov = aligned_ref.rolling(lookback).cov(aligned_asset)
            rolling_var = aligned_ref.rolling(lookback).var()
            symbol_features[f'beta_to_{reference}'] = rolling_cov / rolling_var.replace(0, np.nan)
            
            # Correlation
            symbol_features[f'corr_to_{reference}'] = aligned_ref.rolling(lookback).corr(aligned_asset)
            
            # Relative strength
            ref_perf_24h = ref_df['close'].pct_change(24).reindex(df.index)
            asset_perf_24h = df['close'].pct_change(24)
            symbol_features['relative_strength_24h'] = asset_perf_24h - ref_perf_24h
            
            features[symbol] = symbol_features
        
        return features


# =============================================================================
# Feature Pipeline
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature computation configuration."""
    price_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48, 168])
    volume_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48])
    normalize_features: bool = True
    compute_price: bool = True
    compute_volume: bool = True
    compute_funding: bool = True  # NEW: Enable funding features
    compute_cross_asset: bool = True
    compute_regime: bool = True


class FeaturePipeline:
    """
    Main feature engineering pipeline.
    
    Enhanced for Phase 1: Funding rate integration.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
    
    def compute_features(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        funding_data: Optional[Dict[str, pd.DataFrame]] = None,
        oi_data: Optional[Dict[str, pd.DataFrame]] = None,
        reference_symbol: str = 'BTC-PERP'
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute all features for multiple symbols.
        
        Args:
            ohlcv_data: Dict of {symbol: ohlcv_df}
            funding_data: Dict of {symbol: funding_df} - NEW for Phase 1
            oi_data: Dict of {symbol: oi_df}
            reference_symbol: Reference for cross-asset features
        """
        funding_data = funding_data or {}
        oi_data = oi_data or {}
        
        all_features = {}
        
        # Cross-asset features
        if self.config.compute_cross_asset and len(ohlcv_data) > 1:
            cross_features = CrossAssetFeatures.compute(ohlcv_data, reference=reference_symbol)
        else:
            cross_features = {}
        
        for symbol, df in ohlcv_data.items():
            logger.info(f"Computing features for {symbol}...")
            
            feature_dfs = []
            
            # Price features
            if self.config.compute_price:
                price_features = PriceFeatures.compute(df, self.config.price_lookbacks)
                feature_dfs.append(price_features)
            
            # Volume features
            if self.config.compute_volume:
                vol_features = VolumeVolatilityFeatures.compute(df, self.config.volume_lookbacks)
                feature_dfs.append(vol_features)
            
            # FUNDING FEATURES - KEY FOR PHASE 1
            if self.config.compute_funding and symbol in funding_data:
                funding_features = FundingFeatures.compute(df, funding_data[symbol])
                feature_dfs.append(funding_features)
                logger.info(f"  ✓ Added {len(funding_features.columns)} funding features")
            elif self.config.compute_funding:
                logger.warning(f"  ⚠️ No funding data for {symbol}")
            
            # Regime features
            if self.config.compute_regime:
                regime_features = RegimeFeatures.compute(df)
                feature_dfs.append(regime_features)
            
            # Cross-asset features
            if symbol in cross_features:
                feature_dfs.append(cross_features[symbol])
            
            # Combine
            if feature_dfs:
                combined = pd.concat(feature_dfs, axis=1)
                combined = combined.loc[:, ~combined.columns.duplicated()]
                all_features[symbol] = combined
        
        return all_features
    
    def compute_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = 'return'
    ) -> pd.Series:
        """Compute prediction target."""
        if target_type == 'return':
            target = df['close'].pct_change(horizon).shift(-horizon)
        elif target_type == 'direction':
            forward_return = df['close'].pct_change(horizon).shift(-horizon)
            target = np.sign(forward_return)
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        target.name = f'target_{target_type}_{horizon}h'
        return target
    
    def prepare_ml_dataset(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for ML."""
        X = features.copy()
        y = target.reindex(X.index)
        
        if dropna:
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
        
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].cat.codes
        
        return X, y


# =============================================================================
# Utility
# =============================================================================

def get_feature_importance_names() -> Dict[str, str]:
    """Human-readable feature descriptions."""
    return {
        'return_1h': 'Return over last 1 hour',
        'return_24h': 'Return over last 24 hours',
        'volatility_24h': 'Realized volatility (24h)',
        'rsi_14': 'Relative Strength Index (14 period)',
        'bb_position': 'Bollinger Band position (-1 to 1)',
        'funding_rate_zscore': 'Funding rate z-score (KEY SIGNAL)',
        'funding_rate_bps': 'Funding rate in basis points',
        'cumulative_funding_24h': 'Cumulative funding (24h)',
        'funding_persistence_72h': 'Funding rate persistence (72h)',
        'vol_regime_ratio': 'Short-term vs long-term volatility',
    }