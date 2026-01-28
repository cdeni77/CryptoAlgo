"""
Feature Engineering Module for Crypto Perpetual Futures Trading System.

This module implements all features from design.md Section 7 with strict
point-in-time constraints to prevent lookahead bias.

CRITICAL: All features are computed using ONLY data available at each point in time.
"""

import logging
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# Point-in-Time Normalization (Prevents Lookahead Bias)
# =============================================================================

def normalize_point_in_time(
    series: pd.Series,
    lookback: int = 168,  # 1 week of hourly data
    min_periods: int = 24
) -> pd.Series:
    """
    Normalize using only data available at each point in time.
    
    CRITICAL: This is the CORRECT way to normalize for backtesting.
    Using sklearn's StandardScaler on full data would cause lookahead bias.
    
    Args:
        series: Input series to normalize
        lookback: Rolling window size
        min_periods: Minimum observations required
    
    Returns:
        Z-score normalized series
    """
    rolling_mean = series.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=lookback, min_periods=min_periods).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    return (series - rolling_mean) / rolling_std


def winsorize_point_in_time(
    series: pd.Series,
    lookback: int = 168,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pd.Series:
    """
    Winsorize using rolling quantiles (point-in-time safe).
    """
    rolling_lower = series.rolling(window=lookback, min_periods=24).quantile(lower_pct)
    rolling_upper = series.rolling(window=lookback, min_periods=24).quantile(upper_pct)
    
    return series.clip(lower=rolling_lower, upper=rolling_upper)


# =============================================================================
# Price-Based Features (Section 7.1)
# =============================================================================

class PriceFeatures:
    """Price-derived features for perpetual futures trading."""
    
    DEFAULT_LOOKBACKS = [1, 4, 12, 24, 48, 168]  # hours
    
    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        lookbacks: Optional[List[int]] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute price features with multiple lookback periods.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            lookbacks: List of lookback periods in bars
            normalize: Whether to z-score normalize features
        
        Returns:
            DataFrame of features (same index as input)
        """
        lookbacks = lookbacks or cls.DEFAULT_LOOKBACKS
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            # Returns
            features[f'return_{lb}h'] = df['close'].pct_change(lb)
            
            # Log returns (more suitable for modeling)
            features[f'log_return_{lb}h'] = np.log(df['close'] / df['close'].shift(lb))
            
            # Momentum (rate of change)
            features[f'roc_{lb}h'] = (df['close'] - df['close'].shift(lb)) / df['close'].shift(lb)
            
            # Volatility (realized)
            features[f'volatility_{lb}h'] = df['close'].pct_change().rolling(lb).std()
            
            # Range (high-low as % of close)
            features[f'range_{lb}h'] = (
                df['high'].rolling(lb).max() - df['low'].rolling(lb).min()
            ) / df['close']
            
            # Distance from moving average
            ma = df['close'].rolling(lb).mean()
            features[f'ma_distance_{lb}h'] = (df['close'] - ma) / ma
            
            # Price position within range (0 = at low, 1 = at high)
            rolling_high = df['high'].rolling(lb).max()
            rolling_low = df['low'].rolling(lb).min()
            range_size = rolling_high - rolling_low
            features[f'range_position_{lb}h'] = (df['close'] - rolling_low) / range_size.replace(0, np.nan)
        
        # RSI (Relative Strength Index)
        features['rsi_14'] = cls._compute_rsi(df['close'], 14)
        features['rsi_7'] = cls._compute_rsi(df['close'], 7)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands position
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_upper'] = ma_20 + 2 * std_20
        features['bb_lower'] = ma_20 - 2 * std_20
        features['bb_position'] = (df['close'] - ma_20) / (2 * std_20)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / ma_20
        
        # Trend strength (ADX-like)
        features['trend_strength'] = cls._compute_trend_strength(df, 14)
        
        if normalize:
            # Normalize features that aren't already bounded
            cols_to_normalize = [c for c in features.columns 
                               if not any(x in c for x in ['rsi', 'bb_position', 'range_position'])]
            for col in cols_to_normalize:
                features[f'{col}_z'] = normalize_point_in_time(features[col])
        
        return features
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _compute_trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute trend strength (simplified ADX)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()
        
        return adx


# =============================================================================
# Volume and Volatility Features (Section 7.2)
# =============================================================================

class VolumeVolatilityFeatures:
    """Volume and volatility features."""
    
    DEFAULT_LOOKBACKS = [1, 4, 12, 24, 48]
    
    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        lookbacks: Optional[List[int]] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute volume and volatility features.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            lookbacks: List of lookback periods
            normalize: Whether to normalize features
        """
        lookbacks = lookbacks or cls.DEFAULT_LOOKBACKS
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            # Volume relative to average
            avg_volume = df['volume'].rolling(lb).mean()
            features[f'volume_ratio_{lb}h'] = df['volume'] / avg_volume.replace(0, np.nan)
            
            # Volume trend (is volume increasing?)
            features[f'volume_trend_{lb}h'] = df['volume'].rolling(lb).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                raw=False
            )
            
            # Volume-weighted average price distance
            vwap = (df['close'] * df['volume']).rolling(lb).sum() / df['volume'].rolling(lb).sum()
            features[f'vwap_distance_{lb}h'] = (df['close'] - vwap) / vwap.replace(0, np.nan)
            
            # Volatility ratio (short vs long)
            if lb > 1:
                short_vol = df['close'].pct_change().rolling(lb).std()
                long_vol = df['close'].pct_change().rolling(lb * 4).std()
                features[f'vol_ratio_{lb}h'] = short_vol / long_vol.replace(0, np.nan)
        
        # Parkinson volatility (more efficient estimator using high/low)
        features['parkinson_vol_24h'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['high'] / df['low']) ** 2).rolling(24).mean()
        )
        
        # Garman-Klass volatility (uses OHLC)
        features['gk_vol_24h'] = cls._garman_klass_volatility(df, 24)
        
        # Volume-price correlation (divergence detection)
        features['volume_price_corr_24h'] = df['close'].pct_change().rolling(24).corr(
            df['volume'].pct_change()
        )
        
        # On-Balance Volume trend
        obv = cls._compute_obv(df)
        features['obv_trend_24h'] = obv.rolling(24).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=False
        )
        
        if normalize:
            for col in features.columns:
                if 'ratio' not in col and 'corr' not in col:
                    features[f'{col}_z'] = normalize_point_in_time(features[col])
        
        return features
    
    @staticmethod
    def _garman_klass_volatility(df: pd.DataFrame, window: int) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(df['high'] / df['low']) ** 2
        log_co = np.log(df['close'] / df['open']) ** 2
        
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window).mean())
    
    @staticmethod
    def _compute_obv(df: pd.DataFrame) -> pd.Series:
        """Compute On-Balance Volume."""
        direction = np.sign(df['close'].diff())
        return (direction * df['volume']).cumsum()


# =============================================================================
# Derivatives-Specific Features (Section 7.3)
# =============================================================================

class DerivativesFeatures:
    """Features specific to perpetual futures."""
    
    @classmethod
    def compute(
        cls,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compute derivatives-specific features.
        
        Args:
            df: OHLCV DataFrame
            funding_df: Funding rate DataFrame (columns: rate, mark_price, index_price)
            oi_df: Open interest DataFrame (columns: open_interest_contracts)
            normalize: Whether to normalize features
        
        Returns:
            DataFrame of derivatives features
        """
        features = pd.DataFrame(index=df.index)
        
        # If funding data is available
        if funding_df is not None and not funding_df.empty:
            # Align funding data to OHLCV index
            funding_aligned = funding_df.reindex(df.index, method='ffill')
            
            # Funding rate features
            features['funding_rate'] = funding_aligned['rate']
            features['funding_rate_ma_24h'] = funding_aligned['rate'].rolling(24).mean()
            features['funding_rate_ma_168h'] = funding_aligned['rate'].rolling(168).mean()
            
            # Funding rate z-score (key signal from design.md)
            features['funding_rate_zscore'] = normalize_point_in_time(
                funding_aligned['rate'], lookback=168
            )
            
            # Cumulative funding (carry cost/benefit)
            features['cumulative_funding_24h'] = funding_aligned['rate'].rolling(24).sum()
            features['cumulative_funding_168h'] = funding_aligned['rate'].rolling(168).sum()
            
            # Funding rate persistence (long/short imbalance proxy)
            features['funding_persistence_48h'] = funding_aligned['rate'].rolling(48).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            
            # Basis (perp vs spot) if mark/index available
            if 'mark_price' in funding_aligned.columns and 'index_price' in funding_aligned.columns:
                basis = (funding_aligned['mark_price'] - funding_aligned['index_price']) / funding_aligned['index_price']
                features['basis'] = basis
                features['basis_ma_24h'] = basis.rolling(24).mean()
                features['basis_zscore'] = normalize_point_in_time(basis, lookback=168)
        
        # If open interest data is available
        if oi_df is not None and not oi_df.empty:
            oi_aligned = oi_df.reindex(df.index, method='ffill')
            
            if 'open_interest_contracts' in oi_aligned.columns:
                oi = oi_aligned['open_interest_contracts']
                
                features['oi_change_1h'] = oi.pct_change()
                features['oi_change_24h'] = oi.pct_change(24)
                features['oi_ma_distance'] = (oi - oi.rolling(168).mean()) / oi.rolling(168).mean()
                
                # Price vs OI divergence (potential liquidation signal)
                price_direction = np.sign(df['close'].pct_change(24))
                oi_direction = np.sign(oi.pct_change(24))
                features['price_oi_divergence'] = (price_direction != oi_direction).astype(int)
                
                # OI-weighted returns
                features['oi_weighted_return_24h'] = df['close'].pct_change(24) * (oi / oi.rolling(168).mean())
        
        return features


# =============================================================================
# Cross-Asset Features (Section 7.4)
# =============================================================================

class CrossAssetFeatures:
    """Features capturing cross-asset relationships."""
    
    @classmethod
    def compute(
        cls,
        dfs: Dict[str, pd.DataFrame],
        reference: str = 'BTC-PERP',
        lookback: int = 168
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute cross-asset features.
        
        Args:
            dfs: Dictionary of {symbol: DataFrame} with OHLCV data
            reference: Reference asset (typically BTC)
            lookback: Rolling window for correlations
        
        Returns:
            Dictionary of {symbol: features_df}
        """
        if reference not in dfs:
            logger.warning(f"Reference asset {reference} not in data")
            return {}
        
        ref_df = dfs[reference]
        ref_returns = ref_df['close'].pct_change()
        
        features = {}
        
        for symbol, df in dfs.items():
            if symbol == reference:
                # Reference asset gets self-features
                symbol_features = pd.DataFrame(index=df.index)
                symbol_features['is_reference'] = 1
                features[symbol] = symbol_features
                continue
            
            symbol_features = pd.DataFrame(index=df.index)
            asset_returns = df['close'].pct_change()
            
            # Align indices
            aligned_ref, aligned_asset = ref_returns.align(asset_returns, join='inner')
            
            # Beta to reference
            rolling_cov = aligned_ref.rolling(lookback).cov(aligned_asset)
            rolling_var = aligned_ref.rolling(lookback).var()
            symbol_features[f'beta_to_{reference}'] = rolling_cov / rolling_var.replace(0, np.nan)
            
            # Correlation to reference
            symbol_features[f'corr_to_{reference}'] = aligned_ref.rolling(lookback).corr(aligned_asset)
            
            # Lead-lag relationships (does BTC lead this asset?)
            for lag in [1, 2, 4, 8]:
                lagged_corr = aligned_ref.shift(lag).rolling(lookback).corr(aligned_asset)
                symbol_features[f'lag_{lag}h_corr_to_{reference}'] = lagged_corr
            
            # Relative strength (outperformance vs BTC)
            ref_perf_24h = ref_df['close'].pct_change(24).reindex(df.index)
            asset_perf_24h = df['close'].pct_change(24)
            symbol_features['relative_strength_24h'] = asset_perf_24h - ref_perf_24h
            
            ref_perf_168h = ref_df['close'].pct_change(168).reindex(df.index)
            asset_perf_168h = df['close'].pct_change(168)
            symbol_features['relative_strength_168h'] = asset_perf_168h - ref_perf_168h
            
            # Volatility relative to BTC
            ref_vol = ref_returns.rolling(24).std().reindex(df.index)
            asset_vol = asset_returns.rolling(24).std()
            symbol_features['vol_ratio_vs_btc'] = asset_vol / ref_vol.replace(0, np.nan)
            
            features[symbol] = symbol_features
        
        return features


# =============================================================================
# Regime Detection Features
# =============================================================================

class RegimeFeatures:
    """Market regime detection features."""
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime classification features.
        
        Returns features that help identify:
        - Trending vs mean-reverting
        - High vs low volatility
        - Risk-on vs risk-off
        """
        features = pd.DataFrame(index=df.index)
        
        # Trend detection
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        
        features['trend_sma20_50'] = (sma_20 > sma_50).astype(int)
        features['trend_sma50_200'] = (sma_50 > sma_200).astype(int)
        features['trend_strength'] = (sma_20 - sma_50) / sma_50
        
        # Volatility regime
        vol_short = df['close'].pct_change().rolling(24).std()
        vol_long = df['close'].pct_change().rolling(168).std()
        features['vol_regime_ratio'] = vol_short / vol_long.replace(0, np.nan)
        
        # Classify volatility regime
        features['vol_regime'] = pd.cut(
            features['vol_regime_ratio'],
            bins=[0, 0.7, 1.3, np.inf],
            labels=['low_vol', 'normal_vol', 'high_vol']
        )
        
        # Mean reversion indicators
        rsi = PriceFeatures._compute_rsi(df['close'], 14)
        features['oversold'] = (rsi < 30).astype(int)
        features['overbought'] = (rsi > 70).astype(int)
        
        # Momentum regime
        returns_24h = df['close'].pct_change(24)
        returns_168h = df['close'].pct_change(168)
        
        features['momentum_24h_positive'] = (returns_24h > 0).astype(int)
        features['momentum_168h_positive'] = (returns_168h > 0).astype(int)
        features['momentum_alignment'] = (
            (returns_24h > 0) == (returns_168h > 0)
        ).astype(int)
        
        # Drawdown from recent high
        rolling_max = df['close'].rolling(168).max()
        features['drawdown_from_high'] = (df['close'] - rolling_max) / rolling_max
        
        return features


# =============================================================================
# Feature Pipeline (Main Entry Point)
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    
    # Lookback periods
    price_lookbacks: List[int] = None
    volume_lookbacks: List[int] = None
    
    # Normalization
    normalize_features: bool = True
    
    # Which feature groups to compute
    compute_price: bool = True
    compute_volume: bool = True
    compute_derivatives: bool = True
    compute_cross_asset: bool = True
    compute_regime: bool = True
    
    def __post_init__(self):
        if self.price_lookbacks is None:
            self.price_lookbacks = [1, 4, 12, 24, 48, 168]
        if self.volume_lookbacks is None:
            self.volume_lookbacks = [1, 4, 12, 24, 48]


class FeaturePipeline:
    """
    Main feature engineering pipeline.
    
    Computes all features while ensuring no lookahead bias.
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
            funding_data: Dict of {symbol: funding_df}
            oi_data: Dict of {symbol: oi_df}
            reference_symbol: Symbol to use as reference for cross-asset features
        
        Returns:
            Dict of {symbol: features_df}
        """
        funding_data = funding_data or {}
        oi_data = oi_data or {}
        
        all_features = {}
        
        # Compute cross-asset features first (needs all symbols)
        if self.config.compute_cross_asset and len(ohlcv_data) > 1:
            cross_features = CrossAssetFeatures.compute(
                ohlcv_data, 
                reference=reference_symbol
            )
        else:
            cross_features = {}
        
        # Compute features for each symbol
        for symbol, df in ohlcv_data.items():
            logger.info(f"Computing features for {symbol}...")
            
            feature_dfs = []
            
            # Price features
            if self.config.compute_price:
                price_features = PriceFeatures.compute(
                    df,
                    lookbacks=self.config.price_lookbacks,
                    normalize=self.config.normalize_features
                )
                feature_dfs.append(price_features)
            
            # Volume/volatility features
            if self.config.compute_volume:
                vol_features = VolumeVolatilityFeatures.compute(
                    df,
                    lookbacks=self.config.volume_lookbacks,
                    normalize=self.config.normalize_features
                )
                feature_dfs.append(vol_features)
            
            # Derivatives features
            if self.config.compute_derivatives:
                deriv_features = DerivativesFeatures.compute(
                    df,
                    funding_df=funding_data.get(symbol),
                    oi_df=oi_data.get(symbol),
                    normalize=self.config.normalize_features
                )
                feature_dfs.append(deriv_features)
            
            # Regime features
            if self.config.compute_regime:
                regime_features = RegimeFeatures.compute(df)
                feature_dfs.append(regime_features)
            
            # Cross-asset features
            if symbol in cross_features:
                feature_dfs.append(cross_features[symbol])
            
            # Combine all features
            if feature_dfs:
                combined = pd.concat(feature_dfs, axis=1)
                # Remove duplicate columns if any
                combined = combined.loc[:, ~combined.columns.duplicated()]
                all_features[symbol] = combined
        
        return all_features
    
    def compute_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = 'return'
    ) -> pd.Series:
        """
        Compute prediction target.
        
        CRITICAL: Target is computed using FUTURE data, which is correct.
        The key is that features don't use future data.
        
        Args:
            df: OHLCV DataFrame
            horizon: Prediction horizon in bars
            target_type: 'return', 'direction', or 'volatility'
        
        Returns:
            Target series (shifted appropriately)
        """
        if target_type == 'return':
            # Forward return
            target = df['close'].pct_change(horizon).shift(-horizon)
        
        elif target_type == 'direction':
            # Direction of next move: -1, 0, +1
            forward_return = df['close'].pct_change(horizon).shift(-horizon)
            target = np.sign(forward_return)
        
        elif target_type == 'volatility':
            # Forward realized volatility
            target = df['close'].pct_change().rolling(horizon).std().shift(-horizon)
        
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
        """
        Prepare features and target for ML training.
        
        Args:
            features: Feature DataFrame
            target: Target series
            dropna: Whether to drop rows with NaN
        
        Returns:
            (X, y) tuple ready for model training
        """
        # Align features and target
        X = features.copy()
        y = target.reindex(X.index)
        
        if dropna:
            # Create mask of valid rows
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Convert categorical columns to numeric
        for col in X.select_dtypes(include=['category']).columns:
            X[col] = X[col].cat.codes
        
        return X, y


# =============================================================================
# Utility Functions
# =============================================================================

def get_feature_importance_names() -> Dict[str, str]:
    """Get human-readable descriptions of features."""
    return {
        'return_1h': 'Return over last 1 hour',
        'return_24h': 'Return over last 24 hours',
        'volatility_24h': 'Realized volatility (24h)',
        'rsi_14': 'Relative Strength Index (14 period)',
        'macd_hist': 'MACD histogram',
        'bb_position': 'Bollinger Band position (-1 to 1)',
        'volume_ratio_24h': 'Volume relative to 24h average',
        'funding_rate_zscore': 'Funding rate z-score (key signal)',
        'basis_zscore': 'Basis z-score (perp vs spot)',
        'corr_to_BTC-PERP': 'Correlation to BTC',
        'beta_to_BTC-PERP': 'Beta to BTC',
        'relative_strength_24h': 'Relative performance vs BTC (24h)',
        'vol_regime_ratio': 'Short-term vs long-term volatility',
        'trend_strength': 'Trend strength (SMA20 vs SMA50)',
    }