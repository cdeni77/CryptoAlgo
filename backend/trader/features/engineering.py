"""
Feature Engineering Module - Final Integrated & Scraper-Audited Version
Phase 1: Funding & OI Integration with Zero Lookahead Bias
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =============================================================================
# Point-in-Time Normalization Utilities
# =============================================================================

def normalize_point_in_time(series: pd.Series, lookback: int = 168, min_periods: int = 24) -> pd.Series:
    """Normalize using rolling window to prevent lookahead bias."""
    rolling_mean = series.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=lookback, min_periods=min_periods).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)

def winsorize_point_in_time(series: pd.Series, lookback: int = 168) -> pd.Series:
    """Winsorize using 1%/99% rolling quantiles to handle crypto outliers."""
    lower = series.rolling(window=lookback, min_periods=24).quantile(0.01)
    upper = series.rolling(window=lookback, min_periods=24).quantile(0.99)
    return series.clip(lower=lower, upper=upper)

# =============================================================================
# Feature Blocks
# =============================================================================

class PriceFeatures:
    """Price-derived technical indicators."""
    
    @classmethod
    def compute(cls, df: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for lb in lookbacks:
            features[f'return_{lb}h'] = df['close'].pct_change(lb)
            features[f'log_return_{lb}h'] = np.log(df['close'] / df['close'].shift(lb))
            features[f'volatility_{lb}h'] = df['close'].pct_change().rolling(lb).std()
            
            ma = df['close'].rolling(lb).mean()
            features[f'ma_distance_{lb}h'] = (df['close'] - ma) / ma.replace(0, np.nan)
            
            rolling_high = df['high'].rolling(lb).max()
            rolling_low = df['low'].rolling(lb).min()
            features[f'range_position_{lb}h'] = (df['close'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - ma20) / (2 * std20).replace(0, np.nan)
        features['bb_width'] = (4 * std20) / ma20.replace(0, np.nan)
        
        return features

class VolumeVolatilityFeatures:
    """Volume and Parkinson-style volatility features."""
    
    @classmethod
    def compute(cls, df: pd.DataFrame, lookbacks: List[int]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for lb in lookbacks:
            avg_vol = df['volume'].rolling(lb).mean()
            features[f'volume_ratio_{lb}h'] = df['volume'] / avg_vol.replace(0, np.nan)
        
        # Parkinson Volatility (High/Low based)
        features['parkinson_vol_24h'] = np.sqrt((1/(4*np.log(2))) * (np.log(df['high']/df['low'])**2).rolling(24).mean())
        return features

class FundingFeatures:
    """
    Funding rate features with 1-bar lag to match scraper alignment.
    Ensures funding rate known at T-1 is used to predict T -> T+24.
    """
    
    @classmethod
    def compute(cls, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=ohlcv_df.index)
        if funding_df is None or funding_df.empty: return features

        # Correct Lagging: Resample to hourly and shift 1 bar
        funding_raw = funding_df['rate'].resample('1h').ffill()
        funding_lagged = funding_raw.reindex(ohlcv_df.index, method='ffill').shift(1)
        
        features['funding_rate_bps'] = funding_lagged * 10000
        features['funding_rate_zscore'] = normalize_point_in_time(funding_lagged)
        features['cumulative_funding_24h'] = funding_lagged.rolling(24).sum()
        features['funding_persistence_24h'] = funding_lagged.rolling(24).apply(lambda x: (x > 0).mean())
        features['funding_rate_annualized'] = funding_lagged * 3 * 365
        
        return features

class OpenInterestFeatures:
    """
    Open Interest features with 1-bar lag for scraper safety.
    """
    
    @classmethod
    def compute(cls, ohlcv_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=ohlcv_df.index)
        if oi_df is None or oi_df.empty: return features

        # Correct Lagging: Reindex and shift 1 bar
        oi_col = 'open_interest_contracts' if 'open_interest_contracts' in oi_df.columns else 'open_interest'
        oi_raw = oi_df[oi_col].resample('1h').ffill()
        oi_lagged = oi_raw.reindex(ohlcv_df.index, method='ffill').shift(1)
        
        features['open_interest'] = oi_lagged
        features['oi_change_24h'] = oi_lagged.pct_change(24)
        features['oi_zscore'] = normalize_point_in_time(oi_lagged)
        features['oi_ma_distance_24h'] = (oi_lagged - oi_lagged.rolling(24).mean()) / oi_lagged.rolling(24).mean().replace(0, np.nan)
        
        # Liquidation Cascade Detector (High Vol + Large OI Drop)
        px_std = ohlcv_df['close'].pct_change().rolling(24).std()
        vol_ratio = ohlcv_df['volume'] / ohlcv_df['volume'].rolling(168).mean()
        features['liquidation_cascade_score'] = (
            (px_std > px_std.rolling(168).mean()).astype(int) + 
            (vol_ratio > 3).astype(int) + 
            (features['oi_change_24h'] < -0.05).astype(int)
        )
        return features

class RegimeFeatures:
    """Market regime and trend strength detection."""
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['trend_sma20_50'] = (df['close'].rolling(20).mean() > df['close'].rolling(50).mean()).astype(int)
        features['vol_regime_ratio'] = df['close'].pct_change().rolling(24).std() / df['close'].pct_change().rolling(168).std().replace(0, np.nan)
        features['drawdown_from_high'] = (df['close'] - df['close'].rolling(168).max()) / df['close'].rolling(168).max().replace(0, np.nan)
        features['momentum_168h_positive'] = (df['close'].pct_change(168) > 0).astype(int)
        return features

# =============================================================================
# Main Pipeline Class
# =============================================================================

@dataclass
class FeatureConfig:
    price_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48, 168])
    volume_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48])
    compute_price: bool = True
    compute_volume: bool = True
    compute_funding: bool = True
    compute_oi: bool = True
    compute_regime: bool = True

class FeaturePipeline:
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def compute_features(self, ohlcv_data, funding_data=None, oi_data=None, reference_symbol='BTC-PERP'):
        """Main entry: iterates symbols and returns feature dictionary."""
        all_features = {}
        funding_data = funding_data or {}
        oi_data = oi_data or {}
        
        for symbol, df in ohlcv_data.items():
            logger.info(f"Computing features for {symbol}...")
            f_list = []
            
            if self.config.compute_price: f_list.append(PriceFeatures.compute(df, self.config.price_lookbacks))
            if self.config.compute_volume: f_list.append(VolumeVolatilityFeatures.compute(df, self.config.volume_lookbacks))
            if self.config.compute_funding and symbol in funding_data: f_list.append(FundingFeatures.compute(df, funding_data[symbol]))
            if self.config.compute_oi and symbol in oi_data: f_list.append(OpenInterestFeatures.compute(df, oi_data[symbol]))
            if self.config.compute_regime: f_list.append(RegimeFeatures.compute(df))
            
            if f_list:
                combined = pd.concat(f_list, axis=1)
                # Ensure no duplicated feature columns
                all_features[symbol] = combined.loc[:, ~combined.columns.duplicated()]
        
        return all_features

    def compute_target(self, df: pd.DataFrame, horizon: int = 24, vol_mult: float = 1.8) -> pd.Series:
        """Triple Barrier Method: Labels +1 (TP), -1 (SL), or 0 (Time-out)."""
        target = pd.Series(index=df.index, dtype=float)
        vol = df['close'].pct_change().rolling(24).std()
        
        for i in range(len(df) - horizon):
            entry = df['close'].iloc[i]
            limit = entry * vol.iloc[i] * vol_mult
            if np.isnan(limit) or limit == 0: continue
            
            # Future path starting from i+1 (Zero Lookahead)
            future = df['close'].iloc[i+1 : i+1+horizon]
            
            hit_up = np.where(future >= entry + limit)[0]
            hit_down = np.where(future <= entry - limit)[0]
            
            f_up = hit_up[0] if len(hit_up) > 0 else horizon
            f_down = hit_down[0] if len(hit_down) > 0 else horizon
            
            if f_up < f_down: target.iloc[i] = 1
            elif f_down < f_up: target.iloc[i] = -1
            else: target.iloc[i] = 0
            
        return target

    def prepare_ml_dataset(self, features: pd.DataFrame, target: pd.Series):
        """Final prep: Alignment, Lag Check, and Forward-Fill Imputation."""
        common = features.index.intersection(target.dropna().index)
        X, y = features.loc[common].copy(), target.loc[common]
        
        # Handle gaps via Forward-Fill then Median
        X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(X.median())
        return X[~X.isna().any(axis=1)], y.loc[X.index]