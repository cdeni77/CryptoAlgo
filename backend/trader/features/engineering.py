"""
Feature Engineering Module - v8 with Coin-Specific Feature Blocks
Phase 2: BTC mean-reversion, SOL ecosystem, DOGE sentiment-proxy features

Zero lookahead bias throughout.
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
        
        # ----- RSI 14 (standard) -----
        delta = df['close'].diff()
        gain14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs14 = gain14 / loss14.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs14))
        
        # ----- RSI 6 (fast — needed by FEATURE_COLUMNS) -----
        gain6 = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss6 = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs6 = gain6 / loss6.replace(0, np.nan)
        features['rsi_6'] = 100 - (100 / (1 + rs6))
        
        # ----- MACD -----
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # ----- Bollinger Bands -----
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - ma20) / (2 * std20).replace(0, np.nan)
        features['bb_width'] = (4 * std20) / ma20.replace(0, np.nan)
        # Alias expected by FEATURE_COLUMNS
        features['bb_position_20'] = features['bb_position']
        
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
        features['parkinson_vol_24h'] = np.sqrt(
            (1/(4*np.log(2))) * (np.log(df['high']/df['low'])**2).rolling(24).mean()
        )
        return features

class FundingFeatures:
    """
    Funding rate features with 1-bar lag to match scraper alignment.
    Ensures funding rate known at T-1 is used to predict T -> T+24.
    """
    
    @classmethod
    def compute(cls, ohlcv_df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=ohlcv_df.index)
        if funding_df is None or funding_df.empty:
            return features

        # Correct Lagging: Resample to hourly and shift 1 bar
        funding_raw = funding_df['rate'].resample('1h').ffill()
        funding_lagged = funding_raw.reindex(ohlcv_df.index, method='ffill').shift(1)
        
        features['funding_rate_bps'] = funding_lagged * 10000
        features['funding_rate_zscore'] = normalize_point_in_time(funding_lagged)
        features['cumulative_funding_24h'] = funding_lagged.rolling(24).sum()
        features['cumulative_funding_72h'] = funding_lagged.rolling(72).sum()
        features['funding_persistence_24h'] = funding_lagged.rolling(24).apply(
            lambda x: (x > 0).mean(), raw=False
        )
        features['funding_rate_annualized'] = funding_lagged * 3 * 365
        
        return features

class OpenInterestFeatures:
    """
    Open Interest features with 1-bar lag for scraper safety.
    """
    
    @classmethod
    def compute(cls, ohlcv_df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=ohlcv_df.index)
        if oi_df is None or oi_df.empty:
            return features

        oi_col = 'open_interest_contracts' if 'open_interest_contracts' in oi_df.columns else 'open_interest'
        oi_raw = oi_df[oi_col].resample('1h').ffill()
        oi_lagged = oi_raw.reindex(ohlcv_df.index, method='ffill').shift(1)
        
        features['open_interest'] = oi_lagged
        features['oi_change_1h'] = oi_lagged.pct_change(1)
        features['oi_change_4h'] = oi_lagged.pct_change(4)
        features['oi_change_24h'] = oi_lagged.pct_change(24)
        features['oi_zscore'] = normalize_point_in_time(oi_lagged)
        features['oi_ma_distance_24h'] = (
            (oi_lagged - oi_lagged.rolling(24).mean()) /
            oi_lagged.rolling(24).mean().replace(0, np.nan)
        )
        
        # Liquidation Cascade Detector
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
        features['trend_sma20_50'] = (
            df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
        ).astype(int)
        features['vol_regime_ratio'] = (
            df['close'].pct_change().rolling(24).std() /
            df['close'].pct_change().rolling(168).std().replace(0, np.nan)
        )
        features['drawdown_from_high'] = (
            (df['close'] - df['close'].rolling(168).max()) /
            df['close'].rolling(168).max().replace(0, np.nan)
        )
        features['momentum_168h_positive'] = (df['close'].pct_change(168) > 0).astype(int)
        return features


# =============================================================================
# COIN-SPECIFIC FEATURE BLOCKS (v8)
# =============================================================================

class BTCMeanReversionFeatures:
    """
    BTC-specific: mean-reversion + MIN/MAX breakout features.
    
    Research basis:
    - QuantPedia (2024): BTC trends at local MAX, reverts at local MIN
    - BTC-neutral residual strategies post-2021 Sharpe ~2.3
    - BTC is too efficient for pure momentum at hourly scale (36% WR in backtest)
    - Better alpha from: Bollinger reversals, RSI extremes, volume climax reversals
    """
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df['close']
        
        # --- MIN/MAX proximity (QuantPedia) ---
        for lb in [10, 20, 50]:
            rolling_max = close.rolling(lb * 24).max()  # lb days in hours
            rolling_min = close.rolling(lb * 24).min()
            features[f'at_max_{lb}d'] = (close >= rolling_max * 0.99).astype(int)
            features[f'at_min_{lb}d'] = (close <= rolling_min * 1.01).astype(int)
            features[f'dist_from_max_{lb}d'] = (close - rolling_max) / rolling_max.replace(0, np.nan)
            features[f'dist_from_min_{lb}d'] = (close - rolling_min) / rolling_min.replace(0, np.nan)
        
        # --- Mean-reversion z-scores (multi-horizon) ---
        for lb in [24, 48, 72, 168]:
            ma = close.rolling(lb).mean()
            std = close.rolling(lb).std()
            features[f'zscore_{lb}h'] = (close - ma) / std.replace(0, np.nan)
        
        # --- RSI extremes (contrarian signals) ---
        delta = close.diff()
        for period in [6, 14, 28]:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)
            features[f'rsi_{period}_extreme'] = ((rsi < 20) | (rsi > 80)).astype(int)
        
        # --- Volume climax (reversal signal) ---
        vol_ma = df['volume'].rolling(168).mean()
        vol_std = df['volume'].rolling(168).std()
        features['volume_climax'] = (
            (df['volume'] > vol_ma + 2 * vol_std).astype(int)
        )
        
        # --- Bollinger squeeze (low vol → breakout) ---
        bb_width = (4 * close.rolling(20).std()) / close.rolling(20).mean().replace(0, np.nan)
        bb_width_pct = bb_width.rolling(168).rank(pct=True)
        features['bb_squeeze'] = (bb_width_pct < 0.1).astype(int)
        
        # --- Consecutive candle direction (reversion indicator) ---
        direction = (close.diff() > 0).astype(int)
        features['consecutive_up'] = direction.rolling(6).sum()
        features['consecutive_down'] = (1 - direction).rolling(6).sum()
        
        return features


class SOLEcosystemFeatures:
    """
    SOL-specific: DeFi-correlated momentum + higher-frequency signals.
    
    Research basis:
    - SOL moves on fundamentals: network upgrades, developer activity, ETF flows
    - High beta to BTC but with idiosyncratic alpha from ecosystem events
    - 85% bullish retail sentiment driven, 162M daily transactions
    - Strong golden-cross signals (50d > 200d MA)
    - Higher volatility = wider barriers needed, shorter holds
    """
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        
        # --- Multi-timeframe momentum strength ---
        for lb in [6, 12, 24, 48]:
            mom = close.pct_change(lb)
            mom_ma = mom.rolling(24).mean()
            features[f'momentum_accel_{lb}h'] = mom - mom_ma  # Accelerating or decelerating
        
        # --- Trend consistency (efficiency ratio) ---
        for lb in [24, 72]:
            net_move = abs(close.diff(lb))
            sum_moves = close.diff().abs().rolling(lb).sum()
            features[f'efficiency_ratio_{lb}h'] = net_move / sum_moves.replace(0, np.nan)
        
        # --- Breakout strength ---
        for lb in [24, 72, 168]:
            rolling_high = df['high'].rolling(lb).max()
            rolling_low = df['low'].rolling(lb).min()
            atr_range = rolling_high - rolling_low
            features[f'breakout_strength_{lb}h'] = (
                (close - rolling_high.shift(1)) / atr_range.replace(0, np.nan)
            ).clip(-2, 2)
        
        # --- Volume surge persistence ---
        vol_ratio = df['volume'] / df['volume'].rolling(168).mean().replace(0, np.nan)
        features['vol_surge_persistence'] = (vol_ratio > 2.0).rolling(12).mean()
        
        # --- Volatility term structure (short vs long) ---
        vol_short = returns.rolling(12).std()
        vol_long = returns.rolling(168).std()
        features['vol_term_structure'] = vol_short / vol_long.replace(0, np.nan)
        
        # --- Higher-frequency return autocorrelation ---
        for lag in [1, 2, 4]:
            features[f'ret_autocorr_lag{lag}'] = returns.rolling(48).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
            )
        
        # --- Intraday range expansion ---
        daily_range = (df['high'] - df['low']) / df['close']
        features['range_expansion'] = daily_range / daily_range.rolling(168).mean().replace(0, np.nan)
        
        return features


class DOGESentimentFeatures:
    """
    DOGE-specific: sentiment-proxy and meme-cycle features.
    
    Research basis:
    - DOGE driven by: social media hype, celebrity endorsement, retail sentiment
    - AI handles 89% of trading volume; bots execute 70% of trades
    - Contrarian strategies work: buy during sentiment troughs
    - No smart contracts, no DeFi — pure speculation + community
    - Infinite supply = no supply shock dynamics (unlike BTC)
    - Extreme mean-reversion at daily scale, but momentum at weekly scale
    - Key: detect retail FOMO/panic cycles via price pattern proxies
    """
    
    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        volume = df['volume']
        
        # --- Retail FOMO/panic detector (volume × price acceleration) ---
        price_accel = returns.diff()
        vol_surprise = volume / volume.rolling(168).mean().replace(0, np.nan)
        features['fomo_score'] = (price_accel * vol_surprise).rolling(6).mean()
        features['panic_score'] = (
            (returns < returns.rolling(168).quantile(0.05)).astype(int) *
            vol_surprise
        )
        
        # --- Pump-and-dump cycle detection ---
        # Sharp up followed by sharp down
        ret_6h = close.pct_change(6)
        ret_6h_prev = ret_6h.shift(6)
        features['pump_dump_signal'] = (
            (ret_6h_prev > 0.10) & (ret_6h < -0.05)
        ).astype(int).rolling(24).sum()
        
        # --- Extreme return frequency (meme coin signature) ---
        features['extreme_move_freq_24h'] = (
            (returns.abs() > returns.rolling(168).std() * 2).rolling(24).mean()
        )
        features['extreme_move_freq_72h'] = (
            (returns.abs() > returns.rolling(168).std() * 2).rolling(72).mean()
        )
        
        # --- Asymmetric volatility (downside vs upside) ---
        up_vol = returns.where(returns > 0, 0).rolling(72).std()
        down_vol = returns.where(returns < 0, 0).abs().rolling(72).std()
        features['vol_asymmetry'] = up_vol / down_vol.replace(0, np.nan)
        
        # --- Price memory (autocorrelation decay) ---
        for lag in [1, 6, 12, 24]:
            features[f'autocorr_{lag}h'] = returns.rolling(72).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
            )
        
        # --- Gap from VWAP proxy (volume-weighted) ---
        if 'volume' in df.columns:
            vwap = (df['close'] * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum().replace(0, np.nan)
            features['vwap_distance_24h'] = (close - vwap) / vwap.replace(0, np.nan)
        
        # --- Hype cycle position (distance from 7d high/low) ---
        high_7d = close.rolling(168).max()
        low_7d = close.rolling(168).min()
        range_7d = high_7d - low_7d
        features['hype_cycle_position'] = (close - low_7d) / range_7d.replace(0, np.nan)
        
        # --- Contrarian indicator: consecutive extreme moves ---
        big_move = (returns.abs() > 0.03).astype(int)
        features['consecutive_big_moves'] = big_move.rolling(12).sum()
        
        return features


# =============================================================================
# Coin-Specific Feature Labels
# =============================================================================

# Map symbol prefixes to their coin-specific feature class
COIN_FEATURE_MAP = {
    'BIP': BTCMeanReversionFeatures,     # BTC
    'BTC': BTCMeanReversionFeatures,
    'SLP': SOLEcosystemFeatures,         # SOL
    'SOL': SOLEcosystemFeatures,
    'DOP': DOGESentimentFeatures,         # DOGE
    'DOGE': DOGESentimentFeatures,
}


def get_coin_feature_class(symbol: str):
    """Return coin-specific feature class based on symbol prefix."""
    prefix = symbol.split('-')[0].upper()
    return COIN_FEATURE_MAP.get(prefix, None)


# =============================================================================
# Main Pipeline Class
# =============================================================================

@dataclass
class FeatureConfig:
    price_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48, 72, 168])
    volume_lookbacks: List[int] = field(default_factory=lambda: [1, 4, 12, 24, 48])
    compute_price: bool = True
    compute_volume: bool = True
    compute_funding: bool = True
    compute_oi: bool = True
    compute_regime: bool = True
    compute_coin_specific: bool = True  # v8: coin-specific features


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
            
            if self.config.compute_price:
                f_list.append(PriceFeatures.compute(df, self.config.price_lookbacks))
            if self.config.compute_volume:
                f_list.append(VolumeVolatilityFeatures.compute(df, self.config.volume_lookbacks))
            if self.config.compute_funding and symbol in funding_data:
                f_list.append(FundingFeatures.compute(df, funding_data[symbol]))
            if self.config.compute_oi and symbol in oi_data:
                f_list.append(OpenInterestFeatures.compute(df, oi_data[symbol]))
            if self.config.compute_regime:
                f_list.append(RegimeFeatures.compute(df))
            
            # v8: Coin-specific features
            if self.config.compute_coin_specific:
                coin_cls = get_coin_feature_class(symbol)
                if coin_cls:
                    logger.info(f"  Adding {coin_cls.__name__} for {symbol}")
                    f_list.append(coin_cls.compute(df))
            
            if f_list:
                combined = pd.concat(f_list, axis=1)
                all_features[symbol] = combined.loc[:, ~combined.columns.duplicated()]
        
        return all_features

    def compute_target(self, df: pd.DataFrame, horizon: int = 24, vol_mult: float = 1.8) -> pd.Series:
        """Triple Barrier Method: Labels +1 (TP), -1 (SL), or 0 (Time-out)."""
        close = df['close']
        vol = close.pct_change().rolling(24).std()
        target = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - horizon):
            entry = close.iloc[i]
            v = vol.iloc[i]
            if pd.isna(v) or v == 0:
                continue
            tp = entry * (1 + vol_mult * v)
            sl = entry * (1 - vol_mult * v)
            
            future = close.iloc[i+1:i+1+horizon]
            hit_tp = (future >= tp).any()
            hit_sl = (future <= sl).any()
            
            if hit_tp and hit_sl:
                tp_idx = (future >= tp).idxmax()
                sl_idx = (future <= sl).idxmax()
                target.iloc[i] = 1.0 if tp_idx <= sl_idx else 0.0
            elif hit_tp:
                target.iloc[i] = 1.0
            elif hit_sl:
                target.iloc[i] = 0.0
            else:
                final_ret = (future.iloc[-1] - entry) / entry
                target.iloc[i] = 1.0 if final_ret > 0 else 0.0
        
        return target