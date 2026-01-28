"""
Improved Trading Strategies - Cost-Aware

Key insight from backtesting:
- Coinbase fees: ~0.65% round-trip
- Need signals strong enough to overcome costs
- Trade less frequently, hold longer

The 24h IC values were MUCH stronger than 1h:
- bb_upper/lower: -0.28 (vs -0.06)
- range_48h: +0.15
- momentum_168h: -0.11

Strategy adjustments:
1. Higher entry thresholds (require stronger signals)
2. Longer holding periods (reduce trading frequency)  
3. Multi-factor confirmation (reduce false signals)
4. Time-based exits (not just indicator-based)
"""
import pandas as pd

from datetime import datetime
from typing import Dict, List
from backtesting.engine import Side, Signal, Portfolio


class BaseStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Signal]:
        raise NotImplementedError


class LowFrequencyMeanReversion(BaseStrategy):
    """
    Low-frequency mean reversion strategy.
    
    Changes from original:
    - Much higher entry threshold (2.0 vs 1.0)
    - Minimum hold period (24h) to reduce churn
    - Uses vol_regime_ratio to avoid high-vol periods
    - Position sizing based on signal strength
    """
    
    def __init__(
        self,
        symbols: List[str],
        entry_threshold: float = 2.0,     # Much higher - only extreme moves
        exit_threshold: float = 0.3,       # Exit closer to mean
        min_hold_hours: int = 24,          # Minimum hold period
        position_size: float = 0.15,
        max_vol_ratio: float = 1.5,        # Skip high volatility
    ):
        super().__init__(symbols)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_hold_hours = min_hold_hours
        self.position_size = position_size
        self.max_vol_ratio = max_vol_ratio
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get features
            bb_pos = feat_df.loc[timestamp, 'bb_position'] if 'bb_position' in feat_df.columns else None
            vol_ratio = feat_df.loc[timestamp, 'vol_regime_ratio'] if 'vol_regime_ratio' in feat_df.columns else 1.0
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else 50
            
            if pd.isna(bb_pos):
                continue
            
            if pd.isna(vol_ratio):
                vol_ratio = 1.0
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Check minimum hold period
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                if hours_held < self.min_hold_hours:
                    continue  # Don't exit yet
            
            # Skip high volatility periods for entries
            if current_side == Side.FLAT and vol_ratio > self.max_vol_ratio:
                continue
            
            # Entry signals - require EXTREME moves
            if current_side == Side.FLAT:
                # Strong oversold: bb_pos < -2.0 AND RSI < 25
                if bb_pos < -self.entry_threshold and (pd.isna(rsi) or rsi < 30):
                    confidence = min(1.0, abs(bb_pos) / 3)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                    ))
                    self.entry_times[symbol] = timestamp
                
                # Strong overbought
                elif bb_pos > self.entry_threshold and (pd.isna(rsi) or rsi > 70):
                    confidence = min(1.0, abs(bb_pos) / 3)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                    ))
                    self.entry_times[symbol] = timestamp
            
            # Exit signals - wait for mean reversion
            elif current_side == Side.LONG:
                # Exit when price returns toward mean
                if bb_pos > self.exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
            
            elif current_side == Side.SHORT:
                if bb_pos < -self.exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
        
        return signals


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility breakout strategy.
    
    Based on IC analysis showing:
    - range_48h: +0.15 IC with 24h returns (volatility predicts movement)
    - trend_strength: +0.08 IC (trend continuation)
    
    This is a TREND-FOLLOWING strategy (not mean reversion).
    """
    
    def __init__(
        self,
        symbols: List[str],
        volatility_threshold: float = 1.5,  # Vol ratio threshold
        trend_threshold: float = 40,         # Trend strength threshold
        position_size: float = 0.1,
        hold_hours: int = 48,               # Hold for 2 days
    ):
        super().__init__(symbols)
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.position_size = position_size
        self.hold_hours = hold_hours
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get features
            vol_ratio = feat_df.loc[timestamp, 'vol_regime_ratio'] if 'vol_regime_ratio' in feat_df.columns else None
            trend_strength = feat_df.loc[timestamp, 'trend_strength'] if 'trend_strength' in feat_df.columns else None
            return_24h = feat_df.loc[timestamp, 'return_24h'] if 'return_24h' in feat_df.columns else None
            
            if pd.isna(vol_ratio) or pd.isna(trend_strength) or pd.isna(return_24h):
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Time-based exit
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                if hours_held >= self.hold_hours and current_side != Side.FLAT:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # Entry: High volatility + strong trend + same direction momentum
            if current_side == Side.FLAT:
                if vol_ratio > self.volatility_threshold and trend_strength > self.trend_threshold:
                    # Follow the trend direction
                    if return_24h > 0.02:  # Up trend
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.LONG,
                            confidence=min(1.0, vol_ratio / 2),
                            target_weight=self.position_size,
                        ))
                        self.entry_times[symbol] = timestamp
                    elif return_24h < -0.02:  # Down trend
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.SHORT,
                            confidence=min(1.0, vol_ratio / 2),
                            target_weight=self.position_size,
                        ))
                        self.entry_times[symbol] = timestamp
        
        return signals


class WeeklyMomentumReversal(BaseStrategy):
    """
    Weekly momentum reversal strategy.
    
    Based on IC analysis:
    - momentum_168h_positive: -0.11 IC (weekly momentum reverses)
    - return_168h: -0.09 IC
    
    Trade AGAINST strong weekly moves.
    """
    
    def __init__(
        self,
        symbols: List[str],
        return_threshold: float = 0.10,     # 10% weekly move
        position_size: float = 0.15,
        hold_days: int = 7,                 # Hold for a week
    ):
        super().__init__(symbols)
        self.return_threshold = return_threshold
        self.position_size = position_size
        self.hold_days = hold_days
        self.entry_times: Dict[str, datetime] = {}
        self.last_signal_time: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Only check once per day (reduce computation)
            if symbol in self.last_signal_time:
                hours_since = (timestamp - self.last_signal_time[symbol]).total_seconds() / 3600
                if hours_since < 24:
                    continue
            
            self.last_signal_time[symbol] = timestamp
            
            # Get weekly return
            return_168h = feat_df.loc[timestamp, 'return_168h'] if 'return_168h' in feat_df.columns else None
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else 50
            
            if pd.isna(return_168h):
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Time-based exit
            if symbol in self.entry_times:
                days_held = (timestamp - self.entry_times[symbol]).total_seconds() / 86400
                if days_held >= self.hold_days and current_side != Side.FLAT:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # Entry: Strong weekly move = expect reversal
            if current_side == Side.FLAT:
                # Big weekly drop -> go long (expect bounce)
                if return_168h < -self.return_threshold:
                    confidence = min(1.0, abs(return_168h) / 0.2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                    self.entry_times[symbol] = timestamp
                
                # Big weekly rally -> go short (expect pullback)
                elif return_168h > self.return_threshold:
                    confidence = min(1.0, abs(return_168h) / 0.2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                    self.entry_times[symbol] = timestamp
        
        return signals


class MultiFactorStrategy(BaseStrategy):
    """
    Multi-factor strategy combining multiple signals.
    
    Only trades when MULTIPLE signals agree:
    - BB extreme (mean reversion)
    - RSI extreme (momentum reversal)
    - NOT high volatility regime (avoid whipsaws)
    
    This is the most conservative approach.
    """
    
    def __init__(
        self,
        symbols: List[str],
        bb_threshold: float = 1.5,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        max_vol_ratio: float = 1.3,
        position_size: float = 0.2,
        min_hold_hours: int = 48,
    ):
        super().__init__(symbols)
        self.bb_threshold = bb_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.max_vol_ratio = max_vol_ratio
        self.position_size = position_size
        self.min_hold_hours = min_hold_hours
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio: Portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get all features
            bb_pos = feat_df.loc[timestamp, 'bb_position'] if 'bb_position' in feat_df.columns else None
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else None
            vol_ratio = feat_df.loc[timestamp, 'vol_regime_ratio'] if 'vol_regime_ratio' in feat_df.columns else 1.0
            
            if pd.isna(bb_pos) or pd.isna(rsi):
                continue
            if pd.isna(vol_ratio):
                vol_ratio = 1.0
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Check hold period
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                min_hold_met = hours_held >= self.min_hold_hours
            else:
                min_hold_met = True
            
            # Skip high volatility
            if vol_ratio > self.max_vol_ratio:
                continue
            
            # Count bullish/bearish factors
            bullish_factors = 0
            bearish_factors = 0
            
            if bb_pos < -self.bb_threshold:
                bullish_factors += 1
            if bb_pos > self.bb_threshold:
                bearish_factors += 1
            
            if rsi < self.rsi_oversold:
                bullish_factors += 1
            if rsi > self.rsi_overbought:
                bearish_factors += 1
            
            # Entry: Need 2 factors
            if current_side == Side.FLAT:
                if bullish_factors >= 2:
                    confidence = bullish_factors / 3
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                    self.entry_times[symbol] = timestamp
                
                elif bearish_factors >= 2:
                    confidence = bearish_factors / 3
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                    self.entry_times[symbol] = timestamp
            
            # Exit: When signals neutralize
            elif min_hold_met:
                if current_side == Side.LONG and (bb_pos > 0 or rsi > 50):
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
                
                elif current_side == Side.SHORT and (bb_pos < 0 or rsi < 50):
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
        
        return signals