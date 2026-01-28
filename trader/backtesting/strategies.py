"""
Trading Strategies for Backtesting.

Based on signals identified in feature engineering:
- Mean reversion (Bollinger Bands)
- Momentum reversal (RSI + returns)
"""
import pandas as pd

from datetime import datetime
from typing import Dict, List

from .engine import Side, Signal, Portfolio


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
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Bollinger Band position.
    
    Based on IC analysis showing bb_position has -0.06 IC with 1h returns
    and -0.28 IC with 24h returns (strong mean reversion).
    
    Rules:
    - Long when bb_position < -entry_threshold (oversold)
    - Short when bb_position > +entry_threshold (overbought)
    - Exit when bb_position crosses exit_threshold
    """
    
    def __init__(
        self,
        symbols: List[str],
        entry_threshold: float = 1.0,
        exit_threshold: float = 0.0,
        position_size: float = 0.1,
        use_rsi_filter: bool = False,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        super().__init__(symbols)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.use_rsi_filter = use_rsi_filter
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
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
            
            bb_pos = feat_df.loc[timestamp, 'bb_position'] if 'bb_position' in feat_df.columns else None
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else None
            
            if pd.isna(bb_pos):
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Optional RSI filter
            rsi_long_ok = not self.use_rsi_filter or (rsi is not None and rsi < self.rsi_oversold)
            rsi_short_ok = not self.use_rsi_filter or (rsi is not None and rsi > self.rsi_overbought)
            
            # Entry signals
            if current_side == Side.FLAT:
                if bb_pos < -self.entry_threshold and rsi_long_ok:
                    confidence = min(1.0, abs(bb_pos) / 2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                elif bb_pos > self.entry_threshold and rsi_short_ok:
                    confidence = min(1.0, abs(bb_pos) / 2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
            
            # Exit signals
            elif current_side == Side.LONG:
                if bb_pos > self.exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
            
            elif current_side == Side.SHORT:
                if bb_pos < -self.exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
        
        return signals


class MomentumReversalStrategy(BaseStrategy):
    """
    Momentum reversal strategy.
    
    Based on IC showing 12h-24h returns have negative correlation
    with forward returns (momentum reversal).
    
    Rules:
    - Long when RSI oversold AND recent return is very negative
    - Short when RSI overbought AND recent return is very positive
    - Exit when RSI normalizes
    """
    
    def __init__(
        self,
        symbols: List[str],
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        return_threshold: float = 0.03,
        position_size: float = 0.1,
        exit_rsi: float = 50.0,
    ):
        super().__init__(symbols)
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.return_threshold = return_threshold
        self.position_size = position_size
        self.exit_rsi = exit_rsi
    
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
            
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else None
            ret_24h = feat_df.loc[timestamp, 'return_24h'] if 'return_24h' in feat_df.columns else None
            
            if pd.isna(rsi) or pd.isna(ret_24h):
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            if current_side == Side.FLAT:
                # Oversold with strong down move - expect reversal
                if rsi < self.rsi_oversold and ret_24h < -self.return_threshold:
                    confidence = min(1.0, (self.rsi_oversold - rsi) / 30)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                # Overbought with strong up move - expect reversal
                elif rsi > self.rsi_overbought and ret_24h > self.return_threshold:
                    confidence = min(1.0, (rsi - self.rsi_overbought) / 30)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
            
            # Exit when RSI normalizes
            elif current_side == Side.LONG and rsi > self.exit_rsi:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=Side.FLAT,
                ))
            elif current_side == Side.SHORT and rsi < self.exit_rsi:
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=Side.FLAT,
                ))
        
        return signals


class CombinedStrategy(BaseStrategy):
    """
    Combined strategy using multiple signals.
    
    Requires agreement between indicators for entry.
    """
    
    def __init__(
        self,
        symbols: List[str],
        bb_threshold: float = 1.0,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        position_size: float = 0.1,
    ):
        super().__init__(symbols)
        self.bb_threshold = bb_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.position_size = position_size
    
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
            
            bb_pos = feat_df.loc[timestamp, 'bb_position'] if 'bb_position' in feat_df.columns else None
            rsi = feat_df.loc[timestamp, 'rsi_14'] if 'rsi_14' in feat_df.columns else None
            
            if pd.isna(bb_pos) or pd.isna(rsi):
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Entry requires both BB and RSI to agree
            if current_side == Side.FLAT:
                # Both oversold
                if bb_pos < -self.bb_threshold and rsi < self.rsi_oversold:
                    confidence = min(1.0, (abs(bb_pos) + (self.rsi_oversold - rsi) / 50) / 2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
                # Both overbought
                elif bb_pos > self.bb_threshold and rsi > self.rsi_overbought:
                    confidence = min(1.0, (abs(bb_pos) + (rsi - self.rsi_overbought) / 50) / 2)
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size,
                    ))
            
            # Exit when either returns to neutral
            elif current_side == Side.LONG:
                if bb_pos > 0 or rsi > 50:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
            elif current_side == Side.SHORT:
                if bb_pos < 0 or rsi < 50:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
        
        return signals