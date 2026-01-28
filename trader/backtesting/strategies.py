"""
Funding Rate Strategies - Phase 1

Key strategies based on funding rate signals:

1. FundingArbitrageStrategy: Trade against extreme funding rates
2. FundingAwareMeanReversion: Original mean reversion + funding filter
3. CarryStrategy: Pure carry trade based on cumulative funding

Design.md Section 5.1.1 rationale:
- Extreme funding rates (|z| > 2) tend to revert
- Retail traders systematically overpay for leverage during euphoric/panic periods
- Funding rate arbitrage requires capital commitment but offers edge

Key insight: When funding is extremely positive (z > 2):
- Longs are paying shorts
- Market is overly bullish
- Go SHORT to collect funding AND benefit from likely mean reversion
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum


class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class Signal:
    """Trading signal."""
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        direction: Side,
        confidence: float = 1.0,
        target_weight: float = 1.0,
        reason: str = "",
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.direction = direction
        self.confidence = confidence
        self.target_weight = target_weight
        self.reason = reason


class BaseStrategy:
    """Base strategy class."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        raise NotImplementedError


class FundingArbitrageStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy
    
    This is the PRIMARY strategy for Phase 1.
    
    Logic:
    - When funding_rate_zscore > threshold: Go SHORT
      (Collect funding from longs + bet on mean reversion)
    - When funding_rate_zscore < -threshold: Go LONG
      (Collect funding from shorts + bet on mean reversion)
    - Exit when funding normalizes (crosses zero)
    
    This strategy exploits:
    1. Direct carry (collecting funding payments)
    2. Mean reversion (extreme funding tends to normalize)
    
    Key parameters:
    - entry_threshold: Z-score threshold to enter (default 2.0)
    - exit_threshold: Z-score to exit (default 0.5)
    - min_hold_hours: Minimum hold to collect funding (24h default)
    """
    
    def __init__(
        self,
        symbols: List[str],
        entry_threshold: float = 2.0,      # Only enter on extreme funding
        exit_threshold: float = 0.5,        # Exit when funding normalizes
        min_hold_hours: int = 24,           # Minimum hold for funding collection
        max_hold_hours: int = 168,          # Maximum hold (1 week)
        position_size: float = 0.15,        # 15% of portfolio per position
        use_price_confirmation: bool = True, # Also check price extremes
        bb_confirmation_threshold: float = 1.5,  # BB threshold for confirmation
    ):
        super().__init__(symbols)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.position_size = position_size
        self.use_price_confirmation = use_price_confirmation
        self.bb_confirmation_threshold = bb_confirmation_threshold
        
        # Track entry times for hold period enforcement
        self.entry_times: Dict[str, datetime] = {}
        self.entry_reasons: Dict[str, str] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get funding features
            funding_zscore = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
            funding_rate_bps = self._get_feature(feat_df, timestamp, 'funding_rate_bps')
            
            # Get price features for confirmation
            bb_position = self._get_feature(feat_df, timestamp, 'bb_position')
            rsi = self._get_feature(feat_df, timestamp, 'rsi_14')
            
            if funding_zscore is None:
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Check hold period
            min_hold_met = True
            max_hold_exceeded = False
            
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                min_hold_met = hours_held >= self.min_hold_hours
                max_hold_exceeded = hours_held >= self.max_hold_hours
            
            # === EXIT LOGIC ===
            if current_side != Side.FLAT:
                should_exit = False
                exit_reason = ""
                
                # Max hold exceeded - force exit
                if max_hold_exceeded:
                    should_exit = True
                    exit_reason = "Max hold exceeded"
                
                # Funding normalized - exit
                elif min_hold_met:
                    if current_side == Side.SHORT and funding_zscore < self.exit_threshold:
                        should_exit = True
                        exit_reason = f"Funding normalized (z={funding_zscore:.2f})"
                    elif current_side == Side.LONG and funding_zscore > -self.exit_threshold:
                        should_exit = True
                        exit_reason = f"Funding normalized (z={funding_zscore:.2f})"
                
                if should_exit:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                        reason=exit_reason,
                    ))
                    self.entry_times.pop(symbol, None)
                    self.entry_reasons.pop(symbol, None)
                    continue
            
            # === ENTRY LOGIC ===
            if current_side == Side.FLAT:
                
                # HIGH POSITIVE FUNDING -> SHORT
                if funding_zscore > self.entry_threshold:
                    # Optional: require price confirmation (overbought)
                    if self.use_price_confirmation:
                        if bb_position is None or bb_position < self.bb_confirmation_threshold:
                            continue
                    
                    # Calculate confidence based on z-score extremity
                    confidence = min(1.0, (funding_zscore - self.entry_threshold) / 2 + 0.5)
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"High funding z={funding_zscore:.2f}, rate={funding_rate_bps:.2f}bps",
                    ))
                    self.entry_times[symbol] = timestamp
                    self.entry_reasons[symbol] = "funding_short"
                
                # HIGH NEGATIVE FUNDING -> LONG
                elif funding_zscore < -self.entry_threshold:
                    # Optional: require price confirmation (oversold)
                    if self.use_price_confirmation:
                        if bb_position is None or bb_position > -self.bb_confirmation_threshold:
                            continue
                    
                    confidence = min(1.0, (abs(funding_zscore) - self.entry_threshold) / 2 + 0.5)
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"Low funding z={funding_zscore:.2f}, rate={funding_rate_bps:.2f}bps",
                    ))
                    self.entry_times[symbol] = timestamp
                    self.entry_reasons[symbol] = "funding_long"
        
        return signals
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        """Safely get feature value."""
        if name not in df.columns:
            return None
        val = df.loc[timestamp, name]
        if pd.isna(val):
            return None
        return val


class FundingAwareMeanReversion(BaseStrategy):
    """
    Mean Reversion with Funding Filter
    
    This is your existing mean reversion strategy, enhanced with funding awareness:
    - Only go LONG when funding is not extremely positive
    - Only go SHORT when funding is not extremely negative
    
    This prevents fighting against strong funding momentum.
    """
    
    def __init__(
        self,
        symbols: List[str],
        bb_entry_threshold: float = 1.5,
        bb_exit_threshold: float = 0.2,
        max_funding_zscore: float = 1.5,    # Don't go long if funding > this
        min_funding_zscore: float = -1.5,   # Don't go short if funding < this
        position_size: float = 0.15,
        min_hold_hours: int = 24,
    ):
        super().__init__(symbols)
        self.bb_entry_threshold = bb_entry_threshold
        self.bb_exit_threshold = bb_exit_threshold
        self.max_funding_zscore = max_funding_zscore
        self.min_funding_zscore = min_funding_zscore
        self.position_size = position_size
        self.min_hold_hours = min_hold_hours
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get features
            bb_position = self._get_feature(feat_df, timestamp, 'bb_position')
            funding_zscore = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
            rsi = self._get_feature(feat_df, timestamp, 'rsi_14')
            
            if bb_position is None:
                continue
            
            # Default funding to neutral if not available
            if funding_zscore is None:
                funding_zscore = 0
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Check hold period
            min_hold_met = True
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                min_hold_met = hours_held >= self.min_hold_hours
            
            # === EXIT LOGIC ===
            if current_side != Side.FLAT and min_hold_met:
                if current_side == Side.LONG and bb_position > self.bb_exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                        reason="BB normalized",
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
                    
                elif current_side == Side.SHORT and bb_position < -self.bb_exit_threshold:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                        reason="BB normalized",
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # === ENTRY LOGIC ===
            if current_side == Side.FLAT:
                
                # LONG: Oversold price + favorable funding
                if bb_position < -self.bb_entry_threshold:
                    # Only go long if funding isn't extremely positive
                    if funding_zscore <= self.max_funding_zscore:
                        confidence = min(1.0, abs(bb_position) / 3)
                        
                        # Boost confidence if funding is also favorable (negative)
                        if funding_zscore < 0:
                            confidence = min(1.0, confidence * 1.2)
                        
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.LONG,
                            confidence=confidence,
                            target_weight=self.position_size * confidence,
                            reason=f"BB={bb_position:.2f}, funding_z={funding_zscore:.2f}",
                        ))
                        self.entry_times[symbol] = timestamp
                
                # SHORT: Overbought price + favorable funding
                elif bb_position > self.bb_entry_threshold:
                    # Only go short if funding isn't extremely negative
                    if funding_zscore >= self.min_funding_zscore:
                        confidence = min(1.0, abs(bb_position) / 3)
                        
                        # Boost confidence if funding is also favorable (positive)
                        if funding_zscore > 0:
                            confidence = min(1.0, confidence * 1.2)
                        
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.SHORT,
                            confidence=confidence,
                            target_weight=self.position_size * confidence,
                            reason=f"BB={bb_position:.2f}, funding_z={funding_zscore:.2f}",
                        ))
                        self.entry_times[symbol] = timestamp
        
        return signals
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        if name not in df.columns:
            return None
        val = df.loc[timestamp, name]
        return None if pd.isna(val) else val


class PureFundingCarryStrategy(BaseStrategy):
    """
    Pure Funding Carry Strategy
    
    The simplest funding-based strategy:
    - Go SHORT when cumulative funding is high (collect from longs)
    - Go LONG when cumulative funding is low/negative (collect from shorts)
    
    This is a CARRY trade - we're just collecting funding payments.
    No price-based signals, purely funding driven.
    
    Lower expected Sharpe but very low correlation with price momentum.
    """
    
    def __init__(
        self,
        symbols: List[str],
        cumulative_threshold_bps: float = 10.0,  # 10 bps cumulative over 24h
        position_size: float = 0.1,
        hold_hours: int = 72,  # Hold for 3 days
    ):
        super().__init__(symbols)
        self.cumulative_threshold_bps = cumulative_threshold_bps
        self.position_size = position_size
        self.hold_hours = hold_hours
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get cumulative funding (24h)
            cum_funding = self._get_feature(feat_df, timestamp, 'cumulative_funding_24h')
            
            if cum_funding is None:
                continue
            
            # Convert to bps
            cum_funding_bps = cum_funding * 10000
            
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
                        reason=f"Hold period complete ({hours_held:.0f}h)",
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # Entry
            if current_side == Side.FLAT:
                # High positive cumulative funding -> Short to collect
                if cum_funding_bps > self.cumulative_threshold_bps:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=0.7,  # Lower confidence for pure carry
                        target_weight=self.position_size,
                        reason=f"High cum funding: {cum_funding_bps:.2f}bps/24h",
                    ))
                    self.entry_times[symbol] = timestamp
                
                # High negative cumulative funding -> Long to collect
                elif cum_funding_bps < -self.cumulative_threshold_bps:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=0.7,
                        target_weight=self.position_size,
                        reason=f"Low cum funding: {cum_funding_bps:.2f}bps/24h",
                    ))
                    self.entry_times[symbol] = timestamp
        
        return signals
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        if name not in df.columns:
            return None
        val = df.loc[timestamp, name]
        return None if pd.isna(val) else val


class CombinedFundingPriceStrategy(BaseStrategy):
    """
    Combined Funding + Price Strategy
    
    The most robust strategy - requires BOTH funding AND price signals to align.
    
    Entry requires:
    1. Extreme funding (z > 1.5 or z < -1.5)
    2. Price confirmation (BB position extreme)
    3. RSI confirmation (optional)
    
    This is the highest-conviction, lowest-frequency strategy.
    """
    
    def __init__(
        self,
        symbols: List[str],
        funding_threshold: float = 1.5,
        bb_threshold: float = 1.5,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        require_all_signals: bool = False,  # Require 2 of 3 or all 3
        position_size: float = 0.2,
        min_hold_hours: int = 48,
    ):
        super().__init__(symbols)
        self.funding_threshold = funding_threshold
        self.bb_threshold = bb_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.require_all_signals = require_all_signals
        self.position_size = position_size
        self.min_hold_hours = min_hold_hours
        self.entry_times: Dict[str, datetime] = {}
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features:
                continue
            
            feat_df = features[symbol]
            if timestamp not in feat_df.index:
                continue
            
            # Get all features
            funding_z = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
            bb_pos = self._get_feature(feat_df, timestamp, 'bb_position')
            rsi = self._get_feature(feat_df, timestamp, 'rsi_14')
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Check hold period
            min_hold_met = True
            if symbol in self.entry_times:
                hours_held = (timestamp - self.entry_times[symbol]).total_seconds() / 3600
                min_hold_met = hours_held >= self.min_hold_hours
            
            # Exit when signals neutralize
            if current_side != Side.FLAT and min_hold_met:
                should_exit = False
                
                if current_side == Side.LONG:
                    # Exit long when price normalizes AND funding normalizes
                    if (bb_pos is not None and bb_pos > 0) or \
                       (funding_z is not None and funding_z > 0):
                        should_exit = True
                elif current_side == Side.SHORT:
                    if (bb_pos is not None and bb_pos < 0) or \
                       (funding_z is not None and funding_z < 0):
                        should_exit = True
                
                if should_exit:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # Entry: Count bullish/bearish factors
            if current_side == Side.FLAT:
                bullish_factors = 0
                bearish_factors = 0
                
                # Funding signal
                if funding_z is not None:
                    if funding_z < -self.funding_threshold:
                        bullish_factors += 1
                    elif funding_z > self.funding_threshold:
                        bearish_factors += 1
                
                # BB signal
                if bb_pos is not None:
                    if bb_pos < -self.bb_threshold:
                        bullish_factors += 1
                    elif bb_pos > self.bb_threshold:
                        bearish_factors += 1
                
                # RSI signal
                if rsi is not None:
                    if rsi < self.rsi_oversold:
                        bullish_factors += 1
                    elif rsi > self.rsi_overbought:
                        bearish_factors += 1
                
                # Entry threshold
                required = 3 if self.require_all_signals else 2
                
                if bullish_factors >= required:
                    confidence = bullish_factors / 3
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"{bullish_factors} bullish factors",
                    ))
                    self.entry_times[symbol] = timestamp
                
                elif bearish_factors >= required:
                    confidence = bearish_factors / 3
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"{bearish_factors} bearish factors",
                    ))
                    self.entry_times[symbol] = timestamp
        
        return signals
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        if name not in df.columns:
            return None
        val = df.loc[timestamp, name]
        return None if pd.isna(val) else val