"""
Funding Rate and OI Strategies - VERSION

Key fixes:
1. Added regime filter to avoid shorting in bull markets
2. Improved entry/exit logic with trend awareness
3. Better position sizing based on signal quality
4. Momentum filter to avoid fighting trends
5. Configurable stop-loss per strategy

Design.md Section 5.1.1 rationale:
- Extreme funding rates (|z| > 2) tend to revert
- BUT: In strong trends, mean reversion can be deadly
- Need regime awareness to avoid fighting the trend
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List
from enum import Enum


class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"      # Strong uptrend - avoid shorts
    BULL = "bull"                     # Uptrend - reduce short size
    NEUTRAL = "neutral"               # Range-bound - normal trading
    BEAR = "bear"                     # Downtrend - reduce long size
    STRONG_BEAR = "strong_bear"       # Strong downtrend - avoid longs
    HIGH_VOLATILITY = "high_vol"      # High vol regime - reduce all sizes


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


class RegimeDetector:
    """
    Detect market regime to avoid fighting strong trends.
    
    This is CRITICAL for funding arbitrage strategies that tend to
    short during bull markets (when funding is high).
    """
    
    def __init__(
        self,
        trend_lookback: int = 168,       # 1 week for trend
        momentum_lookback: int = 24,      # 24h for momentum
        volatility_lookback: int = 168,   # 1 week for vol baseline
        strong_trend_threshold: float = 0.15,  # 15% move = strong trend
        trend_threshold: float = 0.05,    # 5% move = trend
        high_vol_multiplier: float = 2.0, # 2x normal vol = high vol
    ):
        self.trend_lookback = trend_lookback
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.strong_trend_threshold = strong_trend_threshold
        self.trend_threshold = trend_threshold
        self.high_vol_multiplier = high_vol_multiplier
    
    def detect_regime(
        self,
        ohlcv: pd.DataFrame,
        timestamp: datetime,
    ) -> MarketRegime:
        """
        Detect current market regime.
        
        Returns regime classification based on:
        1. Price trend over lookback period
        2. Recent momentum
        3. Volatility level
        """
        if timestamp not in ohlcv.index:
            return MarketRegime.NEUTRAL
        
        # Get historical data up to current timestamp
        hist = ohlcv.loc[:timestamp]
        
        if len(hist) < self.trend_lookback:
            return MarketRegime.NEUTRAL
        
        current_price = hist['close'].iloc[-1]
        
        # 1. Calculate trend (price change over lookback)
        lookback_price = hist['close'].iloc[-self.trend_lookback]
        trend_return = (current_price - lookback_price) / lookback_price
        
        # 2. Calculate recent momentum
        if len(hist) >= self.momentum_lookback:
            momentum_price = hist['close'].iloc[-self.momentum_lookback]
            momentum_return = (current_price - momentum_price) / momentum_price
        else:
            momentum_return = 0
        
        # 3. Calculate volatility regime
        returns = hist['close'].pct_change().dropna()
        if len(returns) >= self.volatility_lookback:
            current_vol = returns.iloc[-24:].std() if len(returns) >= 24 else returns.std()
            baseline_vol = returns.iloc[-self.volatility_lookback:].std()
            vol_ratio = current_vol / baseline_vol if baseline_vol > 0 else 1.0
        else:
            vol_ratio = 1.0
        
        # High volatility regime takes precedence
        if vol_ratio > self.high_vol_multiplier:
            return MarketRegime.HIGH_VOLATILITY
        
        # Determine trend regime
        if trend_return > self.strong_trend_threshold and momentum_return > 0:
            return MarketRegime.STRONG_BULL
        elif trend_return > self.trend_threshold and momentum_return > 0:
            return MarketRegime.BULL
        elif trend_return < -self.strong_trend_threshold and momentum_return < 0:
            return MarketRegime.STRONG_BEAR
        elif trend_return < -self.trend_threshold and momentum_return < 0:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL
    
    def get_position_multiplier(
        self,
        regime: MarketRegime,
        signal_direction: Side,
    ) -> float:
        """
        Get position size multiplier based on regime and direction.
        
        Key insight: Don't short in bull markets, don't long in bear markets.
        """
        multipliers = {
            # (regime, direction): multiplier
            (MarketRegime.STRONG_BULL, Side.SHORT): 0.0,   # NO shorts in strong bull
            (MarketRegime.STRONG_BULL, Side.LONG): 1.0,
            (MarketRegime.BULL, Side.SHORT): 0.25,          # Reduced shorts in bull
            (MarketRegime.BULL, Side.LONG): 1.0,
            (MarketRegime.NEUTRAL, Side.SHORT): 1.0,
            (MarketRegime.NEUTRAL, Side.LONG): 1.0,
            (MarketRegime.BEAR, Side.SHORT): 1.0,
            (MarketRegime.BEAR, Side.LONG): 0.25,           # Reduced longs in bear
            (MarketRegime.STRONG_BEAR, Side.LONG): 0.0,    # NO longs in strong bear
            (MarketRegime.STRONG_BEAR, Side.SHORT): 1.0,
            (MarketRegime.HIGH_VOLATILITY, Side.SHORT): 0.5,  # Reduce all in high vol
            (MarketRegime.HIGH_VOLATILITY, Side.LONG): 0.5,
        }
        
        return multipliers.get((regime, signal_direction), 1.0)


class FundingArbitrageStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage Strategy -  VERSION
    
    Key improvements:
    1. Regime filter to avoid shorting in bull markets
    2. Trend confirmation before entry
    3. Dynamic position sizing based on signal quality
    4. Proper stop-loss integration
    
    Logic:
    - When funding_rate_zscore > threshold AND regime allows: Go SHORT
    - When funding_rate_zscore < -threshold AND regime allows: Go LONG
    - Exit when funding normalizes OR trend turns against position
    """
    
    def __init__(
        self,
        symbols: List[str],
        entry_threshold: float = 2.0,      # Z-score threshold to enter
        exit_threshold: float = 0.5,        # Z-score to exit
        min_hold_hours: int = 24,           # Minimum hold for funding collection
        max_hold_hours: int = 168,          # Maximum hold (1 week)
        position_size: float = 0.15,        # 15% of portfolio per position
        use_price_confirmation: bool = True, # Check price extremes
        bb_confirmation_threshold: float = 1.5,
        use_regime_filter: bool = True,     # NEW: Enable regime filtering
        use_trend_filter: bool = True,      # NEW: Check trend before entry
        trend_ma_period: int = 48,          # NEW: MA period for trend
    ):
        super().__init__(symbols)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.position_size = position_size
        self.use_price_confirmation = use_price_confirmation
        self.bb_confirmation_threshold = bb_confirmation_threshold
        self.use_regime_filter = use_regime_filter
        self.use_trend_filter = use_trend_filter
        self.trend_ma_period = trend_ma_period
        
        # Track entry times for hold period enforcement
        self.entry_times: Dict[str, datetime] = {}
        self.entry_reasons: Dict[str, str] = {}
        
        # Regime detector
        self.regime_detector = RegimeDetector()
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        """Safely get feature value."""
        if name not in df.columns:
            return None
        if timestamp not in df.index:
            return None
        val = df.loc[timestamp, name]
        if pd.isna(val):
            return None
        return val
    
    def _check_trend_alignment(
        self,
        ohlcv: pd.DataFrame,
        timestamp: datetime,
        signal_direction: Side,
    ) -> bool:
        """
        Check if signal aligns with or doesn't fight the trend.
        
        For mean reversion strategies, we want to enter when:
        - Price is extended BUT showing signs of reversal
        - NOT when trend is accelerating against us
        """
        if timestamp not in ohlcv.index:
            return True  # Default to allowing
        
        hist = ohlcv.loc[:timestamp]
        if len(hist) < self.trend_ma_period:
            return True
        
        current_price = hist['close'].iloc[-1]
        ma = hist['close'].iloc[-self.trend_ma_period:].mean()
        
        # Calculate short-term momentum (last 4 hours)
        if len(hist) >= 4:
            short_momentum = (current_price - hist['close'].iloc[-4]) / hist['close'].iloc[-4]
        else:
            short_momentum = 0
        
        # For SHORT signals: Allow if price is above MA but momentum is slowing
        if signal_direction == Side.SHORT:
            # Price should be extended above MA
            if current_price < ma:
                return False  # Don't short below MA
            # Short-term momentum should not be strongly positive
            if short_momentum > 0.02:  # 2% in 4 hours = strong momentum
                return False
            return True
        
        # For LONG signals: Allow if price is below MA but momentum is slowing
        elif signal_direction == Side.LONG:
            # Price should be extended below MA
            if current_price > ma:
                return False  # Don't long above MA
            # Short-term momentum should not be strongly negative
            if short_momentum < -0.02:
                return False
            return True
        
        return True
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features or symbol not in ohlcv_data:
                continue
            
            feat_df = features[symbol]
            ohlcv = ohlcv_data[symbol]
            
            if timestamp not in feat_df.index or timestamp not in ohlcv.index:
                continue
            
            # Get current features
            funding_zscore = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
            funding_rate_bps = self._get_feature(feat_df, timestamp, 'funding_rate_bps')
            bb_position = self._get_feature(feat_df, timestamp, 'bb_position')
            
            if funding_zscore is None:
                continue
            
            # Get current position
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Detect market regime
            if self.use_regime_filter:
                regime = self.regime_detector.detect_regime(ohlcv, timestamp)
            else:
                regime = MarketRegime.NEUTRAL
            
            # === EXIT LOGIC ===
            if current_side != Side.FLAT:
                entry_time = self.entry_times.get(symbol)
                hours_held = 0
                if entry_time:
                    hours_held = (timestamp - entry_time).total_seconds() / 3600
                
                min_hold_met = hours_held >= self.min_hold_hours
                max_hold_exceeded = hours_held >= self.max_hold_hours
                
                should_exit = False
                exit_reason = ""
                
                # Max hold time exceeded
                if max_hold_exceeded:
                    should_exit = True
                    exit_reason = f"Max hold time exceeded ({hours_held:.0f}h)"
                
                # Funding normalized - exit
                elif min_hold_met:
                    if current_side == Side.SHORT and funding_zscore < self.exit_threshold:
                        should_exit = True
                        exit_reason = f"Funding normalized (z={funding_zscore:.2f})"
                    elif current_side == Side.LONG and funding_zscore > -self.exit_threshold:
                        should_exit = True
                        exit_reason = f"Funding normalized (z={funding_zscore:.2f})"
                
                # Regime turned strongly against position - early exit
                if not should_exit and self.use_regime_filter:
                    if current_side == Side.SHORT and regime == MarketRegime.STRONG_BULL:
                        should_exit = True
                        exit_reason = "Strong bull regime - exiting short"
                    elif current_side == Side.LONG and regime == MarketRegime.STRONG_BEAR:
                        should_exit = True
                        exit_reason = "Strong bear regime - exiting long"
                
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
                    signal_direction = Side.SHORT
                    
                    # Check regime filter
                    if self.use_regime_filter:
                        regime_multiplier = self.regime_detector.get_position_multiplier(
                            regime, signal_direction
                        )
                        if regime_multiplier == 0:
                            continue  # Skip this signal
                    else:
                        regime_multiplier = 1.0
                    
                    # Check trend alignment
                    if self.use_trend_filter:
                        if not self._check_trend_alignment(ohlcv, timestamp, signal_direction):
                            continue  # Trend not aligned
                    
                    # Optional: require price confirmation (overbought)
                    if self.use_price_confirmation:
                        if bb_position is None or bb_position < self.bb_confirmation_threshold:
                            continue
                    
                    # Calculate confidence based on z-score extremity
                    base_confidence = min(1.0, (funding_zscore - self.entry_threshold) / 2 + 0.5)
                    confidence = base_confidence * regime_multiplier
                    
                    if confidence > 0.1:  # Minimum confidence threshold
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.SHORT,
                            confidence=confidence,
                            target_weight=self.position_size * confidence,
                            reason=f"High funding z={funding_zscore:.2f}, regime={regime.value}",
                        ))
                        self.entry_times[symbol] = timestamp
                        self.entry_reasons[symbol] = "funding_short"
                
                # HIGH NEGATIVE FUNDING -> LONG
                elif funding_zscore < -self.entry_threshold:
                    signal_direction = Side.LONG
                    
                    # Check regime filter
                    if self.use_regime_filter:
                        regime_multiplier = self.regime_detector.get_position_multiplier(
                            regime, signal_direction
                        )
                        if regime_multiplier == 0:
                            continue
                    else:
                        regime_multiplier = 1.0
                    
                    # Check trend alignment
                    if self.use_trend_filter:
                        if not self._check_trend_alignment(ohlcv, timestamp, signal_direction):
                            continue
                    
                    # Optional: require price confirmation (oversold)
                    if self.use_price_confirmation:
                        if bb_position is None or bb_position > -self.bb_confirmation_threshold:
                            continue
                    
                    base_confidence = min(1.0, (abs(funding_zscore) - self.entry_threshold) / 2 + 0.5)
                    confidence = base_confidence * regime_multiplier
                    
                    if confidence > 0.1:
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=symbol,
                            direction=Side.LONG,
                            confidence=confidence,
                            target_weight=self.position_size * confidence,
                            reason=f"Low funding z={funding_zscore:.2f}, regime={regime.value}",
                        ))
                        self.entry_times[symbol] = timestamp
                        self.entry_reasons[symbol] = "funding_long"
        
        return signals


class OIDivergenceStrategy(BaseStrategy):
    """
    OI-Price Divergence Strategy -  VERSION
    
    Trade divergences between price and open interest:
    - Price up + OI down = weak rally, potential reversal (SHORT)
    - Price down + OI up = new shorts entering, potential squeeze (LONG)
    
    Key improvements:
    1. Regime awareness
    2. Confirmation requirements
    3. Better exit logic
    """
    
    def __init__(
        self,
        symbols: List[str],
        divergence_threshold: float = 0.5,
        use_oi_zscore_filter: bool = True,
        oi_zscore_threshold: float = 1.5,
        position_size: float = 0.15,
        min_hold_hours: int = 12,
        max_hold_hours: int = 72,
        use_regime_filter: bool = True,
    ):
        super().__init__(symbols)
        self.divergence_threshold = divergence_threshold
        self.use_oi_zscore_filter = use_oi_zscore_filter
        self.oi_zscore_threshold = oi_zscore_threshold
        self.position_size = position_size
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.use_regime_filter = use_regime_filter
        
        self.entry_times: Dict[str, datetime] = {}
        self.regime_detector = RegimeDetector()
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        """Safely get feature value."""
        if name not in df.columns:
            return None
        if timestamp not in df.index:
            return None
        val = df.loc[timestamp, name]
        if pd.isna(val):
            return None
        return val
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features or symbol not in ohlcv_data:
                continue
            
            feat_df = features[symbol]
            ohlcv = ohlcv_data[symbol]
            
            if timestamp not in feat_df.index:
                continue
            
            # Get features
            oi_price_div = self._get_feature(feat_df, timestamp, 'oi_price_divergence')
            oi_zscore = self._get_feature(feat_df, timestamp, 'oi_zscore')
            oi_change_24h = self._get_feature(feat_df, timestamp, 'oi_change_24h')
            price_change_24h = self._get_feature(feat_df, timestamp, 'price_change_24h')
            
            # Get current position
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Detect regime
            if self.use_regime_filter:
                regime = self.regime_detector.detect_regime(ohlcv, timestamp)
            else:
                regime = MarketRegime.NEUTRAL
            
            # === EXIT LOGIC ===
            if current_side != Side.FLAT:
                entry_time = self.entry_times.get(symbol)
                hours_held = 0
                if entry_time:
                    hours_held = (timestamp - entry_time).total_seconds() / 3600
                
                should_exit = False
                exit_reason = ""
                
                if hours_held >= self.max_hold_hours:
                    should_exit = True
                    exit_reason = "Max hold exceeded"
                elif hours_held >= self.min_hold_hours:
                    # Exit if divergence resolved
                    if oi_price_div is not None and oi_price_div == 0:
                        should_exit = True
                        exit_reason = "Divergence resolved"
                
                if should_exit:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                        reason=exit_reason,
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # === ENTRY LOGIC ===
            if current_side == Side.FLAT and oi_price_div is not None:
                
                # Calculate signal quality
                if price_change_24h is not None and oi_change_24h is not None:
                    
                    # Price up, OI down -> Weak rally, SHORT
                    if price_change_24h > 0.02 and oi_change_24h < -0.02:
                        signal_direction = Side.SHORT
                        
                        if self.use_regime_filter:
                            multiplier = self.regime_detector.get_position_multiplier(
                                regime, signal_direction
                            )
                            if multiplier == 0:
                                continue
                        else:
                            multiplier = 1.0
                        
                        # Additional OI zscore filter
                        if self.use_oi_zscore_filter and oi_zscore is not None:
                            if oi_zscore < self.oi_zscore_threshold:
                                continue
                        
                        confidence = min(1.0, abs(oi_change_24h) * 10) * multiplier
                        
                        if confidence > 0.2:
                            signals.append(Signal(
                                timestamp=timestamp,
                                symbol=symbol,
                                direction=Side.SHORT,
                                confidence=confidence,
                                target_weight=self.position_size * confidence,
                                reason=f"OI divergence: price +{price_change_24h*100:.1f}%, OI {oi_change_24h*100:.1f}%",
                            ))
                            self.entry_times[symbol] = timestamp
                    
                    # Price down, OI up -> New shorts, potential squeeze, LONG
                    elif price_change_24h < -0.02 and oi_change_24h > 0.02:
                        signal_direction = Side.LONG
                        
                        if self.use_regime_filter:
                            multiplier = self.regime_detector.get_position_multiplier(
                                regime, signal_direction
                            )
                            if multiplier == 0:
                                continue
                        else:
                            multiplier = 1.0
                        
                        confidence = min(1.0, oi_change_24h * 10) * multiplier
                        
                        if confidence > 0.2:
                            signals.append(Signal(
                                timestamp=timestamp,
                                symbol=symbol,
                                direction=Side.LONG,
                                confidence=confidence,
                                target_weight=self.position_size * confidence,
                                reason=f"OI divergence: price {price_change_24h*100:.1f}%, OI +{oi_change_24h*100:.1f}%",
                            ))
                            self.entry_times[symbol] = timestamp
        
        return signals


class CombinedOIFundingStrategy(BaseStrategy):
    """
    Combined OI + Funding Strategy -  VERSION
    
    Multi-factor approach requiring confluence of signals:
    1. Extreme funding rate
    2. OI divergence
    3. Price at Bollinger Band extreme
    4. Regime alignment
    
    Only enters when multiple factors align.
    """
    
    def __init__(
        self,
        symbols: List[str],
        oi_divergence_threshold: float = 0.3,
        funding_threshold: float = 1.5,
        bb_threshold: float = 1.5,
        min_factors: int = 2,
        position_size: float = 0.15,
        min_hold_hours: int = 24,
        max_hold_hours: int = 120,
        use_regime_filter: bool = True,
    ):
        super().__init__(symbols)
        self.oi_divergence_threshold = oi_divergence_threshold
        self.funding_threshold = funding_threshold
        self.bb_threshold = bb_threshold
        self.min_factors = min_factors
        self.position_size = position_size
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours
        self.use_regime_filter = use_regime_filter
        
        self.entry_times: Dict[str, datetime] = {}
        self.regime_detector = RegimeDetector()
    
    def _get_feature(self, df: pd.DataFrame, timestamp: datetime, name: str):
        if name not in df.columns:
            return None
        if timestamp not in df.index:
            return None
        val = df.loc[timestamp, name]
        if pd.isna(val):
            return None
        return val
    
    def _count_short_factors(self, feat_df, timestamp) -> tuple:
        """Count factors supporting a SHORT signal."""
        factors = 0
        reasons = []
        
        # Factor 1: High positive funding
        funding_zscore = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
        if funding_zscore is not None and funding_zscore > self.funding_threshold:
            factors += 1
            reasons.append(f"FR z={funding_zscore:.2f}")
        
        # Factor 2: Price/OI divergence (price up, OI down)
        oi_div = self._get_feature(feat_df, timestamp, 'oi_price_divergence')
        price_chg = self._get_feature(feat_df, timestamp, 'price_change_24h')
        oi_chg = self._get_feature(feat_df, timestamp, 'oi_change_24h')
        if oi_div == 1 and price_chg is not None and price_chg > 0:
            factors += 1
            reasons.append("OI divergence")
        
        # Factor 3: Price at upper BB
        bb_pos = self._get_feature(feat_df, timestamp, 'bb_position')
        if bb_pos is not None and bb_pos > self.bb_threshold:
            factors += 1
            reasons.append(f"BB={bb_pos:.2f}")
        
        return factors, reasons
    
    def _count_long_factors(self, feat_df, timestamp) -> tuple:
        """Count factors supporting a LONG signal."""
        factors = 0
        reasons = []
        
        # Factor 1: High negative funding
        funding_zscore = self._get_feature(feat_df, timestamp, 'funding_rate_zscore')
        if funding_zscore is not None and funding_zscore < -self.funding_threshold:
            factors += 1
            reasons.append(f"FR z={funding_zscore:.2f}")
        
        # Factor 2: Price/OI divergence (price down, OI up)
        oi_div = self._get_feature(feat_df, timestamp, 'oi_price_divergence')
        price_chg = self._get_feature(feat_df, timestamp, 'price_change_24h')
        oi_chg = self._get_feature(feat_df, timestamp, 'oi_change_24h')
        if oi_div == 1 and price_chg is not None and price_chg < 0:
            factors += 1
            reasons.append("OI divergence")
        
        # Factor 3: Price at lower BB
        bb_pos = self._get_feature(feat_df, timestamp, 'bb_position')
        if bb_pos is not None and bb_pos < -self.bb_threshold:
            factors += 1
            reasons.append(f"BB={bb_pos:.2f}")
        
        return factors, reasons
    
    def generate_signals(
        self,
        timestamp: datetime,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        portfolio,
    ) -> List[Signal]:
        
        signals = []
        
        for symbol in self.symbols:
            if symbol not in features or symbol not in ohlcv_data:
                continue
            
            feat_df = features[symbol]
            ohlcv = ohlcv_data[symbol]
            
            if timestamp not in feat_df.index:
                continue
            
            current_pos = portfolio.positions.get(symbol)
            current_side = current_pos.side if current_pos else Side.FLAT
            
            # Detect regime
            if self.use_regime_filter:
                regime = self.regime_detector.detect_regime(ohlcv, timestamp)
            else:
                regime = MarketRegime.NEUTRAL
            
            # === EXIT LOGIC ===
            if current_side != Side.FLAT:
                entry_time = self.entry_times.get(symbol)
                hours_held = (timestamp - entry_time).total_seconds() / 3600 if entry_time else 0
                
                should_exit = False
                exit_reason = ""
                
                if hours_held >= self.max_hold_hours:
                    should_exit = True
                    exit_reason = "Max hold exceeded"
                elif hours_held >= self.min_hold_hours:
                    # Check if factors have unwound
                    if current_side == Side.SHORT:
                        factors, _ = self._count_short_factors(feat_df, timestamp)
                        if factors == 0:
                            should_exit = True
                            exit_reason = "Short factors unwound"
                    else:
                        factors, _ = self._count_long_factors(feat_df, timestamp)
                        if factors == 0:
                            should_exit = True
                            exit_reason = "Long factors unwound"
                
                # Regime exit
                if not should_exit and self.use_regime_filter:
                    if current_side == Side.SHORT and regime == MarketRegime.STRONG_BULL:
                        should_exit = True
                        exit_reason = "Strong bull - exiting short"
                    elif current_side == Side.LONG and regime == MarketRegime.STRONG_BEAR:
                        should_exit = True
                        exit_reason = "Strong bear - exiting long"
                
                if should_exit:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.FLAT,
                        reason=exit_reason,
                    ))
                    self.entry_times.pop(symbol, None)
                    continue
            
            # === ENTRY LOGIC ===
            if current_side == Side.FLAT:
                
                # Check SHORT factors
                short_factors, short_reasons = self._count_short_factors(feat_df, timestamp)
                if short_factors >= self.min_factors:
                    signal_direction = Side.SHORT
                    
                    if self.use_regime_filter:
                        multiplier = self.regime_detector.get_position_multiplier(
                            regime, signal_direction
                        )
                        if multiplier == 0:
                            continue
                    else:
                        multiplier = 1.0
                    
                    confidence = (short_factors / 3.0) * multiplier
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.SHORT,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"Combined({short_factors}): {', '.join(short_reasons)}",
                    ))
                    self.entry_times[symbol] = timestamp
                    continue
                
                # Check LONG factors
                long_factors, long_reasons = self._count_long_factors(feat_df, timestamp)
                if long_factors >= self.min_factors:
                    signal_direction = Side.LONG
                    
                    if self.use_regime_filter:
                        multiplier = self.regime_detector.get_position_multiplier(
                            regime, signal_direction
                        )
                        if multiplier == 0:
                            continue
                    else:
                        multiplier = 1.0
                    
                    confidence = (long_factors / 3.0) * multiplier
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction=Side.LONG,
                        confidence=confidence,
                        target_weight=self.position_size * confidence,
                        reason=f"Combined({long_factors}): {', '.join(long_reasons)}",
                    ))
                    self.entry_times[symbol] = timestamp
        
        return signals