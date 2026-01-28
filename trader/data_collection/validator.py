"""
Data validation for the Crypto Perpetual Futures Trading System.

Validates incoming data for quality and integrity before storage.
"""

import logging

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

from .models import (
    OHLCVBar,
    Trade,
    FundingRate,
    OpenInterest,
    OrderBookSnapshot,
    TickerUpdate,
    DataQuality,
    DataValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation thresholds."""
    
    # Price validation
    max_price_change_pct: float = 10.0  # Max % change in 1 minute
    min_price: float = 0.0001  # Minimum valid price
    max_price: float = 1_000_000_000  # Maximum valid price (1 billion)
    
    # Volume validation
    min_volume: float = 0.0
    max_volume: float = 1_000_000_000_000  # 1 trillion
    
    # Timestamp validation
    max_future_seconds: int = 60  # Allow up to 60s in future (clock skew)
    max_past_days: int = 365 * 5  # Data up to 5 years old
    
    # OHLCV specific
    require_ohlc_consistency: bool = True  # high >= low, etc.
    
    # Order book specific
    max_spread_pct: float = 10.0  # Max spread as % of mid
    require_sorted_book: bool = True
    
    # Funding rate specific
    max_funding_rate: float = 0.01  # 1% max funding rate (100 bps)
    
    # Gap detection
    max_gap_minutes: int = 5


class DataValidator:
    """
    Validates incoming market data for quality and integrity.
    
    All validation is performed BEFORE data is stored to ensure
    data quality from the start.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._last_prices: dict = {}  # symbol -> last valid price
        self._last_timestamps: dict = {}  # symbol -> last timestamp
    
    def validate_ohlcv(
        self,
        bar: OHLCVBar,
        previous_bar: Optional[OHLCVBar] = None
    ) -> DataValidationResult:
        """
        Validate OHLCV bar data.
        
        Checks:
        1. Price bounds and OHLC consistency
        2. Volume validity
        3. Timestamp validity
        4. Price continuity with previous bar
        5. Gap detection
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        # 1. Basic price validation
        for field_name, value in [
            ("open", bar.open),
            ("high", bar.high),
            ("low", bar.low),
            ("close", bar.close)
        ]:
            if not self._is_valid_price(value):
                result.add_issue(
                    f"Invalid {field_name} price: {value}",
                    severity="error"
                )
        
        # 2. OHLC consistency
        if self.config.require_ohlc_consistency:
            if bar.high < bar.low:
                result.add_issue(
                    f"High ({bar.high}) < Low ({bar.low})",
                    severity="error"
                )
            if bar.high < bar.open or bar.high < bar.close:
                result.add_issue(
                    f"High ({bar.high}) not the highest",
                    severity="warning"
                )
            if bar.low > bar.open or bar.low > bar.close:
                result.add_issue(
                    f"Low ({bar.low}) not the lowest",
                    severity="warning"
                )
        
        # 3. Volume validation
        if bar.volume < self.config.min_volume:
            result.add_issue(
                f"Volume too low: {bar.volume}",
                severity="warning"
            )
        if bar.volume > self.config.max_volume:
            result.add_issue(
                f"Volume too high: {bar.volume}",
                severity="error"
            )
        
        # 4. Timestamp validation
        ts_result = self._validate_timestamp(bar.event_time, bar.available_time)
        if not ts_result.is_valid:
            result.issues.extend(ts_result.issues)
            result.is_valid = False
        
        # 5. Price continuity
        if previous_bar:
            price_change_pct = abs(bar.open - previous_bar.close) / previous_bar.close * 100
            if price_change_pct > self.config.max_price_change_pct:
                result.add_issue(
                    f"Large price gap: {price_change_pct:.2f}% from previous close",
                    severity="warning"
                )
        
        # 6. Gap detection
        if previous_bar:
            gap = self._detect_gap(previous_bar.event_time, bar.event_time, bar.timeframe)
            if gap:
                result.add_issue(
                    f"Data gap detected: {gap} minutes",
                    severity="warning"
                )
        
        # Update tracking
        if result.is_valid:
            self._last_prices[bar.symbol] = bar.close
            self._last_timestamps[bar.symbol] = bar.event_time
        
        bar.quality = result.quality
        bar.quality_notes = "; ".join(result.issues) if result.issues else None
        
        return result
    
    def validate_trade(self, trade: Trade) -> DataValidationResult:
        """
        Validate individual trade.
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        # Price validation
        if not self._is_valid_price(trade.price):
            result.add_issue(f"Invalid price: {trade.price}", severity="error")
        
        # Size validation
        if trade.size <= 0:
            result.add_issue(f"Invalid size: {trade.size}", severity="error")
        
        # Timestamp validation
        ts_result = self._validate_timestamp(trade.event_time, trade.available_time)
        if not ts_result.is_valid:
            result.issues.extend(ts_result.issues)
            result.is_valid = False
        
        # Price continuity check
        last_price = self._last_prices.get(trade.symbol)
        if last_price:
            price_change_pct = abs(trade.price - last_price) / last_price * 100
            if price_change_pct > self.config.max_price_change_pct:
                result.add_issue(
                    f"Large price move: {price_change_pct:.2f}%",
                    severity="warning"
                )
        
        trade.quality = result.quality
        return result
    
    def validate_funding_rate(self, funding: FundingRate) -> DataValidationResult:
        """
        Validate funding rate data.
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        # Rate bounds
        if abs(funding.rate) > self.config.max_funding_rate:
            result.add_issue(
                f"Extreme funding rate: {funding.rate_bps:.2f} bps",
                severity="warning"
            )
        
        # Price validation
        if not self._is_valid_price(funding.mark_price):
            result.add_issue(f"Invalid mark price: {funding.mark_price}", severity="error")
        
        if not self._is_valid_price(funding.index_price):
            result.add_issue(f"Invalid index price: {funding.index_price}", severity="error")
        
        # Mark-index divergence check
        if funding.index_price > 0:
            basis_pct = abs(funding.basis) * 100
            if basis_pct > 5.0:  # More than 5% basis is suspicious
                result.add_issue(
                    f"Large basis: {basis_pct:.2f}%",
                    severity="warning"
                )
        
        # Timestamp validation
        ts_result = self._validate_timestamp(funding.event_time, funding.available_time)
        if not ts_result.is_valid:
            result.issues.extend(ts_result.issues)
            result.is_valid = False
        
        funding.quality = result.quality
        return result
    
    def validate_orderbook(self, book: OrderBookSnapshot) -> DataValidationResult:
        """
        Validate order book snapshot.
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        # Check non-empty
        if not book.bids or not book.asks:
            result.add_issue("Empty order book side", severity="warning")
            return result
        
        # Validate bid/ask ordering
        if self.config.require_sorted_book:
            # Bids should be descending
            for i in range(len(book.bids) - 1):
                if book.bids[i].price < book.bids[i + 1].price:
                    result.add_issue("Bids not properly sorted", severity="error")
                    break
            
            # Asks should be ascending
            for i in range(len(book.asks) - 1):
                if book.asks[i].price > book.asks[i + 1].price:
                    result.add_issue("Asks not properly sorted", severity="error")
                    break
        
        # Check crossed book
        if book.best_bid and book.best_ask:
            if book.best_bid >= book.best_ask:
                result.add_issue(
                    f"Crossed book: bid {book.best_bid} >= ask {book.best_ask}",
                    severity="error"
                )
        
        # Spread check
        if book.spread_bps and book.spread_bps > self.config.max_spread_pct * 100:
            result.add_issue(
                f"Wide spread: {book.spread_bps:.1f} bps",
                severity="warning"
            )
        
        # Validate individual levels
        for level in book.bids + book.asks:
            if not self._is_valid_price(level.price):
                result.add_issue(f"Invalid price in book: {level.price}", severity="error")
            if level.size < 0:
                result.add_issue(f"Negative size in book: {level.size}", severity="error")
        
        book.quality = result.quality
        return result
    
    def validate_open_interest(self, oi: OpenInterest) -> DataValidationResult:
        """
        Validate open interest data.
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        if oi.open_interest_contracts < 0:
            result.add_issue(
                f"Negative OI: {oi.open_interest_contracts}",
                severity="error"
            )
        
        # Timestamp validation
        ts_result = self._validate_timestamp(oi.event_time, oi.available_time)
        if not ts_result.is_valid:
            result.issues.extend(ts_result.issues)
            result.is_valid = False
        
        oi.quality = result.quality
        return result
    
    def validate_ticker(self, ticker: TickerUpdate) -> DataValidationResult:
        """
        Validate ticker update.
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        
        # Price validation
        for field_name, value in [
            ("price", ticker.price),
            ("best_bid", ticker.best_bid),
            ("best_ask", ticker.best_ask)
        ]:
            if not self._is_valid_price(value):
                result.add_issue(f"Invalid {field_name}: {value}", severity="error")
        
        # Bid/ask consistency
        if ticker.best_bid >= ticker.best_ask:
            result.add_issue(
                f"Crossed ticker: bid {ticker.best_bid} >= ask {ticker.best_ask}",
                severity="error"
            )
        
        return result
    
    def _is_valid_price(self, price: float) -> bool:
        """Check if price is within valid bounds."""
        return (
            price is not None and
            self.config.min_price <= price <= self.config.max_price
        )
    
    def _validate_timestamp(
        self,
        event_time: datetime,
        available_time: datetime
    ) -> DataValidationResult:
        """
        Validate timestamps.
        
        Rules:
        1. available_time >= event_time (can't know about event before it happens)
        2. Neither too far in future
        3. Neither too far in past
        """
        result = DataValidationResult(is_valid=True, quality=DataQuality.VALID)
        now = datetime.utcnow()
        
        # available_time should be >= event_time
        if available_time < event_time:
            result.add_issue(
                f"available_time ({available_time}) < event_time ({event_time})",
                severity="error"
            )
        
        # Check not too far in future
        max_future = now + timedelta(seconds=self.config.max_future_seconds)
        if event_time > max_future:
            result.add_issue(
                f"Event time too far in future: {event_time}",
                severity="error"
            )
        
        # Check not too far in past
        min_past = now - timedelta(days=self.config.max_past_days)
        if event_time < min_past:
            result.add_issue(
                f"Event time too far in past: {event_time}",
                severity="warning"
            )
        
        return result
    
    def _detect_gap(
        self,
        prev_time: datetime,
        curr_time: datetime,
        timeframe: str
    ) -> Optional[int]:
        """
        Detect data gaps.
        
        Returns gap duration in minutes if gap exceeds threshold, None otherwise.
        """
        # Parse timeframe to minutes
        tf_minutes = self._timeframe_to_minutes(timeframe)
        expected_gap = timedelta(minutes=tf_minutes)
        
        actual_gap = curr_time - prev_time
        
        # Allow some tolerance (e.g., 1.5x expected)
        if actual_gap > expected_gap * 1.5:
            gap_minutes = int(actual_gap.total_seconds() / 60)
            if gap_minutes > self.config.max_gap_minutes:
                return gap_minutes
        
        return None
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        multipliers = {
            "m": 1,
            "h": 60,
            "d": 1440,
            "w": 10080,
        }
        
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        
        return value * multipliers.get(unit, 1)


class DataQualityTracker:
    """
    Tracks data quality metrics over time.
    """
    
    def __init__(self):
        self.total_records: int = 0
        self.valid_records: int = 0
        self.suspicious_records: int = 0
        self.invalid_records: int = 0
        self.gaps_detected: int = 0
        self.issues_by_type: dict = {}
    
    def record_validation(self, result: DataValidationResult):
        """Record a validation result."""
        self.total_records += 1
        
        if result.quality == DataQuality.VALID:
            self.valid_records += 1
        elif result.quality == DataQuality.SUSPICIOUS:
            self.suspicious_records += 1
        else:
            self.invalid_records += 1
        
        # Track issues by type
        for issue in result.issues:
            issue_type = issue.split("]")[0] + "]"
            self.issues_by_type[issue_type] = self.issues_by_type.get(issue_type, 0) + 1
    
    @property
    def validity_rate(self) -> float:
        """Percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records * 100
    
    def get_summary(self) -> dict:
        """Get quality summary."""
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "suspicious_records": self.suspicious_records,
            "invalid_records": self.invalid_records,
            "validity_rate": f"{self.validity_rate:.2f}%",
            "issues_by_type": self.issues_by_type,
        }
