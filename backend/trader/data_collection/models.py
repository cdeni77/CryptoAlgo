"""
Data models for the Crypto Perpetual Futures Trading System.

These models enforce strict temporal semantics with bi-temporal timestamps
to prevent lookahead bias and data leakage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class Side(Enum):
    """Trade/order side."""
    BUY = "buy"
    SELL = "sell"
    
    @classmethod
    def from_string(cls, s: str) -> "Side":
        return cls.BUY if s.lower() in ("buy", "bid", "long") else cls.SELL


class DataQuality(Enum):
    """Data quality flags."""
    VALID = "valid"
    SUSPICIOUS = "suspicious"  # Flagged but usable
    INVALID = "invalid"  # Should be excluded


@dataclass
class BiTemporalMixin:
    """
    Mixin for bi-temporal data storage.
    
    Critical for preventing lookahead bias:
    - event_time: When the event actually occurred in the market
    - available_time: When the data became available to our system
    
    For backtesting, we can only use data where:
    decision_time >= available_time
    """
    event_time: datetime  # When event occurred
    available_time: datetime  # When we received/could access the data
    
    def is_available_at(self, query_time: datetime) -> bool:
        """Check if this data point was available at query_time."""
        return query_time >= self.available_time


@dataclass
class OHLCVBar(BiTemporalMixin):
    """
    OHLCV (Open, High, Low, Close, Volume) bar data.
    
    Note: The bar for period [T, T+interval) is only fully available
    at time T+interval. The available_time should reflect this.
    """
    symbol: str
    timeframe: str  # e.g., "1m", "1h", "1d"
    
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Optional: quote volume (in quote currency, usually USD)
    quote_volume: Optional[float] = None
    
    # Number of trades in this bar
    trade_count: Optional[int] = None
    
    # Data quality
    quality: DataQuality = DataQuality.VALID
    quality_notes: Optional[str] = None
    
    # Unique identifier for deduplication
    @property
    def bar_id(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.event_time.isoformat()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trade_count": self.trade_count,
            "quality": self.quality.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCVBar":
        return cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            event_time=datetime.fromisoformat(data["event_time"]),
            available_time=datetime.fromisoformat(data["available_time"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            quote_volume=float(data["quote_volume"]) if data.get("quote_volume") else None,
            trade_count=int(data["trade_count"]) if data.get("trade_count") else None,
            quality=DataQuality(data.get("quality", "valid")),
        )


@dataclass
class Trade(BiTemporalMixin):
    """
    Individual trade record.
    """
    symbol: str
    trade_id: str
    
    price: float
    size: float
    side: Side
    
    # Quality
    quality: DataQuality = DataQuality.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_id": self.trade_id,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "price": self.price,
            "size": self.size,
            "side": self.side.value,
            "quality": self.quality.value,
        }


@dataclass
class FundingRate(BiTemporalMixin):
    """
    Perpetual futures funding rate.
    
    Coinbase specifics:
    - Accrues hourly
    - Settles twice daily (00:00 and 12:00 UTC)
    - Rate is expressed as a percentage (e.g., 0.01 = 0.01%)
    """
    symbol: str
    
    # The funding rate for this period
    rate: float  # As decimal (0.0001 = 0.01%)
    
    # Mark price at funding time (used for payment calculation)
    mark_price: float
    
    # Index price (spot reference)
    index_price: float
    
    # Whether this is a settlement time or just accrual
    is_settlement: bool = False
    
    # Data provenance
    funding_source: str = "coinbase"  # coinbase | binance_proxy

    # Quality
    quality: DataQuality = DataQuality.VALID
    
    @property
    def rate_bps(self) -> float:
        """Return rate in basis points."""
        return self.rate * 10000
    
    @property
    def basis(self) -> float:
        """Calculate basis (mark - index) / index."""
        if self.index_price == 0:
            return 0.0
        return (self.mark_price - self.index_price) / self.index_price
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "rate": self.rate,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "is_settlement": self.is_settlement,
            "funding_source": self.funding_source,
            "quality": self.quality.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FundingRate":
        return cls(
            symbol=data["symbol"],
            event_time=datetime.fromisoformat(data["event_time"]),
            available_time=datetime.fromisoformat(data["available_time"]),
            rate=float(data["rate"]),
            mark_price=float(data.get("mark_price", 0)),
            index_price=float(data.get("index_price", 0)),
            is_settlement=data.get("is_settlement", False),
            funding_source=data.get("funding_source", "coinbase"),
            quality=DataQuality(data.get("quality", "valid")),
        )


@dataclass 
class OpenInterest(BiTemporalMixin):
    """
    Open interest snapshot.
    """
    symbol: str
    
    # Open interest in contracts
    open_interest_contracts: float
    
    # Open interest in base currency (e.g., BTC)
    open_interest_base: Optional[float] = None
    
    # Open interest in quote currency (e.g., USD)
    open_interest_usd: Optional[float] = None
    
    # Quality
    quality: DataQuality = DataQuality.VALID
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "open_interest_contracts": self.open_interest_contracts,
            "open_interest_base": self.open_interest_base,
            "open_interest_usd": self.open_interest_usd,
            "quality": self.quality.value,
        }


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: float
    
    def to_tuple(self) -> tuple:
        return (self.price, self.size)


@dataclass
class OrderBookSnapshot(BiTemporalMixin):
    """
    Order book snapshot at a point in time.
    
    Stores top N levels of bids and asks.
    """
    symbol: str
    
    # Bids (buy orders) - sorted by price descending
    bids: List[OrderBookLevel]
    
    # Asks (sell orders) - sorted by price ascending
    asks: List[OrderBookLevel]
    
    # Quality
    quality: DataQuality = DataQuality.VALID
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        if self.spread and self.mid_price:
            return (self.spread / self.mid_price) * 10000
        return None
    
    def total_bid_size(self, depth: int = None) -> float:
        """Total size on bid side up to depth levels."""
        levels = self.bids[:depth] if depth else self.bids
        return sum(level.size for level in levels)
    
    def total_ask_size(self, depth: int = None) -> float:
        """Total size on ask side up to depth levels."""
        levels = self.asks[:depth] if depth else self.asks
        return sum(level.size for level in levels)
    
    def imbalance(self, depth: int = 5) -> float:
        """
        Order book imbalance ratio.
        
        Returns value in [-1, 1]:
        - Positive: More buying pressure (more bids)
        - Negative: More selling pressure (more asks)
        """
        bid_size = self.total_bid_size(depth)
        ask_size = self.total_ask_size(depth)
        total = bid_size + ask_size
        if total == 0:
            return 0.0
        return (bid_size - ask_size) / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "bids": [level.to_tuple() for level in self.bids],
            "asks": [level.to_tuple() for level in self.asks],
            "quality": self.quality.value,
        }


@dataclass
class TickerUpdate(BiTemporalMixin):
    """
    Real-time ticker update from WebSocket.
    """
    symbol: str
    
    price: float
    best_bid: float
    best_ask: float
    
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_time": self.event_time.isoformat(),
            "available_time": self.available_time.isoformat(),
            "price": self.price,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "volume_24h": self.volume_24h,
            "price_change_24h": self.price_change_24h,
        }


@dataclass
class DataValidationResult:
    """Result of data validation check."""
    is_valid: bool
    quality: DataQuality
    issues: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str, severity: str = "warning"):
        self.issues.append(f"[{severity.upper()}] {issue}")
        if severity == "error":
            self.is_valid = False
            self.quality = DataQuality.INVALID
        elif severity == "warning" and self.quality == DataQuality.VALID:
            self.quality = DataQuality.SUSPICIOUS


# Type aliases for clarity
SymbolData = Dict[str, List[OHLCVBar]]
TradeStream = List[Trade]
FundingHistory = List[FundingRate]
