"""Market data records with bi-temporal timestamps, to prevent lookahead."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class Side(Enum):
    """Trade/order side."""
    BUY = "buy"
    SELL = "sell"


class DataQuality(Enum):
    """Data quality flags.

    UNVALIDATED is the default on every record, so that data which skipped
    validation stays distinct from data that passed. A row still UNVALIDATED
    after ingest is a bypass.
    """
    UNVALIDATED = "unvalidated"  # Never checked — treat as unknown
    VALID = "valid"
    SUSPICIOUS = "suspicious"  # Flagged but usable
    INVALID = "invalid"  # Should be excluded


@dataclass
class BiTemporalMixin:
    """Bi-temporal timestamps.

    A decision may use a record only when decision_time >= available_time.
    """
    event_time: datetime  # When event occurred
    available_time: datetime  # When we received/could access the data


@dataclass
class OHLCVBar(BiTemporalMixin):
    """One OHLCV bar.

    The bar for [T, T+interval) is available only at T+interval.
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

    # The exchange that produced this bar. One symbol's history can blend
    # venues with different microstructure.
    venue: str = 'unknown'
    
    # Data quality
    quality: DataQuality = DataQuality.UNVALIDATED
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
    """One trade."""
    symbol: str
    trade_id: str
    
    price: float
    size: float
    side: Side
    
    # Quality
    quality: DataQuality = DataQuality.UNVALIDATED
    
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
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: float
    
    def to_tuple(self) -> tuple:
        return (self.price, self.size)


@dataclass
class OrderBookSnapshot(BiTemporalMixin):
    """The top N levels of an order book at one instant."""
    symbol: str
    
    # Bids (buy orders) - sorted by price descending
    bids: List[OrderBookLevel]
    
    # Asks (sell orders) - sorted by price ascending
    asks: List[OrderBookLevel]
    
    # Quality
    quality: DataQuality = DataQuality.UNVALIDATED
    
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
    """A real-time ticker update from a WebSocket."""
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


SymbolData = Dict[str, List[OHLCVBar]]
TradeStream = List[Trade]
