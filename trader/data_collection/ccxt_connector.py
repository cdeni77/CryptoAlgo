"""
CCXT Connector for historical data backfill.

Uses CCXT library to fetch historical data from multiple exchanges
for cross-reference and supplementary data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from .models import (
    OHLCVBar,
    FundingRate,
    OpenInterest,
    DataQuality,
)

logger = logging.getLogger(__name__)


class CCXTConnector:
    """
    CCXT-based connector for historical data from multiple exchanges.
    
    Primarily used for:
    1. Historical backfill when Coinbase data is unavailable
    2. Cross-exchange data validation
    3. Supplementary data (funding rates, OI from other exchanges)
    
    Supports proxy configuration for regions where exchanges are blocked.
    """
    
    # Symbol mapping between exchanges
    SYMBOL_MAPPING = {
        # Coinbase symbol -> CCXT unified symbol
        "BTC-PERP": "BTC/USDT:USDT",
        "ETH-PERP": "ETH/USDT:USDT",
        "SOL-PERP": "SOL/USDT:USDT",
        "XRP-PERP": "XRP/USDT:USDT",
        "AVAX-PERP": "AVAX/USDT:USDT",
    }
    
    # Alternative exchanges that may work in restricted regions
    FALLBACK_EXCHANGES = [
        "okx",
        "gate",
        "mexc",
        "kucoin",
        "bitget",
        "kraken",
    ]
    
    def __init__(
        self,
        exchanges: Optional[List[str]] = None,
        rate_limit_enabled: bool = True,
        proxy: Optional[str] = None,
        aiohttp_proxy: Optional[str] = None,
        timeout: int = 30000,
        use_fallbacks: bool = True,
    ):
        """
        Initialize CCXT connector.
        
        Args:
            exchanges: List of exchange IDs to use (default: binance, bybit)
            rate_limit_enabled: Enable built-in rate limiting
            proxy: HTTP/HTTPS proxy URL (e.g., "http://127.0.0.1:7890")
            aiohttp_proxy: Proxy specifically for aiohttp (if different)
            timeout: Request timeout in milliseconds
            use_fallbacks: Try fallback exchanges if primary ones fail
        """
        self.exchange_ids = exchanges or ["binance", "bybit"]
        self.rate_limit_enabled = rate_limit_enabled
        self.proxy = proxy
        self.aiohttp_proxy = aiohttp_proxy or proxy
        self.timeout = timeout
        self.use_fallbacks = use_fallbacks
        self._exchanges: Dict = {}
        self._initialized = False
        self._failed_exchanges: List[str] = []
    
    def _get_exchange_config(self, exchange_id: str) -> Dict[str, Any]:
        """Build exchange configuration with proxy support."""
        config = {
            "enableRateLimit": self.rate_limit_enabled,
            "timeout": self.timeout,
            "options": {
                "defaultType": "swap",  # For perpetuals
                "adjustForTimeDifference": True,
            }
        }
        
        # Add proxy configuration
        if self.proxy:
            config["proxies"] = {
                "http": self.proxy,
                "https": self.proxy,
            }
        
        if self.aiohttp_proxy:
            config["aiohttp_proxy"] = self.aiohttp_proxy
        
        # Exchange-specific configurations
        if exchange_id == "okx":
            config["options"]["defaultType"] = "swap"
        elif exchange_id == "gate":
            config["options"]["defaultType"] = "swap"
        elif exchange_id == "kucoin":
            config["options"]["defaultType"] = "swap"
        elif exchange_id == "mexc":
            config["options"]["defaultType"] = "swap"
        
        return config
    
    async def initialize(self):
        """Initialize exchange connections."""
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            raise ImportError(
                "ccxt package required. Install with: pip install ccxt"
            )
        
        exchanges_to_try = self.exchange_ids.copy()
        
        # Add fallbacks if enabled
        if self.use_fallbacks:
            for fallback in self.FALLBACK_EXCHANGES:
                if fallback not in exchanges_to_try:
                    exchanges_to_try.append(fallback)
        
        for exchange_id in exchanges_to_try:
            try:
                if not hasattr(ccxt, exchange_id):
                    logger.warning(f"Exchange {exchange_id} not supported by CCXT")
                    continue
                
                exchange_class = getattr(ccxt, exchange_id)
                config = self._get_exchange_config(exchange_id)
                exchange = exchange_class(config)
                
                # Load markets with timeout
                await asyncio.wait_for(
                    exchange.load_markets(),
                    timeout=self.timeout / 1000
                )
                
                self._exchanges[exchange_id] = exchange
                logger.info(f"✓ Initialized {exchange_id} with {len(exchange.markets)} markets")
                
                # If we have at least one working exchange, we're good
                if len(self._exchanges) >= 1 and exchange_id in self.exchange_ids:
                    break
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout initializing {exchange_id}")
                self._failed_exchanges.append(exchange_id)
            except Exception as e:
                error_msg = str(e)
                if "451" in error_msg or "403" in error_msg or "restricted" in error_msg.lower():
                    logger.warning(f"Exchange {exchange_id} blocked in your region. Try using a proxy.")
                else:
                    logger.error(f"Failed to initialize {exchange_id}: {e}")
                self._failed_exchanges.append(exchange_id)
                
                # Clean up failed exchange
                try:
                    if hasattr(exchange, 'close'):
                        await exchange.close()
                except:
                    pass
        
        self._initialized = True
        
        if not self._exchanges:
            logger.error(
                "No exchanges available! All exchanges failed to initialize.\n"
                "If exchanges are blocked in your region, try:\n"
                "  1. Use a proxy: CCXTConnector(proxy='http://your-proxy:port')\n"
                "  2. Use a VPN\n"
                f"  3. Try different exchanges: {self.FALLBACK_EXCHANGES}"
            )
    
    def _get_ccxt_symbol(self, coinbase_symbol: str, exchange_id: str) -> Optional[str]:
        """Convert Coinbase symbol to CCXT symbol for specific exchange."""
        base_symbol = self.SYMBOL_MAPPING.get(coinbase_symbol)
        if not base_symbol:
            # Try to construct from Coinbase symbol
            # BTC-PERP -> BTC/USDT:USDT
            base = coinbase_symbol.replace("-PERP", "")
            base_symbol = f"{base}/USDT:USDT"
        
        # Verify symbol exists on exchange
        exchange = self._exchanges.get(exchange_id)
        if exchange and base_symbol in exchange.symbols:
            return base_symbol
        
        # Try alternative formats
        alternatives = [
            f"{coinbase_symbol.replace('-PERP', '')}/USDT",
            f"{coinbase_symbol.replace('-PERP', '')}USDT",
        ]
        
        if exchange:
            for alt in alternatives:
                if alt in exchange.symbols:
                    return alt
        
        return base_symbol  # Return default and let caller handle errors
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        exchange_id: Optional[str] = None,
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV data using CCXT.
        
        Args:
            symbol: Coinbase symbol (e.g., "BTC-PERP")
            timeframe: Candle timeframe (e.g., "1h")
            start: Start datetime
            end: End datetime
            exchange_id: Specific exchange to use (default: first available)
        """
        if not self._initialized:
            await self.initialize()
        
        # Select exchange
        if exchange_id:
            exchange = self._exchanges.get(exchange_id)
            if not exchange:
                logger.error(f"Exchange {exchange_id} not available")
                return []
        else:
            # Use first available exchange
            if not self._exchanges:
                logger.error("No exchanges available")
                return []
            exchange_id = list(self._exchanges.keys())[0]
            exchange = self._exchanges[exchange_id]
        
        ccxt_symbol = self._get_ccxt_symbol(symbol, exchange_id)
        if not ccxt_symbol:
            logger.error(f"Could not map symbol {symbol} for {exchange_id}")
            return []
        
        logger.info(
            f"Fetching {symbol} ({ccxt_symbol}) {timeframe} from {exchange_id} "
            f"({start} to {end})"
        )
        
        all_bars = []
        current_start = start
        
        # Make start and end naive to avoid timezone comparison issues
        start = start.replace(tzinfo=None) if start.tzinfo else start
        end = end.replace(tzinfo=None) if end.tzinfo else end
        
        # CCXT returns timestamps in milliseconds
        since_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        try:
            while since_ms < end_ms:
                # Fetch batch
                ohlcv = await exchange.fetch_ohlcv(
                    ccxt_symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=1000,
                )
                
                if not ohlcv:
                    logger.warning(f"No data returned from {since_ms}")
                    break
                
                # Convert to OHLCVBar objects
                for candle in ohlcv:
                    # CCXT format: [timestamp, open, high, low, close, volume]
                    event_time = datetime.fromtimestamp(candle[0] / 1000)
                    
                    # Skip if past end time
                    if event_time >= end:
                        break
                    
                    # Calculate available_time
                    tf_seconds = self._timeframe_to_seconds(timeframe)
                    available_time = event_time + timedelta(seconds=tf_seconds + 5)
                    
                    bar = OHLCVBar(
                        symbol=symbol,  # Keep original Coinbase symbol
                        timeframe=timeframe,
                        event_time=event_time,
                        available_time=available_time,
                        open=float(candle[1]),
                        high=float(candle[2]),
                        low=float(candle[3]),
                        close=float(candle[4]),
                        volume=float(candle[5]),
                        quality=DataQuality.VALID,
                        quality_notes=f"Source: {exchange_id}",
                    )
                    all_bars.append(bar)
                
                if not ohlcv:
                    break
                
                # Move to next batch
                since_ms = ohlcv[-1][0] + 1
                
                # Progress logging
                if len(all_bars) % 1000 == 0:
                    logger.info(f"Fetched {len(all_bars)} bars from {exchange_id}...")
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV from {exchange_id}: {e}")
        
        logger.info(f"Fetched {len(all_bars)} bars from {exchange_id}")
        return all_bars
    
    async def fetch_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        exchange_id: Optional[str] = None,
    ) -> List[FundingRate]:
        """
        Fetch historical funding rates.
        
        Note: Not all exchanges support historical funding rate queries.
        """
        if not self._initialized:
            await self.initialize()
        
        # Binance is typically best for funding rate history
        exchange_id = exchange_id or "binance"
        exchange = self._exchanges.get(exchange_id)
        
        if not exchange:
            logger.error(f"Exchange {exchange_id} not available")
            return []
        
        ccxt_symbol = self._get_ccxt_symbol(symbol, exchange_id)
        if not ccxt_symbol:
            return []
        
        funding_rates = []
        
        # Make start and end naive
        start = start.replace(tzinfo=None) if start.tzinfo else start
        end = end.replace(tzinfo=None) if end.tzinfo else end
        
        try:
            # Try to fetch funding rate history
            # Note: This is exchange-specific and may not work on all exchanges
            if hasattr(exchange, 'fetch_funding_rate_history'):
                since_ms = int(start.timestamp() * 1000)
                end_ms = int(end.timestamp() * 1000)
                
                while since_ms < end_ms:
                    history = await exchange.fetch_funding_rate_history(
                        ccxt_symbol,
                        since=since_ms,
                        limit=1000,
                    )
                    
                    if not history:
                        break
                    
                    for entry in history:
                        event_time = datetime.fromtimestamp(entry["timestamp"] / 1000)
                        
                        if event_time >= end:
                            break
                        
                        funding = FundingRate(
                            symbol=symbol,
                            event_time=event_time,
                            available_time=event_time + timedelta(seconds=5),
                            rate=float(entry.get("fundingRate", 0)),
                            mark_price=float(entry.get("markPrice", 0)),
                            index_price=float(entry.get("indexPrice", 0)),
                            quality=DataQuality.VALID,
                        )
                        funding_rates.append(funding)
                    
                    since_ms = history[-1]["timestamp"] + 1
            else:
                logger.warning(f"{exchange_id} does not support funding rate history")
                
        except Exception as e:
            logger.error(f"Error fetching funding rates from {exchange_id}: {e}")
        
        logger.info(f"Fetched {len(funding_rates)} funding rates from {exchange_id}")
        return funding_rates
    
    async def fetch_open_interest(
        self,
        symbol: str,
        exchange_id: Optional[str] = None,
    ) -> Optional[OpenInterest]:
        """Fetch current open interest."""
        if not self._initialized:
            await self.initialize()
        
        exchange_id = exchange_id or "binance"
        exchange = self._exchanges.get(exchange_id)
        
        if not exchange:
            return None
        
        ccxt_symbol = self._get_ccxt_symbol(symbol, exchange_id)
        if not ccxt_symbol:
            return None
        
        try:
            if hasattr(exchange, 'fetch_open_interest'):
                oi_data = await exchange.fetch_open_interest(ccxt_symbol)
                
                now = datetime.utcnow().replace(tzinfo=None)  # naive
                return OpenInterest(
                    symbol=symbol,
                    event_time=now,
                    available_time=now,
                    open_interest_contracts=float(oi_data.get("openInterest", 0)),
                    open_interest_usd=float(oi_data.get("openInterestValue", 0)),
                    quality=DataQuality.VALID,
                )
        except Exception as e:
            logger.error(f"Error fetching OI from {exchange_id}: {e}")
        
        return None
    
    async def fetch_multi_exchange_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, List[OHLCVBar]]:
        """
        Fetch OHLCV from all available exchanges.
        
        Useful for cross-validation.
        """
        results = {}
        
        for exchange_id in self._exchanges.keys():
            bars = await self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                exchange_id=exchange_id,
            )
            results[exchange_id] = bars
        
        return results
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60)
    
    async def close(self):
        """Close all exchange connections."""
        for exchange_id, exchange in self._exchanges.items():
            try:
                await exchange.close()
            except Exception as e:
                logger.warning(f"Error closing {exchange_id}: {e}")
        
        self._exchanges.clear()
        self._initialized = False
        logger.info("CCXT connector closed")
    
    def get_available_exchanges(self) -> List[str]:
        """Get list of initialized exchanges."""
        return list(self._exchanges.keys())
    
    async def verify_symbol_availability(self, symbol: str) -> Dict[str, bool]:
        """Check which exchanges have the symbol available."""
        if not self._initialized:
            await self.initialize()
        
        availability = {}
        
        for exchange_id, exchange in self._exchanges.items():
            ccxt_symbol = self._get_ccxt_symbol(symbol, exchange_id)
            availability[exchange_id] = ccxt_symbol in exchange.symbols
        
        return availability