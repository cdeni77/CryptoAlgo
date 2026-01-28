"""
Full Production-Ready run_pipeline.py for Your Coinbase Perps Trading Bot

This script is customized for the top 5 perpetuals you want:
- BTC-PERP
- ETH-PERP
- SOL-PERP
- DOGE-PERP
- XRP-PERP

It will:
1. Automatically detect which of these are actually available on Coinbase Advanced Trade (as of Jan 2026, only BTC-PERP and ETH-PERP are live in the US, but more may be added).
2. Backfill historical data using CCXT (fallback exchanges: OKX → Binance → Bybit → others) - this works even if the contract isn't on Coinbase yet.
3. Start real-time collection via Coinbase WebSocket + REST polling for whatever contracts are live on Coinbase.
4. Log every new candle, ticker, and funding rate in real-time (you can hook your bot logic here).
5. Store everything in SQLite by default (easy switch to TimescaleDB/Redis later).

Just run:
    python run_pipeline.py

Optional flags:
    --backfill-days 30      # default 30 (use smaller values for testing)
    --symbols BTC-PERP,ETH-PERP  # specific symbols to backfill
    --timeframes 1h,1d      # specific timeframes
    --proxy http://127.0.0.1:7890   # strongly recommended in the US
    --db-path ./data/trading.db
    --skip-backfill         # skip backfill, go straight to real-time

Environment variables (recommended):
    export CDP_API_KEY="..."
    export CDP_API_SECRET="..."
    export HTTPS_PROXY="http://127.0.0.1:7890"   # or HTTP_PROXY
    export LOG_LEVEL="INFO"   # or DEBUG
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.pipeline import create_pipeline, PipelineConfig, DataPipeline
from data_collection.models import OHLCVBar, TickerUpdate, FundingRate

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "BTC-PERP",
    "ETH-PERP", 
    "SOL-PERP",
    "DOGE-PERP",
    "XRP-PERP",
]

DEFAULT_TIMEFRAMES = ["1h", "1d"]  # Start with larger timeframes for faster backfill

# Reasonable contract sizes (nano contracts on Coinbase when they exist)
CONTRACT_SIZES = {
    "BTC-PERP": 0.01,
    "ETH-PERP": 0.1,
    "SOL-PERP": 1.0,
    "DOGE-PERP": 5000.0,
    "XRP-PERP": 500.0,
}

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Progress Tracking
# ----------------------------------------------------------------------
class BackfillProgress:
    """Track and display backfill progress."""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.current_symbol = ""
        self.current_timeframe = ""
        self.bars_fetched = 0
        self.start_time = datetime.now()
    
    def update(self, symbol: str, timeframe: str, bars: int = 0):
        self.current_symbol = symbol
        self.current_timeframe = timeframe
        self.bars_fetched = bars
    
    def task_complete(self, symbol: str, timeframe: str, bars: int):
        self.completed_tasks += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        pct = (self.completed_tasks / self.total_tasks) * 100
        logger.info(
            f"✅ [{self.completed_tasks}/{self.total_tasks}] {symbol} {timeframe}: "
            f"{bars} bars | {pct:.0f}% complete | {elapsed:.0f}s elapsed"
        )
    
    def summary(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"🏁 Backfill complete: {self.completed_tasks} tasks in {elapsed:.0f}s")

# ----------------------------------------------------------------------
# Callbacks - Hook your bot logic here!
# ----------------------------------------------------------------------
def on_new_candle(bar: OHLCVBar):
    logger.info(
        f"🕯  NEW CANDLE | {bar.symbol} {bar.timeframe} | "
        f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:.2f}"
    )
    # ←←← Your strategy / indicator updates go here →→→

def on_ticker_update(ticker: TickerUpdate):
    logger.info(
        f"💱 TICKER     | {ticker.symbol} | Price={ticker.price:.2f} | "
        f"Bid={ticker.best_bid:.2f} Ask={ticker.best_ask:.2f}"
    )
    # ←←← Your order execution / risk checks go here →→→

def on_funding_rate(funding: FundingRate):
    logger.info(
        f"🏦 FUNDING    | {funding.symbol} | Rate={funding.rate*100:.6f}% "
        f"({funding.rate_bps:.2f} bps) | Mark={funding.mark_price:.2f}"
    )
    # ←←← Your funding arbitrage / position sizing goes here →→→

# ----------------------------------------------------------------------
# Main pipeline runner
# ----------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Coinbase Perps Top 5 Data Pipeline")
    parser.add_argument("--backfill-days", type=int, default=30, help="Days of historical data to backfill (default: 30)")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols (default: all 5)")
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated list of timeframes (default: 1h,1d)")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL (e.g. http://127.0.0.1:7890)")
    parser.add_argument("--db-path", type=str, default="./data/trading.db", help="SQLite DB path")
    parser.add_argument("--skip-backfill", action="store_true", help="Skip backfill, go straight to real-time")
    parser.add_argument("--backfill-only", action="store_true", help="Only backfill, don't start real-time collection")
    args = parser.parse_args()

    # Parse symbols and timeframes
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    timeframes = args.timeframes.split(",") if args.timeframes else DEFAULT_TIMEFRAMES
    
    logger.info("=" * 60)
    logger.info("🚀 COINBASE PERPS DATA PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Backfill days: {args.backfill_days}")
    logger.info(f"Database: {args.db_path}")
    
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        logger.info(f"🌐 Using proxy: {proxy}")

    api_key = os.environ.get("CDP_API_KEY")
    api_secret = os.environ.get("CDP_API_SECRET")

    if not api_key or not api_secret:
        logger.warning("⚠️  CDP_API_KEY and CDP_API_SECRET not set. Coinbase real-time data will be limited.")
        logger.warning("   Historical backfill via CCXT will still work.")

    # Step 1: Detect which perps are actually live on Coinbase (if credentials provided)
    available_symbols = []
    if api_key and api_secret:
        try:
            rest_client = CoinbaseRESTClient(api_key, api_secret)
            products = await rest_client.get_perpetual_products()
            available_symbols = [p["product_id"] for p in products if p["product_id"] in symbols]
            await rest_client.close()
            logger.info(f"✅ Coinbase live perps in your list: {available_symbols or 'None detected'}")
        except Exception as e:
            logger.warning(f"⚠️  Could not check Coinbase products: {e}")

    if len(available_symbols) < len(symbols):
        logger.info("ℹ️  Some symbols may not be live on Coinbase US yet.")
        logger.info("   Historical backfill via CCXT will still work for all symbols.")

    # Step 2: Create pipeline config
    config = PipelineConfig(
        symbols=symbols,
        timeframes=timeframes,
        coinbase_api_key=api_key,
        coinbase_api_secret=api_secret,
        db_path=args.db_path,
        backfill_days=args.backfill_days,
        proxy=proxy,
        ccxt_exchanges=["okx", "binance", "bybit", "gateio", "mexc", "kucoin"],
        ccxt_use_fallbacks=True,
    )

    pipeline = await create_pipeline(config)

    try:
        # Step 3: Register callbacks
        pipeline.on_ohlcv(on_new_candle)
        pipeline.on_ticker(on_ticker_update)
        pipeline.on_funding(on_funding_rate)

        # Step 4: Backfill historical data
        if not args.skip_backfill:
            total_tasks = len(symbols) * len(timeframes)
            progress = BackfillProgress(total_tasks)
            
            logger.info(f"📥 Starting backfill: {args.backfill_days} days, {len(symbols)} symbols, {len(timeframes)} timeframes")
            logger.info(f"   Total tasks: {total_tasks}")
            
            start_time = datetime.now(timezone.utc) - timedelta(days=args.backfill_days)
            end_time = datetime.now(timezone.utc)
            
            # Backfill each symbol/timeframe combination
            for symbol in symbols:
                for timeframe in timeframes:
                    progress.update(symbol, timeframe)
                    try:
                        await pipeline.backfill(
                            start=start_time,
                            end=end_time,
                            symbols=[symbol],
                            timeframes=[timeframe],
                            use_ccxt=True,
                        )
                        # Get count from database
                        df = pipeline.get_ohlcv(symbol, timeframe, start_time, end_time)
                        bars_count = len(df) if df is not None else 0
                        progress.task_complete(symbol, timeframe, bars_count)
                    except Exception as e:
                        logger.error(f"❌ Failed to backfill {symbol} {timeframe}: {e}")
            
            progress.summary()
            
            # Quality summary
            quality = pipeline.get_quality_summary()
            logger.info(f"📊 Data quality: {quality}")

        if args.backfill_only:
            logger.info("🏁 Backfill-only mode, exiting...")
            return

        # Step 5: Start real-time collection (only works for symbols live on Coinbase)
        if api_key and api_secret:
            await pipeline.start()
            logger.info("🚀 REAL-TIME COLLECTION STARTED - Press Ctrl+C to stop")

            # Run forever
            while True:
                await asyncio.sleep(3600)  # Keep alive
        else:
            logger.info("ℹ️  No Coinbase credentials - skipping real-time collection")
            logger.info("   Set CDP_API_KEY and CDP_API_SECRET to enable real-time data")

    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
    finally:
        await pipeline.stop()
        logger.info("👋 Pipeline shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())