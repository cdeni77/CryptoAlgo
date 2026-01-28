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
2. Backfill 365 days of historical data using CCXT (fallback exchanges: OKX → Binance → Bybit → others) - this works even if the contract isn't on Coinbase yet.
3. Start real-time collection via Coinbase WebSocket + REST polling for whatever contracts are live on Coinbase.
4. Log every new candle, ticker, and funding rate in real-time (you can hook your bot logic here).
5. Store everything in SQLite by default (easy switch to TimescaleDB/Redis later).

Just run:
    python run_pipeline.py

Optional flags:
    --backfill-days 365     # default 365
    --proxy http://127.0.0.1:7890   # strongly recommended in the US
    --db-path ./data/trading.db

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
from data_collection.pipeline import create_pipeline, PipelineConfig, OHLCVBar, TickerUpdate, FundingRate

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SYMBOLS = [
    "BTC-PERP",
    "ETH-PERP", 
    "SOL-PERP",
    "DOGE-PERP",
    "XRP-PERP",
]

TIMEFRAMES = ["5m", "1h", "1d"]  # Optimized for bot: short-term, medium, long-term

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
# Callbacks - Hook your bot logic here!
# ----------------------------------------------------------------------
def on_new_candle(bar: OHLCVBar):
    logger.info(
        f"🕯  NEW CANDLE | {bar.symbol} {bar.timeframe} | "
        f"O={bar.open:.6f} H={bar.high:.6f} L={bar.low:.6f} C={bar.close:.6f} V={bar.volume:.2f}"
    )
    # ←←← Your strategy / indicator updates go here →→→

def on_ticker_update(ticker: TickerUpdate):
    logger.info(
        f"💱 TICKER     | {ticker.symbol} | Price={ticker.price:.6f} | "
        f"Bid={ticker.best_bid:.6f} Ask={ticker.best_ask:.6f}"
    )
    # ←←← Your order execution / risk checks go here →→→

def on_funding_rate(funding: FundingRate):
    logger.info(
        f"🏦 FUNDING    | {funding.symbol} | Rate={funding.rate*100:.6f}% "
        f"({funding.rate_bps:.2f} bps) | Mark={funding.mark_price:.6f}"
    )
    # ←←← Your funding arbitrage / position sizing goes here →→→

# ----------------------------------------------------------------------
# Main pipeline runner
# ----------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Coinbase Perps Top 5 Data Pipeline")
    parser.add_argument("--backfill-days", type=int, default=90, help="Days of historical data to backfill")  # Optimized default
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL (e.g. http://127.0.0.1:7890)")
    parser.add_argument("--db-path", type=str, default="./data/trading.db", help="SQLite DB path")
    args = parser.parse_args()

    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        logger.info(f"🌐 Using proxy: {proxy}")

    api_key = os.environ.get("CDP_API_KEY")
    api_secret = os.environ.get("CDP_API_SECRET")

    if not api_key or not api_secret:
        logger.error("❌ CDP_API_KEY and CDP_API_SECRET must be set in environment variables.")
        sys.exit(1)

    # Step 1: Detect which perps are actually live on Coinbase
    rest_client = CoinbaseRESTClient(api_key, api_secret)
    products = await rest_client.get_perpetual_products()
    available_symbols = [p["product_id"] for p in products if p["product_id"] in SYMBOLS]
    await rest_client.close()

    logger.info(f"✅ Coinbase live perps in your list: {available_symbols or 'None yet'}")
    if len(available_symbols) < len(SYMBOLS):
        logger.warning("⚠  Some symbols not yet live on Coinbase US. Real-time data will only come for live ones. "
                       "Historical backfill via CCXT will still work for all.")

    # Step 2: Create pipeline config
    config = PipelineConfig(
        symbols=SYMBOLS,                          # We want data for all 5
        timeframes=TIMEFRAMES,
        coinbase_api_key=api_key,
        coinbase_api_secret=api_secret,
        db_path=args.db_path,
        backfill_days=args.backfill_days,
        proxy=proxy,
        ccxt_exchanges=["okx", "binance", "bybit", "gateio", "mexc", "kucoin"],  # OKX usually works best behind proxies
        ccxt_use_fallbacks=True,
    )

    pipeline = await create_pipeline(config)

    try:
        # Step 3: Register callbacks
        pipeline.on_ohlcv(on_new_candle)
        pipeline.on_ticker(on_ticker_update)
        pipeline.on_funding(on_funding_rate)  # Enabled for perps bot

        # Step 4: Backfill historical data
        logger.info(f"📥 Starting backfill of {args.backfill_days} days via CCXT...")
        start_time = datetime.now(timezone.utc) - timedelta(days=args.backfill_days)
        await pipeline.backfill(
            start=start_time,
            end=datetime.now(timezone.utc),
            symbols=SYMBOLS,
            timeframes=TIMEFRAMES,
            use_ccxt=True,   # Critical for complete history
        )
        logger.info("📥 Backfill complete!")

        # Step 5: Start real-time collection (only works for symbols live on Coinbase)
        await pipeline.start()
        logger.info("🚀 REAL-TIME COLLECTION STARTED - Press Ctrl+C to stop")

        # Run forever
        while True:
            await asyncio.sleep(3600)  # Keep alive

    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
    finally:
        await pipeline.stop()
        logger.info("👋 Pipeline shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())