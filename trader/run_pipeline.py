"""
Full Production-Ready run_pipeline.py for Your Coinbase Perps Trading Bot

This script dynamically resolves Coinbase's unique "Smart Perp" product IDs
and implements a "Hybrid Backfill" strategy:
1. Try fetching exact contract history from Coinbase (BIP-xxx).
2. If unavailable (contract is too new), fallback to CCXT proxy (BTC/USDT from OKX/Binance).
"""

import argparse
import asyncio
import logging
import os
import sys

from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.pipeline import create_pipeline, PipelineConfig, ensure_naive_utc
from data_collection.models import OHLCVBar, TickerUpdate, FundingRate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Map your desired Assets to Coinbase "Smart Perp" Codes
# The script uses these to find the specific active Contract ID (e.g. BIP -> BIP-20DEC30-CDE)
ASSET_TO_CODE_MAP = {
    "BTC": "BIP",
    "ETH": "ETP",
    "SOL": "SLP",
    "XRP": "XPP",
    "DOGE": "DOP",
}

DEFAULT_TIMEFRAMES = ["1h", "1d"]

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Helper Classes
# ----------------------------------------------------------------------
class BackfillProgress:
    """Track and display backfill progress."""
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = datetime.now()
    
    def update(self, symbol: str, timeframe: str, bars: int = 0):
        pass # Placeholder for UI updates
    
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
# Callbacks
# ----------------------------------------------------------------------
def on_new_candle(bar: OHLCVBar):
    logger.info(f"🕯  NEW CANDLE | {bar.symbol} {bar.timeframe} | C={bar.close:.2f} V={bar.volume:.2f}")

_ticker_last_log: dict[str, datetime] = defaultdict(lambda: datetime.min)
TICKER_LOG_INTERVAL = timedelta(minutes=1)
def on_ticker_update(ticker: TickerUpdate):
    now = datetime.utcnow()
    last_logged = _ticker_last_log[ticker.symbol]
    
    if now - last_logged >= TICKER_LOG_INTERVAL:
        logger.info(
            f"💱 TICKER     | {ticker.symbol} | {ticker.price:.2f} | "
            f"B={ticker.best_bid:.2f} A={ticker.best_ask:.2f}"
        )
        _ticker_last_log[ticker.symbol] = now

def on_funding_rate(funding: FundingRate):
    logger.info(f"🏦 FUNDING    | {funding.symbol} | {funding.rate*100:.6f}%")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Coinbase Perps Data Pipeline")
    parser.add_argument("--backfill-days", type=int, default=30)
    parser.add_argument("--timeframes", type=str, default=None)
    parser.add_argument("--db-path", type=str, default="./data/trading.db")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--backfill-only", action="store_true")
    args = parser.parse_args()

    timeframes = args.timeframes.split(",") if args.timeframes else DEFAULT_TIMEFRAMES
    
    api_key = os.environ.get("CDP_API_KEY")
    api_secret = os.environ.get("CDP_API_SECRET")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    # ------------------------------------------------------------------
    # Step 1: Resolve Active Coinbase Contracts
    # ------------------------------------------------------------------
    logger.info("🔍 Resolving active Coinbase Perpetual contracts...")
    active_symbols = []
    
    if api_key and api_secret:
        try:
            client = CoinbaseRESTClient(api_key, api_secret)
            # Find all futures that match our codes (BIP, ETP, etc)
            target_codes = list(ASSET_TO_CODE_MAP.values())
            products = await client.get_perpetual_products(target_codes=target_codes)
            await client.close()
            
            # Extract the actual Product IDs (e.g. "BIP-20DEC30-CDE")
            active_symbols = [p['product_id'] for p in products]
            
            if active_symbols:
                logger.info(f"✅ Found {len(active_symbols)} active contracts:")
                for s in active_symbols:
                    logger.info(f"   -> {s}")
            else:
                logger.error("❌ No matching perpetuals found! Check API permissions.")
                return

        except Exception as e:
            logger.error(f"⚠️ Failed to resolve symbols: {e}")
            return
    else:
        logger.error("❌ API Keys required to detect active contracts.")
        return

    # ------------------------------------------------------------------
    # Step 2: Initialize Pipeline
    # ------------------------------------------------------------------
    logger.info("⚙️  Initializing Pipeline...")
    config = PipelineConfig(
        symbols=active_symbols, 
        timeframes=timeframes,
        coinbase_api_key=api_key,
        coinbase_api_secret=api_secret,
        db_path=args.db_path,
        backfill_days=args.backfill_days,
        proxy=proxy,
        ccxt_exchanges=["okx", "binance", "bybit"],
        ccxt_use_fallbacks=True,
    )

    pipeline = await create_pipeline(config)

    try:
        pipeline.on_ohlcv(on_new_candle)
        pipeline.on_ticker(on_ticker_update)
        pipeline.on_funding(on_funding_rate)

        # ------------------------------------------------------------------
        # Step 3: Hybrid Backfill (Coinbase Native -> CCXT Fallback)
        # ------------------------------------------------------------------
        if not args.skip_backfill:
            logger.info(f"📥 Starting backfill for {len(active_symbols)} symbols...")
            
            progress = BackfillProgress(len(active_symbols) * len(timeframes))
            
            start_time = ensure_naive_utc(datetime.now(timezone.utc) - timedelta(days=args.backfill_days))
            end_time = ensure_naive_utc(datetime.now(timezone.utc))
            
            for symbol in active_symbols:
                for tf in timeframes:
                    progress.update(symbol, tf)
                    try:
                        # 1. Try Coinbase Native First
                        logger.info(f"   Attempting Coinbase Native fetch for {symbol}...")
                        await pipeline.backfill(start_time, end_time, [symbol], [tf], use_ccxt=False)
                        
                        # Check what we got
                        df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                        cb_count = len(df) if df is not None else 0
                        
                        # 2. Gap Filling Logic
                        if cb_count == 0:
                            # Scenario A: No data at all (Contract too new or API error) -> Full Proxy
                            logger.warning(f"   ⚠️ Coinbase returned 0 bars for {symbol}. Fetching full history via CCXT...")
                            await pipeline.backfill(start_time, end_time, [symbol], [tf], use_ccxt=True)
                        
                        elif cb_count > 0:
                            first_bar_time = df.index.min().to_pydatetime()
                            
                            gap = first_bar_time - start_time
                            if gap > timedelta(hours=12):
                                logger.info(f"   ⚠️ Gap detected: Coinbase data starts {first_bar_time} (missing {gap.days} days).")
                                logger.info(f"      Backfilling pre-history gap via CCXT...")
                                
                                # Fetch CCXT only for the missing period (Start -> First Bar)
                                await pipeline.backfill(
                                    start=start_time, 
                                    end=first_bar_time, 
                                    symbols=[symbol], 
                                    timeframes=[tf], 
                                    use_ccxt=True
                                )
                        
                        # Final count
                        df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                        final_count = len(df) if df is not None else 0
                        progress.task_complete(symbol, tf, final_count)

                    except Exception as e:
                        logger.error(f"❌ Backfill failed for {symbol}: {e}")
            
            progress.summary()

        if args.backfill_only:
            logger.info("🏁 Backfill-only mode, exiting.")
            return

        # ------------------------------------------------------------------
        # Step 4: Real-time Collection
        # ------------------------------------------------------------------
        if api_key:
            await pipeline.start()
            logger.info("🚀 Real-time collection started. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    finally:
        await pipeline.stop()
        logger.info("👋 Pipeline shut down cleanly")

if __name__ == "__main__":
    asyncio.run(main())