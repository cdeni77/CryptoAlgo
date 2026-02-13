#!/usr/bin/env python3
"""
Unified Data Pipeline for Coinbase Perps Trading Bot

This script combines:
1. Dynamic resolution of Coinbase "Smart Perp" product IDs
2. Hybrid OHLCV backfill (Coinbase Native -> CCXT fallback)
3. Funding rate backfill from CCXT exchanges (Binance, OKX, Bybit)
4. Real-time data collection via WebSocket

Usage:
    # Full backfill + real-time collection
    python run_pipeline.py --backfill-days 365

    # Backfill only (no real-time)
    python run_pipeline.py --backfill-days 365 --backfill-only

    # Skip OHLCV, only fetch funding rates
    python run_pipeline.py --funding-only --backfill-days 365

    # Real-time only (skip all backfill)
    python run_pipeline.py --skip-backfill

    # Include open interest data
    python run_pipeline.py --backfill-days 365 --include-oi
"""

import argparse
import asyncio
import logging
import os
import sys

from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.pipeline import create_pipeline, PipelineConfig, ensure_naive_utc
from data_collection.models import OHLCVBar, TickerUpdate, FundingRate, OpenInterest, DataQuality
from data_collection.ccxt_connector import CCXTConnector
from data_collection.storage import SQLiteDatabase

# Configuration

# Map your desired Assets to Coinbase "Smart Perp" Codes
ASSET_TO_CODE_MAP = {
    "BTC": "BIP",
    "ETH": "ETP",
    "SOL": "SLP",
    "XRP": "XPP",
    "DOGE": "DOP",
}

DEFAULT_TIMEFRAMES = ["1h", "1d"]
DEFAULT_SYMBOLS = ["BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP", "DOGE-PERP"]

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# Helper Classes

class BackfillProgress:
    """Track and display backfill progress."""
    
    def __init__(self, total_tasks: int, description: str = "Backfill"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = datetime.now()
        self.description = description
    
    def update(self, symbol: str, timeframe: str = "", bars: int = 0):
        """Update progress (placeholder for UI)."""
        pass
    
    def task_complete(self, symbol: str, timeframe: str = "", count: int = 0):
        """Mark a task as complete."""
        self.completed_tasks += 1
        elapsed = (datetime.now() - self.start_time).total_seconds()
        pct = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 100
        
        tf_str = f" {timeframe}" if timeframe else ""
        logger.info(
            f"✅ [{self.completed_tasks}/{self.total_tasks}] {symbol}{tf_str}: "
            f"{count} records | {pct:.0f}% complete | {elapsed:.0f}s elapsed"
        )
    
    def summary(self):
        """Print summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"🏁 {self.description} complete: {self.completed_tasks} tasks in {elapsed:.0f}s")


# Callbacks for Real-time Data

_ticker_last_log: Dict[str, datetime] = defaultdict(lambda: datetime.min)
TICKER_LOG_INTERVAL = timedelta(minutes=1)


def on_new_candle(bar: OHLCVBar):
    """Callback for new OHLCV candles."""
    logger.info(f"🕯  NEW CANDLE | {bar.symbol} {bar.timeframe} | C={bar.close:.2f} V={bar.volume:.2f}")


def on_ticker_update(ticker: TickerUpdate):
    """Callback for ticker updates (throttled)."""
    now = datetime.utcnow()
    last_logged = _ticker_last_log[ticker.symbol]
    
    if now - last_logged >= TICKER_LOG_INTERVAL:
        logger.info(
            f"💱 TICKER     | {ticker.symbol} | {ticker.price:.2f} | "
            f"B={ticker.best_bid:.2f} A={ticker.best_ask:.2f}"
        )
        _ticker_last_log[ticker.symbol] = now


def on_funding_rate(funding: FundingRate):
    """Callback for funding rate updates."""
    logger.info(f"🏦 FUNDING    | {funding.symbol} | {funding.rate*100:.6f}%")


# Backfill Functions

async def backfill_ohlcv(
    pipeline,
    symbols: List[str],
    timeframes: List[str],
    start_time: datetime,
    end_time: datetime,
):
    """
    Hybrid OHLCV backfill: Coinbase Native -> CCXT fallback.
    """
    print("\n" + "=" * 70)
    print("📊 OHLCV BACKFILL (Hybrid: Coinbase + CCXT)")
    print("=" * 70)
    print(f"Period: {start_time.date()} to {end_time.date()}")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {timeframes}")
    print()
    
    progress = BackfillProgress(len(symbols) * len(timeframes), "OHLCV Backfill")
    
    for symbol in symbols:
        for tf in timeframes:
            progress.update(symbol, tf)
            try:
                # 1. Try Coinbase Native First
                logger.info(f"   Attempting Coinbase Native fetch for {symbol} {tf}...")
                await pipeline.backfill(start_time, end_time, [symbol], [tf], use_ccxt=False)
                
                # Check what we got
                df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                cb_count = len(df) if df is not None and not df.empty else 0
                
                # 2. Gap Filling Logic
                if cb_count == 0:
                    # No data from Coinbase -> Full CCXT fallback
                    logger.warning(f"   ⚠️ Coinbase returned 0 bars for {symbol}. Fetching via CCXT...")
                    await pipeline.backfill(start_time, end_time, [symbol], [tf], use_ccxt=True)
                
                elif cb_count > 0:
                    first_bar_time = df.index.min()
                    if hasattr(first_bar_time, 'to_pydatetime'):
                        first_bar_time = first_bar_time.to_pydatetime()
                    
                    gap = first_bar_time - start_time
                    if gap > timedelta(hours=12):
                        logger.info(f"   ⚠️ Gap detected: Coinbase data starts {first_bar_time.date()} (missing {gap.days} days)")
                        logger.info(f"      Backfilling pre-history gap via CCXT...")
                        await pipeline.backfill(
                            start=start_time,
                            end=first_bar_time,
                            symbols=[symbol],
                            timeframes=[tf],
                            use_ccxt=True
                        )
                
                # Final count
                df = pipeline.get_ohlcv(symbol, tf, start_time, end_time)
                final_count = len(df) if df is not None and not df.empty else 0
                progress.task_complete(symbol, tf, final_count)
                
            except Exception as e:
                logger.error(f"❌ OHLCV backfill failed for {symbol} {tf}: {e}")
                import traceback
                traceback.print_exc()
    
    progress.summary()


async def backfill_funding_rates(
    symbols: List[str],
    start: datetime,
    end: datetime,
    db: SQLiteDatabase,
    proxy: Optional[str] = None,
):
    """
    Backfill funding rates from CCXT exchanges.
    """
    print("\n" + "=" * 70)
    print("🏦 FUNDING RATE BACKFILL")
    print("=" * 70)
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {symbols}")
    print()
    
    # Initialize CCXT connector
    connector = CCXTConnector(
        exchanges=["binance", "okx", "bybit"],
        proxy=proxy,
        use_fallbacks=True,
    )
    
    try:
        await connector.initialize()
        
        available_exchanges = connector.get_available_exchanges()
        if not available_exchanges:
            logger.error("No exchanges available for funding rates! Check network/proxy.")
            return
        
        print(f"✓ Connected to exchanges: {available_exchanges}")
        print()
        
        progress = BackfillProgress(len(symbols), "Funding Rate Backfill")
        total_inserted = 0
        
        for symbol in symbols:
            logger.info(f"\n📊 Processing funding rates for {symbol}")
            
            # Check existing data
            existing_start = db.get_earliest_funding_time(symbol)
            existing_end = db.get_latest_funding_time(symbol)
            
            if existing_start and existing_end:
                logger.info(f"  Existing data: {existing_start.date()} to {existing_end.date()}")
            else:
                logger.info(f"  No existing funding data")
            
            symbol_inserted = 0
            
            if existing_start and existing_end:
                # Prepend if needed
                if start < existing_start:
                    logger.info(f"  📥 Fetching pre-history: {start.date()} to {existing_start.date()}")
                    rates = await connector.fetch_funding_rates(
                        symbol=symbol,
                        start=start,
                        end=existing_start,
                    )
                    if rates:
                        inserted = db.insert_funding_rate_batch(rates)
                        symbol_inserted += inserted
                        logger.info(f"  ✓ Inserted {inserted} pre-history rates")
                
                # Append if needed
                if end > existing_end:
                    logger.info(f"  📥 Fetching new data: {existing_end.date()} to {end.date()}")
                    rates = await connector.fetch_funding_rates(
                        symbol=symbol,
                        start=existing_end,
                        end=end,
                    )
                    if rates:
                        inserted = db.insert_funding_rate_batch(rates)
                        symbol_inserted += inserted
                        logger.info(f"  ✓ Inserted {inserted} new rates")
            else:
                # Full range fetch
                logger.info(f"  📥 Fetching full range: {start.date()} to {end.date()}")
                rates = await connector.fetch_funding_rates(
                    symbol=symbol,
                    start=start,
                    end=end,
                )
                if rates:
                    inserted = db.insert_funding_rate_batch(rates)
                    symbol_inserted += inserted
                    logger.info(f"  ✓ Inserted {inserted} rates")
                else:
                    logger.warning(f"  ⚠️ No funding rates found for {symbol}")
            
            total_inserted += symbol_inserted
            progress.task_complete(symbol, count=symbol_inserted)
        
        progress.summary()
        
        # Final summary
        print("\n📈 Funding Data Summary:")
        stats = db.get_funding_stats()
        for symbol, s in stats.items():
            daily_cost = s['avg_rate_bps'] * 3  # 3 funding periods per day (8h each)
            print(f"  {symbol}: {s['count']} records, avg {s['avg_rate_bps']:.4f} bps/8h (~{daily_cost:.4f} bps/day)")
        
        print(f"\nTotal funding rates inserted: {total_inserted}")
        
    finally:
        await connector.close()


async def backfill_open_interest(
    symbols: List[str],
    start: datetime,
    end: datetime,
    db: SQLiteDatabase,
    proxy: Optional[str] = None,
):
    """
    Backfill historical open interest data from CCXT exchanges.
    """
    print("\n" + "=" * 70)
    print("📊 OPEN INTEREST BACKFILL")
    print("=" * 70)
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {symbols}")
    print()
    
    # Initialize CCXT connector
    connector = CCXTConnector(
        exchanges=["bybit", "okx", "binance"],
        proxy=proxy,
        use_fallbacks=True,
    )
    
    try:
        await connector.initialize()
        
        available_exchanges = connector.get_available_exchanges()
        if not available_exchanges:
            logger.error("No exchanges available for OI! Check network/proxy.")
            return
        
        print(f"✓ Connected to exchanges: {available_exchanges}")
        print()
        
        progress = BackfillProgress(len(symbols), "Open Interest Backfill")
        total_inserted = 0
        
        for symbol in symbols:
            logger.info(f"\n📊 Processing OI history for {symbol}")
            
            # Check existing data
            existing_start = db.get_earliest_oi_time(symbol)
            existing_end = db.get_latest_oi_time(symbol)
            
            if existing_start and existing_end:
                logger.info(f"  Existing OI data: {existing_start.date()} to {existing_end.date()}")
            else:
                logger.info(f"  No existing OI data")
            
            all_history = []
            
            if existing_start and existing_end:
                # Prepend if needed
                if start < existing_start:
                    logger.info(f"  📥 Fetching OI pre-history: {start.date()} to {existing_start.date()}")
                    history = await connector.fetch_open_interest_history(
                        symbol=symbol, timeframe='1h',
                        start=start, end=existing_start, limit=200
                    )
                    if history:
                        all_history.extend(history)
                
                # Append if needed
                if end > existing_end:
                    logger.info(f"  📥 Fetching new OI data: {existing_end.date()} to {end.date()}")
                    history = await connector.fetch_open_interest_history(
                        symbol=symbol, timeframe='1h',
                        start=existing_end, end=end, limit=200
                    )
                    if history:
                        all_history.extend(history)
                
                if not all_history:
                    logger.info(f"  ✓ OI data already up to date")
            else:
                # Full range fetch
                logger.info(f"  📥 Fetching full OI range: {start.date()} to {end.date()}")
                all_history = await connector.fetch_open_interest_history(
                    symbol=symbol, timeframe='1h',
                    start=start, end=end, limit=200
                )
            
            if all_history:
                # Parse dicts to OpenInterest objects
                oi_list = []
                for entry in all_history:
                    try:
                        event_time = datetime.fromtimestamp(entry['timestamp'] / 1000).replace(tzinfo=None)
                        contracts = float(entry.get('openInterestAmount') or entry.get('baseVolume') or 0)
                        value = float(entry.get('openInterestValue') or entry.get('quoteVolume') or 0)
                        oi = OpenInterest(
                            symbol=symbol, event_time=event_time,
                            available_time=event_time + timedelta(seconds=5),
                            open_interest_contracts=contracts,
                            open_interest_usd=value,
                            quality=DataQuality.VALID,
                        )
                        oi_list.append(oi)
                    except Exception as e:
                        logger.warning(f"Failed to parse OI entry: {e}")
                
                if oi_list:
                    try:
                        inserted = db.insert_open_interest_batch(oi_list)
                        total_inserted += inserted
                        logger.info(f"  ✓ Inserted {inserted} OI records")
                    except Exception as e:
                        logger.error(f"Failed to insert OI batch: {e}")
            
            progress.task_complete(symbol, count=len(all_history))
        
        progress.summary()
        print(f"\nTotal OI records inserted: {total_inserted}")
        
    finally:
        await connector.close()


# Symbol Resolution

async def resolve_coinbase_symbols(api_key: str, api_secret: str) -> List[str]:
    """
    Resolve active Coinbase Perpetual contract IDs.
    
    Returns list of product IDs like ["BIP-20DEC30-CDE", "ETP-20DEC30-CDE", ...]
    """
    logger.info("🔍 Resolving active Coinbase Perpetual contracts...")
    
    try:
        client = CoinbaseRESTClient(api_key, api_secret)
        target_codes = list(ASSET_TO_CODE_MAP.values())
        products = await client.get_perpetual_products(target_codes=target_codes)
        await client.close()
        
        active_symbols = [p['product_id'] for p in products]
        
        if active_symbols:
            logger.info(f"✅ Found {len(active_symbols)} active contracts:")
            for s in active_symbols:
                logger.info(f"   -> {s}")
        else:
            logger.warning("⚠️ No matching perpetuals found from Coinbase API")
        
        return active_symbols
        
    except Exception as e:
        logger.error(f"⚠️ Failed to resolve Coinbase symbols: {e}")
        return []


# Main

async def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Pipeline for Coinbase Perps Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --backfill-days 365
  python run_pipeline.py --backfill-days 365 --backfill-only
  python run_pipeline.py --funding-only --backfill-days 365
  python run_pipeline.py --skip-backfill
  python run_pipeline.py --backfill-days 365 --include-oi
        """
    )
    
    # Time range
    parser.add_argument("--backfill-days", type=int, default=365, help="Days of history to fetch")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD), overrides --backfill-days")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    
    # Data types
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated timeframes (default: 1h,1d)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (default: auto-detect from Coinbase)")
    
    # Modes
    parser.add_argument("--skip-backfill", action="store_true", help="Skip all backfill, only run real-time")
    parser.add_argument("--backfill-only", action="store_true", help="Only backfill, don't start real-time")
    parser.add_argument("--funding-only", action="store_true", help="Only backfill funding rates")
    parser.add_argument("--ohlcv-only", action="store_true", help="Only backfill OHLCV data")
    parser.add_argument("--include-oi", action="store_true", help="Also fetch open interest data")
    
    # Paths
    parser.add_argument("--db-path", type=str, default="./data/trading.db", help="Database path")
    
    # Network
    parser.add_argument("--proxy", type=str, help="HTTP proxy URL for CCXT exchanges")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start and args.end:
        start_time = datetime.strptime(args.start, "%Y-%m-%d")
        end_time = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_time = datetime.strptime(args.start, "%Y-%m-%d")
        end_time = datetime.utcnow()
    else:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=args.backfill_days)
    
    # Ensure naive UTC
    start_time = ensure_naive_utc(start_time)
    end_time = ensure_naive_utc(end_time)
    
    timeframes = args.timeframes.split(",") if args.timeframes else DEFAULT_TIMEFRAMES
    proxy = args.proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    
    # API credentials
    api_key = os.environ.get("COINBASE_API_KEY")
    api_secret = os.environ.get("COINBASE_API_SECRET")
    
    print("=" * 70)
    print("🚀 UNIFIED DATA PIPELINE - Coinbase Perps Trading")
    print("=" * 70)
    print(f"Date Range: {start_time.date()} to {end_time.date()} ({args.backfill_days} days)")
    print(f"Timeframes: {timeframes}")
    print(f"Database: {args.db_path}")
    print(f"Proxy: {proxy or 'None'}")
    print()
    
    # Step 1: Resolve Symbols
    if args.symbols:
        # User-specified symbols
        symbols = [s.strip() for s in args.symbols.split(",")]
        logger.info(f"Using user-specified symbols: {symbols}")
    elif api_key and api_secret:
        # Auto-detect from Coinbase
        symbols = await resolve_coinbase_symbols(api_key, api_secret)
        if not symbols:
            logger.warning("Falling back to default symbols")
            symbols = DEFAULT_SYMBOLS
    else:
        # Default symbols (for funding-only mode without API keys)
        symbols = DEFAULT_SYMBOLS
        logger.info(f"Using default symbols: {symbols}")
    
    if not symbols:
        logger.error("❌ No symbols to process!")
        return
    
    # Step 2: Initialize Database
    db = SQLiteDatabase(args.db_path)
    db.initialize()
    logger.info(f"✓ Database initialized: {args.db_path}")
    
    # Step 3: Backfill Data
    pipeline = None
    
    if not args.skip_backfill:
        
        # OHLCV Backfill
        if not args.funding_only:
            if api_key and api_secret:
                # Initialize pipeline for OHLCV
                config = PipelineConfig(
                    symbols=symbols,
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
                
                # Register callbacks
                pipeline.on_ohlcv(on_new_candle)
                pipeline.on_ticker(on_ticker_update)
                pipeline.on_funding(on_funding_rate)
                
                await backfill_ohlcv(pipeline, symbols, timeframes, start_time, end_time)
            else:
                logger.warning("⚠️ No Coinbase API keys - skipping OHLCV backfill")
        
        # Funding Rate Backfill
        if not args.ohlcv_only:
            await backfill_funding_rates(symbols, start_time, end_time, db, proxy)
        
        # Open Interest Backfill (optional)
        if args.include_oi and not args.ohlcv_only:
            await backfill_open_interest(symbols, start_time, end_time, db, proxy)
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("📊 DATA SUMMARY")
    print("=" * 70)
    
    # OHLCV summary
    print("\n📈 OHLCV Data:")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT symbol, timeframe, COUNT(*) as count, 
                   MIN(event_time) as earliest, MAX(event_time) as latest
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)
        for row in cursor.fetchall():
            print(f"  {row['symbol']} {row['timeframe']}: {row['count']} bars "
                  f"({row['earliest'][:10] if row['earliest'] else 'N/A'} to "
                  f"{row['latest'][:10] if row['latest'] else 'N/A'})")
    
    # Funding summary
    print("\n🏦 Funding Rates:")
    stats = db.get_funding_stats()
    if stats:
        for symbol, s in stats.items():
            daily_cost = s['avg_rate_bps'] * 3
            print(f"  {symbol}: {s['count']} records, avg {s['avg_rate_bps']:.4f} bps/8h (~{daily_cost:.2f} bps/day)")
    else:
        print("  No funding rate data")
    
    # OI summary
    if args.include_oi:
        print("\n📊 Open Interest:")
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, COUNT(*) as count, AVG(open_interest_contracts) as avg_oi
                FROM open_interest
                GROUP BY symbol
            """)
            rows = cursor.fetchall()
            if rows:
                for row in rows:
                    print(f"  {row['symbol']}: {row['count']} records, avg OI: {row['avg_oi']:,.0f} contracts")
            else:
                print("  No open interest data")
    
    # Step 5: Real-time Collection
    if args.backfill_only:
        logger.info("\n🏁 Backfill-only mode, exiting.")
        db.close()
        if pipeline:
            await pipeline.stop()
        return
    
    if not api_key or not api_secret:
        logger.warning("\n⚠️ No API keys - cannot start real-time collection")
        db.close()
        return
    
    # Ensure pipeline is initialized
    if pipeline is None:
        config = PipelineConfig(
            symbols=symbols,
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
        pipeline.on_ohlcv(on_new_candle)
        pipeline.on_ticker(on_ticker_update)
        pipeline.on_funding(on_funding_rate)
    
    try:
        await pipeline.start()
        logger.info("\n🚀 Real-time collection started. Press Ctrl+C to stop.")
        
        while True:
            await asyncio.sleep(3600)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopped by user")
    finally:
        if pipeline:
            await pipeline.stop()
        db.close()
        logger.info("👋 Pipeline shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())
