#!/usr/bin/env python3
"""
Simple Test Script for Data Collection Pipeline

This script tests the pipeline components without requiring API credentials.
It uses CCXT to fetch public data from Binance.

Usage:
    cd crypto_trading_system
    pip install aiohttp ccxt pandas numpy
    python test_pipeline.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_ccxt_connector():
    """Test 1: Fetch historical data via CCXT."""
    logger.info("=" * 60)
    logger.info("TEST 1: CCXT Historical Data Fetch")
    logger.info("=" * 60)
    
    from data_collection.ccxt_connector import CCXTConnector
    
    connector = CCXTConnector(exchanges=["binance"])
    
    try:
        await connector.initialize()
        logger.info("✓ CCXT connector initialized")
        
        # Fetch 24 hours of hourly BTC data
        end = datetime.utcnow()
        start = end - timedelta(hours=24)
        
        logger.info(f"Fetching BTC-PERP 1h data from {start} to {end}...")
        
        bars = await connector.fetch_ohlcv(
            symbol="BTC-PERP",
            timeframe="1h",
            start=start,
            end=end,
            exchange_id="binance"
        )
        
        if bars:
            logger.info(f"✓ Fetched {len(bars)} bars")
            logger.info(f"  First bar: {bars[0].event_time} - O:{bars[0].open:.2f} H:{bars[0].high:.2f} L:{bars[0].low:.2f} C:{bars[0].close:.2f}")
            logger.info(f"  Last bar:  {bars[-1].event_time} - O:{bars[-1].open:.2f} H:{bars[-1].high:.2f} L:{bars[-1].low:.2f} C:{bars[-1].close:.2f}")
            return True
        else:
            logger.error("✗ No bars returned")
            return False
            
    except Exception as e:
        logger.error(f"✗ CCXT test failed: {e}")
        return False
    finally:
        await connector.close()


async def test_database():
    """Test 2: Database storage and retrieval."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Database Storage")
    logger.info("=" * 60)
    
    from data_collection.storage import SQLiteDatabase
    from data_collection.models import OHLCVBar, DataQuality
    
    db_path = "/home/claude/crypto_trading_system/data/test.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    db = SQLiteDatabase(db_path)
    
    try:
        db.initialize()
        logger.info("✓ Database initialized")
        
        # Create test bar
        now = datetime.utcnow()
        test_bar = OHLCVBar(
            symbol="BTC-PERP",
            timeframe="1h",
            event_time=now - timedelta(hours=1),
            available_time=now - timedelta(minutes=55),
            open=100000.0,
            high=100500.0,
            low=99500.0,
            close=100200.0,
            volume=1234.56,
            quality=DataQuality.VALID,
        )
        
        # Insert
        success = db.insert_ohlcv(test_bar)
        if success:
            logger.info("✓ Inserted test bar")
        else:
            logger.error("✗ Failed to insert")
            return False
        
        # Retrieve
        df = db.get_ohlcv(
            symbol="BTC-PERP",
            timeframe="1h",
            start=now - timedelta(hours=2),
            end=now,
        )
        
        if not df.empty:
            logger.info(f"✓ Retrieved {len(df)} bar(s)")
            logger.info(f"  Data:\n{df}")
            return True
        else:
            logger.error("✗ No data retrieved")
            return False
            
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        return False
    finally:
        db.close()


async def test_validator():
    """Test 3: Data validation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Data Validation")
    logger.info("=" * 60)
    
    from data_collection.validator import DataValidator, ValidationConfig
    from data_collection.models import OHLCVBar, DataQuality
    
    validator = DataValidator(ValidationConfig())
    
    now = datetime.utcnow()
    
    # Test valid bar
    valid_bar = OHLCVBar(
        symbol="BTC-PERP",
        timeframe="1h",
        event_time=now - timedelta(hours=1),
        available_time=now,
        open=100000.0,
        high=100500.0,
        low=99500.0,
        close=100200.0,
        volume=1234.56,
    )
    
    result = validator.validate_ohlcv(valid_bar)
    if result.is_valid:
        logger.info("✓ Valid bar passed validation")
    else:
        logger.error(f"✗ Valid bar failed: {result.issues}")
        return False
    
    # Test invalid bar (high < low)
    invalid_bar = OHLCVBar(
        symbol="BTC-PERP",
        timeframe="1h",
        event_time=now - timedelta(hours=1),
        available_time=now,
        open=100000.0,
        high=99000.0,  # Invalid: high < low
        low=99500.0,
        close=100200.0,
        volume=1234.56,
    )
    
    result = validator.validate_ohlcv(invalid_bar)
    if not result.is_valid:
        logger.info(f"✓ Invalid bar correctly rejected: {result.issues}")
    else:
        logger.error("✗ Invalid bar should have failed validation")
        return False
    
    return True


async def test_queue():
    """Test 4: Message queue."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Message Queue")
    logger.info("=" * 60)
    
    from data_collection.queue import InMemoryQueue
    
    queue = InMemoryQueue()
    
    try:
        await queue.start()
        logger.info("✓ Queue started")
        
        # Publish messages
        for i in range(5):
            await queue.publish("test_channel", {"value": i})
        logger.info("✓ Published 5 messages")
        
        # Retrieve messages
        messages = await queue.get_batch("test_channel", max_messages=10, timeout_ms=500)
        
        if len(messages) == 5:
            logger.info(f"✓ Retrieved {len(messages)} messages")
            return True
        else:
            logger.error(f"✗ Expected 5 messages, got {len(messages)}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Queue test failed: {e}")
        return False
    finally:
        await queue.close()


async def test_full_pipeline():
    """Test 5: Full pipeline integration (CCXT backfill only, no Coinbase API)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Full Pipeline Integration")
    logger.info("=" * 60)
    
    from data_collection.pipeline import DataPipeline, PipelineConfig
    
    config = PipelineConfig(
        symbols=["BTC-PERP"],
        timeframes=["1h"],
        db_path="/home/claude/crypto_trading_system/data/pipeline_test.db",
        backfill_days=1,  # Just 1 day for testing
        # No API credentials - will use CCXT only
    )
    
    pipeline = DataPipeline(config)
    
    try:
        await pipeline.initialize()
        logger.info("✓ Pipeline initialized")
        
        # Backfill using CCXT
        end = datetime.utcnow()
        start = end - timedelta(days=1)
        
        logger.info("Running backfill...")
        await pipeline.backfill(
            start=start,
            end=end,
            symbols=["BTC-PERP"],
            timeframes=["1h"],
            use_ccxt=True,
        )
        
        # Query the data
        df = pipeline.get_ohlcv(
            symbol="BTC-PERP",
            timeframe="1h",
            start=start,
            end=end,
        )
        
        if not df.empty:
            logger.info(f"✓ Pipeline backfill successful: {len(df)} bars")
            logger.info(f"\nSample data:\n{df.head()}")
            
            # Test point-in-time query
            pit_query_time = end - timedelta(hours=6)
            df_pit = pipeline.get_ohlcv(
                symbol="BTC-PERP",
                timeframe="1h",
                start=start,
                end=pit_query_time,
                as_of=pit_query_time,
            )
            logger.info(f"✓ Point-in-time query: {len(df_pit)} bars as of {pit_query_time}")
            
            # Quality summary
            quality = pipeline.get_quality_summary()
            logger.info(f"✓ Data quality: {quality['validity_rate']}")
            
            return True
        else:
            logger.error("✗ No data after backfill")
            return False
            
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await pipeline.stop()


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("CRYPTO TRADING SYSTEM - DATA PIPELINE TESTS")
    logger.info("=" * 60)
    
    results = {}
    
    # Run tests
    results["CCXT Connector"] = await test_ccxt_connector()
    results["Database"] = await test_database()
    results["Validator"] = await test_validator()
    results["Queue"] = await test_queue()
    results["Full Pipeline"] = await test_full_pipeline()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.info("SOME TESTS FAILED - see above for details")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
