#!/usr/bin/env python3
"""
Minimal Test Script - Tests core components without external dependencies.

This tests the parts that don't require aiohttp/ccxt:
- Data models
- Validator
- Database (SQLite)
- Queue (in-memory)

Run: python3 test_minimal.py
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_models():
    """Test data models."""
    logger.info("TEST: Data Models")
    
    from data_collection.models import OHLCVBar, FundingRate, DataQuality
    
    now = datetime.utcnow()
    
    # Test OHLCVBar
    bar = OHLCVBar(
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
    
    assert bar.symbol == "BTC-PERP"
    assert bar.bar_id == f"BTC-PERP_1h_{bar.event_time.isoformat()}"
    
    # Test serialization
    bar_dict = bar.to_dict()
    bar_restored = OHLCVBar.from_dict(bar_dict)
    assert bar_restored.close == bar.close
    
    # Test bi-temporal check
    assert bar.is_available_at(now)
    assert not bar.is_available_at(now - timedelta(hours=2))
    
    # Test FundingRate
    funding = FundingRate(
        symbol="BTC-PERP",
        event_time=now,
        available_time=now,
        rate=0.0001,  # 0.01%
        mark_price=100000.0,
        index_price=99950.0,
    )
    
    assert funding.rate_bps == 1.0  # 1 basis point
    assert abs(funding.basis - 0.0005) < 0.0001  # ~0.05%
    
    logger.info("✓ Data models OK")
    return True


def test_validator():
    """Test data validator."""
    logger.info("TEST: Validator")
    
    from data_collection.validator import DataValidator, ValidationConfig
    from data_collection.models import OHLCVBar, OrderBookSnapshot, OrderBookLevel, DataQuality
    
    validator = DataValidator(ValidationConfig())
    now = datetime.utcnow()
    
    # Valid bar
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
    assert result.is_valid, f"Valid bar should pass: {result.issues}"
    
    # Invalid bar (high < low)
    invalid_bar = OHLCVBar(
        symbol="BTC-PERP",
        timeframe="1h",
        event_time=now - timedelta(hours=1),
        available_time=now,
        open=100000.0,
        high=99000.0,  # Invalid!
        low=99500.0,
        close=100200.0,
        volume=1234.56,
    )
    
    result = validator.validate_ohlcv(invalid_bar)
    assert not result.is_valid, "Invalid bar should fail"
    assert any("High" in issue for issue in result.issues)
    
    # Valid orderbook
    book = OrderBookSnapshot(
        symbol="BTC-PERP",
        event_time=now,
        available_time=now,
        bids=[OrderBookLevel(100000, 1.0), OrderBookLevel(99999, 2.0)],
        asks=[OrderBookLevel(100001, 1.5), OrderBookLevel(100002, 2.5)],
    )
    
    result = validator.validate_orderbook(book)
    assert result.is_valid
    assert book.spread == 1.0
    assert book.imbalance(2) != 0  # Should calculate imbalance
    
    logger.info("✓ Validator OK")
    return True


def test_database():
    """Test SQLite database."""
    logger.info("TEST: Database")
    
    from data_collection.storage import SQLiteDatabase
    from data_collection.models import OHLCVBar, FundingRate, DataQuality
    
    # Use temp path
    db_path = "/tmp/test_trading.db"
    db = SQLiteDatabase(db_path)
    
    try:
        db.initialize()
        
        now = datetime.utcnow()
        
        # Insert OHLCV bars
        bars = []
        for i in range(10):
            bar = OHLCVBar(
                symbol="BTC-PERP",
                timeframe="1h",
                event_time=now - timedelta(hours=10-i),
                available_time=now - timedelta(hours=10-i) + timedelta(minutes=5),
                open=100000.0 + i * 100,
                high=100000.0 + i * 100 + 50,
                low=100000.0 + i * 100 - 50,
                close=100000.0 + i * 100 + 25,
                volume=1000.0 + i * 10,
            )
            bars.append(bar)
        
        inserted = db.insert_ohlcv_batch(bars)
        assert inserted == 10, f"Expected 10 inserted, got {inserted}"
        
        # Query all
        df = db.get_ohlcv(
            symbol="BTC-PERP",
            timeframe="1h",
            start=now - timedelta(hours=12),
            end=now,
        )
        assert len(df) == 10, f"Expected 10 rows, got {len(df)}"
        
        # Point-in-time query (critical for backtesting!)
        # Query as if we're 5 hours ago - should only see bars available then
        pit_time = now - timedelta(hours=5)
        df_pit = db.get_ohlcv(
            symbol="BTC-PERP",
            timeframe="1h",
            start=now - timedelta(hours=12),
            end=now,
            as_of=pit_time,  # Only data available at this time
        )
        
        # Bars 0-5 should be available (event_time + 5min < pit_time)
        # This tests our anti-lookahead-bias mechanism
        assert len(df_pit) < len(df), f"Point-in-time should return fewer rows"
        logger.info(f"  Full query: {len(df)} bars, Point-in-time: {len(df_pit)} bars")
        
        # Test funding rates
        funding = FundingRate(
            symbol="BTC-PERP",
            event_time=now,
            available_time=now,
            rate=0.0001,
            mark_price=100000.0,
            index_price=99950.0,
        )
        
        assert db.insert_funding_rate(funding)
        
        df_funding = db.get_funding_rates(
            symbol="BTC-PERP",
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
        )
        assert len(df_funding) == 1
        
        logger.info("✓ Database OK")
        return True
        
    finally:
        db.close()
        Path(db_path).unlink(missing_ok=True)


async def test_queue():
    """Test in-memory message queue."""
    logger.info("TEST: Message Queue")
    
    from data_collection.queue import InMemoryQueue, Channels
    
    queue = InMemoryQueue(max_size=100)
    
    try:
        await queue.start()
        
        # Publish messages
        for i in range(10):
            success = await queue.publish("test_channel", {"index": i, "value": f"msg_{i}"})
            assert success
        
        # Get batch
        messages = await queue.get_batch("test_channel", max_messages=5, timeout_ms=100)
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}"
        
        # Get remaining
        messages = await queue.get_batch("test_channel", max_messages=10, timeout_ms=100)
        assert len(messages) == 5, f"Expected 5 remaining, got {len(messages)}"
        
        # Test subscriber callback
        received = []
        
        async def callback(msg):
            received.append(msg.data)
        
        await queue.subscribe("callback_channel", callback)
        
        # Publish and wait for processing
        await queue.publish("callback_channel", {"test": "data"})
        await asyncio.sleep(0.1)  # Allow dispatch loop to process
        
        assert len(received) == 1
        assert received[0]["test"] == "data"
        
        logger.info("✓ Message Queue OK")
        return True
        
    finally:
        await queue.close()


def test_config():
    """Test configuration system."""
    logger.info("TEST: Configuration")
    
    from config import Config, get_config, reset_config
    
    reset_config()
    config = get_config()
    
    assert config.instruments.perpetual_symbols == ["BTC-PERP", "ETH-PERP"]
    assert config.data_collection.primary_timeframe == "1h"
    assert config.coinbase.public_rate_limit == 10
    
    logger.info("✓ Configuration OK")
    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("MINIMAL PIPELINE TESTS (no external dependencies)")
    logger.info("=" * 60)
    
    results = {}
    
    # Sync tests
    results["Models"] = test_models()
    results["Validator"] = test_validator()
    results["Database"] = test_database()
    results["Config"] = test_config()
    
    # Async tests
    results["Queue"] = await test_queue()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL TESTS PASSED!")
        logger.info("\nTo test with live data, install dependencies:")
        logger.info("  pip install aiohttp ccxt pandas numpy")
        logger.info("  python test_pipeline.py")
    else:
        logger.info("SOME TESTS FAILED")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
