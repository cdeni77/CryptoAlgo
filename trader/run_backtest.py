#!/usr/bin/env python3
"""
Run Backtest v4 - ACTUAL Coinbase US Perps Fees (User Confirmed)

ACTUAL FEE STRUCTURE (per user):
- Trading fee: 0.1% per trade (on entry)
- Funding: Hourly, variable (typically 0.01-0.05% per hour when positive)
- No separate maker/taker distinction mentioned

This means:
- Round-trip cost: ~0.2% (0.1% entry + 0.1% exit)
- Plus funding costs if holding with positive funding rate

Run: python run_backtest_v4.py
"""

import sys
import warnings
import logging

from pathlib import Path
from datetime import datetime, timedelta

from data_collection.storage import SQLiteDatabase
from features.engineering import FeaturePipeline, FeatureConfig
from backtesting.engine import Backtester, CostModel
from backtesting.strategies import MeanReversionStrategy, MomentumReversalStrategy, CombinedStrategy
from backtesting.strategies_v2 import (
    LowFrequencyMeanReversion,
    WeeklyMomentumReversal,
    MultiFactorStrategy,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = "./data/trading.db"


def load_data(symbols: list, timeframe: str = "1h"):
    """Load OHLCV data from database."""
    db = SQLiteDatabase(DB_PATH)
    db.initialize()
    
    end = datetime.utcnow()
    start = end - timedelta(days=365)
    
    ohlcv_data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            ohlcv_data[symbol] = df
    
    db.close()
    return ohlcv_data


def compute_features(ohlcv_data: dict):
    """Compute features."""
    config = FeatureConfig(
        price_lookbacks=[1, 4, 12, 24, 48, 168],
        volume_lookbacks=[1, 4, 12, 24, 48],
        normalize_features=True,
    )
    
    pipeline = FeaturePipeline(config)
    return pipeline.compute_features(ohlcv_data)


def run_backtest(strategy, ohlcv_data, features, name: str, cost_model: CostModel):
    """Run backtest and return results."""
    
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.25,
        max_leverage=3.0,
    )
    
    portfolio, metrics = backtester.run(
        strategy=strategy,
        ohlcv_data=ohlcv_data,
        features=features,
    )
    
    return portfolio, metrics, name


def main():
    print("=" * 70)
    print("🎯 BACKTEST v4 - ACTUAL Coinbase US Perps Fees")
    print("=" * 70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  ACTUAL FEE STRUCTURE (Coinbase US Perpetual Futures)      │
    ├─────────────────────────────────────────────────────────────┤
    │  Trading Fee:   0.10% per trade                            │
    │  Round-trip:    ~0.20% (entry + exit)                      │
    │  Funding:       Variable, hourly (~0.01-0.05%/hr typical)  │
    │  Slippage:      ~0.02% estimate                            │
    │                                                             │
    │  Total round-trip: ~0.22%                                  │
    │                                                             │
    │  This is MUCH better than original 0.6% assumption,        │
    │  but still significant for high-frequency strategies.      │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    # Load data
    print("📊 Loading data...")
    symbols = ['BTC-PERP', 'ETH-PERP', 'SOL-PERP', 'DOGE-PERP', 'XRP-PERP']
    ohlcv_data = load_data(symbols)
    
    if not ohlcv_data:
        print("No data! Run backfill first.")
        return
    
    print(f"Loaded {len(ohlcv_data)} symbols, {len(list(ohlcv_data.values())[0])} bars each")
    
    # Compute features
    print("Computing features...")
    features = compute_features(ohlcv_data)
    
    # ACTUAL cost model for Coinbase US Perpetual Futures
    # User confirmed: 0.1% per trade
    cost_model = CostModel(
        maker_fee_bps=10,     # 0.10% - same for all orders on US perps
        taker_fee_bps=10,     # 0.10% - same for all orders on US perps  
        base_slippage_bps=2,  # ~0.02% slippage estimate
        volatility_slippage_multiplier=1.5,
        size_impact_coefficient=0.02,
    )
    
    print("\n" + "=" * 70)
    print("RUNNING STRATEGIES WITH ACTUAL 0.1% FEES")
    print("=" * 70)
    
    all_strategies = [
        # Original strategies
        (MeanReversionStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=1.0,
            exit_threshold=0.0,
            position_size=0.1,
        ), "MeanRev BB (entry=1.0)"),
        
        (MeanReversionStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=1.5,
            exit_threshold=0.2,
            position_size=0.15,
        ), "MeanRev BB (entry=1.5)"),
        
        (MeanReversionStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=2.0,
            exit_threshold=0.3,
            position_size=0.15,
        ), "MeanRev BB (entry=2.0)"),
        
        (MeanReversionStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=1.0,
            exit_threshold=0.0,
            position_size=0.1,
            use_rsi_filter=True,
            rsi_oversold=35,
            rsi_overbought=65,
        ), "MeanRev BB+RSI Filter"),
        
        (MomentumReversalStrategy(
            symbols=list(ohlcv_data.keys()),
            rsi_oversold=30,
            rsi_overbought=70,
            return_threshold=0.03,
            position_size=0.1,
        ), "Momentum Reversal"),
        
        (CombinedStrategy(
            symbols=list(ohlcv_data.keys()),
            bb_threshold=1.0,
            rsi_oversold=35,
            rsi_overbought=65,
            position_size=0.1,
        ), "Combined BB+RSI"),
        
        # Low frequency strategies (better for 0.1% fees)
        (LowFrequencyMeanReversion(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=1.5,
            exit_threshold=0.2,
            min_hold_hours=24,
            position_size=0.15,
        ), "LowFreq MeanRev (24h hold)"),
        
        (LowFrequencyMeanReversion(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=2.0,
            exit_threshold=0.3,
            min_hold_hours=48,
            position_size=0.15,
        ), "LowFreq MeanRev (48h hold)"),
        
        (WeeklyMomentumReversal(
            symbols=list(ohlcv_data.keys()),
            return_threshold=0.08,
            position_size=0.15,
            hold_days=5,
        ), "Weekly Reversal (8%, 5d)"),
        
        (WeeklyMomentumReversal(
            symbols=list(ohlcv_data.keys()),
            return_threshold=0.10,
            position_size=0.15,
            hold_days=7,
        ), "Weekly Reversal (10%, 7d)"),
        
        (MultiFactorStrategy(
            symbols=list(ohlcv_data.keys()),
            bb_threshold=1.5,
            rsi_oversold=30,
            rsi_overbought=70,
            position_size=0.15,
            min_hold_hours=48,
        ), "MultiFactor (48h hold)"),
    ]
    
    results = []
    
    for strategy, name in all_strategies:
        _, metrics, _ = run_backtest(strategy, ohlcv_data, features, name, cost_model)
        results.append((name, metrics))
        
        sharpe_str = f"{metrics.sharpe_ratio:>6.2f}" if abs(metrics.sharpe_ratio) < 100 else "  N/A"
        print(f"  {name:<30} Return: {metrics.total_return*100:>7.2f}% | "
              f"Sharpe: {sharpe_str} | Trades: {metrics.num_trades:>4}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("📊 STRATEGY COMPARISON (0.1% Trading Fee)")
    print("=" * 70)
    
    print(f"\n{'Strategy':<32} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print("-" * 75)
    
    for name, m in sorted(results, key=lambda x: x[1].sharpe_ratio, reverse=True):
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"{name:<32} {m.total_return*100:>8.2f}% {sharpe_str:>8} "
              f"{m.max_drawdown*100:>7.2f}% {m.num_trades:>7} {m.win_rate*100:>5.1f}%")
    
    # Cost analysis
    print("\n" + "=" * 70)
    print("💰 COST ANALYSIS (0.1% per trade = 0.22% round-trip with slippage)")
    print("=" * 70)
    
    print(f"\n{'Strategy':<32} {'Trades':>7} {'Cost Drag':>10} {'Gross Est':>10}")
    print("-" * 65)
    
    for name, m in results:
        # Round-trip cost: 0.1% + 0.1% + 0.02% slippage = 0.22%
        cost_drag_pct = m.num_trades * 0.0022 * 100
        gross_est = m.total_return * 100 + cost_drag_pct
        print(f"{name:<32} {m.num_trades:>7} {cost_drag_pct:>9.2f}% {gross_est:>9.2f}%")
    
    # Profitability analysis
    profitable = [(n, m) for n, m in results if m.total_return > 0]
    positive_sharpe = [(n, m) for n, m in results if m.sharpe_ratio > 0]
    
    print("\n" + "=" * 70)
    print("📈 PROFITABILITY SUMMARY")
    print("=" * 70)
    
    print(f"\n   Profitable strategies: {len(profitable)}/{len(results)}")
    print(f"   Positive Sharpe:       {len(positive_sharpe)}/{len(results)}")
    
    if profitable:
        print(f"\n   ✅ Profitable strategies:")
        for name, m in sorted(profitable, key=lambda x: x[1].total_return, reverse=True):
            print(f"      - {name}: {m.total_return*100:+.2f}% (Sharpe: {m.sharpe_ratio:.2f})")
    
    # Break-even analysis
    print("\n" + "=" * 70)
    print("📐 BREAK-EVEN ANALYSIS")
    print("=" * 70)
    
    print("""
    With 0.22% round-trip costs, you need:
    
    ┌─────────────────────────────────────────────────────────────┐
    │  Trades/Month  │  Monthly Cost Drag  │  Required Edge/Trade │
    ├────────────────┼─────────────────────┼──────────────────────┤
    │       10       │        2.2%         │       > 0.22%        │
    │       20       │        4.4%         │       > 0.22%        │
    │       50       │       11.0%         │       > 0.22%        │
    │      100       │       22.0%         │       > 0.22%        │
    └─────────────────────────────────────────────────────────────┘
    
    Key insight: Each trade needs to generate > 0.22% profit to break even.
    With IC (information coefficient) of ~0.06, expected return per trade
    is roughly 0.03-0.10%. This is LESS than the 0.22% cost.
    
    Solutions:
    1. Trade LESS frequently (only highest-conviction signals)
    2. Hold LONGER (let winners run to overcome fixed costs)
    3. Use stronger signals (combine multiple factors)
    4. Accept that this fee structure limits viable strategies
    """)
    
    # Funding rate consideration
    print("\n" + "=" * 70)
    print("⏰ FUNDING RATE CONSIDERATIONS")
    print("=" * 70)
    
    print("""
    Hourly funding rates add additional costs/benefits:
    
    - Positive funding (longs pay shorts): ~0.01-0.05% per hour
    - 24-hour cost if funding is 0.03%: 0.72% (!!)
    - 7-day cost if funding is 0.03%: 5.04% (!!)
    
    This means:
    - SHORT-TERM trades (< 24h): Funding is minor (~0.2-0.7%)
    - LONG-TERM trades (> 7d): Funding can DOMINATE returns
    
    Strategy implications:
    - Mean reversion (short holds): Funding is manageable
    - Trend following (long holds): Must account for funding
    - Consider shorting when funding is highly positive (funding arb)
    """)
    
    # Recommendations
    print("\n" + "=" * 70)
    print("📋 RECOMMENDATIONS")
    print("=" * 70)
    
    if profitable:
        best = max(profitable, key=lambda x: x[1].sharpe_ratio)
        print(f"""
    ✅ Some strategies show profitability with 0.1% fees!
    
    Best strategy: {best[0]}
    - Return: {best[1].total_return*100:.2f}%
    - Sharpe: {best[1].sharpe_ratio:.2f}
    - Trades: {best[1].num_trades}
    
    Next steps:
    1. Run walk-forward validation to confirm robustness
    2. Paper trade for 2-4 weeks before going live
    3. Start with 25-50% of backtest position sizes
    4. Monitor funding rates closely for held positions
        """)
    else:
        print("""
    ⚠️  No strategy was profitable in this 90-day period.
    
    At 0.1% per trade (0.22% round-trip), the fee structure is
    MUCH more manageable than 0.6%, but still challenging.
    
    The issue is likely:
    1. 90 days of data is insufficient (need 1+ year)
    2. Market regime was unfavorable for mean reversion
    3. Signal strength (IC ~0.06) is borderline for these costs
    
    Recommended actions:
    1. Backfill 1 year of data: python run_pipeline.py --backfill-days 365
    2. Add funding rate data to features (can improve signals)
    3. Consider a funding rate arbitrage strategy
    4. Focus on highest-conviction signals only
        """)


if __name__ == "__main__":
    main()