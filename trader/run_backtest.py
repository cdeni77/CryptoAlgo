#!/usr/bin/env python3
"""
Phase 1 Backtest - Funding Rate Strategies

This script runs backtests with the new funding-aware strategies:
1. FundingArbitrageStrategy - Pure funding arbitrage
2. FundingAwareMeanReversion - BB + funding filter
3. CombinedFundingPriceStrategy - Multi-factor

Prerequisites:
1. Run backfill_funding.py to get funding data
2. Have OHLCV data in database

Usage:
    python run_backtest_phase1.py
"""

import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.storage import SQLiteDatabase
from features.engineering import FeaturePipeline, FeatureConfig
from backtesting.engine import Backtester, CostModel
from backtesting.strategies import (
    FundingArbitrageStrategy,
    FundingAwareMeanReversion,
    PureFundingCarryStrategy,
    CombinedFundingPriceStrategy,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "./data/trading.db"


def load_data(db: SQLiteDatabase, symbols: list, timeframe: str = "1h"):
    """Load OHLCV data from database."""
    end = datetime.utcnow()
    start = end - timedelta(days=1000)
    
    ohlcv_data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            ohlcv_data[symbol] = df
            logger.info(f"  Loaded {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
    
    return ohlcv_data


def load_funding_data(db: SQLiteDatabase, symbols: list):
    """Load funding rate data from database."""
    end = datetime.utcnow()
    start = end - timedelta(days=1000)
    
    funding_data = {}
    for symbol in symbols:
        df = db.get_funding_rates(symbol, start, end)
        if not df.empty:
            funding_data[symbol] = df
            logger.info(f"  Loaded {symbol} funding: {len(df)} records ({df.index.min().date()} to {df.index.max().date()})")
    
    return funding_data


def compute_features(ohlcv_data: dict, funding_data: dict):
    """Compute features including funding."""
    config = FeatureConfig(
        price_lookbacks=[1, 4, 12, 24, 48, 168],
        volume_lookbacks=[1, 4, 12, 24, 48],
        normalize_features=True,
        compute_funding=True,  # Enable funding features
    )
    
    pipeline = FeaturePipeline(config)
    return pipeline.compute_features(ohlcv_data, funding_data=funding_data)


def analyze_funding_features(features: dict):
    """Analyze funding feature statistics."""
    print("\n" + "=" * 70)
    print("📊 FUNDING FEATURE ANALYSIS")
    print("=" * 70)
    
    for symbol, feat_df in features.items():
        funding_cols = [c for c in feat_df.columns if 'funding' in c.lower()]
        
        if not funding_cols:
            print(f"\n{symbol}: No funding features (need to run backfill_funding.py)")
            continue
        
        print(f"\n{symbol}:")
        print(f"  Funding features: {len(funding_cols)}")
        
        # Key funding stats
        if 'funding_rate_zscore' in feat_df.columns:
            zscore = feat_df['funding_rate_zscore'].dropna()
            if len(zscore) > 0:
                print(f"\n  funding_rate_zscore:")
                print(f"    Mean: {zscore.mean():.4f}")
                print(f"    Std:  {zscore.std():.4f}")
                print(f"    Min:  {zscore.min():.4f}")
                print(f"    Max:  {zscore.max():.4f}")
                print(f"    % > 2.0 (extreme positive): {(zscore > 2.0).mean()*100:.2f}%")
                print(f"    % < -2.0 (extreme negative): {(zscore < -2.0).mean()*100:.2f}%")
        
        if 'funding_rate_bps' in feat_df.columns:
            rate = feat_df['funding_rate_bps'].dropna()
            if len(rate) > 0:
                print(f"\n  funding_rate_bps:")
                print(f"    Mean: {rate.mean():.4f} bps/8h")
                print(f"    Annual equiv: {rate.mean() * 3 * 365:.2f} bps/year")
        
        if 'cumulative_funding_24h' in feat_df.columns:
            cum = feat_df['cumulative_funding_24h'].dropna() * 10000
            if len(cum) > 0:
                print(f"\n  cumulative_funding_24h (bps):")
                print(f"    Mean: {cum.mean():.4f}")
                print(f"    Max:  {cum.max():.4f}")


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
    print("🎯 PHASE 1 BACKTEST - FUNDING RATE STRATEGIES")
    print("=" * 70)
    
    # Initialize database
    db = SQLiteDatabase(DB_PATH)
    db.initialize()
    
    # Check for funding data
    funding_stats = db.get_funding_stats()
    
    if not funding_stats:
        print("""
    ⚠️  NO FUNDING DATA FOUND!
    
    Please run the funding backfill first:
    
        python backfill_funding.py --days 365
    
    This will fetch historical funding rates from Binance/OKX.
        """)
        db.close()
        return
    
    print("\n📊 Funding Data Available:")
    for symbol, stats in funding_stats.items():
        print(f"  {symbol}: {stats['count']} records, avg {stats['avg_rate_bps']:.4f} bps/8h")
    
    # Load data
    print("\n📊 Loading OHLCV data...")
    symbols = list(funding_stats.keys())
    ohlcv_data = load_data(db, symbols)
    
    if not ohlcv_data:
        print("No OHLCV data! Run your pipeline backfill first.")
        db.close()
        return
    
    print(f"\nLoaded {len(ohlcv_data)} symbols")
    
    # Load funding data
    print("\n📊 Loading funding data...")
    funding_data = load_funding_data(db, symbols)
    
    # Compute features
    print("\n🔧 Computing features (including funding)...")
    features = compute_features(ohlcv_data, funding_data)
    
    # Analyze funding features
    analyze_funding_features(features)
    
    # Cost model (Coinbase US Perps: 0.1% per trade)
    cost_model = CostModel(
        maker_fee_bps=10,
        taker_fee_bps=10,
        base_slippage_bps=2,
        volatility_slippage_multiplier=1.5,
        size_impact_coefficient=0.02,
    )
    
    print("\n" + "=" * 70)
    print("📈 RUNNING FUNDING STRATEGIES")
    print("=" * 70)
    
    # Define strategies
    strategies = [
        # Pure Funding Arbitrage (primary Phase 1 strategy)
        (FundingArbitrageStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=2.0,
            exit_threshold=0.5,
            min_hold_hours=24,
            max_hold_hours=168,
            position_size=0.15,
            use_price_confirmation=True,
            bb_confirmation_threshold=1.5,
        ), "Funding Arb (z>2, BB confirm)"),
        
        (FundingArbitrageStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=1.5,
            exit_threshold=0.3,
            min_hold_hours=24,
            max_hold_hours=168,
            position_size=0.15,
            use_price_confirmation=False,  # No price confirmation
        ), "Funding Arb (z>1.5, no confirm)"),
        
        (FundingArbitrageStrategy(
            symbols=list(ohlcv_data.keys()),
            entry_threshold=2.5,
            exit_threshold=0.5,
            min_hold_hours=48,
            max_hold_hours=336,  # 2 weeks
            position_size=0.2,
            use_price_confirmation=True,
        ), "Funding Arb (z>2.5, 48h hold)"),
        
        # Funding-Aware Mean Reversion
        (FundingAwareMeanReversion(
            symbols=list(ohlcv_data.keys()),
            bb_entry_threshold=1.5,
            bb_exit_threshold=0.2,
            max_funding_zscore=1.5,
            min_funding_zscore=-1.5,
            position_size=0.15,
            min_hold_hours=24,
        ), "Funding-Aware MeanRev"),
        
        (FundingAwareMeanReversion(
            symbols=list(ohlcv_data.keys()),
            bb_entry_threshold=2.0,
            bb_exit_threshold=0.3,
            max_funding_zscore=1.0,  # More restrictive
            min_funding_zscore=-1.0,
            position_size=0.15,
            min_hold_hours=48,
        ), "Funding-Aware MeanRev (strict)"),
        
        # Pure Carry
        (PureFundingCarryStrategy(
            symbols=list(ohlcv_data.keys()),
            cumulative_threshold_bps=10.0,
            position_size=0.1,
            hold_hours=72,
        ), "Pure Carry (10bps, 72h)"),
        
        (PureFundingCarryStrategy(
            symbols=list(ohlcv_data.keys()),
            cumulative_threshold_bps=15.0,
            position_size=0.15,
            hold_hours=120,
        ), "Pure Carry (15bps, 120h)"),
        
        # Combined Multi-Factor
        (CombinedFundingPriceStrategy(
            symbols=list(ohlcv_data.keys()),
            funding_threshold=1.5,
            bb_threshold=1.5,
            rsi_oversold=35,
            rsi_overbought=65,
            require_all_signals=False,  # 2 of 3
            position_size=0.2,
            min_hold_hours=48,
        ), "Combined (2 of 3 factors)"),
        
        (CombinedFundingPriceStrategy(
            symbols=list(ohlcv_data.keys()),
            funding_threshold=1.5,
            bb_threshold=1.5,
            rsi_oversold=30,
            rsi_overbought=70,
            require_all_signals=True,  # All 3
            position_size=0.25,
            min_hold_hours=72,
        ), "Combined (all 3 factors)"),
    ]
    
    results = []
    
    for strategy, name in strategies:
        try:
            _, metrics, _ = run_backtest(strategy, ohlcv_data, features, name, cost_model)
            results.append((name, metrics))
            
            sharpe_str = f"{metrics.sharpe_ratio:>6.2f}" if abs(metrics.sharpe_ratio) < 100 else "  N/A"
            print(f"  {name:<35} Return: {metrics.total_return*100:>7.2f}% | "
                  f"Sharpe: {sharpe_str} | Trades: {metrics.num_trades:>4}")
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 STRATEGY COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<37} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print("-" * 80)
    
    for name, m in sorted(results, key=lambda x: x[1].sharpe_ratio, reverse=True):
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"{name:<37} {m.total_return*100:>8.2f}% {sharpe_str:>8} "
              f"{m.max_drawdown*100:>7.2f}% {m.num_trades:>7} {m.win_rate*100:>5.1f}%")
    
    # Best strategy
    if results:
        profitable = [(n, m) for n, m in results if m.total_return > 0]
        positive_sharpe = [(n, m) for n, m in results if m.sharpe_ratio > 0]
        
        print("\n" + "=" * 70)
        print("📈 SUMMARY")
        print("=" * 70)
        
        print(f"\n   Profitable strategies: {len(profitable)}/{len(results)}")
        print(f"   Positive Sharpe:       {len(positive_sharpe)}/{len(results)}")
        
        if profitable:
            best = max(profitable, key=lambda x: x[1].sharpe_ratio)
            print(f"\n   ✅ Best strategy: {best[0]}")
            print(f"      - Return: {best[1].total_return*100:+.2f}%")
            print(f"      - Sharpe: {best[1].sharpe_ratio:.2f}")
            print(f"      - Max DD: {best[1].max_drawdown*100:.2f}%")
            print(f"      - Trades: {best[1].num_trades}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("📋 PHASE 1 ANALYSIS")
    print("=" * 70)
    
    print("""
    Key insights from funding-based strategies:
    
    1. FUNDING ARBITRAGE: Captures carry + mean reversion
       - Best when funding is extremely positive (z > 2) -> SHORT
       - Collects funding from longs + price mean reversion
    
    2. FUNDING-AWARE MEAN REVERSION: Filters bad trades
       - Don't go LONG when funding is extremely positive
       - Don't go SHORT when funding is extremely negative
       - Prevents fighting market momentum
    
    3. COMBINED STRATEGIES: Highest conviction
       - Require multiple signals to align
       - Lower frequency, higher quality trades
    
    NEXT STEPS:
    - If strategies are profitable: Move to walk-forward validation
    - If not profitable: Check funding data quality, try different thresholds
    - Consider: Funding data from CCXT may not perfectly match Coinbase
    """)
    
    db.close()


if __name__ == "__main__":
    main()