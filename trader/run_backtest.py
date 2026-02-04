#!/usr/bin/env python3
"""
Per-Coin Backtest Runner

Runs backtests for EACH coin individually to find optimal parameters per asset.
This is better than running all coins together because:
1. Each coin has different volatility/funding dynamics
2. Optimal thresholds vary by asset (BTC vs DOGE)
3. Uses full available history per coin (BTC 6yr, SOL 3.5yr)

Usage:
    python run_backtest_per_coin.py
    python run_backtest_per_coin.py --symbols BIP-20DEC30-CDE ETP-20DEC30-CDE
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from data_collection.storage import SQLiteDatabase
from backtesting.engine import Backtester, CostModel, PerformanceMetrics
from backtesting.strategies import (
    FundingArbitrageStrategy,
    FundingAwareMeanReversion,
    PureFundingCarryStrategy,
    CombinedFundingPriceStrategy,
    OIDivergenceStrategy,
    CombinedOIFundingStrategy,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/trading.db"
DEFAULT_FEATURES_DIR = "./data/features"


@dataclass
class CoinResult:
    """Results for a single coin."""
    symbol: str
    strategy_name: str
    params: Dict
    metrics: PerformanceMetrics
    data_start: datetime
    data_end: datetime
    num_bars: int


def load_features_from_csv(features_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load features for a single symbol."""
    # Try different filename patterns
    patterns = [
        f"{symbol.replace('-', '_')}_features.csv",
        f"{symbol}_features.csv",
    ]
    
    for pattern in patterns:
        filepath = features_dir / pattern
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
    
    return None


def load_ohlcv_from_db(db_path: str, symbol: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
    """Load OHLCV for a single symbol."""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    query = """
        SELECT event_time, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND timeframe = ?
        ORDER BY event_time ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe), parse_dates=['event_time'])
    conn.close()
    
    if df.empty:
        return None
    
    df.set_index('event_time', inplace=True)
    return df


def align_data(ohlcv: pd.DataFrame, features: pd.DataFrame) -> tuple:
    """Align OHLCV and features by timestamp."""
    common_idx = features.index.intersection(ohlcv.index)
    if len(common_idx) == 0:
        return None, None
    return ohlcv.loc[common_idx], features.loc[common_idx]


def check_features(features: pd.DataFrame) -> Dict[str, bool]:
    """Check which feature types are available."""
    cols = features.columns.tolist()
    return {
        'funding': any('funding' in c.lower() for c in cols),
        'oi': any('oi' in c.lower() or 'open_interest' in c.lower() for c in cols),
        'price': any('bb_' in c.lower() or 'rsi' in c.lower() for c in cols),
    }


def run_single_backtest(
    symbol: str,
    strategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> PerformanceMetrics:
    """Run backtest for single coin."""
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.5,  # Higher since single coin
        max_leverage=3.0,
    )
    
    _, metrics = backtester.run(
        strategy=strategy,
        ohlcv_data={symbol: ohlcv},
        features={symbol: features},
    )
    
    return metrics


def get_funding_arb_strategies(symbol: str) -> List[tuple]:
    """Get Funding Arbitrage strategies with various thresholds."""
    strategies = []
    
    # Different entry thresholds
    for entry_thresh in [1.2, 1.5, 1.8, 2.0, 2.5]:
        for use_confirm in [True, False]:
            name = f"FundingArb(z>{entry_thresh}"
            if use_confirm:
                name += ",BB)"
            else:
                name += ")"
            
            strategies.append((
                FundingArbitrageStrategy(
                    symbols=[symbol],
                    entry_threshold=entry_thresh,
                    exit_threshold=0.5,
                    min_hold_hours=24,
                    max_hold_hours=168,
                    position_size=0.3,
                    use_price_confirmation=use_confirm,
                    bb_confirmation_threshold=1.5,
                ),
                name,
                {'entry_threshold': entry_thresh, 'use_price_confirmation': use_confirm}
            ))
    
    return strategies


def get_oi_strategies(symbol: str) -> List[tuple]:
    """Get OI-based strategies."""
    strategies = []
    
    for div_thresh in [0.3, 0.5, 0.7]:
        for use_zscore in [True, False]:
            name = f"OI_Div(>{div_thresh}"
            if use_zscore:
                name += ",z-filter)"
            else:
                name += ")"
            
            strategies.append((
                OIDivergenceStrategy(
                    symbols=[symbol],
                    divergence_threshold=div_thresh,
                    use_oi_zscore_filter=use_zscore,
                    oi_zscore_threshold=1.5,
                    position_size=0.3,
                    min_hold_hours=12,
                    max_hold_hours=72,
                ),
                name,
                {'divergence_threshold': div_thresh, 'use_oi_zscore_filter': use_zscore}
            ))
    
    return strategies


def get_combined_strategies(symbol: str) -> List[tuple]:
    """Get combined strategies."""
    strategies = []
    
    for min_factors in [2, 3]:
        for funding_thresh in [1.0, 1.5, 2.0]:
            name = f"Combined({min_factors}factors,FR>{funding_thresh})"
            
            strategies.append((
                CombinedOIFundingStrategy(
                    symbols=[symbol],
                    oi_divergence_threshold=0.3,
                    funding_threshold=funding_thresh,
                    bb_threshold=1.5,
                    min_factors=min_factors,
                    position_size=0.3,
                    min_hold_hours=24,
                    max_hold_hours=120,
                ),
                name,
                {'min_factors': min_factors, 'funding_threshold': funding_thresh}
            ))
    
    return strategies


def run_coin_analysis(
    symbol: str,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> List[CoinResult]:
    """Run all strategy variations for a single coin."""
    
    results = []
    available = check_features(features)
    
    data_start = ohlcv.index.min()
    data_end = ohlcv.index.max()
    num_bars = len(ohlcv)
    
    strategies = []
    
    # Add strategies based on available features
    if available['funding']:
        strategies.extend(get_funding_arb_strategies(symbol))
    
    if available['oi']:
        strategies.extend(get_oi_strategies(symbol))
    
    if available['funding'] and available['oi']:
        strategies.extend(get_combined_strategies(symbol))
    
    # Run each strategy
    for strategy, name, params in strategies:
        try:
            metrics = run_single_backtest(symbol, strategy, ohlcv, features, cost_model)
            
            results.append(CoinResult(
                symbol=symbol,
                strategy_name=name,
                params=params,
                metrics=metrics,
                data_start=data_start,
                data_end=data_end,
                num_bars=num_bars,
            ))
        except Exception as e:
            logger.warning(f"  Error running {name}: {e}")
    
    return results


def print_coin_results(symbol: str, results: List[CoinResult]):
    """Print results for a single coin."""
    if not results:
        print(f"  No results for {symbol}")
        return
    
    # Sort by Sharpe
    results = sorted(results, key=lambda x: x.metrics.sharpe_ratio, reverse=True)
    
    print(f"\n  {'Strategy':<35} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print(f"  {'-'*80}")
    
    for r in results[:10]:  # Top 10
        m = r.metrics
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"  {r.strategy_name:<35} {m.total_return*100:>8.2f}% {sharpe_str:>8} "
              f"{m.max_drawdown*100:>7.2f}% {m.num_trades:>7} {m.win_rate*100:>5.1f}%")
    
    # Best strategy summary
    best = results[0]
    print(f"\n  ✅ BEST: {best.strategy_name}")
    print(f"     Params: {best.params}")
    print(f"     Sharpe: {best.metrics.sharpe_ratio:.2f}, Return: {best.metrics.total_return*100:.2f}%")
    print(f"     Trades: {best.metrics.num_trades}, Win Rate: {best.metrics.win_rate*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Per-coin backtest analysis")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--symbols", type=str, nargs="+", help="Specific symbols to test")
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    print("=" * 90)
    print("🎯 PER-COIN BACKTEST ANALYSIS")
    print("=" * 90)
    print("\nFinding optimal strategy parameters for each coin individually.\n")
    
    # Cost model
    cost_model = CostModel(
        maker_fee_bps=10,
        taker_fee_bps=10,
        base_slippage_bps=2,
    )
    
    # Find available symbols from feature files
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = []
        for f in features_dir.glob("*_features.csv"):
            symbol = f.stem.replace("_features", "").replace("_", "-")
            symbols.append(symbol)
    
    if not symbols:
        print("❌ No feature files found!")
        return
    
    print(f"Symbols to analyze: {symbols}\n")
    
    # Store all results
    all_results: Dict[str, List[CoinResult]] = {}
    best_per_coin: Dict[str, CoinResult] = {}
    
    # Process each coin
    for symbol in symbols:
        print("=" * 90)
        print(f"📊 {symbol}")
        print("=" * 90)
        
        # Load data
        features = load_features_from_csv(features_dir, symbol)
        if features is None:
            print(f"  ❌ No features found for {symbol}")
            continue
        
        ohlcv = load_ohlcv_from_db(args.db_path, symbol)
        if ohlcv is None:
            print(f"  ❌ No OHLCV data found for {symbol}")
            continue
        
        # Align
        ohlcv, features = align_data(ohlcv, features)
        if ohlcv is None:
            print(f"  ❌ No overlapping data for {symbol}")
            continue
        
        print(f"  Data: {ohlcv.index.min().date()} to {ohlcv.index.max().date()} ({len(ohlcv)} bars)")
        
        available = check_features(features)
        print(f"  Features: Funding={available['funding']}, OI={available['oi']}, Price={available['price']}")
        
        # Run analysis
        results = run_coin_analysis(symbol, ohlcv, features, cost_model)
        all_results[symbol] = results
        
        # Print results
        print_coin_results(symbol, results)
        
        # Track best
        if results:
            best = max(results, key=lambda x: x.metrics.sharpe_ratio)
            best_per_coin[symbol] = best
    
    # Final Summary
    print("\n" + "=" * 90)
    print("📈 FINAL SUMMARY - BEST STRATEGY PER COIN")
    print("=" * 90)
    
    print(f"\n{'Symbol':<20} {'Best Strategy':<30} {'Sharpe':>8} {'Return':>10} {'Trades':>7} {'Data Range':<25}")
    print("-" * 100)
    
    for symbol, best in sorted(best_per_coin.items()):
        m = best.metrics
        date_range = f"{best.data_start.date()} to {best.data_end.date()}"
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"{symbol:<20} {best.strategy_name:<30} {sharpe_str:>8} {m.total_return*100:>9.2f}% "
              f"{m.num_trades:>7} {date_range:<25}")
    
    # Recommendations
    print("\n" + "=" * 90)
    print("🎯 RECOMMENDATIONS")
    print("=" * 90)
    
    good_coins = [(s, r) for s, r in best_per_coin.items() 
                  if r.metrics.sharpe_ratio > 0.5 and r.metrics.num_trades >= 10]
    weak_coins = [(s, r) for s, r in best_per_coin.items() 
                  if 0 < r.metrics.sharpe_ratio <= 0.5]
    bad_coins = [(s, r) for s, r in best_per_coin.items() 
                 if r.metrics.sharpe_ratio <= 0]
    
    if good_coins:
        print("\n✅ TRADE THESE (Sharpe > 0.5, sufficient trades):")
        for symbol, r in good_coins:
            print(f"   {symbol}: {r.strategy_name} (Sharpe={r.metrics.sharpe_ratio:.2f})")
    
    if weak_coins:
        print("\n⚠️  PAPER TRADE FIRST (Weak edge):")
        for symbol, r in weak_coins:
            print(f"   {symbol}: {r.strategy_name} (Sharpe={r.metrics.sharpe_ratio:.2f})")
    
    if bad_coins:
        print("\n❌ SKIP THESE (No edge):")
        for symbol, r in bad_coins:
            print(f"   {symbol}: Best Sharpe={r.metrics.sharpe_ratio:.2f}")
    
    # Output optimal params
    print("\n" + "=" * 90)
    print("📋 OPTIMAL PARAMETERS FOR DEPLOYMENT")
    print("=" * 90)
    
    print("\nCopy this config for paper trading:\n")
    print("COIN_CONFIGS = {")
    for symbol, best in best_per_coin.items():
        if best.metrics.sharpe_ratio > 0:
            print(f"    '{symbol}': {{")
            print(f"        'strategy': '{best.strategy_name}',")
            print(f"        'params': {best.params},")
            print(f"        'expected_sharpe': {best.metrics.sharpe_ratio:.2f},")
            print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()