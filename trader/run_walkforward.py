#!/usr/bin/env python3
"""
Walk-Forward Validation for Winning Strategies

This script performs proper out-of-sample testing:
1. Split data chronologically (no peeking at future)
2. Train/optimize on first portion
3. Test on held-out future data
4. Roll forward and repeat

This tells us if backtest results are real or just overfitting.

Usage:
    python run_walkforward.py
"""

import sys
import warnings
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from backtesting.engine import Backtester, CostModel, PerformanceMetrics
from backtesting.strategies import (
    FundingArbitrageStrategy,
    CombinedOIFundingStrategy,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"


@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class WalkForwardResult:
    """Results from one walk-forward window."""
    window: WalkForwardWindow
    strategy_name: str
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    best_params: Dict


def load_features_from_csv(features_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load pre-computed features from CSV files."""
    features = {}
    
    for csv_file in features_dir.glob("*_features.csv"):
        filename = csv_file.stem
        symbol = filename.replace("_features", "").replace("_", "-")
        
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        features[symbol] = df
        logger.info(f"  Loaded {symbol}: {len(df)} rows")
    
    return features


def load_ohlcv_from_db(symbols: List[str], db_path: str) -> Dict[str, pd.DataFrame]:
    """Load OHLCV data from SQLite database."""
    import sqlite3
    
    ohlcv_data = {}
    conn = sqlite3.connect(db_path)
    
    for symbol in symbols:
        query = """
            SELECT event_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = '1h'
            ORDER BY event_time ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol,), parse_dates=['event_time'])
        
        if not df.empty:
            df.set_index('event_time', inplace=True)
            ohlcv_data[symbol] = df
    
    conn.close()
    return ohlcv_data


def generate_windows(
    start_date: datetime,
    end_date: datetime,
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
) -> List[WalkForwardWindow]:
    """
    Generate walk-forward windows.
    
    Default: 180 days training, 60 days testing, 60 day steps
    This gives us non-overlapping test periods.
    """
    windows = []
    window_id = 0
    current_start = start_date
    
    while True:
        train_end = current_start + timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=test_days)
        
        if test_end > end_date:
            break
        
        windows.append(WalkForwardWindow(
            train_start=current_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            window_id=window_id,
        ))
        
        window_id += 1
        current_start += timedelta(days=step_days)
    
    return windows


def filter_data_by_time(
    ohlcv_data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    start: datetime,
    end: datetime,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Filter data to a specific time range."""
    filtered_ohlcv = {}
    filtered_features = {}
    
    for symbol in ohlcv_data.keys():
        if symbol not in features:
            continue
        
        ohlcv_mask = (ohlcv_data[symbol].index >= start) & (ohlcv_data[symbol].index < end)
        feat_mask = (features[symbol].index >= start) & (features[symbol].index < end)
        
        filtered_ohlcv[symbol] = ohlcv_data[symbol][ohlcv_mask].copy()
        filtered_features[symbol] = features[symbol][feat_mask].copy()
    
    return filtered_ohlcv, filtered_features


def run_strategy_backtest(
    strategy,
    ohlcv_data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    cost_model: CostModel,
) -> PerformanceMetrics:
    """Run backtest and return metrics."""
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.25,
        max_leverage=3.0,
    )
    
    _, metrics = backtester.run(
        strategy=strategy,
        ohlcv_data=ohlcv_data,
        features=features,
    )
    
    return metrics


def create_funding_arb_strategy(symbols: List[str], params: Dict):
    """Create FundingArbitrageStrategy with given params."""
    return FundingArbitrageStrategy(
        symbols=symbols,
        entry_threshold=params.get('entry_threshold', 2.0),
        exit_threshold=params.get('exit_threshold', 0.5),
        min_hold_hours=params.get('min_hold_hours', 24),
        max_hold_hours=params.get('max_hold_hours', 168),
        position_size=params.get('position_size', 0.15),
        use_price_confirmation=params.get('use_price_confirmation', True),
        bb_confirmation_threshold=params.get('bb_confirmation_threshold', 1.5),
    )


def create_combined_strategy(symbols: List[str], params: Dict):
    """Create CombinedOIFundingStrategy with given params."""
    return CombinedOIFundingStrategy(
        symbols=symbols,
        oi_divergence_threshold=params.get('oi_divergence_threshold', 0.3),
        funding_threshold=params.get('funding_threshold', 1.5),
        bb_threshold=params.get('bb_threshold', 1.5),
        min_factors=params.get('min_factors', 3),
        position_size=params.get('position_size', 0.2),
        min_hold_hours=params.get('min_hold_hours', 48),
    )


# Parameter grids to test
FUNDING_ARB_PARAMS = [
    {'entry_threshold': 1.8, 'exit_threshold': 0.5, 'use_price_confirmation': True, 'bb_confirmation_threshold': 1.5},
    {'entry_threshold': 2.0, 'exit_threshold': 0.5, 'use_price_confirmation': True, 'bb_confirmation_threshold': 1.5},
    {'entry_threshold': 2.2, 'exit_threshold': 0.5, 'use_price_confirmation': True, 'bb_confirmation_threshold': 1.5},
    {'entry_threshold': 2.5, 'exit_threshold': 0.5, 'use_price_confirmation': True, 'bb_confirmation_threshold': 1.5},
    {'entry_threshold': 2.0, 'exit_threshold': 0.3, 'use_price_confirmation': True, 'bb_confirmation_threshold': 1.5},
    {'entry_threshold': 2.0, 'exit_threshold': 0.5, 'use_price_confirmation': True, 'bb_confirmation_threshold': 2.0},
]

COMBINED_PARAMS = [
    {'oi_divergence_threshold': 0.3, 'funding_threshold': 1.5, 'bb_threshold': 1.5, 'min_factors': 3},
    {'oi_divergence_threshold': 0.4, 'funding_threshold': 1.5, 'bb_threshold': 1.5, 'min_factors': 3},
    {'oi_divergence_threshold': 0.3, 'funding_threshold': 2.0, 'bb_threshold': 1.5, 'min_factors': 3},
    {'oi_divergence_threshold': 0.3, 'funding_threshold': 1.5, 'bb_threshold': 2.0, 'min_factors': 3},
]


def optimize_on_train(
    strategy_type: str,
    param_grid: List[Dict],
    symbols: List[str],
    train_ohlcv: Dict[str, pd.DataFrame],
    train_features: Dict[str, pd.DataFrame],
    cost_model: CostModel,
) -> Tuple[Dict, PerformanceMetrics]:
    """Find best parameters on training data."""
    best_params = None
    best_sharpe = -np.inf
    best_metrics = None
    
    for params in param_grid:
        if strategy_type == 'funding_arb':
            strategy = create_funding_arb_strategy(symbols, params)
        else:
            strategy = create_combined_strategy(symbols, params)
        
        metrics = run_strategy_backtest(strategy, train_ohlcv, train_features, cost_model)
        
        # Use Sharpe as optimization target (could also use Calmar or other metrics)
        if metrics.sharpe_ratio > best_sharpe:
            best_sharpe = metrics.sharpe_ratio
            best_params = params
            best_metrics = metrics
    
    return best_params, best_metrics


def run_walk_forward(
    strategy_type: str,
    strategy_name: str,
    param_grid: List[Dict],
    windows: List[WalkForwardWindow],
    ohlcv_data: Dict[str, pd.DataFrame],
    features: Dict[str, pd.DataFrame],
    cost_model: CostModel,
) -> List[WalkForwardResult]:
    """Run walk-forward validation for a strategy."""
    results = []
    symbols = list(ohlcv_data.keys())
    
    for window in windows:
        logger.info(f"\n  Window {window.window_id + 1}: "
                   f"Train {window.train_start.date()} to {window.train_end.date()}, "
                   f"Test {window.test_start.date()} to {window.test_end.date()}")
        
        # Split data
        train_ohlcv, train_features = filter_data_by_time(
            ohlcv_data, features, window.train_start, window.train_end
        )
        test_ohlcv, test_features = filter_data_by_time(
            ohlcv_data, features, window.test_start, window.test_end
        )
        
        # Check we have enough data
        min_train_bars = min(len(df) for df in train_ohlcv.values()) if train_ohlcv else 0
        min_test_bars = min(len(df) for df in test_ohlcv.values()) if test_ohlcv else 0
        
        if min_train_bars < 100 or min_test_bars < 50:
            logger.warning(f"    Skipping - insufficient data (train={min_train_bars}, test={min_test_bars})")
            continue
        
        # Optimize on training data
        best_params, train_metrics = optimize_on_train(
            strategy_type, param_grid, symbols,
            train_ohlcv, train_features, cost_model
        )
        
        if best_params is None:
            logger.warning(f"    Skipping - no valid params found")
            continue
        
        # Test on out-of-sample data with best params
        if strategy_type == 'funding_arb':
            test_strategy = create_funding_arb_strategy(symbols, best_params)
        else:
            test_strategy = create_combined_strategy(symbols, best_params)
        
        test_metrics = run_strategy_backtest(test_strategy, test_ohlcv, test_features, cost_model)
        
        results.append(WalkForwardResult(
            window=window,
            strategy_name=strategy_name,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            best_params=best_params,
        ))
        
        train_sharpe = train_metrics.sharpe_ratio if train_metrics else 0
        test_sharpe = test_metrics.sharpe_ratio if test_metrics else 0
        test_return = test_metrics.total_return * 100 if test_metrics else 0
        
        logger.info(f"    Train Sharpe: {train_sharpe:>6.2f} | "
                   f"Test Sharpe: {test_sharpe:>6.2f} | "
                   f"Test Return: {test_return:>7.2f}% | "
                   f"Params: entry={best_params.get('entry_threshold', best_params.get('oi_divergence_threshold', 'N/A'))}")
    
    return results


def analyze_results(results: List[WalkForwardResult], strategy_name: str):
    """Analyze walk-forward results."""
    if not results:
        print(f"\n  {strategy_name}: No valid results")
        return None
    
    train_sharpes = [r.train_metrics.sharpe_ratio for r in results]
    test_sharpes = [r.test_metrics.sharpe_ratio for r in results]
    train_returns = [r.train_metrics.total_return for r in results]
    test_returns = [r.test_metrics.total_return for r in results]
    
    # Aggregate test performance (multiply returns)
    cumulative_test_return = np.prod([1 + r for r in test_returns]) - 1
    
    # Sharpe decay (how much worse is OOS vs IS)
    avg_train_sharpe = np.mean(train_sharpes)
    avg_test_sharpe = np.mean(test_sharpes)
    sharpe_decay = avg_train_sharpe - avg_test_sharpe
    
    # Win rate (% of windows with positive OOS return)
    pct_profitable = sum(1 for r in test_returns if r > 0) / len(test_returns) * 100
    pct_positive_sharpe = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100
    
    stats = {
        'strategy': strategy_name,
        'num_windows': len(results),
        'avg_train_sharpe': avg_train_sharpe,
        'avg_test_sharpe': avg_test_sharpe,
        'sharpe_decay': sharpe_decay,
        'cumulative_test_return': cumulative_test_return,
        'avg_test_return': np.mean(test_returns),
        'test_return_std': np.std(test_returns),
        'pct_profitable_windows': pct_profitable,
        'pct_positive_sharpe': pct_positive_sharpe,
        'test_sharpes': test_sharpes,
        'test_returns': test_returns,
    }
    
    return stats


def print_summary(all_stats: List[Dict]):
    """Print formatted summary."""
    print("\n" + "=" * 85)
    print("📊 WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 85)
    
    print(f"\n{'Strategy':<35} {'Windows':>8} {'Train SR':>10} {'Test SR':>10} {'Decay':>8} {'OOS Ret':>10} {'Win%':>7}")
    print("-" * 85)
    
    for stats in all_stats:
        if stats is None:
            continue
        print(f"{stats['strategy']:<35} "
              f"{stats['num_windows']:>8} "
              f"{stats['avg_train_sharpe']:>10.2f} "
              f"{stats['avg_test_sharpe']:>10.2f} "
              f"{stats['sharpe_decay']:>8.2f} "
              f"{stats['cumulative_test_return']*100:>9.2f}% "
              f"{stats['pct_profitable_windows']:>6.1f}%")
    
    print("\n" + "-" * 85)
    print("Key metrics:")
    print("  - Train SR: Average Sharpe ratio on training data (in-sample)")
    print("  - Test SR: Average Sharpe ratio on test data (out-of-sample)")
    print("  - Decay: Train SR - Test SR (lower is better, <0.5 is good)")
    print("  - OOS Ret: Cumulative return across all test windows")
    print("  - Win%: Percentage of test windows with positive return")


def main():
    print("=" * 85)
    print("🔄 WALK-FORWARD VALIDATION")
    print("=" * 85)
    print("\nThis validates if backtest results hold out-of-sample.")
    print("We train on past data, then test on future data we haven't seen.\n")
    
    # Load data
    print("📊 Loading data...")
    features = load_features_from_csv(FEATURES_DIR)
    
    if not features:
        print("❌ No feature files found!")
        return
    
    symbols = list(features.keys())
    ohlcv_data = load_ohlcv_from_db(symbols, DB_PATH)
    
    # Align
    for symbol in list(features.keys()):
        if symbol in ohlcv_data:
            common_idx = features[symbol].index.intersection(ohlcv_data[symbol].index)
            features[symbol] = features[symbol].loc[common_idx]
            ohlcv_data[symbol] = ohlcv_data[symbol].loc[common_idx]
    
    # Get date range
    all_dates = []
    for df in features.values():
        all_dates.extend(df.index.tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    total_days = (end_date - start_date).days
    
    print(f"\n📅 Data range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    print(f"📈 Symbols: {len(symbols)}")
    
    # Generate windows
    # With ~365 days of data, use:
    # - 180 days train (6 months)
    # - 60 days test (2 months)
    # - 60 days step (non-overlapping test periods)
    # This gives us about 3 windows
    
    windows = generate_windows(
        start_date=start_date,
        end_date=end_date,
        train_days=180,
        test_days=60,
        step_days=60,
    )
    
    print(f"🔄 Generated {len(windows)} walk-forward windows")
    
    if len(windows) < 2:
        print("\n⚠️  Not enough data for meaningful walk-forward validation.")
        print("    Need at least 240 days (180 train + 60 test) for 1 window.")
        print("    Current data: {total_days} days")
        
        # Try with smaller windows
        print("\n    Trying with smaller windows (90 train, 30 test)...")
        windows = generate_windows(
            start_date=start_date,
            end_date=end_date,
            train_days=90,
            test_days=30,
            step_days=30,
        )
        print(f"    Generated {len(windows)} windows with smaller periods")
    
    if len(windows) < 2:
        print("\n❌ Still not enough windows. Need more historical data.")
        return
    
    # Cost model
    cost_model = CostModel(
        maker_fee_bps=10,
        taker_fee_bps=10,
        base_slippage_bps=2,
    )
    
    # Run walk-forward for each strategy
    all_results = []
    
    print("\n" + "=" * 85)
    print("🎯 STRATEGY 1: Funding Arbitrage (z>2, BB confirm)")
    print("=" * 85)
    
    funding_results = run_walk_forward(
        strategy_type='funding_arb',
        strategy_name='Funding Arb (z>2, BB)',
        param_grid=FUNDING_ARB_PARAMS,
        windows=windows,
        ohlcv_data=ohlcv_data,
        features=features,
        cost_model=cost_model,
    )
    funding_stats = analyze_results(funding_results, 'Funding Arb (z>2, BB)')
    all_results.append(funding_stats)
    
    print("\n" + "=" * 85)
    print("🎯 STRATEGY 2: Combined OI+FR+Price (all 3 factors)")
    print("=" * 85)
    
    combined_results = run_walk_forward(
        strategy_type='combined',
        strategy_name='Combined OI+FR+Price',
        param_grid=COMBINED_PARAMS,
        windows=windows,
        ohlcv_data=ohlcv_data,
        features=features,
        cost_model=cost_model,
    )
    combined_stats = analyze_results(combined_results, 'Combined OI+FR+Price')
    all_results.append(combined_stats)
    
    # Print summary
    print_summary([s for s in all_results if s is not None])
    
    # Interpretation
    print("\n" + "=" * 85)
    print("📋 INTERPRETATION")
    print("=" * 85)
    
    valid_results = [s for s in all_results if s is not None]
    
    if not valid_results:
        print("\n❌ No valid results to analyze.")
        return
    
    for stats in valid_results:
        print(f"\n{stats['strategy']}:")
        
        # Evaluate quality
        if stats['sharpe_decay'] < 0.3:
            decay_quality = "✅ Excellent (minimal overfitting)"
        elif stats['sharpe_decay'] < 0.5:
            decay_quality = "✅ Good (some overfitting, acceptable)"
        elif stats['sharpe_decay'] < 1.0:
            decay_quality = "⚠️  Moderate overfitting"
        else:
            decay_quality = "❌ Severe overfitting"
        
        if stats['avg_test_sharpe'] > 0.5:
            oos_quality = "✅ Strong OOS performance"
        elif stats['avg_test_sharpe'] > 0:
            oos_quality = "✅ Positive OOS performance"
        else:
            oos_quality = "❌ Negative OOS performance"
        
        if stats['pct_profitable_windows'] >= 60:
            consistency = "✅ Consistent (profitable in most windows)"
        elif stats['pct_profitable_windows'] >= 40:
            consistency = "⚠️  Mixed results"
        else:
            consistency = "❌ Inconsistent"
        
        print(f"  Sharpe decay: {stats['sharpe_decay']:.2f} - {decay_quality}")
        print(f"  OOS Sharpe: {stats['avg_test_sharpe']:.2f} - {oos_quality}")
        print(f"  Consistency: {stats['pct_profitable_windows']:.0f}% - {consistency}")
        
        # Per-window breakdown
        print(f"\n  Per-window test returns:")
        for i, (ret, sharpe) in enumerate(zip(stats['test_returns'], stats['test_sharpes'])):
            status = "✅" if ret > 0 else "❌"
            print(f"    Window {i+1}: {ret*100:>7.2f}% return, {sharpe:>6.2f} Sharpe {status}")
    
    # Final recommendation
    print("\n" + "=" * 85)
    print("🎯 RECOMMENDATION")
    print("=" * 85)
    
    # Find best strategy by OOS Sharpe
    best = max(valid_results, key=lambda x: x['avg_test_sharpe'])
    
    if best['avg_test_sharpe'] > 0 and best['sharpe_decay'] < 1.0:
        print(f"""
    ✅ {best['strategy']} shows genuine edge:
       - Positive out-of-sample Sharpe: {best['avg_test_sharpe']:.2f}
       - Acceptable overfitting (decay: {best['sharpe_decay']:.2f})
       - {best['pct_profitable_windows']:.0f}% of test windows profitable
    
    NEXT STEPS:
    1. Paper trade this strategy for 2-4 weeks
    2. Monitor for regime changes
    3. Start with small position sizes (50% of backtest)
    4. Set strict drawdown limits (e.g., stop at -10%)
        """)
    elif best['avg_test_sharpe'] > 0:
        print(f"""
    ⚠️  {best['strategy']} shows weak edge:
       - Positive but low OOS Sharpe: {best['avg_test_sharpe']:.2f}
       - Significant overfitting (decay: {best['sharpe_decay']:.2f})
    
    RECOMMENDATIONS:
    1. Need more data for robust validation
    2. Consider simpler strategy with fewer parameters
    3. Paper trade before any live deployment
        """)
    else:
        print(f"""
    ❌ No strategy shows reliable out-of-sample edge.
    
    RECOMMENDATIONS:
    1. The backtest results were likely overfitted
    2. Need different approach or more data
    3. Consider:
       - Longer holding periods (reduce cost drag)
       - Different signals
       - Different asset selection
        """)


if __name__ == "__main__":
    main()