#!/usr/bin/env python3
"""
Per-Coin Walk-Forward Validation with Automatic Window Sizing

Automatically calculates optimal train/test windows based on available data:
- More data = larger windows, more windows
- Less data = smaller windows, fewer windows
- Ensures minimum statistical significance

Rules:
- Minimum 60 days training, 30 days testing
- Target 4-8 walk-forward windows per coin
- Non-overlapping test periods

Usage:
    python run_walkforward_per_coin.py
    python run_walkforward_per_coin.py --symbols BIP-20DEC30-CDE
    python run_walkforward_per_coin.py --min-windows 3 --max-windows 10
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from backtesting.engine import Backtester, CostModel, PerformanceMetrics
from backtesting.strategies import (
    FundingArbitrageStrategy,
    OIDivergenceStrategy,
    CombinedOIFundingStrategy,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class WindowConfig:
    """Calculated window configuration for a coin."""
    train_days: int
    test_days: int
    step_days: int
    num_windows: int
    total_days: int
    
    def __str__(self):
        return f"{self.train_days}d train / {self.test_days}d test / {self.step_days}d step → {self.num_windows} windows"


@dataclass 
class WindowResult:
    """Result from one window."""
    window: WalkForwardWindow
    best_params: Dict
    train_sharpe: float
    train_return: float
    train_trades: int
    test_sharpe: float
    test_return: float
    test_trades: int


@dataclass
class CoinWalkForwardResult:
    """Complete walk-forward results for one coin."""
    symbol: str
    strategy_type: str
    window_config: WindowConfig
    windows: List[WindowResult]
    data_start: datetime
    data_end: datetime
    
    # Aggregated stats
    avg_train_sharpe: float = 0.0
    avg_test_sharpe: float = 0.0
    sharpe_decay: float = 0.0
    cumulative_test_return: float = 0.0
    pct_profitable_windows: float = 0.0
    total_test_trades: int = 0
    avg_trades_per_window: float = 0.0
    best_overall_params: Dict = field(default_factory=dict)
    param_stability: float = 0.0  # How often same params chosen
    
    def compute_stats(self):
        """Compute aggregate statistics."""
        if not self.windows:
            return
        
        train_sharpes = [w.train_sharpe for w in self.windows]
        test_sharpes = [w.test_sharpe for w in self.windows]
        test_returns = [w.test_return for w in self.windows]
        test_trades = [w.test_trades for w in self.windows]
        
        self.avg_train_sharpe = np.mean(train_sharpes)
        self.avg_test_sharpe = np.mean(test_sharpes)
        self.sharpe_decay = self.avg_train_sharpe - self.avg_test_sharpe
        self.cumulative_test_return = np.prod([1 + r for r in test_returns]) - 1
        self.pct_profitable_windows = sum(1 for r in test_returns if r > 0) / len(test_returns) * 100
        self.total_test_trades = sum(test_trades)
        self.avg_trades_per_window = np.mean(test_trades)
        
        # Parameter stability - how often was the same param set chosen?
        param_strings = [str(sorted(w.best_params.items())) for w in self.windows]
        unique_params = set(param_strings)
        most_common_count = max(param_strings.count(p) for p in unique_params)
        self.param_stability = most_common_count / len(self.windows)
        
        # Most common params
        param_counts = {}
        for w in self.windows:
            key = str(sorted(w.best_params.items()))
            param_counts[key] = param_counts.get(key, 0) + 1
        
        most_common_key = max(param_counts, key=param_counts.get)
        # Find the actual params dict
        for w in self.windows:
            if str(sorted(w.best_params.items())) == most_common_key:
                self.best_overall_params = w.best_params.copy()
                break


# =============================================================================
# Automatic Window Calculation
# =============================================================================

def calculate_window_config(
    total_days: int,
    min_windows: int = 3,
    max_windows: int = 10,
    min_train_days: int = 60,
    min_test_days: int = 30,
) -> Optional[WindowConfig]:
    """
    Automatically calculate optimal window configuration based on data length.
    
    Strategy:
    - Target 4-8 windows for good statistical power
    - Train period should be 2-3x test period
    - Larger datasets get larger windows (more robust)
    - Smaller datasets get smaller windows (more windows)
    
    Returns None if not enough data.
    """
    
    # Minimum data required
    min_required = min_train_days + min_test_days + min_test_days  # At least 2 windows
    if total_days < min_required:
        return None
    
    # Calculate based on data length tiers
    if total_days >= 1800:  # 5+ years
        train_days = 365
        test_days = 90
        step_days = 90
    elif total_days >= 1095:  # 3-5 years
        train_days = 270
        test_days = 90
        step_days = 90
    elif total_days >= 730:  # 2-3 years
        train_days = 180
        test_days = 60
        step_days = 60
    elif total_days >= 365:  # 1-2 years
        train_days = 120
        test_days = 45
        step_days = 45
    elif total_days >= 240:  # 8-12 months
        train_days = 90
        test_days = 30
        step_days = 30
    else:  # 4-8 months
        train_days = 60
        test_days = 30
        step_days = 30
    
    # Calculate number of windows
    # Formula: num_windows = (total_days - train_days - test_days) / step_days + 1
    usable_days = total_days - train_days
    num_windows = max(1, int((usable_days - test_days) / step_days) + 1)
    
    # Adjust if too few or too many windows
    iterations = 0
    while num_windows < min_windows and iterations < 10:
        # Reduce window sizes to get more windows
        if train_days > min_train_days:
            train_days = max(min_train_days, train_days - 30)
        if test_days > min_test_days:
            test_days = max(min_test_days, test_days - 15)
        step_days = test_days  # Keep non-overlapping
        
        usable_days = total_days - train_days
        num_windows = max(1, int((usable_days - test_days) / step_days) + 1)
        iterations += 1
    
    while num_windows > max_windows and iterations < 20:
        # Increase window sizes to reduce windows
        train_days += 30
        test_days += 15
        step_days = test_days
        
        usable_days = total_days - train_days
        num_windows = max(1, int((usable_days - test_days) / step_days) + 1)
        iterations += 1
    
    # Final check
    if num_windows < 2:
        # Try minimum config
        train_days = min_train_days
        test_days = min_test_days
        step_days = min_test_days
        usable_days = total_days - train_days
        num_windows = max(1, int((usable_days - test_days) / step_days) + 1)
    
    if num_windows < 2:
        return None
    
    return WindowConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        num_windows=num_windows,
        total_days=total_days,
    )


def generate_windows(
    start_date: datetime,
    end_date: datetime,
    config: WindowConfig,
) -> List[WalkForwardWindow]:
    """Generate walk-forward windows from config."""
    windows = []
    window_id = 0
    current_start = start_date
    
    while True:
        train_end = current_start + timedelta(days=config.train_days)
        test_start = train_end
        test_end = test_start + timedelta(days=config.test_days)
        
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
        current_start += timedelta(days=config.step_days)
    
    return windows


# =============================================================================
# Data Loading
# =============================================================================

def load_features_from_csv(features_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load features for a single symbol."""
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


def filter_data_by_time(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter data to time range."""
    ohlcv_mask = (ohlcv.index >= start) & (ohlcv.index < end)
    feat_mask = (features.index >= start) & (features.index < end)
    return ohlcv[ohlcv_mask].copy(), features[feat_mask].copy()


def check_features(features: pd.DataFrame) -> Dict[str, bool]:
    """Check available feature types."""
    cols = features.columns.tolist()
    return {
        'funding': any('funding' in c.lower() for c in cols),
        'oi': any('oi' in c.lower() or 'open_interest' in c.lower() for c in cols),
        'price': any('bb_' in c.lower() or 'rsi' in c.lower() for c in cols),
    }


# =============================================================================
# Strategy Configuration
# =============================================================================

FUNDING_ARB_PARAMS = [
    {'entry_threshold': 1.2, 'exit_threshold': 0.5, 'use_price_confirmation': False},
    {'entry_threshold': 1.5, 'exit_threshold': 0.5, 'use_price_confirmation': False},
    {'entry_threshold': 1.5, 'exit_threshold': 0.5, 'use_price_confirmation': True},
    {'entry_threshold': 1.8, 'exit_threshold': 0.5, 'use_price_confirmation': True},
    {'entry_threshold': 2.0, 'exit_threshold': 0.5, 'use_price_confirmation': True},
    {'entry_threshold': 2.0, 'exit_threshold': 0.3, 'use_price_confirmation': True},
    {'entry_threshold': 2.5, 'exit_threshold': 0.5, 'use_price_confirmation': True},
]

OI_DIVERGENCE_PARAMS = [
    {'divergence_threshold': 0.3, 'use_oi_zscore_filter': False},
    {'divergence_threshold': 0.3, 'use_oi_zscore_filter': True},
    {'divergence_threshold': 0.5, 'use_oi_zscore_filter': False},
    {'divergence_threshold': 0.5, 'use_oi_zscore_filter': True},
    {'divergence_threshold': 0.7, 'use_oi_zscore_filter': True},
]

COMBINED_PARAMS = [
    {'funding_threshold': 1.0, 'min_factors': 2},
    {'funding_threshold': 1.5, 'min_factors': 2},
    {'funding_threshold': 1.5, 'min_factors': 3},
    {'funding_threshold': 2.0, 'min_factors': 2},
    {'funding_threshold': 2.0, 'min_factors': 3},
]


def create_strategy(strategy_type: str, symbol: str, params: Dict):
    """Create strategy instance."""
    if strategy_type == 'funding_arb':
        return FundingArbitrageStrategy(
            symbols=[symbol],
            entry_threshold=params.get('entry_threshold', 2.0),
            exit_threshold=params.get('exit_threshold', 0.5),
            use_price_confirmation=params.get('use_price_confirmation', True),
            bb_confirmation_threshold=1.5,
            min_hold_hours=24,
            max_hold_hours=168,
            position_size=0.3,
        )
    elif strategy_type == 'oi_divergence':
        return OIDivergenceStrategy(
            symbols=[symbol],
            divergence_threshold=params.get('divergence_threshold', 0.5),
            use_oi_zscore_filter=params.get('use_oi_zscore_filter', True),
            oi_zscore_threshold=1.5,
            position_size=0.3,
            min_hold_hours=12,
            max_hold_hours=72,
        )
    elif strategy_type == 'combined':
        return CombinedOIFundingStrategy(
            symbols=[symbol],
            oi_divergence_threshold=0.3,
            funding_threshold=params.get('funding_threshold', 1.5),
            bb_threshold=1.5,
            min_factors=params.get('min_factors', 2),
            position_size=0.3,
            min_hold_hours=24,
            max_hold_hours=120,
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def get_param_grid(strategy_type: str) -> List[Dict]:
    """Get parameter grid for strategy type."""
    if strategy_type == 'funding_arb':
        return FUNDING_ARB_PARAMS
    elif strategy_type == 'oi_divergence':
        return OI_DIVERGENCE_PARAMS
    elif strategy_type == 'combined':
        return COMBINED_PARAMS
    return []


# =============================================================================
# Backtesting
# =============================================================================

def run_backtest(
    symbol: str,
    strategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> PerformanceMetrics:
    """Run single backtest."""
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.5,
        max_leverage=3.0,
    )
    
    _, metrics = backtester.run(
        strategy=strategy,
        ohlcv_data={symbol: ohlcv},
        features={symbol: features},
    )
    
    return metrics


def run_walk_forward_for_strategy(
    symbol: str,
    strategy_type: str,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    windows: List[WalkForwardWindow],
    window_config: WindowConfig,
    cost_model: CostModel,
) -> CoinWalkForwardResult:
    """Run walk-forward for one strategy type on one coin."""
    
    param_grid = get_param_grid(strategy_type)
    window_results = []
    
    for window in windows:
        # Split data
        train_ohlcv, train_features = filter_data_by_time(
            ohlcv, features, window.train_start, window.train_end
        )
        test_ohlcv, test_features = filter_data_by_time(
            ohlcv, features, window.test_start, window.test_end
        )
        
        # Check data sufficiency (at least 100 bars train, 50 test)
        if len(train_ohlcv) < 100 or len(test_ohlcv) < 50:
            continue
        
        # Find best params on training data
        best_params = None
        best_train_sharpe = -np.inf
        best_train_return = 0
        best_train_trades = 0
        
        for params in param_grid:
            try:
                strategy = create_strategy(strategy_type, symbol, params)
                metrics = run_backtest(symbol, strategy, train_ohlcv, train_features, cost_model)
                
                if metrics.sharpe_ratio > best_train_sharpe:
                    best_train_sharpe = metrics.sharpe_ratio
                    best_train_return = metrics.total_return
                    best_train_trades = metrics.num_trades
                    best_params = params
            except Exception:
                continue
        
        if best_params is None:
            continue
        
        # Test with best params
        try:
            strategy = create_strategy(strategy_type, symbol, best_params)
            test_metrics = run_backtest(symbol, strategy, test_ohlcv, test_features, cost_model)
            
            window_results.append(WindowResult(
                window=window,
                best_params=best_params,
                train_sharpe=best_train_sharpe,
                train_return=best_train_return,
                train_trades=best_train_trades,
                test_sharpe=test_metrics.sharpe_ratio,
                test_return=test_metrics.total_return,
                test_trades=test_metrics.num_trades,
            ))
        except Exception:
            continue
    
    # Create result object
    result = CoinWalkForwardResult(
        symbol=symbol,
        strategy_type=strategy_type,
        window_config=window_config,
        windows=window_results,
        data_start=ohlcv.index.min(),
        data_end=ohlcv.index.max(),
    )
    result.compute_stats()
    
    return result


# =============================================================================
# Output / Reporting
# =============================================================================

def print_coin_results(results: List[CoinWalkForwardResult]):
    """Print results for one coin."""
    if not results:
        return
    
    print(f"  {'Strategy':<15} {'Windows':>8} {'Train SR':>10} {'Test SR':>10} {'Decay':>8} {'OOS Ret':>10} {'Win%':>7} {'Trades':>7} {'Stability':>10}")
    print(f"  {'-'*100}")
    
    for r in results:
        if not r.windows:
            continue
        print(f"  {r.strategy_type:<15} {len(r.windows):>8} {r.avg_train_sharpe:>10.2f} "
              f"{r.avg_test_sharpe:>10.2f} {r.sharpe_decay:>8.2f} "
              f"{r.cumulative_test_return*100:>9.2f}% {r.pct_profitable_windows:>6.1f}% "
              f"{r.total_test_trades:>7} {r.param_stability*100:>9.1f}%")
    
    # Best strategy
    valid = [r for r in results if r.windows and r.avg_test_sharpe > -np.inf]
    if valid:
        best = max(valid, key=lambda x: x.avg_test_sharpe)
        
        # Quality assessment
        if best.avg_test_sharpe > 0.5 and best.sharpe_decay < 0.5 and best.param_stability > 0.5:
            quality = "✅ STRONG"
        elif best.avg_test_sharpe > 0.3 and best.sharpe_decay < 0.7:
            quality = "✅ GOOD"
        elif best.avg_test_sharpe > 0 and best.sharpe_decay < 1.0:
            quality = "⚠️  MODERATE"
        else:
            quality = "❌ WEAK"
        
        print(f"  Best: {best.strategy_type} {quality}")
        print(f"  └─ Test Sharpe: {best.avg_test_sharpe:.2f}, Decay: {best.sharpe_decay:.2f}, Param Stability: {best.param_stability*100:.0f}%")
        print(f"  └─ Optimal params: {best.best_overall_params}")
        
        # Per-window breakdown
        print(f"  Window details:")
        for w in best.windows:
            status = "✅" if w.test_return > 0 else "❌"
            print(f"    W{w.window.window_id}: {w.window.train_start.date()} → {w.window.test_end.date()} | "
                  f"Test: {w.test_return*100:+.2f}%, SR={w.test_sharpe:.2f}, {w.test_trades} trades {status}")


def main():
    parser = argparse.ArgumentParser(description="Per-coin walk-forward with automatic window sizing")
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR))
    parser.add_argument("--symbols", type=str, nargs="+", help="Specific symbols")
    parser.add_argument("--min-windows", type=int, default=3, help="Minimum walk-forward windows")
    parser.add_argument("--max-windows", type=int, default=10, help="Maximum walk-forward windows")
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    print("=" * 105)
    print("🔄 PER-COIN WALK-FORWARD VALIDATION (Auto Window Sizing)")
    print("=" * 105)
    print(f"Target: {args.min_windows}-{args.max_windows} windows per coin (auto-calculated based on data length)")
    
    # Cost model
    cost_model = CostModel(
        maker_fee_bps=10,
        taker_fee_bps=10,
        base_slippage_bps=2,
    )
    
    # Find symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = []
        for f in features_dir.glob("*_features.csv"):
            symbol = f.stem.replace("_features", "").replace("_", "-")
            symbols.append(symbol)
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    print(f"Symbols: {symbols}")
    
    # Store all results
    all_results: Dict[str, List[CoinWalkForwardResult]] = {}
    
    # Process each coin
    for symbol in symbols:
        print("=" * 105)
        print(f"📊 {symbol}")
        print("=" * 105)
        
        # Load data
        features = load_features_from_csv(features_dir, symbol)
        if features is None:
            print(f"  ❌ No features found")
            continue
        
        ohlcv = load_ohlcv_from_db(args.db_path, symbol)
        if ohlcv is None:
            print(f"  ❌ No OHLCV data found")
            continue
        
        # Align
        common_idx = features.index.intersection(ohlcv.index)
        if len(common_idx) == 0:
            print(f"  ❌ No overlapping data")
            continue
        
        ohlcv = ohlcv.loc[common_idx]
        features = features.loc[common_idx]
        
        data_start = ohlcv.index.min()
        data_end = ohlcv.index.max()
        total_days = (data_end - data_start).days
        
        print(f"  Data: {data_start.date()} to {data_end.date()} ({total_days} days, {len(ohlcv)} bars)")
        
        # Calculate optimal window config
        window_config = calculate_window_config(
            total_days=total_days,
            min_windows=args.min_windows,
            max_windows=args.max_windows,
        )
        
        if window_config is None:
            print(f"  ❌ Not enough data for walk-forward (need at least 120 days)")
            continue
        
        print(f"  Window config: {window_config}")
        
        # Generate windows
        windows = generate_windows(data_start, data_end, window_config)
        print(f"  Generated {len(windows)} windows")
        
        if len(windows) < 2:
            print(f"  ⚠️  Only {len(windows)} window(s) - results may not be reliable")
        
        # Check available features
        available = check_features(features)
        print(f"  Features: Funding={available['funding']}, OI={available['oi']}, Price={available['price']}")
        
        # Determine which strategies to run
        strategy_types = []
        if available['funding']:
            strategy_types.append('funding_arb')
        if available['oi']:
            strategy_types.append('oi_divergence')
        if available['funding'] and available['oi']:
            strategy_types.append('combined')
        
        if not strategy_types:
            print(f"  ❌ No applicable strategies")
            continue
        
        # Run walk-forward for each strategy type
        coin_results = []
        for strategy_type in strategy_types:
            print(f"  Running {strategy_type}...", end=" ", flush=True)
            result = run_walk_forward_for_strategy(
                symbol, strategy_type, ohlcv, features, windows, window_config, cost_model
            )
            coin_results.append(result)
            print(f"done ({len(result.windows)} valid windows)")
        
        all_results[symbol] = coin_results
        print_coin_results(coin_results)
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("=" * 105)
    print("📈 FINAL SUMMARY")
    print("=" * 105)
    
    print(f"{'Symbol':<20} {'Data Days':>10} {'Windows':>8} {'Strategy':<15} {'Test SR':>10} {'Decay':>8} {'Win%':>7} {'Status':<12}")
    print("-" * 105)
    
    deployment_configs = {}
    
    for symbol, results in all_results.items():
        valid = [r for r in results if r.windows and len(r.windows) >= 2]
        if not valid:
            wc = results[0].window_config if results else None
            days = wc.total_days if wc else "?"
            print(f"{symbol:<20} {days:>10} {'N/A':>8} {'N/A':<15} {'N/A':>10} {'N/A':>8} {'N/A':>7} {'❌ No data':<12}")
            continue
        
        best = max(valid, key=lambda x: x.avg_test_sharpe)
        
        # Status determination
        if (best.avg_test_sharpe > 0.5 and 
            best.sharpe_decay < 0.5 and 
            best.total_test_trades >= 10 and
            best.param_stability >= 0.4):
            status = "✅ DEPLOY"
        elif (best.avg_test_sharpe > 0.2 and 
              best.sharpe_decay < 0.8 and
              best.total_test_trades >= 5):
            status = "⚠️  PAPER"
        elif best.avg_test_sharpe > 0:
            status = "⚠️  WEAK"
        else:
            status = "❌ SKIP"
        
        print(f"{symbol:<20} {best.window_config.total_days:>10} {len(best.windows):>8} "
              f"{best.strategy_type:<15} {best.avg_test_sharpe:>10.2f} "
              f"{best.sharpe_decay:>8.2f} {best.pct_profitable_windows:>6.1f}% {status:<12}")
        
        if best.avg_test_sharpe > 0:
            deployment_configs[symbol] = {
                'strategy': best.strategy_type,
                'params': best.best_overall_params,
                'expected_sharpe': round(best.avg_test_sharpe, 2),
                'sharpe_decay': round(best.sharpe_decay, 2),
                'windows_tested': len(best.windows),
                'param_stability': round(best.param_stability, 2),
                'data_days': best.window_config.total_days,
            }
    
    # Output config
    print("=" * 105)
    print("📋 DEPLOYMENT CONFIG")
    print("=" * 105)
    
    print("# Copy this for paper trading:")
    print("COIN_CONFIGS = {")
    for symbol, config in sorted(deployment_configs.items(), key=lambda x: -x[1]['expected_sharpe']):
        print(f"    '{symbol}': {{")
        for k, v in config.items():
            if isinstance(v, str):
                print(f"        '{k}': '{v}',")
            elif isinstance(v, dict):
                print(f"        '{k}': {v},")
            else:
                print(f"        '{k}': {v},")
        print(f"    }},")
    print("}")
    
    # Interpretation
    print("=" * 105)
    print("📊 INTERPRETATION GUIDE")
    print("=" * 105)
    print("""
    Test SR (Sharpe):  Out-of-sample risk-adjusted return
                       > 0.5 = Strong, 0.2-0.5 = Moderate, < 0.2 = Weak
    
    Decay:             Train Sharpe - Test Sharpe (overfitting measure)
                       < 0.3 = Excellent, 0.3-0.5 = Good, 0.5-1.0 = Moderate, > 1.0 = Severe
    
    Win%:              Percentage of test windows with positive return
                       > 60% = Consistent, 40-60% = Mixed, < 40% = Inconsistent
    
    Param Stability:   How often the same params were chosen across windows
                       > 60% = Stable, 40-60% = Moderate, < 40% = Unstable (possible overfit)
    
    Status:
    ✅ DEPLOY  = Strong edge, ready for paper trading
    ⚠️  PAPER  = Some edge, needs more validation
    ⚠️  WEAK   = Marginal edge, high risk
    ❌ SKIP    = No reliable edge detected
    """)


if __name__ == "__main__":
    main()