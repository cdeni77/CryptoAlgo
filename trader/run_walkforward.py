#!/usr/bin/env python3
"""
Per-Coin Walk-Forward Validation with Auto Window Sizing - UPDATED VERSION

Updates from original:
- Uses fixed backtester with stop-loss, liquidation, funding
- Tests regime_filter and trend_filter parameters
- Updated param grids based on backtest results

Automatically calculates optimal train/test windows based on available data:
- More data = larger windows, more windows
- Less data = smaller windows, fewer windows
- Ensures minimum statistical significance

Usage:
    python run_walkforward.py
    python run_walkforward.py --symbols BIP-20DEC30-CDE
    python run_walkforward.py --min-windows 3 --max-windows 10
"""

import argparse
import sys
import warnings
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Add parent to path for imports
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
# Parameter Grids - UPDATED with regime/trend filters
# =============================================================================

FUNDING_ARB_PARAMS = [
    # Best performers from backtest - with regime filter
    {'entry_threshold': 2.5, 'use_regime_filter': True, 'use_trend_filter': False},
    {'entry_threshold': 2.0, 'use_regime_filter': True, 'use_trend_filter': True},
    {'entry_threshold': 2.0, 'use_regime_filter': True, 'use_trend_filter': False},
    {'entry_threshold': 1.5, 'use_regime_filter': True, 'use_trend_filter': True},
    {'entry_threshold': 1.5, 'use_regime_filter': True, 'use_trend_filter': False},
    # Without regime filter for comparison
    {'entry_threshold': 2.0, 'use_regime_filter': False, 'use_trend_filter': True},
    {'entry_threshold': 1.5, 'use_regime_filter': False, 'use_trend_filter': True},
]

OI_DIVERGENCE_PARAMS = [
    {'divergence_threshold': 0.5, 'use_oi_zscore_filter': True, 'use_regime_filter': True},
    {'divergence_threshold': 0.3, 'use_oi_zscore_filter': True, 'use_regime_filter': True},
    {'divergence_threshold': 0.5, 'use_oi_zscore_filter': False, 'use_regime_filter': True},
    {'divergence_threshold': 0.3, 'use_oi_zscore_filter': False, 'use_regime_filter': False},
]

COMBINED_PARAMS = [
    # Best from backtest
    {'min_factors': 2, 'funding_threshold': 1.0, 'use_regime_filter': True},
    {'min_factors': 2, 'funding_threshold': 1.5, 'use_regime_filter': True},
    {'min_factors': 2, 'funding_threshold': 2.0, 'use_regime_filter': True},
    # 3 factors (stricter)
    {'min_factors': 3, 'funding_threshold': 1.0, 'use_regime_filter': True},
    {'min_factors': 3, 'funding_threshold': 1.5, 'use_regime_filter': True},
]


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
    param_stability: float = 0.0
    
    def compute_stats(self):
        """Compute aggregate statistics."""
        if not self.windows:
            return
        
        train_sharpes = [w.train_sharpe for w in self.windows]
        test_sharpes = [w.test_sharpe for w in self.windows]
        test_returns = [w.test_return for w in self.windows]
        test_trades = [w.test_trades for w in self.windows]
        
        self.avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0
        self.avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0
        self.sharpe_decay = self.avg_train_sharpe - self.avg_test_sharpe
        
        # Cumulative return (compounded)
        self.cumulative_test_return = np.prod([1 + r for r in test_returns]) - 1 if test_returns else 0
        
        self.pct_profitable_windows = sum(1 for r in test_returns if r > 0) / len(test_returns) * 100 if test_returns else 0
        self.total_test_trades = sum(test_trades)
        self.avg_trades_per_window = np.mean(test_trades) if test_trades else 0
        
        # Parameter stability
        param_strs = [str(sorted(w.best_params.items())) for w in self.windows]
        if param_strs:
            most_common = max(set(param_strs), key=param_strs.count)
            self.param_stability = param_strs.count(most_common) / len(param_strs) * 100
            # Find the actual params
            for w in self.windows:
                if str(sorted(w.best_params.items())) == most_common:
                    self.best_overall_params = w.best_params
                    break


# =============================================================================
# Data Loading
# =============================================================================

def load_features_from_csv(features_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """Load precomputed features for a symbol."""
    possible_names = [
        f"{symbol}_features.csv",
        f"{symbol.replace('-', '_')}_features.csv",
        f"{symbol.lower()}_features.csv",
        f"{symbol.lower().replace('-', '_')}_features.csv",
    ]
    
    for name in possible_names:
        path = features_dir / name
        if path.exists():
            try:
                df_header = pd.read_csv(path, nrows=0)
                columns = df_header.columns.tolist()
                
                datetime_col = None
                for col in ['timestamp', 'event_time', 'datetime', 'Timestamp', 'Event_time']:
                    if col in columns:
                        datetime_col = col
                        break
                
                if datetime_col is None:
                    for col in columns:
                        if col.startswith('Unnamed'):
                            datetime_col = col
                            break
                
                if datetime_col:
                    df = pd.read_csv(path, parse_dates=[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                    return df
                
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index)
                return df
                    
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue
    return None


def load_ohlcv_from_db(db_path: str, symbol: str, timeframe: str = '1h') -> Optional[pd.DataFrame]:
    """Load OHLCV data from database."""
    try:
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
    except Exception as e:
        logger.error(f"Error loading OHLCV for {symbol}: {e}")
        return None


def check_features(features: pd.DataFrame) -> Dict[str, bool]:
    """Check which feature types are available."""
    cols = features.columns.tolist()
    return {
        'funding': any('funding' in c.lower() for c in cols),
        'oi': any('oi' in c.lower() or 'open_interest' in c.lower() for c in cols),
        'price': any('bb_' in c.lower() or 'rsi' in c.lower() for c in cols),
    }


# =============================================================================
# Window Configuration
# =============================================================================

def calculate_window_config(
    total_days: int,
    min_windows: int = 3,
    max_windows: int = 10,
) -> Optional[WindowConfig]:
    """
    Calculate optimal window configuration based on data length.
    
    Rules:
    - Training period should be 3-4x test period
    - Minimum 180 days training (6 months)
    - Minimum 60 days testing (2 months)
    - Test periods don't overlap (step = test_days)
    """
    MIN_TRAIN_DAYS = 180  # 6 months minimum training
    MIN_TEST_DAYS = 60    # 2 months minimum testing
    TRAIN_TEST_RATIO = 3  # Train should be 3x test
    
    MIN_TOTAL = MIN_TRAIN_DAYS + MIN_TEST_DAYS * 2  # At least 2 windows
    
    if total_days < MIN_TOTAL:
        return None
    
    # Start with reasonable defaults based on data length
    if total_days >= 1800:  # 5+ years
        train_days = 365  # 1 year training
        test_days = 90    # 3 months testing
    elif total_days >= 1000:  # ~3 years
        train_days = 270  # 9 months training
        test_days = 90    # 3 months testing
    elif total_days >= 500:  # ~1.5 years
        train_days = 180  # 6 months training
        test_days = 60    # 2 months testing
    else:  # Less data
        train_days = MIN_TRAIN_DAYS
        test_days = MIN_TEST_DAYS
    
    # Calculate how many windows fit with non-overlapping test periods
    # Layout: [train][test1][test2][test3]...
    # Total = train_days + test_days * num_windows
    available_for_test = total_days - train_days
    num_windows = available_for_test // test_days
    
    # Clamp to min/max
    num_windows = min(max_windows, max(min_windows, num_windows))
    
    # Step = test_days for non-overlapping test periods
    step_days = test_days
    
    return WindowConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        num_windows=num_windows,
        total_days=total_days,
    )


def generate_windows(
    data_start: datetime,
    data_end: datetime,
    config: WindowConfig,
) -> List[WalkForwardWindow]:
    """Generate walk-forward windows based on config."""
    windows = []
    
    train_delta = timedelta(days=config.train_days)
    test_delta = timedelta(days=config.test_days)
    step_delta = timedelta(days=config.step_days)
    
    current_train_start = data_start
    window_id = 0
    
    while True:
        train_end = current_train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        
        if test_end > data_end:
            break
        
        windows.append(WalkForwardWindow(
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            window_id=window_id,
        ))
        
        current_train_start += step_delta
        window_id += 1
        
        if window_id >= config.num_windows:
            break
    
    return windows


def filter_data_by_time(
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter data to time range."""
    mask = (ohlcv.index >= start) & (ohlcv.index < end)
    return ohlcv[mask], features[mask]


# =============================================================================
# Strategy Creation - UPDATED with new parameters
# =============================================================================

def create_strategy(strategy_type: str, symbol: str, params: Dict):
    """Create strategy instance with given parameters."""
    if strategy_type == 'funding_arb':
        return FundingArbitrageStrategy(
            symbols=[symbol],
            entry_threshold=params.get('entry_threshold', 2.0),
            exit_threshold=params.get('exit_threshold', 0.5),
            use_price_confirmation=params.get('use_price_confirmation', True),
            bb_confirmation_threshold=1.5,
            min_hold_hours=24,
            max_hold_hours=168,
            position_size=0.15,  # Reduced from 0.3
            use_regime_filter=params.get('use_regime_filter', True),  # NEW
            use_trend_filter=params.get('use_trend_filter', False),   # NEW
        )
    elif strategy_type == 'oi_divergence':
        return OIDivergenceStrategy(
            symbols=[symbol],
            divergence_threshold=params.get('divergence_threshold', 0.5),
            use_oi_zscore_filter=params.get('use_oi_zscore_filter', True),
            oi_zscore_threshold=1.5,
            position_size=0.15,
            min_hold_hours=12,
            max_hold_hours=72,
            use_regime_filter=params.get('use_regime_filter', True),  # NEW
        )
    elif strategy_type == 'combined':
        return CombinedOIFundingStrategy(
            symbols=[symbol],
            oi_divergence_threshold=0.3,
            funding_threshold=params.get('funding_threshold', 1.5),
            bb_threshold=1.5,
            min_factors=params.get('min_factors', 2),
            position_size=0.15,
            min_hold_hours=24,
            max_hold_hours=120,
            use_regime_filter=params.get('use_regime_filter', True),  # NEW
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
# Backtesting - UPDATED with fixed engine settings
# =============================================================================

def run_backtest(
    symbol: str,
    strategy,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> PerformanceMetrics:
    """Run single backtest with fixed engine."""
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.3,
        max_leverage=3.0,
        default_stop_loss_pct=0.15,  # 15% stop loss
        apply_funding=True,           # Apply funding rates
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
        
        # Check data sufficiency
        if len(train_ohlcv) < 500 or len(test_ohlcv) < 100:
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
                
                # Prefer higher Sharpe, but require minimum trades
                if metrics.num_trades >= 3 and metrics.sharpe_ratio > best_train_sharpe:
                    best_train_sharpe = metrics.sharpe_ratio
                    best_train_return = metrics.total_return
                    best_train_trades = metrics.num_trades
                    best_params = params
            except Exception as e:
                logger.debug(f"Error with params {params}: {e}")
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
        except Exception as e:
            logger.debug(f"Error testing window {window.window_id}: {e}")
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
    
    print(f"\n  {'Strategy':<15} {'Train SR':>10} {'Test SR':>10} {'Decay':>8} "
          f"{'Cum Ret':>10} {'Win%':>8} {'Trades':>8} {'Param Stab':>10}")
    print(f"  {'-'*95}")
    
    for r in results:
        if not r.windows:
            print(f"  {r.strategy_type:<15} {'No valid windows':>30}")
            continue
        
        # Highlight good results
        status = ""
        if r.avg_test_sharpe > 0.5 and r.pct_profitable_windows >= 50:
            status = "✅"
        elif r.avg_test_sharpe > 0 and r.pct_profitable_windows >= 40:
            status = "⚠️"
        else:
            status = "❌"
        
        print(f"  {r.strategy_type:<15} {r.avg_train_sharpe:>10.2f} {r.avg_test_sharpe:>10.2f} "
              f"{r.sharpe_decay:>8.2f} {r.cumulative_test_return*100:>9.1f}% "
              f"{r.pct_profitable_windows:>7.1f}% {r.total_test_trades:>8} "
              f"{r.param_stability:>9.0f}% {status}")
    
    # Best strategy
    valid_results = [r for r in results if r.windows]
    if valid_results:
        best = max(valid_results, key=lambda x: x.avg_test_sharpe)
        print(f"\n  🏆 Best OOS: {best.strategy_type}")
        print(f"     Avg Test Sharpe: {best.avg_test_sharpe:.2f}")
        print(f"     Cumulative Test Return: {best.cumulative_test_return*100:.1f}%")
        print(f"     Best Params: {best.best_overall_params}")
        print(f"     Param Stability: {best.param_stability:.0f}%")
        
        # Overfitting check
        if best.sharpe_decay > 0.5:
            print(f"     ⚠️  HIGH DECAY: Train-Test Sharpe gap = {best.sharpe_decay:.2f}")
        if best.pct_profitable_windows < 50:
            print(f"     ⚠️  LOW WIN RATE: Only {best.pct_profitable_windows:.0f}% of windows profitable")


def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation (UPDATED)")
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR))
    parser.add_argument("--symbols", type=str, nargs="+", help="Specific symbols")
    parser.add_argument("--min-windows", type=int, default=3, help="Minimum walk-forward windows")
    parser.add_argument("--max-windows", type=int, default=10, help="Maximum walk-forward windows")
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    print("=" * 105)
    print("🔄 WALK-FORWARD VALIDATION (UPDATED with Regime/Trend Filters)")
    print("=" * 105)
    print(f"\nSettings:")
    print(f"  - Target {args.min_windows}-{args.max_windows} windows per coin (auto-sized)")
    print(f"  - Stop-loss: 15%")
    print(f"  - Leverage limit: 3x")
    print(f"  - Funding rates applied")
    print(f"  - Testing regime_filter and trend_filter parameters")
    print()
    
    # Cost model (Coinbase fees)
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
    
    print(f"Symbols: {symbols}\n")
    
    all_results: Dict[str, List[CoinWalkForwardResult]] = {}
    
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
        
        # Calculate window config
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
        
        # Check features
        available = check_features(features)
        print(f"  Features: Funding={available['funding']}, OI={available['oi']}, Price={available['price']}")
        
        # Determine strategies
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
        
        # Run walk-forward
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
    print("\n" + "=" * 105)
    print("📈 FINAL SUMMARY")
    print("=" * 105)
    
    print(f"\n{'Symbol':<20} {'Strategy':<15} {'Test SR':>10} {'Decay':>8} {'Win%':>8} {'Cum Ret':>10} {'Status':<10}")
    print("-" * 90)
    
    for symbol, results in all_results.items():
        valid = [r for r in results if r.windows and len(r.windows) >= 2]
        if not valid:
            print(f"{symbol:<20} {'No valid results':<15}")
            continue
        
        best = max(valid, key=lambda x: x.avg_test_sharpe)
        
        # Status
        if best.avg_test_sharpe > 0.5 and best.pct_profitable_windows >= 50 and best.sharpe_decay < 0.5:
            status = "✅ DEPLOY"
        elif best.avg_test_sharpe > 0 and best.pct_profitable_windows >= 40:
            status = "⚠️ MAYBE"
        else:
            status = "❌ SKIP"
        
        print(f"{symbol:<20} {best.strategy_type:<15} {best.avg_test_sharpe:>10.2f} "
              f"{best.sharpe_decay:>8.2f} {best.pct_profitable_windows:>7.0f}% "
              f"{best.cumulative_test_return*100:>9.1f}% {status:<10}")
    
    # Deployment recommendations
    print("\n" + "=" * 105)
    print("🚀 DEPLOYMENT RECOMMENDATIONS")
    print("=" * 105)
    
    deployable = []
    for symbol, results in all_results.items():
        valid = [r for r in results if r.windows and len(r.windows) >= 2]
        if valid:
            best = max(valid, key=lambda x: x.avg_test_sharpe)
            if best.avg_test_sharpe > 0.3 and best.pct_profitable_windows >= 50:
                deployable.append((symbol, best))
    
    if deployable:
        print("\nStrategies ready for paper trading:")
        for symbol, result in deployable:
            print(f"\n  {symbol}:")
            print(f"    Strategy: {result.strategy_type}")
            print(f"    Params: {result.best_overall_params}")
            print(f"    Expected Sharpe: {result.avg_test_sharpe:.2f}")
            print(f"    OOS Win Rate: {result.pct_profitable_windows:.0f}%")
    else:
        print("\n⚠️  No strategies meet deployment criteria.")
        print("    Consider: loosening entry thresholds, more data, or different instruments.")


if __name__ == "__main__":
    main()