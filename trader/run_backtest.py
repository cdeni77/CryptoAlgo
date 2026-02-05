"""
Per-Coin Backtest Analysis - FIXED VERSION

Uses fixed backtesting engine with:
- Proper stop-loss enforcement
- Liquidation simulation
- Leverage limits
- Regime-aware strategies

Usage:
    python run_backtest.py --db-path ./data/trading.db --features-dir ./data/features
"""

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Import fixed modules
from backtesting.engine import (
    Backtester,
    CostModel,
    PerformanceMetrics,
)
from backtesting.strategies import (
    FundingArbitrageStrategy,
    OIDivergenceStrategy,
    CombinedOIFundingStrategy,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = "./data/trading.db"
DEFAULT_FEATURES_DIR = "./data/features"


@dataclass
class CoinResult:
    """Result for a single strategy on a single coin."""
    symbol: str
    strategy_name: str
    params: Dict
    metrics: PerformanceMetrics
    data_start: datetime
    data_end: datetime
    num_bars: int


def load_features(features_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
    """
    Load precomputed features for a symbol.
    
    Handles various CSV formats:
    - 'timestamp' column
    - 'event_time' column  
    - First column as index
    - Unnamed index column
    """
    # Try different filename formats
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
                # First, read just the header to inspect columns
                df_header = pd.read_csv(path, nrows=0)
                columns = df_header.columns.tolist()
                
                # Identify the datetime column
                datetime_col = None
                possible_ts_cols = ['timestamp', 'event_time', 'datetime', 'date', 'time', 'Timestamp', 'Event_time']
                
                for col in possible_ts_cols:
                    if col in columns:
                        datetime_col = col
                        break
                
                # Check for unnamed index column (common when saving with index=True)
                if datetime_col is None:
                    for col in columns:
                        if col.startswith('Unnamed'):
                            datetime_col = col
                            break
                
                # If still no datetime column found, try the first column
                if datetime_col is None and len(columns) > 0:
                    # Read a sample to check if first column looks like datetime
                    df_sample = pd.read_csv(path, nrows=5)
                    first_col = columns[0]
                    try:
                        pd.to_datetime(df_sample[first_col].iloc[0])
                        datetime_col = first_col
                    except:
                        pass
                
                # If we found a datetime column, parse it
                if datetime_col:
                    df = pd.read_csv(path, parse_dates=[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                    df.index.name = 'timestamp'  # Normalize index name
                    logger.info(f"Loaded features from {path} using '{datetime_col}' as datetime column")
                    return df
                
                # Last resort: try reading with index_col=0
                df = pd.read_csv(path, index_col=0)
                try:
                    df.index = pd.to_datetime(df.index)
                    df.index.name = 'timestamp'
                    logger.info(f"Loaded features from {path} using first column as index")
                    return df
                except Exception as e:
                    logger.warning(f"Could not parse index as datetime: {e}")
                    # Return anyway, might still work
                    return df
                    
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue
    
    return None


def load_ohlcv(db_path: str, symbol: str, timeframe: str = '1h') -> Optional[pd.DataFrame]:
    """Load OHLCV data from database."""
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
    stop_loss_pct: float = 0.15,
) -> PerformanceMetrics:
    """Run backtest for single coin with fixed engine."""
    backtester = Backtester(
        initial_capital=100000,
        cost_model=cost_model,
        max_position_pct=0.3,      # Max 30% per position
        max_leverage=3.0,          # Max 3x leverage
        default_stop_loss_pct=stop_loss_pct,
        apply_funding=True,        # Apply funding rates
    )
    
    _, metrics = backtester.run(
        strategy=strategy,
        ohlcv_data={symbol: ohlcv},
        features={symbol: features},
    )
    
    return metrics


def get_funding_arb_strategies(symbol: str) -> List[tuple]:
    """Get Funding Arbitrage strategies with regime filter."""
    strategies = []
    
    # Test different entry thresholds with regime filter ON
    for entry_thresh in [1.5, 2.0, 2.5]:
        for use_regime in [True, False]:
            for use_trend in [True, False]:
                # Skip non-regime + non-trend (that's the broken version)
                if not use_regime and not use_trend:
                    continue
                
                name = f"FundingArb(z>{entry_thresh}"
                if use_regime:
                    name += ",Regime"
                if use_trend:
                    name += ",Trend"
                name += ")"
                
                strategies.append((
                    FundingArbitrageStrategy(
                        symbols=[symbol],
                        entry_threshold=entry_thresh,
                        exit_threshold=0.5,
                        min_hold_hours=24,
                        max_hold_hours=168,
                        position_size=0.15,
                        use_price_confirmation=True,
                        bb_confirmation_threshold=1.5,
                        use_regime_filter=use_regime,
                        use_trend_filter=use_trend,
                    ),
                    name,
                    {
                        'entry_threshold': entry_thresh,
                        'use_regime_filter': use_regime,
                        'use_trend_filter': use_trend,
                    }
                ))
    
    return strategies


def get_oi_strategies(symbol: str) -> List[tuple]:
    """Get OI-based strategies with regime filter."""
    strategies = []
    
    for use_regime in [True, False]:
        name = "OIDivergence"
        if use_regime:
            name += "(Regime)"
        else:
            name += "(NoRegime)"
        
        strategies.append((
            OIDivergenceStrategy(
                symbols=[symbol],
                divergence_threshold=0.5,
                use_oi_zscore_filter=True,
                oi_zscore_threshold=1.5,
                position_size=0.15,
                min_hold_hours=12,
                max_hold_hours=72,
                use_regime_filter=use_regime,
            ),
            name,
            {'use_regime_filter': use_regime}
        ))
    
    return strategies


def get_combined_strategies(symbol: str) -> List[tuple]:
    """Get combined strategies with different factor requirements."""
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
                    position_size=0.15,
                    min_hold_hours=24,
                    max_hold_hours=120,
                    use_regime_filter=True,  # Always use regime filter
                ),
                name,
                {
                    'min_factors': min_factors,
                    'funding_threshold': funding_thresh,
                }
            ))
    
    return strategies


def analyze_coin(
    symbol: str,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cost_model: CostModel,
) -> List[CoinResult]:
    """Run all strategies on a single coin."""
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
            
            logger.info(f"  {name}: Return={metrics.total_return*100:.2f}%, "
                       f"Sharpe={metrics.sharpe_ratio:.2f}, "
                       f"StopLosses={metrics.num_stop_losses}")
            
        except Exception as e:
            logger.warning(f"  Error running {name}: {e}")
    
    return results


def print_coin_results(symbol: str, results: List[CoinResult]):
    """Print results for a single coin."""
    if not results:
        print(f"  No results for {symbol}")
        return
    
    # Sort by Sharpe ratio
    results = sorted(results, key=lambda x: x.metrics.sharpe_ratio, reverse=True)
    
    print(f"\n  {'Strategy':<40} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'Trades':>7} {'Win%':>6} {'Stops':>6} {'Liq':>5}")
    print(f"  {'-'*100}")
    
    for r in results[:15]:  # Top 15
        m = r.metrics
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"  {r.strategy_name:<40} {m.total_return*100:>8.2f}% {sharpe_str:>8} "
              f"{m.max_drawdown*100:>7.2f}% {m.num_trades:>7} {m.win_rate*100:>5.1f}% "
              f"{m.num_stop_losses:>6} {m.num_liquidations:>5}")
    
    # Best strategy summary
    best = results[0]
    print(f"\n  ✅ BEST: {best.strategy_name}")
    print(f"     Params: {best.params}")
    print(f"     Sharpe: {best.metrics.sharpe_ratio:.2f}, Return: {best.metrics.total_return*100:.2f}%")
    print(f"     Trades: {best.metrics.num_trades}, Win Rate: {best.metrics.win_rate*100:.1f}%")
    print(f"     Stop Losses: {best.metrics.num_stop_losses}, Liquidations: {best.metrics.num_liquidations}")
    print(f"     Fees Paid: ${best.metrics.total_fees_paid:.2f}, Funding Paid: ${best.metrics.total_funding_paid:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Per-coin backtest analysis (FIXED)")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--symbols", type=str, nargs="+", help="Specific symbols to test")
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop loss percentage (default 15%)")
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    
    print("=" * 100)
    print("🎯 PER-COIN BACKTEST ANALYSIS (FIXED VERSION)")
    print("=" * 100)
    print("\nKey fixes applied:")
    print("  ✓ Stop-loss enforcement (default 15%)")
    print("  ✓ Liquidation simulation (5% maintenance margin)")
    print("  ✓ Regime filter (avoids shorting in bull markets)")
    print("  ✓ Trend filter (confirms reversal signals)")
    print("  ✓ Leverage limits (max 3x)")
    print("  ✓ Funding rate application to positions")
    print()
    
    # Cost model with Coinbase fees
    cost_model = CostModel(
        maker_fee_bps=10,    # 0.10%
        taker_fee_bps=10,    # 0.10%
        base_slippage_bps=2, # 0.02%
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
        print(f"   Looking in: {features_dir}")
        return
    
    print(f"Symbols to analyze: {symbols}\n")
    
    all_results = {}
    
    for symbol in symbols:
        print("=" * 100)
        print(f"📊 {symbol}")
        print("=" * 100)
        
        # Load data
        features = load_features(features_dir, symbol)
        if features is None:
            print(f"  ❌ No features found for {symbol}")
            continue
        
        # Debug: print feature columns and index info
        print(f"  Features loaded: {len(features)} rows, {len(features.columns)} columns")
        print(f"  Index type: {type(features.index).__name__}, range: {features.index.min()} to {features.index.max()}")
        
        ohlcv = load_ohlcv(args.db_path, symbol)
        if ohlcv is None:
            print(f"  ❌ No OHLCV data found for {symbol}")
            continue
        
        # Align data
        ohlcv, features = align_data(ohlcv, features)
        if ohlcv is None:
            print(f"  ❌ No overlapping data for {symbol}")
            continue
        
        print(f"  Data: {ohlcv.index.min().date()} to {ohlcv.index.max().date()} ({len(ohlcv)} bars)")
        
        available = check_features(features)
        print(f"  Features: Funding={available['funding']}, OI={available['oi']}, Price={available['price']}")
        
        # Run analysis
        results = analyze_coin(symbol, ohlcv, features, cost_model)
        all_results[symbol] = results
        
        # Print results
        print_coin_results(symbol, results)
    
    # Summary across all coins
    print("\n" + "=" * 100)
    print("📈 SUMMARY ACROSS ALL COINS")
    print("=" * 100)
    
    # Find best strategy per coin
    print(f"\n{'Symbol':<25} {'Best Strategy':<45} {'Return':>10} {'Sharpe':>8}")
    print("-" * 95)
    
    for symbol, results in all_results.items():
        if results:
            best = max(results, key=lambda x: x.metrics.sharpe_ratio)
            print(f"{symbol:<25} {best.strategy_name:<45} "
                  f"{best.metrics.total_return*100:>9.2f}% {best.metrics.sharpe_ratio:>8.2f}")
    
    # Overall statistics
    all_sharpes = []
    all_returns = []
    for results in all_results.values():
        for r in results:
            if abs(r.metrics.sharpe_ratio) < 100:  # Filter valid Sharpes
                all_sharpes.append(r.metrics.sharpe_ratio)
            all_returns.append(r.metrics.total_return)
    
    if all_sharpes:
        print(f"\nOverall Statistics (all strategies, all coins):")
        print(f"  Average Sharpe: {sum(all_sharpes)/len(all_sharpes):.2f}")
        print(f"  Best Sharpe: {max(all_sharpes):.2f}")
        print(f"  Median Return: {sorted(all_returns)[len(all_returns)//2]*100:.2f}%")
        print(f"  % Profitable: {sum(1 for r in all_returns if r > 0)/len(all_returns)*100:.1f}%")


if __name__ == "__main__":
    main()