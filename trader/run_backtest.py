#!/usr/bin/env python3
"""
Backtest Runner - Loads Pre-Computed Features

This script:
1. Loads pre-computed features from CSV files (from compute_features.py)
2. Loads OHLCV data from database for price reference
3. Runs multiple strategies and compares results

Prerequisites:
1. Run compute_features.py to generate feature CSVs
2. Features should be in data/features/<SYMBOL>_features.csv

Usage:
    python run_backtest.py
    python run_backtest.py --features-dir ./data/features
    python run_backtest.py --symbols BTC-PERP ETH-PERP
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.storage import SQLiteDatabase
from backtesting.engine import Backtester, CostModel
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

# Default paths
DEFAULT_DB_PATH = "./data/trading.db"
DEFAULT_FEATURES_DIR = "./data/features"


def find_feature_files(features_dir: Path) -> Dict[str, Path]:
    """Find all feature CSV files in directory."""
    feature_files = {}
    
    # Look for *_features.csv files
    for f in features_dir.glob("*_features.csv"):
        # Extract symbol from filename (e.g., BTC_PERP_features.csv -> BTC-PERP)
        symbol = f.stem.replace("_features", "").replace("_", "-")
        feature_files[symbol] = f
    
    return feature_files


def load_features_from_csv(features_dir: Path, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load pre-computed features from CSV files."""
    features_dir = Path(features_dir)
    
    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        logger.error("Please run compute_features.py first to generate features.")
        return {}
    
    feature_files = find_feature_files(features_dir)
    
    if not feature_files:
        logger.error(f"No feature files found in {features_dir}")
        logger.error("Expected files like: BTC_PERP_features.csv")
        return {}
    
    logger.info(f"Found {len(feature_files)} feature files:")
    for symbol, path in feature_files.items():
        logger.info(f"  {symbol}: {path.name}")
    
    # Filter by requested symbols if specified
    if symbols:
        feature_files = {s: p for s, p in feature_files.items() if s in symbols}
    
    # Load features
    features = {}
    for symbol, filepath in feature_files.items():
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            features[symbol] = df
            logger.info(f"  Loaded {symbol}: {len(df)} rows, {len(df.columns)} features")
        except Exception as e:
            logger.error(f"  Failed to load {symbol}: {e}")
    
    return features


def load_ohlcv_from_db(db: SQLiteDatabase, symbols: List[str], timeframe: str = "1h") -> Dict[str, pd.DataFrame]:
    """Load OHLCV data from database."""
    end = datetime.utcnow()
    start = end - timedelta(days=1000)  # Get all available
    
    ohlcv_data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            ohlcv_data[symbol] = df
            logger.info(f"  Loaded OHLCV {symbol}: {len(df)} bars")
    
    return ohlcv_data


def align_data(ohlcv_data: Dict[str, pd.DataFrame], features: Dict[str, pd.DataFrame]) -> tuple:
    """Ensure OHLCV and features are aligned by timestamp."""
    aligned_ohlcv = {}
    aligned_features = {}
    
    for symbol in features.keys():
        if symbol not in ohlcv_data:
            logger.warning(f"No OHLCV data for {symbol}, skipping")
            continue
        
        feat_df = features[symbol]
        ohlcv_df = ohlcv_data[symbol]
        
        # Find common timestamps
        common_idx = feat_df.index.intersection(ohlcv_df.index)
        
        if len(common_idx) == 0:
            logger.warning(f"No overlapping timestamps for {symbol}")
            continue
        
        aligned_features[symbol] = feat_df.loc[common_idx]
        aligned_ohlcv[symbol] = ohlcv_df.loc[common_idx]
        
        logger.info(f"  Aligned {symbol}: {len(common_idx)} common timestamps")
    
    return aligned_ohlcv, aligned_features


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


def print_results_table(results: List[tuple]):
    """Print formatted results table."""
    print(f"{'Strategy':<40} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    print("-" * 85)
    
    for name, m in sorted(results, key=lambda x: x[1].sharpe_ratio, reverse=True):
        sharpe_str = f"{m.sharpe_ratio:.2f}" if abs(m.sharpe_ratio) < 100 else "N/A"
        print(f"{name:<40} {m.total_return*100:>8.2f}% {sharpe_str:>8} "
              f"{m.max_drawdown*100:>7.2f}% {m.num_trades:>7} {m.win_rate*100:>5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run backtests with pre-computed features")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Database path")
    parser.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR, help="Features directory")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to backtest (default: all)")
    parser.add_argument("--initial-capital", type=float, default=100000, help="Initial capital")
    args = parser.parse_args()
    
    print("=" * 85)
    print("🎯 BACKTEST RUNNER - Pre-Computed Features")
    print("=" * 85)
    
    # =========================================================================
    # 1. Load Pre-Computed Features
    # =========================================================================
    print("📊 Loading pre-computed features...")
    features = load_features_from_csv(Path(args.features_dir), args.symbols)
    
    if not features:
        print("❌ No features loaded! Please run compute_features.py first:")
        print("   python compute_features.py")
        return
    
    symbols = list(features.keys())
    print(f"Symbols to backtest: {symbols}")
    
    # =========================================================================
    # 2. Load OHLCV Data
    # =========================================================================
    print("📈 Loading OHLCV data from database...")
    db = SQLiteDatabase(args.db_path)
    db.initialize()
    
    ohlcv_data = load_ohlcv_from_db(db, symbols)
    
    if not ohlcv_data:
        print("❌ No OHLCV data loaded!")
        db.close()
        return
    
    # =========================================================================
    # 3. Align Data
    # =========================================================================
    print("🔗 Aligning OHLCV and features...")
    ohlcv_data, features = align_data(ohlcv_data, features)
    
    if not features:
        print("❌ No aligned data!")
        db.close()
        return
    
    # =========================================================================
    # 4. Check Available Features
    # =========================================================================
    print("📋 Available feature categories:")
    sample_symbol = list(features.keys())[0]
    sample_features = features[sample_symbol].columns.tolist()
    
    has_funding = any('funding' in f.lower() for f in sample_features)
    has_oi = any('oi' in f.lower() or 'open_interest' in f.lower() for f in sample_features)
    has_price = any(f in sample_features for f in ['bb_position', 'rsi_14', 'macd'])
    
    print(f"  - Price/Technical features: {'✅' if has_price else '❌'}")
    print(f"  - Funding rate features: {'✅' if has_funding else '❌'}")
    print(f"  - Open Interest features: {'✅' if has_oi else '❌'}")
    print(f"  - Total features: {len(sample_features)}")
    
    # =========================================================================
    # 5. Setup Cost Model
    # =========================================================================
    # Coinbase US Perps: 0.1% per trade
    cost_model = CostModel(
        maker_fee_bps=10,
        taker_fee_bps=10,
        base_slippage_bps=2,
        volatility_slippage_multiplier=1.5,
        size_impact_coefficient=0.02,
    )
    
    # =========================================================================
    # 6. Define Strategies
    # =========================================================================
    print("=" * 85)
    print("📈 RUNNING STRATEGIES")
    print("=" * 85)
    
    strategies = []
    
    # --- Funding-Based Strategies ---
    if has_funding:
        strategies.extend([
            (FundingArbitrageStrategy(
                symbols=symbols,
                entry_threshold=2.0,
                exit_threshold=0.5,
                min_hold_hours=24,
                max_hold_hours=168,
                position_size=0.15,
                use_price_confirmation=True,
            ), "Funding Arb (z>2, BB confirm)"),
            
            (FundingArbitrageStrategy(
                symbols=symbols,
                entry_threshold=1.5,
                exit_threshold=0.3,
                min_hold_hours=24,
                max_hold_hours=168,
                position_size=0.15,
                use_price_confirmation=False,
            ), "Funding Arb (z>1.5, no confirm)"),
            
            (FundingAwareMeanReversion(
                symbols=symbols,
                bb_entry_threshold=1.5,
                bb_exit_threshold=0.2,
                max_funding_zscore=1.5,
                min_funding_zscore=-1.5,
                position_size=0.15,
                min_hold_hours=24,
            ), "Funding-Aware MeanRev"),
            
            (PureFundingCarryStrategy(
                symbols=symbols,
                cumulative_threshold_bps=10.0,
                position_size=0.1,
                hold_hours=72,
            ), "Pure Carry (10bps, 72h)"),
        ])
    
    # --- OI-Based Strategies ---
    if has_oi:
        strategies.extend([
            (OIDivergenceStrategy(
                symbols=symbols,
                divergence_threshold=0.5,
                use_oi_zscore_filter=True,
                oi_zscore_threshold=1.5,
                use_liquidation_filter=False,
                position_size=0.15,
                min_hold_hours=12,
                max_hold_hours=72,
            ), "OI Divergence (div>0.5)"),
            
            (OIDivergenceStrategy(
                symbols=symbols,
                divergence_threshold=0.3,
                use_oi_zscore_filter=False,
                position_size=0.15,
                min_hold_hours=12,
                max_hold_hours=48,
            ), "OI Divergence (div>0.3, no filter)"),
            
            (OIDivergenceStrategy(
                symbols=symbols,
                divergence_threshold=0.7,
                use_oi_zscore_filter=True,
                oi_zscore_threshold=2.0,
                position_size=0.2,
                min_hold_hours=24,
                max_hold_hours=96,
            ), "OI Divergence (div>0.7, strict)"),
        ])
    
    # --- Combined Strategies ---
    if has_funding and has_price:
        strategies.extend([
            (CombinedFundingPriceStrategy(
                symbols=symbols,
                funding_threshold=1.5,
                bb_threshold=1.5,
                rsi_oversold=35,
                rsi_overbought=65,
                require_all_signals=False,
                position_size=0.2,
                min_hold_hours=48,
            ), "Combined FR+Price (2 of 3)"),
        ])
    
    if has_oi and has_funding:
        strategies.extend([
            (CombinedOIFundingStrategy(
                symbols=symbols,
                oi_divergence_threshold=0.3,
                funding_threshold=1.0,
                bb_threshold=1.0,
                min_factors=2,
                position_size=0.2,
                min_hold_hours=24,
                max_hold_hours=120,
            ), "Combined OI+FR+Price (2 of 3)"),
            
            (CombinedOIFundingStrategy(
                symbols=symbols,
                oi_divergence_threshold=0.5,
                funding_threshold=1.5,
                bb_threshold=1.5,
                min_factors=3,
                position_size=0.25,
                min_hold_hours=48,
                max_hold_hours=168,
            ), "Combined OI+FR+Price (all 3)"),
        ])
    
    if not strategies:
        print("❌ No strategies could be configured with available features!")
        db.close()
        return
    
    print(f"Running {len(strategies)} strategies...")
    
    # =========================================================================
    # 7. Run Backtests
    # =========================================================================
    results = []
    
    for strategy, name in strategies:
        try:
            _, metrics, _ = run_backtest(strategy, ohlcv_data, features, name, cost_model)
            results.append((name, metrics))
            
            sharpe_str = f"{metrics.sharpe_ratio:>6.2f}" if abs(metrics.sharpe_ratio) < 100 else "  N/A"
            print(f"  ✓ {name:<40} Return: {metrics.total_return*100:>7.2f}% | "
                  f"Sharpe: {sharpe_str} | Trades: {metrics.num_trades:>4}")
        except Exception as e:
            logger.error(f"  ✗ Error running {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # 8. Results Summary
    # =========================================================================
    print("=" * 85)
    print("📊 STRATEGY COMPARISON")
    print("=" * 85)
    
    print_results_table(results)
    
    # =========================================================================
    # 9. Analysis
    # =========================================================================
    if results:
        profitable = [(n, m) for n, m in results if m.total_return > 0]
        positive_sharpe = [(n, m) for n, m in results if m.sharpe_ratio > 0]
        
        print("=" * 85)
        print("📈 SUMMARY")
        print("=" * 85)
        
        print(f"   Total strategies tested: {len(results)}")
        print(f"   Profitable strategies:   {len(profitable)}/{len(results)}")
        print(f"   Positive Sharpe:         {len(positive_sharpe)}/{len(results)}")
        
        if profitable:
            best = max(profitable, key=lambda x: x[1].sharpe_ratio)
            print(f"   ✅ Best strategy: {best[0]}")
            print(f"      - Return: {best[1].total_return*100:+.2f}%")
            print(f"      - Sharpe: {best[1].sharpe_ratio:.2f}")
            print(f"      - Max DD: {best[1].max_drawdown*100:.2f}%")
            print(f"      - Trades: {best[1].num_trades}")
            print(f"      - Win Rate: {best[1].win_rate*100:.1f}%")
        
        # Category analysis
        print("   📊 Performance by Strategy Type:")
        
        funding_strats = [(n, m) for n, m in results if 'Funding' in n or 'Carry' in n]
        oi_strats = [(n, m) for n, m in results if 'OI' in n]
        combined_strats = [(n, m) for n, m in results if 'Combined' in n]
        
        for category, strats in [("Funding-based", funding_strats), 
                                  ("OI-based", oi_strats), 
                                  ("Combined", combined_strats)]:
            if strats:
                avg_return = np.mean([m.total_return for _, m in strats])
                avg_sharpe = np.mean([m.sharpe_ratio for _, m in strats])
                print(f"      {category}: Avg Return={avg_return*100:.2f}%, Avg Sharpe={avg_sharpe:.2f}")
    
    # =========================================================================
    # 10. Next Steps
    # =========================================================================
    print("=" * 85)
    print("📋 NEXT STEPS")
    print("=" * 85)
    
    print("""
    1. If strategies are profitable:
       - Run walk-forward validation: python run_validation.py
       - Paper trade the best strategy
    
    2. If strategies are not profitable:
       - Check feature quality with compute_features.py
       - Adjust strategy parameters
       - Consider different entry/exit thresholds
    
    3. To improve:
       - Combine top-performing strategies
       - Add position sizing based on volatility
       - Implement regime detection
    """)
    
    db.close()


if __name__ == "__main__":
    main()