#!/usr/bin/env python3
"""
Feature Engineering Test Script (v2 - Fixed)

This script:
1. Loads collected OHLCV data
2. Computes all features from design.md
3. Shows feature statistics and correlations
4. Prepares a sample ML dataset
5. Exports features to CSV

Run: python compute_features.py
"""

import sys
import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from datetime import datetime, timedelta

from data_collection.storage import SQLiteDatabase
from features.engineering import (
    FeaturePipeline,
    FeatureConfig,
)

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH = "./data/trading.db"
EXPORT_DIR = Path("./data/features")


def load_ohlcv_data(db: SQLiteDatabase, symbols: list, timeframe: str = "1h") -> dict:
    """Load OHLCV data for multiple symbols."""
    end = datetime.utcnow()
    start = end - timedelta(days=365)  # Get all available data
    
    data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            data[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
    
    return data


def analyze_feature_coverage(features: pd.DataFrame, name: str) -> pd.DataFrame:
    """Analyze NaN coverage for features."""
    coverage = pd.DataFrame({
        'non_null': features.notna().sum(),
        'null': features.isna().sum(),
        'coverage_pct': (features.notna().sum() / len(features) * 100).round(1)
    })
    return coverage.sort_values('coverage_pct', ascending=False)


def main():
    print("=" * 70)
    print("🔧 FEATURE ENGINEERING - Crypto Trading System")
    print("=" * 70)
    
    # Initialize
    db = SQLiteDatabase(DB_PATH)
    db.initialize()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("\n1️⃣  LOADING DATA")
    print("-" * 50)
    
    # Get available symbols from database
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = '1h'")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Available symbols: {symbols}")
    
    ohlcv_data = load_ohlcv_data(db, symbols, "1h")
    
    if not ohlcv_data:
        print("No data available! Run backfill first.")
        return
    
    # 2. Compute features
    print("\n2️⃣  COMPUTING FEATURES")
    print("-" * 50)
    
    config = FeatureConfig(
        price_lookbacks=[1, 4, 12, 24, 48, 168],
        volume_lookbacks=[1, 4, 12, 24, 48],
        normalize_features=True,
        compute_price=True,
        compute_volume=True,
        compute_derivatives=True,
        compute_cross_asset=len(ohlcv_data) > 1,
        compute_regime=True,
    )
    
    pipeline = FeaturePipeline(config)
    
    # Compute features
    all_features = pipeline.compute_features(
        ohlcv_data,
        reference_symbol='BTC-PERP' if 'BTC-PERP' in ohlcv_data else list(ohlcv_data.keys())[0]
    )
    
    for symbol, features in all_features.items():
        print(f"  {symbol}: {features.shape[1]} features, {features.shape[0]} rows")
    
    # 3. Feature coverage analysis
    print("\n3️⃣  FEATURE COVERAGE ANALYSIS")
    print("-" * 50)
    
    example_symbol = 'BTC-PERP' if 'BTC-PERP' in all_features else list(all_features.keys())[0]
    example_features = all_features[example_symbol]
    
    coverage = analyze_feature_coverage(example_features, example_symbol)
    
    # Show features with good coverage (>80%)
    good_coverage = coverage[coverage['coverage_pct'] >= 80]
    poor_coverage = coverage[coverage['coverage_pct'] < 80]
    
    print(f"\n{example_symbol}:")
    print(f"  Features with >=80% coverage: {len(good_coverage)}")
    print(f"  Features with <80% coverage: {len(poor_coverage)}")
    
    # Show worst coverage features
    print(f"\n  Lowest coverage features (need more history):")
    for feat, row in coverage.tail(10).iterrows():
        print(f"    {feat:<45} {row['coverage_pct']:>5.1f}%")
    
    # 4. Feature statistics (using features with good coverage)
    print("\n4️⃣  FEATURE STATISTICS (features with >=80% coverage)")
    print("-" * 50)
    
    # Select only numeric columns with good coverage
    good_feature_names = good_coverage.index.tolist()
    numeric_cols = example_features[good_feature_names].select_dtypes(include=[np.number]).columns
    
    # Use rows where we have data (not requiring ALL features to be non-null)
    # Instead, compute stats per-feature
    key_features = [
        'return_1h', 'return_24h', 'volatility_24h', 'rsi_14', 'macd_hist', 
        'bb_position', 'volume_ratio_24h', 'trend_strength',
        'vol_regime_ratio', 'drawdown_from_high', 'range_position_24h'
    ]
    
    available_key_features = [f for f in key_features if f in numeric_cols]
    
    if available_key_features:
        print(f"\nKey feature statistics for {example_symbol}:")
        stats_data = []
        for feat in available_key_features:
            series = example_features[feat].dropna()
            if len(series) > 0:
                stats_data.append({
                    'feature': feat,
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'skew': series.skew()
                })
        
        stats_df = pd.DataFrame(stats_data).set_index('feature')
        print(stats_df.round(4).to_string())
    
    # 5. Feature correlations with forward returns
    print("\n5️⃣  FEATURE CORRELATIONS WITH FORWARD RETURNS")
    print("-" * 50)
    
    # Compute target (1h and 24h forward return)
    example_df = ohlcv_data[example_symbol]
    target_1h = pipeline.compute_target(example_df, horizon=1, target_type='return')
    target_24h = pipeline.compute_target(example_df, horizon=24, target_type='return')
    
    # Compute IC for each feature individually (not requiring all features to be non-null)
    correlations_1h = {}
    correlations_24h = {}
    
    for col in numeric_cols:
        # Get non-null values for this feature
        feat_series = example_features[col].dropna()
        
        # Align with targets
        common_idx_1h = feat_series.index.intersection(target_1h.dropna().index)
        common_idx_24h = feat_series.index.intersection(target_24h.dropna().index)
        
        if len(common_idx_1h) > 100:
            corr = feat_series.loc[common_idx_1h].corr(target_1h.loc[common_idx_1h])
            if not np.isnan(corr):
                correlations_1h[col] = corr
        
        if len(common_idx_24h) > 100:
            corr = feat_series.loc[common_idx_24h].corr(target_24h.loc[common_idx_24h])
            if not np.isnan(corr):
                correlations_24h[col] = corr
    
    if correlations_1h:
        print(f"\nTop 20 features by |IC| with 1h forward return ({example_symbol}):")
        sorted_corr = sorted(correlations_1h.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        for feat, corr in sorted_corr:
            direction = "📈" if corr > 0 else "📉"
            strength = "***" if abs(corr) > 0.05 else "**" if abs(corr) > 0.03 else "*" if abs(corr) > 0.02 else ""
            print(f"  {direction} {feat:<45} IC: {corr:+.4f} {strength}")
    
    if correlations_24h:
        print(f"\nTop 20 features by |IC| with 24h forward return ({example_symbol}):")
        sorted_corr = sorted(correlations_24h.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        for feat, corr in sorted_corr:
            direction = "📈" if corr > 0 else "📉"
            strength = "***" if abs(corr) > 0.05 else "**" if abs(corr) > 0.03 else "*" if abs(corr) > 0.02 else ""
            print(f"  {direction} {feat:<45} IC: {corr:+.4f} {strength}")
    
    print("\n  Legend: *** IC>0.05 (strong), ** IC>0.03 (moderate), * IC>0.02 (weak)")
    
    # 6. Prepare ML dataset (using subset of features with good coverage)
    print("\n6️⃣  PREPARING ML DATASET")
    print("-" * 50)
    
    # Use only features with good coverage for ML
    ml_features = example_features[good_feature_names].select_dtypes(include=[np.number])
    
    # Prepare dataset
    X, y = pipeline.prepare_ml_dataset(ml_features, target_1h)
    
    print(f"  Using {len(good_feature_names)} features with >=80% coverage")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    if len(y) > 0:
        print(f"  Target stats: mean={y.mean()*100:.4f}%, std={y.std()*100:.4f}%")
        print(f"  Target distribution: {(y > 0).sum()} up ({(y > 0).mean()*100:.1f}%), "
              f"{(y < 0).sum()} down ({(y < 0).mean()*100:.1f}%)")
    
    # 7. Cross-asset features summary
    if len(all_features) > 1:
        print("\n7️⃣  CROSS-ASSET FEATURES SUMMARY")
        print("-" * 50)
        
        print(f"\n{'Symbol':<12} {'Beta':<8} {'Corr':<8} {'RelStr24h':<12} {'VolRatio':<10}")
        print("-" * 50)
        
        for symbol, features in all_features.items():
            beta = features.get('beta_to_BTC-PERP', pd.Series([np.nan])).dropna().mean()
            corr = features.get('corr_to_BTC-PERP', pd.Series([np.nan])).dropna().mean()
            rel_str = features.get('relative_strength_24h', pd.Series([np.nan])).dropna().mean()
            vol_ratio = features.get('vol_ratio_vs_btc', pd.Series([np.nan])).dropna().mean()
            
            if symbol == 'BTC-PERP':
                print(f"{symbol:<12} {'(ref)':<8} {'1.000':<8} {'0.000':<12} {'1.000':<10}")
            else:
                print(f"{symbol:<12} {beta:<8.3f} {corr:<8.3f} {rel_str*100:<12.2f}% {vol_ratio:<10.3f}")
    
    # 8. Export features
    print("\n8️⃣  EXPORTING FEATURES")
    print("-" * 50)
    
    for symbol, features in all_features.items():
        # Export full features
        filename = EXPORT_DIR / f"{symbol.replace('-', '_')}_features.csv"
        features.to_csv(filename)
        print(f"  Exported: {filename}")
        
        # Export ML-ready dataset (only good coverage features)
        symbol_df = ohlcv_data[symbol]
        target = pipeline.compute_target(symbol_df, horizon=1, target_type='return')
        
        good_cols = [c for c in features.columns 
                    if features[c].notna().sum() / len(features) >= 0.8]
        ml_feats = features[good_cols].select_dtypes(include=[np.number])
        
        X, y = pipeline.prepare_ml_dataset(ml_feats, target)
        
        if len(X) > 0:
            ml_dataset = X.copy()
            ml_dataset['target_return_1h'] = y
            
            filename_ml = EXPORT_DIR / f"{symbol.replace('-', '_')}_ml_dataset.csv"
            ml_dataset.to_csv(filename_ml)
            print(f"  Exported: {filename_ml} ({len(ml_dataset)} samples, {ml_dataset.shape[1]-1} features)")
    
    # 9. Summary
    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    
    print(f"\nData summary:")
    print(f"  Symbols: {len(all_features)}")
    print(f"  Total features computed: {example_features.shape[1]}")
    print(f"  Features with good coverage (>=80%): {len(good_feature_names)}")
    print(f"  ML-ready samples per symbol: ~{len(X)}")
    
    # Key findings
    if correlations_1h:
        top_ic = max(abs(v) for v in correlations_1h.values())
        print(f"\n  Highest |IC| (1h): {top_ic:.4f}")
        if top_ic > 0.03:
            print("  ✅ Some predictive signal detected!")
        else:
            print("  ⚠️  Weak predictive signal - may need more features or different approach")
    
    print(f"\nExported to: {EXPORT_DIR.absolute()}")
    
    print("\n📋 NEXT STEPS:")
    print("  1. Review features with IC > 0.02 - these have predictive value")
    print("  2. Build baseline model with top features")
    print("  3. Implement walk-forward validation")
    print("  4. Consider adding funding rate data for stronger signals")
    
    db.close()


if __name__ == "__main__":
    main()