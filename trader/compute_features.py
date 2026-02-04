#!/usr/bin/env python3
"""
Feature Engineering Test Script (v3 - With OI Support)

This script:
1. Loads collected OHLCV data
2. Loads funding rate data
3. Loads open interest data (NEW)
4. Computes all features from design.md
5. Shows feature statistics and correlations
6. Prepares a sample ML dataset
7. Exports features to CSV

Run: python compute_features.py
"""

import sys
import numpy as np
import pandas as pd
import warnings
import sqlite3

from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from data_collection.storage import SQLiteDatabase
from features.engineering import (
    FeaturePipeline,
    FeatureConfig,
)

DB_PATH = "./data/trading.db"
EXPORT_DIR = Path("./data/features")


def load_ohlcv_data(db: SQLiteDatabase, symbols: list, timeframe: str = "1h") -> dict:
    """Load OHLCV data for multiple symbols."""
    end = datetime.utcnow()
    start = end - timedelta(days=2190)  # Get all available data
    
    data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            data[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
    
    return data


def load_funding_data(db: SQLiteDatabase, symbols: list) -> dict:
    """Load funding rate data from database."""
    funding_data = {}
    
    for symbol in symbols:
        try:
            funding_df = db.get_funding_rates(symbol)
            
            if not funding_df.empty:
                funding_df = funding_df[['rate']]  # Keep only rate column
                funding_df = funding_df.sort_index()
                funding_data[symbol] = funding_df
                print(f"  Loaded funding for {symbol}: {len(funding_df)} rates")
            else:
                print(f"  No funding data for {symbol}")
        except Exception as e:
            print(f"  Error loading funding for {symbol}: {e}")
    
    return funding_data


def load_oi_data(db: SQLiteDatabase, symbols: list) -> dict:
    """Load open interest data from database."""
    oi_data = {}
    end = datetime.utcnow()
    start = end - timedelta(days=2190)
    
    for symbol in symbols:
        try:
            oi_df = db.get_open_interest(symbol, start, end)
            
            if not oi_df.empty:
                oi_df = oi_df.sort_index()
                oi_data[symbol] = oi_df
                print(f"  Loaded OI for {symbol}: {len(oi_df)} records ({oi_df.index.min().date()} to {oi_df.index.max().date()})")
            else:
                print(f"  No OI data for {symbol}")
        except Exception as e:
            print(f"  Error loading OI for {symbol}: {e}")
    
    return oi_data


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
    print("🔧 FEATURE ENGINEERING - Crypto Trading System (with OI)")
    print("=" * 70)
    
    # Initialize
    db = SQLiteDatabase(DB_PATH)
    db.initialize()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("\n1️⃣  LOADING DATA")
    print("-" * 50)
    
    # Get available symbols from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = '1h'")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Available symbols: {symbols}")
    
    # Load OHLCV
    print("\n  Loading OHLCV data...")
    ohlcv_data = load_ohlcv_data(db, symbols, "1h")
    
    if not ohlcv_data:
        print("No OHLCV data available! Run backfill first.")
        return
    
    # Load funding data
    print("\n  Loading funding data...")
    funding_data = load_funding_data(db, list(ohlcv_data.keys()))
    
    # Load OI data (NEW)
    print("\n  Loading open interest data...")
    oi_data = load_oi_data(db, list(ohlcv_data.keys()))
    
    # 2. Compute features
    print("\n2️⃣  COMPUTING FEATURES")
    print("-" * 50)
    
    config = FeatureConfig(
        price_lookbacks=[1, 4, 12, 24, 48, 168],
        volume_lookbacks=[1, 4, 12, 24, 48],
        normalize_features=True,
        compute_price=True,
        compute_volume=True,
        compute_cross_asset=len(ohlcv_data) > 1,
        compute_regime=True,
        compute_funding=True,
        compute_oi=True,  # Enable OI features
    )
    
    pipeline = FeaturePipeline(config)
    
    # Find reference symbol (prefer BTC)
    ref_symbol = None
    for s in ['BTC-PERP', 'BIP-20DEC30-CDE']:
        if s in ohlcv_data:
            ref_symbol = s
            break
    if not ref_symbol:
        ref_symbol = list(ohlcv_data.keys())[0]
    
    # Compute features
    all_features = pipeline.compute_features(
        ohlcv_data,
        funding_data=funding_data,
        oi_data=oi_data,
        reference_symbol=ref_symbol
    )
    
    for symbol, features in all_features.items():
        print(f"  {symbol}: {features.shape[1]} features, {features.shape[0]} rows")
    
    # 3. Feature coverage analysis
    print("\n3️⃣  FEATURE COVERAGE ANALYSIS")
    print("-" * 50)
    
    example_symbol = ref_symbol
    example_features = all_features[example_symbol]
    
    coverage = analyze_feature_coverage(example_features, example_symbol)
    
    # Show features with good coverage (>80%)
    good_coverage = coverage[coverage['coverage_pct'] >= 80]
    poor_coverage = coverage[coverage['coverage_pct'] < 80]
    
    print(f"\n{example_symbol}:")
    print(f"  Features with >=80% coverage: {len(good_coverage)}")
    print(f"  Features with <80% coverage: {len(poor_coverage)}")
    
    # Show feature categories
    all_cols = example_features.columns.tolist()
    funding_cols = [c for c in all_cols if 'funding' in c.lower()]
    oi_cols = [c for c in all_cols if 'oi' in c.lower() or 'open_interest' in c.lower() or 'liquidation' in c.lower()]
    price_cols = [c for c in all_cols if any(x in c.lower() for x in ['return', 'rsi', 'macd', 'bb_', 'ma_', 'volatility'])]
    
    print(f"\n  Feature categories:")
    print(f"    - Price/Technical: {len(price_cols)}")
    print(f"    - Funding: {len(funding_cols)}")
    print(f"    - Open Interest: {len(oi_cols)}")
    
    # Show worst coverage features
    print(f"\n  Lowest coverage features (need more history):")
    for feat, row in coverage.tail(10).iterrows():
        print(f"    {feat:<45} {row['coverage_pct']:>5.1f}%")
    
    # 4. Feature statistics
    print("\n4️⃣  FEATURE STATISTICS (features with >=80% coverage)")
    print("-" * 50)
    
    good_feature_names = good_coverage.index.tolist()
    numeric_cols = example_features[good_feature_names].select_dtypes(include=[np.number]).columns
    
    # Key features including OI
    key_features = [
        'return_1h', 'return_24h', 'volatility_24h', 'rsi_14', 'macd_hist', 
        'bb_position', 'volume_ratio_24h', 'trend_strength',
        'vol_regime_ratio', 'drawdown_from_high', 'range_position_24h',
        # Funding features
        'funding_rate_zscore', 'funding_rate_bps', 'cumulative_funding_24h',
        # OI features
        'open_interest', 'oi_change_24h', 'oi_zscore', 'liquidation_cascade_score',
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
    
    # 5. OI-specific analysis
    if oi_cols:
        print("\n5️⃣  OPEN INTEREST FEATURE ANALYSIS")
        print("-" * 50)
        
        for symbol, feat_df in all_features.items():
            oi_features = [c for c in feat_df.columns if 'oi' in c.lower() or 'open_interest' in c.lower() or 'liquidation' in c.lower()]
            
            if not oi_features:
                print(f"\n{symbol}: No OI features (need to run backfill with --include-oi)")
                continue
            
            print(f"\n{symbol}:")
            print(f"  OI features available: {len(oi_features)}")
            
            if 'oi_zscore' in feat_df.columns:
                zscore = feat_df['oi_zscore'].dropna()
                if len(zscore) > 0:
                    print(f"\n  oi_zscore:")
                    print(f"    Mean: {zscore.mean():.4f}")
                    print(f"    Std:  {zscore.std():.4f}")
                    print(f"    % > 2.0 (high OI): {(zscore > 2.0).mean()*100:.2f}%")
                    print(f"    % < -2.0 (low OI): {(zscore < -2.0).mean()*100:.2f}%")
            
            if 'oi_change_24h' in feat_df.columns:
                change = feat_df['oi_change_24h'].dropna() * 100
                if len(change) > 0:
                    print(f"\n  oi_change_24h (%):")
                    print(f"    Mean: {change.mean():.2f}%")
                    print(f"    Max:  {change.max():.2f}%")
                    print(f"    Min:  {change.min():.2f}%")
            
            if 'liquidation_cascade_score' in feat_df.columns:
                cascade = feat_df['liquidation_cascade_score'].dropna()
                if len(cascade) > 0:
                    print(f"\n  liquidation_cascade_score:")
                    print(f"    Score >= 2 events: {(cascade >= 2).sum()} ({(cascade >= 2).mean()*100:.2f}%)")
    
    # 6. Feature correlations with forward returns
    print("\n6️⃣  FEATURE CORRELATIONS WITH FORWARD RETURNS")
    print("-" * 50)
    
    example_df = ohlcv_data[example_symbol]
    target_1h = pipeline.compute_target(example_df, horizon=1, target_type='return')
    target_24h = pipeline.compute_target(example_df, horizon=24, target_type='return')
    
    correlations_1h = {}
    correlations_24h = {}
    
    for col in numeric_cols:
        feat_series = example_features[col].dropna()
        
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
            # Highlight OI and funding features
            tag = ""
            if 'oi' in feat.lower() or 'open_interest' in feat.lower():
                tag = "[OI] "
            elif 'funding' in feat.lower():
                tag = "[FR] "
            print(f"  {direction} {tag}{feat:<42} IC: {corr:+.4f} {strength}")
    
    if correlations_24h:
        print(f"\nTop 20 features by |IC| with 24h forward return ({example_symbol}):")
        sorted_corr = sorted(correlations_24h.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        for feat, corr in sorted_corr:
            direction = "📈" if corr > 0 else "📉"
            strength = "***" if abs(corr) > 0.05 else "**" if abs(corr) > 0.03 else "*" if abs(corr) > 0.02 else ""
            tag = ""
            if 'oi' in feat.lower() or 'open_interest' in feat.lower():
                tag = "[OI] "
            elif 'funding' in feat.lower():
                tag = "[FR] "
            print(f"  {direction} {tag}{feat:<42} IC: {corr:+.4f} {strength}")
    
    print("\n  Legend: *** IC>0.05 (strong), ** IC>0.03 (moderate), * IC>0.02 (weak)")
    print("  Tags: [OI] = Open Interest, [FR] = Funding Rate")
    
    # 7. Prepare ML dataset
    print("\n7️⃣  PREPARING ML DATASET")
    print("-" * 50)
    
    ml_features = example_features[good_feature_names].select_dtypes(include=[np.number])
    X, y = pipeline.prepare_ml_dataset(ml_features, target_1h)
    
    print(f"  Using {len(good_feature_names)} features with >=80% coverage")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    if len(y) > 0:
        print(f"  Target stats: mean={y.mean()*100:.4f}%, std={y.std()*100:.4f}%")
        print(f"  Target distribution: {(y > 0).sum()} up ({(y > 0).mean()*100:.1f}%), "
              f"{(y < 0).sum()} down ({(y < 0).mean()*100:.1f}%)")
    
    # 8. Export features
    print("\n8️⃣  EXPORTING FEATURES")
    print("-" * 50)
    
    for symbol, features in all_features.items():
        # Export full features
        filename = EXPORT_DIR / f"{symbol.replace('-', '_').replace('/', '_')}_features.csv"
        features.to_csv(filename)
        print(f"  Exported: {filename}")
        
        # Export ML-ready dataset
        symbol_df = ohlcv_data[symbol]
        target = pipeline.compute_target(symbol_df, horizon=1, target_type='return')
        
        good_cols = [c for c in features.columns 
                    if features[c].notna().sum() / len(features) >= 0.8]
        ml_feats = features[good_cols].select_dtypes(include=[np.number])
        
        X, y = pipeline.prepare_ml_dataset(ml_feats, target)
        
        if len(X) > 0:
            ml_dataset = X.copy()
            ml_dataset['target_return_1h'] = y
            
            filename_ml = EXPORT_DIR / f"{symbol.replace('-', '_').replace('/', '_')}_ml_dataset.csv"
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
    
    # Data availability
    print(f"\nData availability:")
    print(f"  OHLCV: {len(ohlcv_data)} symbols")
    print(f"  Funding: {len(funding_data)} symbols")
    print(f"  Open Interest: {len(oi_data)} symbols")
    
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
    print("  2. If OI features are missing, run: python run_pipeline.py --backfill-days 365 --include-oi")
    print("  3. Build baseline model with top features")
    print("  4. Implement walk-forward validation")
    
    db.close()


if __name__ == "__main__":
    main()