#!/usr/bin/env python3
"""
Feature Engineering Script (v8 - Golden Version)
================================================
Combines:
1. Original Working Logic (Stats, Coverage, IC Analysis)
2. Robust Data Cleaning (Fixes 0-row errors)
3. Strategy Alignment (Exports 1.8x Triple Barrier Targets for the Bot)
"""

import os
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
from core.coin_profiles import get_coin_profile, CoinProfile

DB_PATH = "./data/trading.db"
EXPORT_DIR = Path("./data/features")
LOOKBACK_DAYS = int(os.getenv("FEATURE_LOOKBACK_DAYS", "2190"))


# 1. ROBUST LOADERS (UTC Enforced)

def load_ohlcv_data(db: SQLiteDatabase, symbols: list, timeframe: str = "1h") -> dict:
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    data = {}
    for symbol in symbols:
        df = db.get_ohlcv(symbol, timeframe, start, end)
        if not df.empty:
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            data[symbol] = df
            print(f"  Loaded {symbol}: {len(df)} bars")
    return data

def load_funding_data(db: SQLiteDatabase, symbols: list) -> dict:
    funding_data = {}
    for symbol in symbols:
        try:
            df = db.get_funding_rates(symbol)
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                df = df[['rate']].sort_index()
                df = df[~df.index.duplicated(keep='first')]
                funding_data[symbol] = df
                print(f"  Loaded funding for {symbol}: {len(df)} rates")
        except Exception:
            pass
    return funding_data

def load_oi_data(db: SQLiteDatabase, symbols: list) -> dict:
    oi_data = {}
    end = datetime.utcnow()
    start = end - timedelta(days=LOOKBACK_DAYS)
    for symbol in symbols:
        try:
            df = db.get_open_interest(symbol, start, end)
            if not df.empty:
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]
                oi_data[symbol] = df
                print(f"  Loaded OI for {symbol}: {len(df)} records")
        except Exception:
            pass
    return oi_data

# 2. ANALYSIS HELPERS

def analyze_feature_coverage(features: pd.DataFrame) -> pd.DataFrame:
    coverage = pd.DataFrame({
        'non_null': features.notna().sum(),
        'null': features.isna().sum(),
        'coverage_pct': (features.notna().sum() / len(features) * 100).round(1)
    })
    return coverage.sort_values('coverage_pct', ascending=False)

def prepare_robust_dataset(features: pd.DataFrame, target: pd.Series):
    """
    Robust cleaning that prevents '0 rows' errors.
    """
    # 1. Align Indexes
    common = features.index.intersection(target.dropna().index)
    if len(common) == 0:
        # Fallback: Naive alignment if UTC mismatch occurs
        f_idx = features.index.tz_localize(None)
        t_idx = target.index.tz_localize(None)
        common_naive = f_idx.intersection(t_idx)
        if len(common_naive) > 0:
            X, y = features.copy(), target.copy()
            X.index, y.index = f_idx, t_idx
            X, y = X.loc[common_naive], y.loc[common_naive]
        else:
            return pd.DataFrame(), pd.Series()
    else:
        X, y = features.loc[common].copy(), target.loc[common]

    # 2. Drop Sparse Columns (>20% Missing)
    missing_pct = X.isna().mean()
    bad_cols = missing_pct[missing_pct > 0.20].index
    if len(bad_cols) > 0:
        X = X.drop(columns=bad_cols)

    # 3. Aggressive Imputation (Forward Fill -> 0)
    X = X.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    return X, y


def resolve_label_horizon(profile: CoinProfile) -> int:
    """Prefer execution hold horizon; fallback to explicit label horizon."""
    if profile.max_hold_hours and profile.max_hold_hours > 0:
        return int(profile.max_hold_hours)
    return int(profile.label_forward_hours)


def compute_profile_target(ohlcv: pd.DataFrame, profile: CoinProfile) -> pd.Series:
    """Triple-barrier labels with profile TP/SL multipliers and neutral timeout class.

    Label values:
      1  -> take-profit first touch
      -1 -> stop-loss first touch
      0  -> timeout/no touch within horizon
    """
    horizon = resolve_label_horizon(profile)
    target = pd.Series(index=ohlcv.index, dtype=float)
    vol = ohlcv['close'].pct_change().rolling(24).std().ffill()

    for idx in range(len(ohlcv) - horizon):
        ts = ohlcv.index[idx]
        row_vol = vol.iloc[idx]
        if pd.isna(row_vol) or row_vol <= 0:
            continue

        entry = ohlcv['close'].iloc[idx]
        tp = entry * (1 + profile.vol_mult_tp * row_vol)
        sl = entry * (1 - profile.vol_mult_sl * row_vol)
        future = ohlcv.iloc[idx + 1: idx + 1 + horizon]

        label = 0.0
        for _, bar in future.iterrows():
            tp_hit = bar['high'] >= tp
            sl_hit = bar['low'] <= sl
            if tp_hit and not sl_hit:
                label = 1.0
                break
            if sl_hit and not tp_hit:
                label = -1.0
                break
            if tp_hit and sl_hit:
                label = -1.0
                break

        target.iloc[idx] = label

    return target

# 3. MAIN SCRIPT

def main():
    print("=" * 70)
    print("🔧 FEATURE ENGINEERING - 5 Coin Pipeline (Robust)")
    print("=" * 70)
    
    db = SQLiteDatabase(DB_PATH)
    db.initialize()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load Data ---
    print("\n1️⃣  LOADING DATA")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe = '1h'")
    symbols = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Symbols Found: {symbols}")
    
    ohlcv_data = load_ohlcv_data(db, symbols)
    funding_data = load_funding_data(db, list(ohlcv_data.keys()))
    oi_data = load_oi_data(db, list(ohlcv_data.keys()))
    
    if not ohlcv_data:
        print("❌ No data found.")
        return

    # --- 2. Compute Features ---
    print("\n2️⃣  COMPUTING FEATURES")
    config = FeatureConfig(
        price_lookbacks=[1, 4, 12, 24, 48, 168],
        volume_lookbacks=[1, 4, 12, 24, 48],
        compute_price=True,
        compute_volume=True,
        compute_regime=True,
        compute_funding=True,
        compute_oi=True
    )
    pipeline = FeaturePipeline(config)
    
    # Pick Reference (Prefer BTC, else first available)
    ref_symbol = next((s for s in ['BTC-PERP', 'BIP-20DEC30-CDE'] if s in ohlcv_data), list(ohlcv_data.keys())[0])
    
    all_features = pipeline.compute_features(
        ohlcv_data, funding_data, oi_data, reference_symbol=ref_symbol
    )
    
    # Force UTC alignment on features immediately
    for s in all_features:
        if all_features[s].index.tz is None:
            all_features[s].index = all_features[s].index.tz_localize('UTC')
        else:
            all_features[s].index = all_features[s].index.tz_convert('UTC')
            
    for s, df in all_features.items():
        print(f"  {s}: {df.shape[1]} features, {len(df)} rows")

    # --- 3. Analysis (Reference Symbol) ---
    print(f"\n3️⃣  ANALYSIS ({ref_symbol})")
    print("-" * 50)
    
    ref_feats = all_features[ref_symbol]
    
    # A. Stats
    key_feats = ['return_1h', 'rsi_14', 'vol_regime_ratio', 'funding_rate_zscore', 'oi_zscore']
    stats = []
    for k in key_feats:
        if k in ref_feats.columns:
            s = ref_feats[k].dropna()
            stats.append({'feat': k, 'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max()})
    if stats:
        print(pd.DataFrame(stats).set_index('feat').round(4).to_string())

    # B. Correlations (Using Raw Returns for insight)
    print("\n  Feature Correlations (with 24h Forward Returns):")
    raw_target = ohlcv_data[ref_symbol]['close'].pct_change(24).shift(-24)
    if raw_target.index.tz is None: raw_target.index = raw_target.index.tz_localize('UTC')
    
    correlations = {}
    for col in ref_feats.select_dtypes(include=[np.number]).columns:
        # Quick align for stats
        idx = ref_feats.index.intersection(raw_target.dropna().index)
        if len(idx) > 500:
            corr = ref_feats.loc[idx, col].corr(raw_target.loc[idx])
            if not np.isnan(corr): correlations[col] = corr
            
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for f, c in sorted_corr:
        print(f"    {'📈' if c>0 else '📉'} {f:<35} IC: {c:+.4f}")

    # --- 4. Export & Prep ML Data ---
    print("\n4️⃣  EXPORTING DATA (All Symbols)")
    print("-" * 50)
    
    for symbol in symbols:
        if symbol not in all_features: continue
        
        feats = all_features[symbol]
        ohlcv = ohlcv_data[symbol]
        
        # 1. Export Features
        f_path = EXPORT_DIR / f"{symbol.replace('-', '_')}_features.csv"
        feats.to_csv(f_path)
        
        profile = get_coin_profile(symbol)
        target = compute_profile_target(ohlcv, profile)
        
        # Ensure UTC match
        if target.index.tz is None: target.index = target.index.tz_localize('UTC')
        
        # Robust Cleaning
        X_final, y_final = prepare_robust_dataset(feats, target)
        
        if len(X_final) > 0:
            ml_df = X_final.copy()
            ml_df['target_tb'] = y_final
            ml_path = EXPORT_DIR / f"{symbol.replace('-', '_')}_ml_dataset.csv"
            ml_df.to_csv(ml_path)
            
            tp_rate = (y_final == 1).mean()
            sl_rate = (y_final == -1).mean()
            timeout_rate = (y_final == 0).mean()
            print(
                f"  ✅ {symbol}: Saved {len(ml_df)} rows | "
                f"TP={tp_rate:.1%} SL={sl_rate:.1%} Timeout={timeout_rate:.1%} "
                f"(TP={profile.vol_mult_tp}x, SL={profile.vol_mult_sl}x, H={resolve_label_horizon(profile)}h)"
            )
        else:
            print(f"  ❌ {symbol}: Failed to generate ML dataset (0 rows)")

    print("\n" + "=" * 70)
    print("✅ DONE. Pipeline Complete.")
    print("=" * 70)
    db.close()

if __name__ == "__main__":
    main()
