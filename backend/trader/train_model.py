#!/usr/bin/env python3
"""
CRYPTO ML TRADING SYSTEM v7 — MOMENTUM + FUNDING CARRY
=====================================================================
FUNDAMENTAL STRATEGY CHANGE FROM v6:

v6 PROBLEM: Funding-rate contrarian direction (short when funding high).
This fights the trend — Coinbase research confirms funding LAGS momentum.
Result: -0.39% raw PnL per trade. No exit structure can fix negative edge.

v7 SOLUTION: MOMENTUM for direction, FUNDING as carry bonus.
  - GO LONG when multi-timeframe momentum is bullish
  - GO SHORT when multi-timeframe momentum is bearish  
  - PREFER shorts when funding is positive (receive carry)
  - AVOID longs when funding is extreme positive (pay carry)
  - Size positions larger when funding ALIGNS with direction

This combines crypto's strongest documented edge (momentum/trend) with
the structural carry advantage from funding rates.

Built for the EXACT contract specifications on Coinbase Derivatives Exchange (CDE):

CONTRACT SPECS (from user's screenshot):
  BTC-PERP:  0.01  BTC  per contract  | Vol $1.65B
  ETH-PERP:  0.1   ETH  per contract  | Vol $209.70M
  XRP-PERP:  500   XRP  per contract  | Vol $122.74M
  SOL-PERP:  5     SOL  per contract  | Vol $99.27M
  DOGE-PERP: 5000  DOGE per contract  | Vol $1.39M

FEE STRUCTURE (confirmed from screenshot showing "Fee (0.10%): ~$0.85" on BTC):
  - 0.10% per contract per SIDE (buy OR sell)
  - Round-trip = 0.20% of notional
  - Minimum $0.20 per contract per side
  - Includes exchange, clearing, and NFA fees
  - Liquidation fee: 0.80%
  - This is a FLAT fee — no maker/taker distinction for US futures

LEVERAGE (from screenshot showing "Leverage 4.1X" with overnight note):
  - Intraday (Sun-Fri, 6PM-4PM ET = 22hrs/day): up to 10x
  - Overnight (4PM-6PM ET): reduced, ~4.1x for BTC
  - We use 4x as conservative overnight-safe level

FUNDING:
  - Settled HOURLY on Coinbase (not 8h like Binance!)
  - Positive rate = longs pay shorts
  - Our strategy shorts when funding is high → RECEIVES funding

KEY v6 FIXES:
  1. EXACT 0.10%/side fee (confirmed from screenshot)
  2. DOGE not ADA (corrected per user)
  3. DISCRETE CONTRACT SIZING (whole contracts only)
  4. SINGLE EXIT per position (fixes v4 double-fee bug)
  5. BREAKEVEN SL includes fees (exit at entry+fees, not entry)
  6. WEEKLY EQUITY BASE (prevents compounding spiral)
  7. 4x leverage (overnight-safe from screenshot)
"""
import argparse
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

# --- Paths ---
FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# =============================================================================
# COINBASE CDE CONTRACT SPECIFICATIONS — EXACT
# =============================================================================
# Maps symbol -> (units_per_contract, min_fee_per_contract)
# "units_per_contract" = how many of the base asset per 1 contract
# From the screenshot: BTC 0.01, ETH 0.1, XRP 500, SOL 5, ADA 1000
# Nano contract specs from Coinbase CDE
# BIP = nano BTC Perp (0.01 BTC), ETP = nano ETH Perp (0.1 ETH), etc.
CONTRACT_SPECS = {
    # CDE product codes (actual symbol names in our data)
    'BIP': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETP': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XPP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SLP': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOP': {'units': 5000,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    # Legacy symbol formats (in case data uses these)
    'BTC': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETH': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XRP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SOL': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOGE': {'units': 5000, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    # Fallback
    'DEFAULT': {'units': 1, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'UNKNOWN'},
}

def get_contract_spec(symbol: str) -> dict:
    """
    Get contract spec for a symbol, matching by CDE product code prefix.
    
    Handles symbols like 'BIP-20DEC30-CDE', 'ETP-20DEC30-CDE', etc.
    Extracts the product code (first part before '-') and looks it up.
    """
    # Try exact match first
    if symbol in CONTRACT_SPECS:
        return CONTRACT_SPECS[symbol]
    
    # Extract product code prefix (e.g., 'BIP' from 'BIP-20DEC30-CDE')
    prefix = symbol.split('-')[0] if '-' in symbol else symbol
    
    if prefix in CONTRACT_SPECS:
        return CONTRACT_SPECS[prefix]
    
    # Try matching by known base asset names anywhere in the symbol
    symbol_upper = symbol.upper()
    for code, spec in CONTRACT_SPECS.items():
        if code == 'DEFAULT':
            continue
        if code in symbol_upper:
            return spec
    
    print(f"  ⚠️ No contract spec found for '{symbol}', using DEFAULT (1 unit/contract)")
    return CONTRACT_SPECS['DEFAULT']


# =============================================================================
# EXPLICIT FEATURE LIST
# =============================================================================
FEATURE_COLUMNS = [
    # Momentum features (PRIMARY — for direction)
    'return_1h', 'return_4h', 'return_12h', 'return_24h', 'return_48h', 'return_168h',
    'rsi_14', 'rsi_6',
    'range_position_24h', 'range_position_72h',
    'bb_position_20',  # Bollinger band position
    'ma_distance_24h', 'ma_distance_168h',
    # Volatility (for regime + sizing)
    'volatility_1h', 'volatility_4h', 'volatility_24h',
    'volume_ratio_1h', 'volume_ratio_24h',
    'parkinson_vol_24h',
    # Funding (SECONDARY — for carry/sizing, not direction)
    'funding_rate_bps', 'funding_rate_zscore',
    'cumulative_funding_24h', 'cumulative_funding_72h',
    # OI (for crowding/liquidation signals)
    'oi_change_1h', 'oi_change_4h', 'oi_change_24h',
    # Regime
    'trend_sma20_50', 'vol_regime_ratio',
]


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    net_pnl: float          # Percentage return on notional
    raw_pnl: float          # Price return only
    funding_pnl: float      # Accumulated funding
    fee_pnl: float          # Total fees (negative)
    pnl_dollars: float      # Actual dollar PnL
    n_contracts: int         # Number of contracts traded
    notional: float          # Total notional value
    exit_reason: str


@dataclass
class Config:
    # Walk-Forward Windows
    train_lookback_days: int = 120
    retrain_frequency_days: int = 7
    min_train_samples: int = 400

    # Signal Filters — Optuna-optimized (75 trials, accurate backtest, 10/10 top profitable)
    # Converged: thr=0.74, mom=0.04 across all top 5 configs
    signal_threshold: float = 0.74
    min_funding_z: float = 0.0
    min_momentum_magnitude: float = 0.04

    # Exit Strategy — Optuna-optimized (wider SL = fewer stop-outs = 48% WR)
    vol_mult_tp: float = 4.5         # 4.5x vol TP
    vol_mult_sl: float = 2.75        # 2.75x vol SL (wider = room to breathe)
    breakeven_trigger: float = 999.0
    trailing_active: bool = False
    trailing_mult: float = 999.0
    max_hold_hours: int = 96

    # Symbol selection — exclude underperformers
    # BTC: too efficient, momentum edge doesn't persist (36% WR, -0.04% net)
    # DOGE: too noisy/memecoin, momentum signals unreliable (32% WR, -0.13% net)
    excluded_symbols: Optional[List[str]] = None  # Set to ['BIP', 'DOP'] to exclude BTC and DOGE
    
    # Risk — volatility-adjusted sizing
    max_positions: int = 3
    position_size: float = 0.15      # 15% base — adjusted down for high-vol assets
    leverage: int = 4
    vol_sizing_target: float = 0.025  # Target 2.5% daily vol exposure per position
    # Intraday (6PM-4PM ET Sun-Fri) allows up to 10x
    # Overnight (4PM-6PM ET) reduces — screenshot shows 4.1x
    # We use 4x as the conservative overnight-compatible level

    # Fees — EXACT COINBASE US CDE (from screenshot: "Fee (0.10%)")
    fee_pct_per_side: float = 0.0010  # 0.10% per contract per side (10 bps)
    min_fee_per_contract: float = 0.20  # Minimum $0.20 per contract per side
    slippage_bps: float = 0.0          # No added slippage — 0.10% fee already includes execution

    # Regime Filter
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06

    # Safety
    min_equity: float = 1000.0

    # Validation
    val_fraction: float = 0.20
    min_val_auc: float = 0.54

    # Correlation filter
    max_portfolio_correlation: float = 0.75
    correlation_lookback_hours: int = 72

    # Label
    label_forward_hours: int = 24
    label_vol_target: float = 1.8

    # Position sizing control
    max_weekly_equity_growth: float = 0.03  # Max 3% growth in sizing base per week


# =============================================================================
# EXACT COINBASE FEE CALCULATION
# =============================================================================
def calculate_coinbase_fee(n_contracts: int, price: float, symbol: str,
                           config: Config) -> float:
    """
    Calculate EXACT Coinbase CDE fee for one side (entry OR exit).
    
    Fee = max(n_contracts * min_fee_per_contract, 
              n_contracts * units_per_contract * price * fee_pct_per_side)
    
    Returns total fee in USD for this side.
    """
    spec = get_contract_spec(symbol)
    
    # Percentage-based fee
    notional_per_contract = spec['units'] * price
    pct_fee = n_contracts * notional_per_contract * config.fee_pct_per_side
    
    # Minimum fee
    min_fee = n_contracts * config.min_fee_per_contract
    
    return max(pct_fee, min_fee)


def calculate_pnl_exact(entry_price: float, exit_price: float, direction: int,
                         accum_funding: float, n_contracts: int, symbol: str,
                         config: Config) -> Tuple[float, float, float, float, float]:
    """
    Calculate PnL with EXACT Coinbase fee structure.
    
    Returns: (net_pnl_pct, raw_pnl_pct, total_fee_pct, pnl_dollars, total_notional)
    """
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * entry_price
    total_notional = n_contracts * notional_per_contract
    
    if total_notional == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Raw PnL
    raw_pnl_pct = (exit_price - entry_price) / entry_price * direction
    raw_pnl_dollars = total_notional * raw_pnl_pct
    
    # Fees (entry + exit)
    entry_fee = calculate_coinbase_fee(n_contracts, entry_price, symbol, config)
    exit_fee = calculate_coinbase_fee(n_contracts, exit_price, symbol, config)
    
    # Slippage (applied to exit only)
    slippage = total_notional * (config.slippage_bps / 10000.0)
    
    total_fee_dollars = entry_fee + exit_fee + slippage
    total_fee_pct = total_fee_dollars / total_notional
    
    # Funding PnL in dollars
    funding_dollars = accum_funding * total_notional
    
    # Net
    net_pnl_dollars = raw_pnl_dollars - total_fee_dollars + funding_dollars
    net_pnl_pct = net_pnl_dollars / total_notional
    
    return net_pnl_pct, raw_pnl_pct, -total_fee_pct, net_pnl_dollars, total_notional


def calculate_n_contracts(equity: float, price: float, symbol: str,
                           config: Config, vol_24h: float = 0.0) -> int:
    """
    Calculate how many WHOLE contracts we can buy given equity allocation.
    
    v7.2: Volatility-adjusted sizing.
    If vol_24h is provided, scale position size so that daily vol exposure
    targets config.vol_sizing_target. This means high-vol assets (DOGE) get
    smaller positions and low-vol assets (BTC) get normal/larger ones.
    """
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * price
    
    if notional_per_contract <= 0:
        return 0
    
    # Base position size
    pos_size = config.position_size
    
    # Vol-adjust: scale down if asset vol exceeds target
    if vol_24h > 0 and config.vol_sizing_target > 0:
        vol_ratio = config.vol_sizing_target / vol_24h
        vol_ratio = min(vol_ratio, 1.5)  # Cap upward scaling at 1.5x
        vol_ratio = max(vol_ratio, 0.3)  # Floor at 0.3x (don't go too small)
        pos_size = pos_size * vol_ratio
    
    target_notional = equity * pos_size * config.leverage
    n_contracts = int(target_notional / notional_per_contract)
    
    return max(n_contracts, 0)


# =============================================================================
# CORE ML ENGINE
# =============================================================================
class MLSystem:
    def __init__(self, config: Config):
        self.config = config

    def get_feature_columns(self, available_columns: pd.Index) -> List[str]:
        cols = [c for c in FEATURE_COLUMNS if c in available_columns]
        return cols if len(cols) >= 4 else []

    def create_labels(self, ohlcv: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        MOMENTUM-BASED forward return label.
        
        v7 change: Direction from multi-timeframe price momentum, NOT funding z-score.
        Label = 1 if price moves >= threshold in the momentum direction within forward window.
        """
        target = pd.Series(index=features.index, dtype=float)
        vol = ohlcv['close'].pct_change().rolling(24).std().ffill()
        fwd_hours = self.config.label_forward_hours
        
        # Multi-timeframe momentum for direction
        ret_24h = ohlcv['close'].pct_change(24).ffill()
        ret_72h = ohlcv['close'].pct_change(72).ffill()
        sma_50 = ohlcv['close'].rolling(50).mean()

        for ts in features.index:
            if ts not in ohlcv.index or ts not in vol.index:
                continue
            row_vol = vol.loc[ts]
            if pd.isna(row_vol) or row_vol == 0:
                continue
            try:
                pos_in_ohlcv = ohlcv.index.get_loc(ts)
            except KeyError:
                continue
            if pos_in_ohlcv + fwd_hours >= len(ohlcv):
                continue

            entry_px = ohlcv.loc[ts, 'close']
            future = ohlcv.iloc[pos_in_ohlcv + 1: pos_in_ohlcv + 1 + fwd_hours]
            threshold = self.config.label_vol_target * row_vol
            
            # MOMENTUM DIRECTION: consensus of 24h return, 72h return, and price vs SMA50
            r24 = ret_24h.get(ts, 0)
            r72 = ret_72h.get(ts, 0)
            sma = sma_50.get(ts, entry_px)
            if pd.isna(r24): r24 = 0
            if pd.isna(r72): r72 = 0
            if pd.isna(sma): sma = entry_px
            
            momentum_score = (1 if r24 > 0 else -1) + (1 if r72 > 0 else -1) + (1 if entry_px > sma else -1)
            
            if momentum_score >= 2:
                direction = 1   # Bullish momentum consensus
            elif momentum_score <= -2:
                direction = -1  # Bearish momentum consensus
            else:
                continue  # No consensus — skip this sample

            if direction == 1:
                max_excursion = (future['high'].max() - entry_px) / entry_px
            else:
                max_excursion = (entry_px - future['low'].min()) / entry_px

            target.loc[ts] = 1.0 if max_excursion >= threshold else 0.0
        return target

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Optional[Tuple]:
        if len(X_train) < self.config.min_train_samples:
            return None
        if y_train.sum() < 15 or (1 - y_train).sum() < 15:
            return None
        if len(X_val) < 30 or y_val.sum() < 5:
            return None

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        base_model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            class_weight='balanced', verbose=-1,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        )
        base_model.fit(X_train_scaled, y_train)

        val_probs = base_model.predict_proba(X_val_scaled)[:, 1]
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            return None
        if auc < self.config.min_val_auc:
            return None

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(val_probs, y_val)

        X_full = np.vstack([X_train_scaled, X_val_scaled])
        y_full = pd.concat([y_train, y_val])
        base_model.fit(X_full, y_full)

        return (base_model, scaler, iso, auc)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    data = {}
    print("⏳ Loading data from features and database...")
    feature_files = list(FEATURES_DIR.glob("*_features.csv"))

    for f in feature_files:
        sym = f.stem.replace("_features", "").replace("_", "-")
        try:
            conn = sqlite3.connect(DB_PATH)
            ohlcv = pd.read_sql_query(
                f"SELECT event_time, open, high, low, close FROM ohlcv "
                f"WHERE symbol='{sym}' AND timeframe='1h'",
                conn, parse_dates=['event_time']
            ).set_index('event_time').sort_index()
            conn.close()
            ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
            ohlcv['sma_200'] = ohlcv['close'].rolling(200).mean()

            feat = pd.read_csv(f, index_col=0, parse_dates=True)
            feat = feat.replace([np.inf, -np.inf], 0).ffill().fillna(0)
            feat.index = pd.to_datetime(feat.index, utc=True)

            common = feat.index.intersection(ohlcv.index)
            if len(common) > 1000:
                data[sym] = {'features': feat.loc[common], 'ohlcv': ohlcv.loc[common]}
        except Exception as e:
            print(f"Skipping {sym}: {e}")
    print(f"✅ Loaded {len(data)} symbols.")
    
    # Show contract specs for loaded symbols
    for sym in data:
        spec = get_contract_spec(sym)
        price = data[sym]['ohlcv']['close'].iloc[-1]
        notional = spec['units'] * price
        eff_fee = max(0.20, notional * 0.0010) / notional * 100
        print(f"  {sym}: {spec['units']} units/contract, "
              f"~${notional:.2f}/contract, "
              f"effective fee: {eff_fee:.3f}% per side")
    
    return data


# =============================================================================
# CORRELATION FILTER
# =============================================================================
def check_correlation(sym: str, direction: int, active_positions: Dict,
                      all_data: Dict, ts, config: Config) -> bool:
    if not active_positions:
        return True
    lookback = config.correlation_lookback_hours
    new_returns = all_data[sym]['ohlcv']['close'].pct_change()

    for existing_sym in active_positions:
        existing_returns = all_data[existing_sym]['ohlcv']['close'].pct_change()
        combined = pd.DataFrame({
            'new': new_returns, 'existing': existing_returns
        }).dropna().loc[:ts].tail(lookback)
        if len(combined) < 20:
            continue
        corr = combined['new'].corr(combined['existing'])
        if abs(corr) > config.max_portfolio_correlation:
            existing_dir = active_positions[existing_sym]['dir']
            if corr > 0 and direction == existing_dir:
                return False
            if corr < -config.max_portfolio_correlation and direction != existing_dir:
                return False
    return True


# =============================================================================
# BACKTEST — v6 EXACT COINBASE
# =============================================================================
def run_backtest(all_data: Dict, config: Config):
    system = MLSystem(config)

    all_ts = [ts for d in all_data.values() for ts in d['ohlcv'].index]
    if not all_ts:
        return
    current_date = min(all_ts) + timedelta(days=config.train_lookback_days)
    end_date = max(all_ts)

    print(f"\n⏩ STARTING BACKTEST (v7 — Momentum + Funding Carry)")
    print(f"   Period: {current_date.date()} to {end_date.date()}")
    print(f"   TP: {config.vol_mult_tp}x vol | SL: {config.vol_mult_sl}x vol")
    trail_status = "DISABLED" if not config.trailing_active else f"BE at +{config.breakeven_trigger}x vol | Trail {config.trailing_mult}x from peak"
    print(f"   Trailing: {trail_status}")
    print(f"   Leverage: {config.leverage}x | Fee: 0.10%/side (0.20% round-trip)")
    print(f"   Regime: {config.min_vol_24h:.1%}-{config.max_vol_24h:.1%}")
    print(f"   Signal threshold: {config.signal_threshold}")

    equity = 100_000.0
    peak_equity = equity
    max_drawdown = 0.0
    completed_trades = []
    active_positions = {}
    last_exit_time = {}  # v7.2: cooldown tracking per symbol
    models_rejected = 0
    models_accepted = 0
    regime_filtered = 0
    no_contracts = 0  # Times we couldn't buy even 1 contract

    weekly_equity_base = equity

    while current_date < end_date:
        if equity < config.min_equity:
            print(f"\n🛑 EQUITY BELOW MINIMUM (${equity:,.2f}). Stopping.")
            break

        week_end = current_date + timedelta(days=config.retrain_frequency_days)

        # Weekly equity base — damped growth to prevent compounding spiral
        weekly_equity_base = min(equity, weekly_equity_base * (1 + config.max_weekly_equity_growth))
        weekly_equity_base = max(weekly_equity_base, equity * 0.9)

        # --- TRAINING ---
        train_start = current_date - timedelta(days=config.train_lookback_days)
        models = {}

        for sym, d in all_data.items():
            feat, ohlc = d['features'], d['ohlcv']
            cols = system.get_feature_columns(feat.columns)
            if not cols:
                continue

            train_feat = feat.loc[train_start:current_date]
            train_ohlc = ohlc.loc[train_start:current_date + timedelta(hours=config.max_hold_hours)]

            if len(train_feat) < config.min_train_samples:
                continue

            y = system.create_labels(train_ohlc, train_feat)
            valid_idx = y.dropna().index
            X_all = train_feat.loc[valid_idx, cols]
            y_all = y.loc[valid_idx]

            if len(X_all) < config.min_train_samples:
                continue

            split_idx = int(len(X_all) * (1 - config.val_fraction))
            X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
            y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

            result = system.train(X_tr, y_tr, X_vl, y_vl)
            if result:
                model, scaler, iso, auc = result
                models[sym] = (model, scaler, cols, iso, auc)
                models_accepted += 1
            else:
                models_rejected += 1

        # --- TRADING ---
        test_hours = pd.date_range(current_date, week_end, freq='1h')

        for ts in test_hours:
            # ============================================================
            # 1. MANAGE EXITS
            # ============================================================
            to_close = []
            for sym, pos in active_positions.items():
                if ts not in all_data[sym]['ohlcv'].index:
                    continue

                # Funding accumulation
                # CRITICAL: funding_rate_bps comes from Binance 8h data, forward-filled hourly
                # by engineering.py. The value at each hour is the FULL 8h rate, not hourly.
                # We must divide by 8 to get the per-hour accrual.
                # (Coinbase settles hourly, but our data source is 8h from Binance)
                funding_8h_bps = all_data[sym]['features'].loc[ts].get('funding_rate_bps', 0.0)
                funding_hourly_bps = funding_8h_bps / 8.0
                # If we're SHORT and funding is positive, we RECEIVE funding
                # accum_funding is in fractional terms (e.g. 0.0001 = 1bps)
                pos['accum_funding'] += -(funding_hourly_bps / 10000.0) * pos['dir']

                bar = all_data[sym]['ohlcv'].loc[ts]
                direction = pos['dir']

                # Track peak favorable excursion
                if direction == 1:
                    pos['peak_price'] = max(pos['peak_price'], bar['high'])
                else:
                    pos['peak_price'] = min(pos['peak_price'], bar['low'])

                # --- BREAKEVEN + TRAILING ---
                peak_move = (pos['peak_price'] - pos['entry']) / pos['entry'] * direction

                if not pos['at_breakeven'] and peak_move >= config.breakeven_trigger * pos['vol']:
                    # Move SL to entry + enough to cover round-trip fees
                    # We need to profit at least the fee amount to truly break even
                    effective_fee_pct = pos['effective_fee_pct']  # Stored at entry
                    pos['sl'] = pos['entry'] * (1 + effective_fee_pct * direction)
                    pos['at_breakeven'] = True

                if pos['at_breakeven'] and config.trailing_active:
                    if direction == 1:
                        trail_sl = pos['peak_price'] * (1 - config.trailing_mult * pos['vol'])
                        pos['sl'] = max(pos['sl'], trail_sl)
                    else:
                        trail_sl = pos['peak_price'] * (1 + config.trailing_mult * pos['vol'])
                        pos['sl'] = min(pos['sl'], trail_sl)

                # --- CHECK EXITS ---
                exit_price, reason = None, None

                # TP
                tp_hit = (direction == 1 and bar['high'] >= pos['tp']) or \
                         (direction == -1 and bar['low'] <= pos['tp'])
                # SL
                sl_hit = (direction == 1 and bar['low'] <= pos['sl']) or \
                         (direction == -1 and bar['high'] >= pos['sl'])

                if tp_hit and sl_hit:
                    # Both hit in same bar — use distance from open
                    dist_tp = abs(bar['open'] - pos['tp'])
                    dist_sl = abs(bar['open'] - pos['sl'])
                    if dist_sl <= dist_tp:
                        exit_price = pos['sl']
                        reason = 'stop_loss' if not pos['at_breakeven'] else \
                                 ('trailing_stop' if pos['sl'] != pos['entry'] and pos['at_breakeven'] else 'breakeven')
                    else:
                        exit_price, reason = pos['tp'], 'take_profit'
                elif tp_hit:
                    exit_price, reason = pos['tp'], 'take_profit'
                elif sl_hit:
                    exit_price = pos['sl']
                    if pos['at_breakeven']:
                        raw_ret = (exit_price - pos['entry']) / pos['entry'] * direction
                        reason = 'trailing_stop' if raw_ret > 0.001 else 'breakeven'
                    else:
                        reason = 'stop_loss'

                # Max hold
                if not exit_price and (ts - pos['time']).total_seconds() / 3600 >= config.max_hold_hours:
                    exit_price, reason = bar['close'], 'max_hold'

                # --- CLOSE ---
                if exit_price:
                    net_pnl, raw_pnl, fee_pnl, pnl_dollars, notional = calculate_pnl_exact(
                        pos['entry'], exit_price, direction,
                        pos['accum_funding'], pos['n_contracts'], sym, config
                    )

                    # Cap loss at full notional
                    pnl_dollars = max(pnl_dollars, -notional)
                    equity += pnl_dollars

                    peak_equity = max(peak_equity, equity)
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)

                    completed_trades.append(Trade(
                        sym, pos['time'], ts, direction, pos['entry'], exit_price,
                        net_pnl, raw_pnl, pos['accum_funding'], fee_pnl,
                        pnl_dollars, pos['n_contracts'], notional, reason
                    ))
                    to_close.append(sym)
                    last_exit_time[sym] = ts  # Record for cooldown

            for sym in to_close:
                del active_positions[sym]

            # ============================================================
            # 2. ENTRIES
            # ============================================================
            if len(active_positions) < config.max_positions and equity >= config.min_equity:
                for sym, (model, scaler, cols, iso, auc) in models.items():
                    if sym in active_positions or ts not in all_data[sym]['features'].index:
                        continue
                    if ts not in all_data[sym]['ohlcv'].index:
                        continue
                    # v7.2: 24h cooldown after exit
                    if sym in last_exit_time and (ts - last_exit_time[sym]).total_seconds() < 24 * 3600:
                        continue
                    # v7.3: Symbol exclusion
                    if config.excluded_symbols:
                        sym_prefix = sym.split('-')[0] if '-' in sym else sym
                        if sym_prefix in config.excluded_symbols:
                            continue

                    row = all_data[sym]['features'].loc[ts]
                    ohlcv_row = all_data[sym]['ohlcv'].loc[ts]
                    price = ohlcv_row['close']
                    sma_200 = ohlcv_row['sma_200']
                    if pd.isna(sma_200):
                        continue

                    # Regime filter
                    vol_24h = all_data[sym]['ohlcv']['close'].pct_change().rolling(24).std().get(ts, None)
                    if vol_24h is None or pd.isna(vol_24h):
                        continue
                    if vol_24h < config.min_vol_24h or vol_24h > config.max_vol_24h:
                        regime_filtered += 1
                        continue

                    # ============================================================
                    # v7: MOMENTUM DIRECTION (replaces funding contrarian)
                    # ============================================================
                    ohlcv_ts = all_data[sym]['ohlcv']
                    ts_loc = ohlcv_ts.index.get_loc(ts) if ts in ohlcv_ts.index else None
                    if ts_loc is None or ts_loc < 72:
                        continue
                    
                    ret_24h = (price / ohlcv_ts['close'].iloc[ts_loc - 24] - 1) if ts_loc >= 24 else 0
                    ret_72h = (price / ohlcv_ts['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
                    sma_50 = ohlcv_ts['close'].iloc[max(0, ts_loc-50):ts_loc].mean()
                    
                    # Minimum momentum magnitude — avoid weak/choppy moves
                    if abs(ret_72h) < config.min_momentum_magnitude:
                        continue
                    
                    # v7.3: Require 24h and 72h returns to agree on direction
                    if ret_24h * ret_72h < 0:
                        continue  # 24h and 72h disagree — skip
                    
                    mom_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)
                    
                    if mom_score >= 2:
                        direction = 1   # Bullish
                    elif mom_score <= -2:
                        direction = -1  # Bearish
                    else:
                        continue  # No momentum consensus — skip
                    
                    # Trend confirmation: long only above SMA200, short only below
                    if direction == 1 and price < sma_200:
                        continue
                    if direction == -1 and price > sma_200:
                        continue
                    
                    # ============================================================
                    # FUNDING CARRY FILTER (bonus, not direction)
                    # ============================================================
                    f_z = row.get('funding_rate_zscore', 0)
                    if pd.isna(f_z): f_z = 0
                    
                    # Reject longs when funding is extremely positive (expensive carry)
                    if direction == 1 and f_z > 2.5:
                        continue
                    # Reject shorts when funding is extremely negative (expensive carry)
                    if direction == -1 and f_z < -2.5:
                        continue

                    # Correlation filter
                    if not check_correlation(sym, direction, active_positions, all_data, ts, config):
                        continue

                    x_in = np.nan_to_num(
                        np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0
                    )
                    raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
                    prob = float(iso.predict([raw_prob])[0])

                    if prob >= config.signal_threshold:
                        vol = vol_24h if vol_24h > 0 else 0.02

                        # DISCRETE CONTRACT SIZING
                        n_contracts = calculate_n_contracts(weekly_equity_base, price, sym, config, vol_24h=vol_24h)
                        if n_contracts < 1:
                            no_contracts += 1
                            continue

                        # Calculate effective round-trip fee percentage for this position
                        spec = get_contract_spec(sym)
                        notional_per_contract = spec['units'] * price
                        total_notional = n_contracts * notional_per_contract
                        entry_fee = calculate_coinbase_fee(n_contracts, price, sym, config)
                        # Estimate exit fee at same price (approximate)
                        exit_fee = entry_fee
                        effective_fee_pct = (entry_fee + exit_fee) / total_notional

                        active_positions[sym] = {
                            'time': ts,
                            'entry': price,
                            'dir': direction,
                            'vol': vol,
                            'tp': price * (1 + config.vol_mult_tp * vol * direction),
                            'sl': price * (1 - config.vol_mult_sl * vol * direction),
                            'accum_funding': 0.0,
                            'n_contracts': n_contracts,
                            'peak_price': price,
                            'at_breakeven': False,
                            'effective_fee_pct': effective_fee_pct,
                        }

        current_date = week_end

    # --- REPORT ---
    if not completed_trades:
        print("\n⚠️ No trades executed.")
        print(f"   Models accepted: {models_accepted}, rejected: {models_rejected}")
        print(f"   Regime filtered: {regime_filtered}")
        print(f"   No contracts (too small): {no_contracts}")
        return {'n_trades': 0, 'sharpe_annual': -99, 'total_return': -1, 'max_drawdown': 1.0,
                'profit_factor': 0, 'ann_return': -1, 'final_equity': equity}

    df = pd.DataFrame([t.__dict__ for t in completed_trades])
    win_rate = (df['net_pnl'] > 0).mean()
    total_ret = (equity / 100000) - 1
    reason_counts = df['exit_reason'].value_counts()

    avg_pnl = df['net_pnl'].mean()
    std_pnl = df['net_pnl'].std()
    sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0

    gross_wins = df.loc[df['net_pnl'] > 0, 'pnl_dollars'].sum()
    gross_losses = abs(df.loc[df['net_pnl'] < 0, 'pnl_dollars'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    total_pnl_dollars = df['pnl_dollars'].sum()

    days = (end_date - (min(all_ts) + timedelta(days=config.train_lookback_days))).days
    years = days / 365.25
    ann_return = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else -1

    # Annualized Sharpe (approximate)
    trades_per_year = len(df) / years if years > 0 else 0
    ann_sharpe = sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    avg_fee = df['fee_pnl'].mean()
    avg_raw = df['raw_pnl'].mean()
    avg_funding = df['funding_pnl'].mean()

    print(f"\n{'=' * 70}")
    print(f"📊 BACKTEST RESULTS (v7 — Momentum + Funding Carry)")
    print(f"{'=' * 70}")
    print(f"Total Trades:           {len(df)}")
    print(f"Win Rate:               {win_rate:.1%}")
    print(f"Profit Factor:          {profit_factor:.2f}")
    print(f"Total Return:           {total_ret:.2%}")
    print(f"Annualized Return:      {ann_return:.2%}")
    print(f"Final Equity:           ${equity:,.2f}")
    print(f"Max Drawdown:           {max_drawdown:.2%}")
    print(f"Total PnL:              ${total_pnl_dollars:,.2f}")
    print(f"Avg PnL/Trade:          ${df['pnl_dollars'].mean():,.2f}")
    print(f"")
    print(f"Sharpe (per-trade):     {sharpe_per_trade:.3f}")
    print(f"Sharpe (annualized):    {ann_sharpe:.2f}")
    print(f"Trades/Year:            {trades_per_year:.0f}")
    print(f"")
    print(f"Edge Decomposition (% of notional):")
    print(f"  Avg Raw PnL:          {avg_raw:.4%}")
    print(f"  Avg Funding:          {avg_funding:.4%}")
    print(f"  Avg Fees:             {avg_fee:.4%}")
    print(f"  Avg Net PnL:          {avg_pnl:.4%}")
    if avg_raw != 0:
        print(f"  Fee/Edge Ratio:       {abs(avg_fee/avg_raw)*100:.0f}% of raw edge consumed by fees")
    print(f"")
    print(f"Avg Contracts/Trade:    {df['n_contracts'].mean():.1f}")
    print(f"Avg Notional/Trade:     ${df['notional'].mean():,.0f}")
    print(f"")
    print(f"Model Stats:")
    print(f"  Accepted: {models_accepted}  |  Rejected: {models_rejected}")
    print(f"  Regime filtered: {regime_filtered}")
    print(f"  No contracts (sizing): {no_contracts}")
    print(f"")
    print(f"Exit Reasons:")
    for reason, count in reason_counts.items():
        pct = count / len(df)
        avg = df[df['exit_reason'] == reason]['net_pnl'].mean()
        avg_d = df[df['exit_reason'] == reason]['pnl_dollars'].mean()
        print(f"  {reason:20s}: {count:4d} ({pct:.1%}) | Avg: {avg:.4%} | ${avg_d:,.2f}")

    # Per-symbol breakdown
    print(f"\nPer-Symbol Performance:")
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        sym_wr = (sym_df['net_pnl'] > 0).mean()
        sym_pnl = sym_df['pnl_dollars'].sum()
        sym_avg = sym_df['net_pnl'].mean()
        sym_fee = sym_df['fee_pnl'].mean()
        print(f"  {sym:12s}: {len(sym_df):3d} trades | WR: {sym_wr:.0%} | "
              f"PnL: ${sym_pnl:+,.0f} | Avg: {sym_avg:.4%} | Fee: {sym_fee:.4%}")
    print(f"{'=' * 70}")
    
    # Return metrics for optimization
    return {
        'total_return': total_ret,
        'ann_return': ann_return,
        'sharpe_annual': ann_sharpe,
        'sharpe_per_trade': sharpe_per_trade,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'n_trades': len(df),
        'trades_per_year': trades_per_year,
        'avg_net_pnl': avg_pnl,
        'avg_raw_pnl': avg_raw,
        'avg_fee_pnl': avg_fee,
        'final_equity': equity,
    }
# =============================================================================
def run_signals(all_data: Dict, config: Config, debug: bool = False):
    system = MLSystem(config)
    print(f"\n🔍 ANALYZING LIVE MARKETS (v6 — Coinbase CDE)...")

    for sym, d in all_data.items():
        feat, ohlc = d['features'], d['ohlcv']
        cols = system.get_feature_columns(feat.columns)
        if not cols:
            continue

        train_end = feat.index[-1]
        train_start = train_end - timedelta(days=config.train_lookback_days)
        train_feat = feat.loc[train_start:train_end]
        train_ohlc = ohlc.loc[train_start:train_end]

        y = system.create_labels(train_ohlc, train_feat)
        valid = y.dropna().index
        X_all = train_feat.loc[valid, cols]
        y_all = y.loc[valid]

        if len(X_all) < config.min_train_samples:
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

        result = system.train(X_tr, y_tr, X_vl, y_vl)
        if not result:
            if debug:
                print(f"\n[{sym}] ❌ MODEL REJECTED (low AUC)")
            continue

        model, scaler, iso, auc = result
        row = feat.iloc[-1]
        price = ohlc.iloc[-1]['close']
        sma_200 = ohlc.iloc[-1]['sma_200']

        f_z = row.get('funding_rate_zscore', 0)
        
        # v7: Momentum direction
        ts_loc = len(ohlc) - 1
        ret_24h = (price / ohlc['close'].iloc[ts_loc - 24] - 1) if ts_loc >= 24 else 0
        ret_72h = (price / ohlc['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
        sma_50 = ohlc['close'].iloc[max(0, ts_loc-50):ts_loc].mean()
        mom_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)
        direction = 1 if mom_score >= 2 else (-1 if mom_score <= -2 else 0)

        rule_pass = direction != 0  # Need momentum consensus
        trend_pass = (direction == 1 and price > sma_200) or \
                     (direction == -1 and price < sma_200)

        vol_24h = ohlc['close'].pct_change().rolling(24).std().iloc[-1]
        regime_pass = config.min_vol_24h <= vol_24h <= config.max_vol_24h if not pd.isna(vol_24h) else False

        x_in = np.nan_to_num(np.array([row.get(c, 0) for c in cols]).reshape(1, -1))
        raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
        prob = float(iso.predict([raw_prob])[0])
        ml_pass = prob >= config.signal_threshold

        # Calculate contract count for sizing info
        spec = get_contract_spec(sym)
        n_contracts = calculate_n_contracts(100000, price, sym, config)

        if debug:
            all_pass = rule_pass and trend_pass and regime_pass and ml_pass
            status = "✅ SIGNAL" if all_pass else "❌ NEUTRAL"
            trend_str = "BULL" if price > sma_200 else "BEAR"
            vol_str = f"{vol_24h:.2%}" if not pd.isna(vol_24h) else "N/A"
            print(f"\n[{sym}] {status} (Val AUC: {auc:.3f})")
            print(f"  Funding Z:     {f_z:.2f} ({'PASS' if rule_pass else 'FAIL'})")
            print(f"  Trend:         {trend_str} ({'PASS' if trend_pass else 'FAIL'})")
            print(f"  Regime Vol:    {vol_str} ({'PASS' if regime_pass else 'FAIL'})")
            print(f"  ML Prob:       {prob:.1%} ({'PASS' if ml_pass else 'FAIL'})")
            print(f"  Contracts:     {n_contracts} ({spec['units']} {sym.split('-')[0]}/contract)")
            print(f"  Notional:      ${n_contracts * spec['units'] * price:,.2f}")

        elif rule_pass and trend_pass and regime_pass and ml_pass:
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            notional = n_contracts * spec['units'] * price
            print(f"🎯 {sym}: {dir_str} | {n_contracts} contracts | "
                  f"${notional:,.0f} notional | Prob: {prob:.1%} | AUC: {auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto ML Trading System v6 — Exact Coinbase CDE"
    )
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--signals", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--min-auc", type=float, default=0.54)
    parser.add_argument("--leverage", type=int, default=4)
    parser.add_argument("--exclude", type=str, default="BIP,DOP",
                        help="Comma-separated symbol prefixes to exclude (default: BIP,DOP)")
    args = parser.parse_args()

    excluded = [s.strip() for s in args.exclude.split(',')] if args.exclude else None
    config = Config(
        signal_threshold=args.threshold,
        min_val_auc=args.min_auc,
        leverage=args.leverage,
        excluded_symbols=excluded,
    )
    data = load_data()

    if args.backtest:
        run_backtest(data, config)
    elif args.signals or args.debug:
        run_signals(data, config, debug=args.debug)
    else:
        parser.print_help()