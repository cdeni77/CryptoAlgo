"""
Crypto ML Trading System v8 — Per-Coin Profiles + Model Persistence

KEY CHANGES from v7:
  - Per-coin profiles with tuned thresholds, exits, hyperparameters
  - All 5 coins trade (no exclusions by default)
  - Coin-specific extra features feed into ML model
  - Model saving to disk after training (joblib)
  - All coins use momentum strategy with per-coin parameter tuning

Coinbase CDE fee model: 0.10% per side, $0.20 minimum per contract.
Funding: Binance 8h data, divided by 8 for hourly accrual.
"""
import argparse
import joblib
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

from core.coin_profiles import (
    get_coin_profile, save_model, load_model, list_saved_models,
    COIN_PROFILES, BASE_FEATURES, CoinProfile, MODELS_DIR,
)

warnings.filterwarnings('ignore')


# --- Paths ---
FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"

# COINBASE CDE CONTRACT SPECIFICATIONS — EXACT
CONTRACT_SPECS = {
    'BIP': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETP': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XPP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SLP': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOP': {'units': 5000,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'BTC': {'units': 0.01,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'BTC'},
    'ETH': {'units': 0.10,  'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'ETH'},
    'XRP': {'units': 500,   'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'XRP'},
    'SOL': {'units': 5,     'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'SOL'},
    'DOGE': {'units': 5000, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'DOGE'},
    'DEFAULT': {'units': 1, 'min_fee_usd': 0.20, 'fee_pct': 0.0010, 'base': 'UNKNOWN'},
}

def get_contract_spec(symbol: str) -> dict:
    if symbol in CONTRACT_SPECS:
        return CONTRACT_SPECS[symbol]
    prefix = symbol.split('-')[0] if '-' in symbol else symbol
    if prefix in CONTRACT_SPECS:
        return CONTRACT_SPECS[prefix]
    symbol_upper = symbol.upper()
    for code, spec in CONTRACT_SPECS.items():
        if code == 'DEFAULT':
            continue
        if code in symbol_upper:
            return spec
    print(f"  ⚠️ No contract spec found for '{symbol}', using DEFAULT (1 unit/contract)")
    return CONTRACT_SPECS['DEFAULT']


# BASE FEATURE LIST (fallback — coin_profiles provides per-coin lists)
FEATURE_COLUMNS = BASE_FEATURES


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    net_pnl: float
    raw_pnl: float
    funding_pnl: float
    fee_pnl: float
    pnl_dollars: float
    n_contracts: int
    notional: float
    exit_reason: str


@dataclass
class Config:
    # Walk-Forward Windows
    train_lookback_days: int = 120
    retrain_frequency_days: int = 7
    min_train_samples: int = 400

    # Signal Filters (defaults — overridden per-coin by profiles)
    signal_threshold: float = 0.80
    min_funding_z: float = 0.0
    min_momentum_magnitude: float = 0.07

    # Exit Strategy (defaults — overridden per-coin by profiles)
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    breakeven_trigger: float = 999.0
    trailing_active: bool = False
    trailing_mult: float = 999.0
    max_hold_hours: int = 96

    # Symbol selection — v8: no exclusions by default
    excluded_symbols: Optional[List[str]] = None

    # Risk
    max_positions: int = 5
    position_size: float = 0.15
    leverage: int = 4
    vol_sizing_target: float = 0.025

    # Fees — EXACT COINBASE US CDE
    fee_pct_per_side: float = 0.0010
    min_fee_per_contract: float = 0.20
    slippage_bps: float = 0.0

    # Regime Filter (defaults — overridden per-coin)
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06

    # Safety
    min_equity: float = 1000.0
    cooldown_hours: float = 24.0

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
    max_weekly_equity_growth: float = 0.03

    # Entry quality filters
    min_signal_edge: float = 0.02
    max_ensemble_std: float = 0.12

    # Walk-forward leakage protection / evaluation
    train_embargo_hours: int = 24
    oos_eval_days: int = 60


def calculate_coinbase_fee(n_contracts: int, price: float, symbol: str,
                           config: Config) -> float:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * price
    pct_fee = n_contracts * notional_per_contract * config.fee_pct_per_side
    min_fee = n_contracts * config.min_fee_per_contract
    return max(pct_fee, min_fee)


def calculate_pnl_exact(entry_price: float, exit_price: float, direction: int,
                         accum_funding: float, n_contracts: int, symbol: str,
                         config: Config) -> Tuple[float, float, float, float, float]:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * entry_price
    total_notional = n_contracts * notional_per_contract
    if total_notional == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    raw_pnl_pct = (exit_price - entry_price) / entry_price * direction
    raw_pnl_dollars = total_notional * raw_pnl_pct
    entry_fee = calculate_coinbase_fee(n_contracts, entry_price, symbol, config)
    exit_fee = calculate_coinbase_fee(n_contracts, exit_price, symbol, config)
    slippage = total_notional * (config.slippage_bps / 10000.0)
    total_fee_dollars = entry_fee + exit_fee + slippage
    total_fee_pct = total_fee_dollars / total_notional
    funding_dollars = accum_funding * total_notional
    net_pnl_dollars = raw_pnl_dollars - total_fee_dollars + funding_dollars
    net_pnl_pct = net_pnl_dollars / total_notional
    return net_pnl_pct, raw_pnl_pct, -total_fee_pct, net_pnl_dollars, total_notional


def calculate_n_contracts(equity: float, price: float, symbol: str,
                           config: Config, vol_24h: float = 0.0,
                           profile: Optional[CoinProfile] = None) -> int:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * price
    if notional_per_contract <= 0:
        return 0
    pos_size = profile.position_size if profile else config.position_size
    vol_target = profile.vol_sizing_target if profile else config.vol_sizing_target
    if vol_24h > 0 and vol_target > 0:
        vol_ratio = vol_target / vol_24h
        vol_ratio = min(vol_ratio, 1.5)
        vol_ratio = max(vol_ratio, 0.3)
        pos_size = pos_size * vol_ratio
    target_notional = equity * pos_size * config.leverage
    n_contracts = int(target_notional / notional_per_contract)
    return max(n_contracts, 0)


class MLSystem:
    def __init__(self, config: Config):
        self.config = config

    def get_feature_columns(self, available_columns: pd.Index,
                            coin_features: Optional[List[str]] = None) -> List[str]:
        feature_list = coin_features if coin_features else FEATURE_COLUMNS
        cols = [c for c in feature_list if c in available_columns]
        return cols if len(cols) >= 4 else []

    def create_labels(self, ohlcv: pd.DataFrame, features: pd.DataFrame,
                      profile: Optional[CoinProfile] = None) -> pd.Series:
        """
        MOMENTUM-BASED forward return label (v7 original, v8 cleaned).
        
        Direction from multi-timeframe momentum consensus.
        Label = 1 if price moves >= threshold in momentum direction within forward window.
        """
        forward_hours = profile.label_forward_hours if profile else self.config.label_forward_hours
        vol_target = profile.label_vol_target if profile else self.config.label_vol_target

        target = pd.Series(index=features.index, dtype=float)
        vol = ohlcv['close'].pct_change().rolling(24).std().ffill()

        # Pre-compute momentum series (vectorized — matches v7 exactly)
        ret_24h_series = ohlcv['close'].pct_change(24).ffill()
        ret_72h_series = ohlcv['close'].pct_change(72).ffill()
        sma_50_series = ohlcv['close'].rolling(50).mean()

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
            if pos_in_ohlcv + forward_hours >= len(ohlcv):
                continue

            entry_px = ohlcv.loc[ts, 'close']
            future = ohlcv.iloc[pos_in_ohlcv + 1: pos_in_ohlcv + 1 + forward_hours]
            threshold = vol_target * row_vol

            r24 = ret_24h_series.get(ts, 0)
            r72 = ret_72h_series.get(ts, 0)
            sma = sma_50_series.get(ts, entry_px)
            if pd.isna(r24): r24 = 0
            if pd.isna(r72): r72 = 0
            if pd.isna(sma): sma = entry_px

            momentum_score = (1 if r24 > 0 else -1) + (1 if r72 > 0 else -1) + (1 if entry_px > sma else -1)

            if momentum_score >= 2:
                direction = 1
            elif momentum_score <= -2:
                direction = -1
            else:
                continue

            if direction == 1:
                max_excursion = (future['high'].max() - entry_px) / entry_px
            else:
                max_excursion = (entry_px - future['low'].min()) / entry_px

            target.loc[ts] = 1.0 if max_excursion >= threshold else 0.0

        return target

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              profile: Optional[CoinProfile] = None) -> Optional[Tuple]:
        if len(X_train) < self.config.min_train_samples:
            return None
        if y_train.sum() < 15 or (1 - y_train).sum() < 15:
            return None
        if len(X_val) < 30 or y_val.sum() < 5:
            return None

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_est = profile.n_estimators if profile else 100
        depth = profile.max_depth if profile else 3
        lr = profile.learning_rate if profile else 0.05
        min_child = profile.min_child_samples if profile else 20

        base_model = lgb.LGBMClassifier(
            n_estimators=n_est, 
            max_depth=depth, 
            learning_rate=lr,
            class_weight='balanced', 
            verbose=-1,
            min_child_samples=min_child, 
            reg_alpha=0.1, 
            reg_lambda=0.1,
            n_jobs=1  
        )
        base_model.fit(X_train_scaled, y_train)

        val_probs = base_model.predict_proba(X_val_scaled)[:, 1]
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            return None

        min_auc = profile.min_val_auc if profile else self.config.min_val_auc
        if auc < min_auc:
            return None

        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(val_probs, y_val)

        X_full = np.vstack([X_train_scaled, X_val_scaled])
        y_full = pd.concat([y_train, y_val])
        base_model.fit(X_full, y_full)

        return (base_model, scaler, iso, auc)


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

    for sym in data:
        spec = get_contract_spec(sym)
        price = data[sym]['ohlcv']['close'].iloc[-1]
        notional = spec['units'] * price
        eff_fee = max(0.20, notional * 0.0010) / notional * 100
        profile = get_coin_profile(sym)
        print(f"  {sym}: {spec['units']} units/contract, "
              f"~${notional:.2f}/contract, "
              f"effective fee: {eff_fee:.3f}% per side | "
              f"profile: {profile.name} momentum")

    return data


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


# BACKTEST — v8 PER-COIN PROFILES
def _get_profile(symbol: str, profile_overrides: Optional[Dict[str, CoinProfile]] = None) -> CoinProfile:
    if profile_overrides:
        prefix = symbol.split('-')[0].upper()
        for profile in profile_overrides.values():
            if prefix in profile.prefixes:
                return profile
    return get_coin_profile(symbol)


def run_backtest(all_data: Dict, config: Config,
                 profile_overrides: Optional[Dict[str, CoinProfile]] = None):
    system = MLSystem(config)

    all_ts = [ts for d in all_data.values() for ts in d['ohlcv'].index]
    if not all_ts:
        return
    current_date = min(all_ts) + timedelta(days=config.train_lookback_days)
    end_date = max(all_ts)

    print(f"\n⏩ STARTING BACKTEST (v8 — Per-Coin Profiles)")
    print(f"   Period: {current_date.date()} to {end_date.date()}")
    print(f"   Leverage: {config.leverage}x | Fee: 0.10%/side (0.20% round-trip)")
    print(f"   Coin profiles:")
    for sym in all_data:
        p = _get_profile(sym, profile_overrides)
        print(f"     {sym}: {p.name} (momentum) | thresh={p.signal_threshold} | "
              f"TP={p.vol_mult_tp}x SL={p.vol_mult_sl}x | hold={p.max_hold_hours}h")

    equity = 100_000.0
    peak_equity = equity
    max_drawdown = 0.0
    completed_trades = []
    active_positions = {}
    last_exit_time = {}
    models_rejected = 0
    models_accepted = 0
    regime_filtered = 0
    no_contracts = 0

    weekly_equity_base = equity

    while current_date < end_date:
        if equity < config.min_equity:
            print(f"\n🛑 EQUITY BELOW MINIMUM (${equity:,.2f}). Stopping.")
            break

        week_end = current_date + timedelta(days=config.retrain_frequency_days)

        weekly_equity_base = min(equity, weekly_equity_base * (1 + config.max_weekly_equity_growth))
        weekly_equity_base = max(weekly_equity_base, equity * 0.9)

        # --- TRAINING (ENSEMBLE: 3 lookback offsets for stability) ---
        ensemble_offsets = [0, 3, 6]
        all_ensemble_models = {}

        for offset in ensemble_offsets:
            train_start = current_date - timedelta(days=config.train_lookback_days + offset)

            for sym, d in all_data.items():
                profile = _get_profile(sym, profile_overrides)
                feat, ohlc = d['features'], d['ohlcv']
                cols = system.get_feature_columns(feat.columns, profile.feature_columns)
                if not cols:
                    continue

                embargo_hours = max(config.train_embargo_hours, profile.label_forward_hours, 1)
                train_end = current_date - timedelta(hours=embargo_hours)
                if train_end <= train_start:
                    continue

                train_feat = feat.loc[train_start:train_end]
                train_ohlc = ohlc.loc[train_start:train_end + timedelta(hours=profile.label_forward_hours)]

                if len(train_feat) < config.min_train_samples:
                    continue

                y = system.create_labels(train_ohlc, train_feat, profile=profile)
                valid_idx = y.dropna().index
                X_all = train_feat.loc[valid_idx, cols]
                y_all = y.loc[valid_idx]

                if len(X_all) < config.min_train_samples:
                    continue

                split_idx = int(len(X_all) * (1 - config.val_fraction))
                X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
                y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

                result = system.train(X_tr, y_tr, X_vl, y_vl, profile=profile)
                if result:
                    model, scaler, iso, auc = result
                    if sym not in all_ensemble_models:
                        all_ensemble_models[sym] = []
                    all_ensemble_models[sym].append((model, scaler, cols, iso, auc))
                    models_accepted += 1
                else:
                    models_rejected += 1

        models = {}
        for sym, ensemble_list in all_ensemble_models.items():
            if ensemble_list:
                models[sym] = ensemble_list

        # --- TRADING ---
        test_start = current_date + timedelta(hours=1)
        test_hours = pd.date_range(test_start, week_end, freq='1h')

        for ts in test_hours:
            # 1. MANAGE EXITS
            to_close = []
            for sym, pos in active_positions.items():
                if ts not in all_data[sym]['ohlcv'].index:
                    continue

                pos_profile = _get_profile(sym, profile_overrides)

                funding_8h_bps = all_data[sym]['features'].loc[ts].get('funding_rate_bps', 0.0)
                if pd.isna(funding_8h_bps):
                    funding_8h_bps = 0.0
                funding_hourly_bps = funding_8h_bps / 8.0
                pos['accum_funding'] += -(funding_hourly_bps / 10000.0) * pos['dir']

                bar = all_data[sym]['ohlcv'].loc[ts]
                direction = pos['dir']

                if direction == 1:
                    pos['peak_price'] = max(pos['peak_price'], bar['high'])
                else:
                    pos['peak_price'] = min(pos['peak_price'], bar['low'])

                peak_move = (pos['peak_price'] - pos['entry']) / pos['entry'] * direction

                if not pos['at_breakeven'] and peak_move >= config.breakeven_trigger * pos['vol']:
                    effective_fee_pct = pos['effective_fee_pct']
                    pos['sl'] = pos['entry'] * (1 + effective_fee_pct * direction)
                    pos['at_breakeven'] = True

                if pos['at_breakeven'] and config.trailing_active:
                    if direction == 1:
                        trail_sl = pos['peak_price'] * (1 - config.trailing_mult * pos['vol'])
                        pos['sl'] = max(pos['sl'], trail_sl)
                    else:
                        trail_sl = pos['peak_price'] * (1 + config.trailing_mult * pos['vol'])
                        pos['sl'] = min(pos['sl'], trail_sl)

                exit_price, reason = None, None

                tp_hit = (direction == 1 and bar['high'] >= pos['tp']) or \
                         (direction == -1 and bar['low'] <= pos['tp'])
                sl_hit = (direction == 1 and bar['low'] <= pos['sl']) or \
                         (direction == -1 and bar['high'] >= pos['sl'])

                if tp_hit and sl_hit:
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

                # Max hold — use per-coin profile
                if not exit_price and (ts - pos['time']).total_seconds() / 3600 >= pos_profile.max_hold_hours:
                    exit_price, reason = bar['close'], 'max_hold'

                if exit_price:
                    net_pnl, raw_pnl, fee_pnl, pnl_dollars, notional = calculate_pnl_exact(
                        pos['entry'], exit_price, direction,
                        pos['accum_funding'], pos['n_contracts'], sym, config
                    )
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
                    last_exit_time[sym] = ts

            for sym in to_close:
                del active_positions[sym]

            # 2. ENTRIES
            if len(active_positions) < config.max_positions and equity >= config.min_equity:
                candidates = []
                for sym, ensemble_models in models.items():
                    if sym in active_positions or ts not in all_data[sym]['features'].index:
                        continue
                    if ts not in all_data[sym]['ohlcv'].index:
                        continue

                    profile = _get_profile(sym, profile_overrides)

                    # Cooldown — use per-coin profile
                    if sym in last_exit_time and \
                       (ts - last_exit_time[sym]).total_seconds() < profile.cooldown_hours * 3600:
                        continue

                    # Symbol exclusion (CLI override)
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

                    # Regime filter — per-coin thresholds
                    vol_24h = all_data[sym]['ohlcv']['close'].pct_change().rolling(24).std().get(ts, None)
                    if vol_24h is None or pd.isna(vol_24h):
                        continue
                    if vol_24h < profile.min_vol_24h or vol_24h > profile.max_vol_24h:
                        regime_filtered += 1
                        continue

                    # ── Momentum direction (all coins) ──
                    ts_loc = all_data[sym]['ohlcv'].index.get_loc(ts)

                    # v7 guard: need at least 72 bars of history
                    if ts_loc < 72:
                        continue

                    ohlcv_ts = all_data[sym]['ohlcv']
                    ret_24h = (price / ohlcv_ts['close'].iloc[ts_loc - 24] - 1)
                    ret_72h = (price / ohlcv_ts['close'].iloc[ts_loc - 72] - 1)
                    sma_50 = ohlcv_ts['close'].iloc[max(0, ts_loc - 50):ts_loc].mean()

                    # v7.3: Minimum momentum magnitude
                    if abs(ret_72h) < profile.min_momentum_magnitude:
                        continue

                    # v7.3: Require 24h and 72h returns to agree on direction
                    if ret_24h * ret_72h < 0:
                        continue

                    momentum_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)

                    if momentum_score >= 2:
                        direction = 1
                    elif momentum_score <= -2:
                        direction = -1
                    else:
                        continue

                    # Trend filter: long only above SMA200, short only below
                    if direction == 1 and price < sma_200:
                        continue
                    if direction == -1 and price > sma_200:
                        continue

                    # Funding carry filter
                    f_z = row.get('funding_rate_zscore', 0)
                    if pd.isna(f_z):
                        f_z = 0
                    if direction == 1 and f_z > 2.5:
                        continue
                    if direction == -1 and f_z < -2.5:
                        continue

                    # Correlation filter
                    if not check_correlation(sym, direction, active_positions, all_data, ts, config):
                        continue

                    probs = []
                    for (model, scaler, cols, iso, auc) in models[sym]:
                        x_in = np.nan_to_num(
                            np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0
                        )
                        raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
                        cal_prob = float(iso.predict([raw_prob])[0])
                        probs.append(cal_prob)
                    prob = float(np.mean(probs))
                    prob_std = float(np.std(probs))

                    signal_cutoff = profile.signal_threshold + config.min_signal_edge
                    if prob < signal_cutoff or prob_std > config.max_ensemble_std:
                        continue

                    edge_score = (prob - signal_cutoff) / max(0.01, prob_std + 0.01)
                    momentum_strength = abs(ret_72h)
                    rank_score = edge_score + (0.2 * momentum_strength)
                    candidates.append((rank_score, sym, profile, price, vol_24h, direction))

                slots = max(config.max_positions - len(active_positions), 0)
                for _, sym, profile, price, vol_24h, direction in sorted(candidates, key=lambda x: x[0], reverse=True)[:slots]:
                    vol = vol_24h if vol_24h > 0 else 0.02

                    n_contracts = calculate_n_contracts(
                        weekly_equity_base, price, sym, config,
                        vol_24h=vol_24h, profile=profile
                    )
                    if n_contracts < 1:
                        no_contracts += 1
                        continue

                    spec = get_contract_spec(sym)
                    notional_per_contract = spec['units'] * price
                    total_notional = n_contracts * notional_per_contract
                    entry_fee = calculate_coinbase_fee(n_contracts, price, sym, config)
                    exit_fee = entry_fee
                    effective_fee_pct = (entry_fee + exit_fee) / total_notional

                    # Per-coin TP/SL
                    active_positions[sym] = {
                        'time': ts,
                        'entry': price,
                        'dir': direction,
                        'vol': vol,
                        'tp': price * (1 + profile.vol_mult_tp * vol * direction),
                        'sl': price * (1 - profile.vol_mult_sl * vol * direction),
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

    trades_per_year = len(df) / years if years > 0 else 0
    ann_sharpe = sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    avg_fee = df['fee_pnl'].mean()
    avg_raw = df['raw_pnl'].mean()
    avg_funding = df['funding_pnl'].mean()

    print(f"\n{'=' * 70}")
    print(f"📊 BACKTEST RESULTS (v8 — Per-Coin Profiles)")
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

    print(f"\nPer-Symbol Performance:")
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        sym_wr = (sym_df['net_pnl'] > 0).mean()
        sym_pnl = sym_df['pnl_dollars'].sum()
        sym_avg = sym_df['net_pnl'].mean()
        sym_fee = sym_df['fee_pnl'].mean()
        p = _get_profile(sym, profile_overrides)
        print(f"  {sym:20s}: {len(sym_df):3d} trades | WR: {sym_wr:.0%} | "
              f"PnL: ${sym_pnl:+,.0f} | Avg: {sym_avg:.4%} | Fee: {sym_fee:.4%} | "
              f"[{p.name}/momentum]")
    print(f"{'=' * 70}")

    # ── Save final models from last walk-forward window ──
    print(f"\n💾 Saving final models...")
    saved_count = 0
    for sym, ensemble_list in models.items():
        if ensemble_list:
            best = max(ensemble_list, key=lambda x: x[4])
            model, scaler, cols, iso, auc = best
            profile = get_coin_profile(sym)
            save_model(
                symbol=sym,
                model=model,
                scaler=scaler,
                calibrator=iso,
                feature_columns=cols,
                auc=auc,
                profile_name=profile.name,
                extra_meta={
                    'strategy': 'momentum',
                    'signal_threshold': profile.signal_threshold,
                    'n_ensemble': len(ensemble_list),
                    'backtest_trades': len(df[df['symbol'] == sym]) if sym in df['symbol'].values else 0,
                },
            )
            saved_count += 1
    print(f"✅ Saved {saved_count} models to {MODELS_DIR}/")

    oos_cutoff = end_date - timedelta(days=max(config.oos_eval_days, 1))
    oos_df = df[df['exit_time'] >= oos_cutoff]
    oos_sharpe = -99.0
    oos_return = -1.0
    if len(oos_df) >= 5:
        oos_avg = oos_df['net_pnl'].mean()
        oos_std = oos_df['net_pnl'].std()
        oos_sharpe = oos_avg / oos_std if oos_std > 0 else 0.0
        oos_return = oos_df['pnl_dollars'].sum() / 100000.0

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
        'oos_trades': int(len(oos_df)),
        'oos_sharpe': oos_sharpe,
        'oos_return': oos_return,
    }


def retrain_models(all_data: Dict, config: Config, target_dir: Optional[Path] = None, train_window_days: int = 90) -> Dict:
    system = MLSystem(config)
    metrics = {}
    symbols_trained = 0
    train_end_global = pd.Timestamp.now(tz='UTC')
    train_start_global = train_end_global - pd.Timedelta(days=train_window_days)

    for sym, d in all_data.items():
        profile = get_coin_profile(sym)
        feat, ohlc = d['features'], d['ohlcv']
        cols = system.get_feature_columns(feat.columns, profile.feature_columns)
        if not cols:
            continue

        train_feat = feat.loc[feat.index >= train_start_global]
        train_ohlc = ohlc.loc[ohlc.index >= train_start_global]

        y = system.create_labels(train_ohlc, train_feat, profile=profile)
        valid = y.dropna().index
        X_all = train_feat.loc[valid, cols]
        y_all = y.loc[valid]
        if len(X_all) < config.min_train_samples:
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

        result = system.train(X_tr, y_tr, X_vl, y_vl, profile=profile)
        if not result:
            continue

        model, scaler, iso, auc = result
        symbol_train_start = X_all.index.min()
        symbol_train_end = X_all.index.max()
        model_metrics = {
            'auc': float(auc),
            'train_samples': int(len(X_all)),
            'val_samples': int(len(X_vl)),
        }
        save_model(
            symbol=sym,
            model=model,
            scaler=scaler,
            calibrator=iso,
            feature_columns=cols,
            auc=auc,
            profile_name=profile.name,
            target_dir=target_dir,
            extra_meta={
                'strategy': 'momentum',
                'signal_threshold': profile.signal_threshold,
                'train_start': symbol_train_start.isoformat() if symbol_train_start is not None else None,
                'train_end': symbol_train_end.isoformat() if symbol_train_end is not None else None,
                'metrics': model_metrics,
            },
        )
        symbols_trained += 1
        metrics[sym] = model_metrics

    return {
        'symbols_total': len(all_data),
        'symbols_trained': symbols_trained,
        'train_start': train_start_global.isoformat(),
        'train_end': train_end_global.isoformat(),
        'metrics': metrics,
    }


# LIVE SIGNALS — v8
def run_signals(all_data: Dict, config: Config, debug: bool = False):
    system = MLSystem(config)
    print(f"\n🔍 ANALYZING LIVE MARKETS (v8 — Per-Coin Profiles)...")

    for sym, d in all_data.items():
        profile = get_coin_profile(sym)
        feat, ohlc = d['features'], d['ohlcv']
        cols = system.get_feature_columns(feat.columns, profile.feature_columns)
        if not cols:
            if debug:
                print(f"\n[{sym}] ❌ Not enough features (need ≥4, got {len(cols)})")
            continue

        train_end = feat.index[-1]
        train_start = train_end - timedelta(days=config.train_lookback_days)
        train_feat = feat.loc[train_start:train_end]
        train_ohlc = ohlc.loc[train_start:train_end]

        y = system.create_labels(train_ohlc, train_feat, profile=profile)
        valid = y.dropna().index
        X_all = train_feat.loc[valid, cols]
        y_all = y.loc[valid]

        if len(X_all) < config.min_train_samples:
            if debug:
                print(f"\n[{sym}] ❌ Insufficient samples ({len(X_all)} < {config.min_train_samples})")
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

        result = system.train(X_tr, y_tr, X_vl, y_vl, profile=profile)
        if not result:
            if debug:
                print(f"\n[{sym}] ❌ MODEL REJECTED (low AUC < {profile.min_val_auc})")
            continue

        model, scaler, iso, auc = result

        # ── Save model to disk ──
        save_model(
            symbol=sym,
            model=model,
            scaler=scaler,
            calibrator=iso,
            feature_columns=cols,
            auc=auc,
            profile_name=profile.name,
            extra_meta={
                'strategy': 'momentum',
                'signal_threshold': profile.signal_threshold,
                'train_samples': len(X_all),
            },
        )

        row = feat.iloc[-1]
        price = ohlc.iloc[-1]['close']
        sma_200 = ohlc.iloc[-1]['sma_200']

        f_z = row.get('funding_rate_zscore', 0)
        if pd.isna(f_z):
            f_z = 0

        ts_loc = len(ohlc) - 1

        # ── Momentum direction (all coins) ──
        ret_24h = (price / ohlc['close'].iloc[ts_loc - 24] - 1) if ts_loc >= 24 else 0
        ret_72h = (price / ohlc['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
        sma_50 = ohlc['close'].iloc[max(0, ts_loc - 50):ts_loc].mean() if ts_loc >= 10 else price
        momentum_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)
        if momentum_score >= 2:
            direction = 1
        elif momentum_score <= -2:
            direction = -1
        else:
            if debug:
                print(f"\n[{sym}] ⏸️  No momentum consensus (score={momentum_score})")
            continue

        # Trend filter: long only above SMA200, short only below
        if direction == 1 and price < sma_200 and not pd.isna(sma_200):
            if debug:
                print(f"\n[{sym}] ⏸️  Long rejected: price < SMA200")
            continue
        if direction == -1 and price > sma_200 and not pd.isna(sma_200):
            if debug:
                print(f"\n[{sym}] ⏸️  Short rejected: price > SMA200")
            continue

        # Funding carry filter
        if direction == 1 and f_z > 2.5:
            continue
        if direction == -1 and f_z < -2.5:
            continue

        # ML prediction
        x_in = np.nan_to_num(
            np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0
        )
        raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
        prob = float(iso.predict([raw_prob])[0])
        ml_pass = prob >= profile.signal_threshold

        vol_24h = ohlc['close'].pct_change().rolling(24).std().iloc[-1]
        regime_pass = profile.min_vol_24h <= vol_24h <= profile.max_vol_24h if not pd.isna(vol_24h) else False

        # Momentum gate
        ret_72h = (price / ohlc['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
        momentum_pass = abs(ret_72h) >= profile.min_momentum_magnitude

        if debug:
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            print(f"\n[{sym}] ({profile.name})")
            print(f"  Price: ${price:,.2f} | SMA200: ${sma_200:,.2f}" if not pd.isna(sma_200) else f"  Price: ${price:,.2f}")
            print(f"  Direction: {dir_str}")
            print(f"  Raw prob: {raw_prob:.3f} → Calibrated: {prob:.3f} (thresh: {profile.signal_threshold})")
            print(f"  AUC: {auc:.3f}")
            print(f"  Gates: ML={'✅' if ml_pass else '❌'} | Regime={'✅' if regime_pass else '❌'} | Mom={'✅' if momentum_pass else '❌'}")
            print(f"  Funding z-score: {f_z:.2f}")
            print(f"  24h Vol: {vol_24h:.4f}" if not pd.isna(vol_24h) else "  24h Vol: N/A")

        if ml_pass and regime_pass and momentum_pass:
            n_contracts = calculate_n_contracts(
                100_000, price, sym, config,
                vol_24h=vol_24h if not pd.isna(vol_24h) else 0.02,
                profile=profile
            )
            spec = get_contract_spec(sym)
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            notional = n_contracts * spec['units'] * price
            print(f"🎯 {sym} [{profile.name}]: {dir_str} | "
                  f"{n_contracts} contracts | ${notional:,.0f} notional | "
                  f"Prob: {prob:.1%} | AUC: {auc:.3f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto ML Trading System v8 — Per-Coin Profiles"
    )
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--signals", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Default signal threshold (overridden by per-coin profiles)")
    parser.add_argument("--min-auc", type=float, default=0.54)
    parser.add_argument("--leverage", type=int, default=4)
    parser.add_argument("--tp", type=float, default=5.5, help="Default TP vol multiplier")
    parser.add_argument("--sl", type=float, default=3.0, help="Default SL vol multiplier")
    parser.add_argument("--momentum", type=float, default=0.07, help="Default min 72h momentum magnitude")
    parser.add_argument("--hold", type=int, default=96, help="Default max hold hours")
    parser.add_argument("--cooldown", type=float, default=24, help="Default hours cooldown after exit")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Require prob >= threshold + min-edge")
    parser.add_argument("--max-ensemble-std", type=float, default=0.12, help="Max std across ensemble probs")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated symbol prefixes to exclude (default: none)")
    args = parser.parse_args()

    excluded = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None
    config = Config(
        signal_threshold=args.threshold,
        min_val_auc=args.min_auc,
        leverage=args.leverage,
        vol_mult_tp=args.tp,
        vol_mult_sl=args.sl,
        min_momentum_magnitude=args.momentum,
        max_hold_hours=args.hold,
        cooldown_hours=args.cooldown,
        min_signal_edge=args.min_edge,
        max_ensemble_std=args.max_ensemble_std,
        excluded_symbols=excluded,
    )
    data = load_data()

    if args.backtest:
        run_backtest(data, config)
    elif args.signals or args.debug:
        run_signals(data, config, debug=args.debug)
    else:
        parser.print_help()
