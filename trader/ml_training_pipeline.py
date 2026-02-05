"""
Parameter Sweep - Find the Sweet Spot

Target: ~0.3-0.5 trades/day = 1 trade every 2-3 days = ~120-180 trades/year

Testing:
1. Different funding thresholds (1.5, 1.75, 2.0)
2. Longs only vs both directions (longs performed better: 58.8% vs 55.4%)
3. Regime filters
4. Equal stop/TP (1:1 R:R needs 52%+ WR)
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"


@dataclass
class Config:
    """Test configuration."""
    funding_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    longs_only: bool
    block_bull_shorts: bool
    block_bear_longs: bool
    max_hold_hours: int
    cooldown_hours: int


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    regime: str


def run_backtest(
    config: Config,
    symbols: List[str],
    features_dir: Path,
    db_path: str,
) -> Tuple[List[Trade], Dict]:
    """Run backtest with given config."""
    
    # Load data
    data = {}
    for symbol in symbols:
        features = load_features(features_dir, symbol)
        ohlcv = load_ohlcv(db_path, symbol)
        
        if features is not None and ohlcv is not None:
            common = features.index.intersection(ohlcv.index)
            if len(common) > 1000:
                data[symbol] = {'features': features.loc[common], 'ohlcv': ohlcv.loc[common]}
    
    if not data:
        return [], {}
    
    all_ts = sorted(set().union(*[set(d['ohlcv'].index) for d in data.values()]))
    
    positions = {}
    cooldowns = {}
    trades = []
    
    for ts in all_ts:
        for symbol, d in data.items():
            if ts not in d['ohlcv'].index:
                continue
            
            ohlcv = d['ohlcv']
            features = d['features']
            
            price = ohlcv.loc[ts, 'close']
            high = ohlcv.loc[ts, 'high']
            low = ohlcv.loc[ts, 'low']
            
            # Manage position
            if symbol in positions:
                pos = positions[symbol]
                exit_reason, exit_price = check_exit(pos, high, low, price, ts, config)
                
                if exit_reason:
                    pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
                    
                    trades.append(Trade(
                        entry_time=pos['entry_time'],
                        exit_time=ts,
                        symbol=symbol,
                        direction=pos['direction'],
                        entry_price=pos['entry_price'],
                        exit_price=exit_price,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        regime=pos['regime'],
                    ))
                    
                    del positions[symbol]
                    cooldowns[symbol] = ts
                    continue
            
            # Check new entry
            if symbol not in positions:
                if symbol in cooldowns:
                    if (ts - cooldowns[symbol]).total_seconds() / 3600 < config.cooldown_hours:
                        continue
                
                if len(positions) >= 3:
                    continue
                
                signal = check_entry(ts, features, ohlcv, config)
                
                if signal:
                    if signal['direction'] == 1:
                        stop = price * (1 - config.stop_loss_pct)
                        tp = price * (1 + config.take_profit_pct)
                    else:
                        stop = price * (1 + config.stop_loss_pct)
                        tp = price * (1 - config.take_profit_pct)
                    
                    positions[symbol] = {
                        'entry_time': ts,
                        'entry_price': price,
                        'direction': signal['direction'],
                        'stop': stop,
                        'tp': tp,
                        'regime': signal['regime'],
                    }
    
    # Close remaining
    for symbol, pos in positions.items():
        if symbol in data:
            last_ts = data[symbol]['ohlcv'].index[-1]
            last_price = data[symbol]['ohlcv'].loc[last_ts, 'close']
            pnl_pct = (last_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
            
            trades.append(Trade(
                entry_time=pos['entry_time'],
                exit_time=last_ts,
                symbol=symbol,
                direction=pos['direction'],
                entry_price=pos['entry_price'],
                exit_price=last_price,
                pnl_pct=pnl_pct,
                exit_reason='end',
                regime=pos['regime'],
            ))
    
    metrics = calculate_metrics(trades, all_ts)
    return trades, metrics


def check_entry(ts, features, ohlcv, config: Config) -> Optional[Dict]:
    """Check for entry signal."""
    if ts not in features.index:
        return None
    
    row = features.loc[ts]
    funding_z = get_val(row, 'funding_rate_zscore', 0)
    
    if abs(funding_z) < config.funding_threshold:
        return None
    
    direction = -1 if funding_z > 0 else 1
    regime = get_regime(ohlcv, ts)
    
    # Longs only filter
    if config.longs_only and direction == -1:
        return None
    
    # Regime filters
    if config.block_bull_shorts and direction == -1 and regime in ['bull', 'strong_bull']:
        return None
    
    if config.block_bear_longs and direction == 1 and regime in ['bear', 'strong_bear']:
        return None
    
    return {'direction': direction, 'regime': regime, 'funding_z': funding_z}


def check_exit(pos, high, low, close, ts, config: Config) -> Tuple[Optional[str], float]:
    """Check exit conditions."""
    if pos['direction'] == 1:
        if low <= pos['stop']:
            return 'stop_loss', pos['stop']
        if high >= pos['tp']:
            return 'take_profit', pos['tp']
    else:
        if high >= pos['stop']:
            return 'stop_loss', pos['stop']
        if low <= pos['tp']:
            return 'take_profit', pos['tp']
    
    hold_hours = (ts - pos['entry_time']).total_seconds() / 3600
    if hold_hours >= config.max_hold_hours:
        return 'max_hold', close
    
    return None, close


def get_regime(ohlcv, ts) -> str:
    """Get regime."""
    if ts not in ohlcv.index:
        return 'neutral'
    
    loc = ohlcv.index.get_loc(ts)
    if loc < 168:
        return 'neutral'
    
    hist = ohlcv.iloc[max(0, loc-168):loc+1]
    current = hist['close'].iloc[-1]
    ma = hist['close'].mean()
    trend = (current - ma) / ma
    
    if trend > 0.12:
        return 'strong_bull'
    elif trend > 0.04:
        return 'bull'
    elif trend < -0.12:
        return 'strong_bear'
    elif trend < -0.04:
        return 'bear'
    return 'neutral'


def get_val(row, key, default=0):
    if key in row.index:
        val = row[key]
        return float(val) if pd.notna(val) else default
    return default


def calculate_metrics(trades: List[Trade], timestamps: List) -> Dict:
    """Calculate metrics."""
    if not trades:
        return {'n_trades': 0, 'trades_per_day': 0}
    
    n_days = len(timestamps) / 24
    pnls = [t.pnl_pct for t in trades]
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    
    metrics = {
        'n_trades': len(trades),
        'trades_per_day': len(trades) / n_days,
        'win_rate': len(wins) / len(trades) if trades else 0,
        'avg_win': np.mean([t.pnl_pct * 100 for t in wins]) if wins else 0,
        'avg_loss': np.mean([t.pnl_pct * 100 for t in losses]) if losses else 0,
        'total_return': sum(pnls) * 100,
        'avg_return': np.mean(pnls) * 100,
    }
    
    if wins and losses:
        metrics['realized_rr'] = abs(metrics['avg_win'] / metrics['avg_loss'])
    else:
        metrics['realized_rr'] = 0
    
    gross_win = sum(t.pnl_pct for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_pct for t in losses)) if losses else 1
    metrics['profit_factor'] = gross_win / gross_loss if gross_loss > 0 else 0
    
    # EV
    if metrics['win_rate'] > 0:
        metrics['ev_per_trade'] = metrics['win_rate'] * metrics['avg_win'] + (1 - metrics['win_rate']) * metrics['avg_loss']
    else:
        metrics['ev_per_trade'] = 0
    
    # By exit
    for reason in ['take_profit', 'stop_loss', 'max_hold']:
        r_trades = [t for t in trades if t.exit_reason == reason]
        metrics[f'pct_{reason}'] = len(r_trades) / len(trades) * 100 if trades else 0
    
    return metrics


def load_features(features_dir, symbol):
    for name in [f"{symbol}_features.csv", f"{symbol.replace('-', '_')}_features.csv"]:
        path = features_dir / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in ['event_time', 'timestamp', 'datetime']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df.set_index(col, inplace=True)
                        return df
                return pd.read_csv(path, index_col=0, parse_dates=True)
            except:
                pass
    return None


def load_ohlcv(db_path, symbol):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT event_time, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = '1h' ORDER BY event_time",
            conn, params=(symbol,), parse_dates=['event_time']
        )
        conn.close()
        df.set_index('event_time', inplace=True)
        return df
    except:
        return None


def main():
    features_dir = Path(FEATURES_DIR)
    
    # Find symbols
    symbols = []
    for f in features_dir.glob("*_features.csv"):
        symbols.append(f.stem.replace("_features", "").replace("_", "-"))
    
    print("=" * 90)
    print("PARAMETER SWEEP - Finding the Sweet Spot")
    print("=" * 90)
    print(f"Target: ~0.3-0.5 trades/day (1 trade every 2-3 days)")
    print(f"Symbols: {symbols}")
    print()
    
    # Parameter combinations to test
    configs = [
        # Threshold | SL | TP | Longs Only | Block Bull Shorts | Block Bear Longs | Max Hold | Cooldown
        
        # Test different thresholds with equal SL/TP
        Config(1.5, 0.08, 0.08, False, True, False, 72, 12),   # z>1.5, 8%/8%, block bull shorts
        Config(1.75, 0.08, 0.08, False, True, False, 72, 12),  # z>1.75
        Config(2.0, 0.08, 0.08, False, True, False, 72, 12),   # z>2.0
        
        # Test longs only (they performed better)
        Config(1.5, 0.08, 0.08, True, False, False, 72, 12),   # z>1.5 longs only
        Config(1.75, 0.08, 0.08, True, False, False, 72, 12),  # z>1.75 longs only
        
        # Test tighter stops with higher TP (1.5:1 R:R)
        Config(1.5, 0.06, 0.09, False, True, True, 72, 12),    # Block both adverse regimes
        Config(1.75, 0.06, 0.09, False, True, True, 72, 12),
        
        # Test wider stops
        Config(1.5, 0.10, 0.10, False, True, False, 96, 12),   # 10%/10%, longer hold
        Config(1.75, 0.10, 0.10, False, True, False, 96, 12),
        
        # Very selective - higher threshold, block adverse regimes
        Config(1.75, 0.08, 0.08, False, True, True, 72, 8),    # Block both
        Config(2.0, 0.10, 0.10, False, True, True, 96, 8),     # z>2.0, more room
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        desc = f"z>{config.funding_threshold} SL{config.stop_loss_pct:.0%}/TP{config.take_profit_pct:.0%}"
        if config.longs_only:
            desc += " LONGS"
        if config.block_bull_shorts:
            desc += " !bull"
        if config.block_bear_longs:
            desc += " !bear"
        
        print(f"Testing [{i+1}/{len(configs)}]: {desc}...", end=" ", flush=True)
        
        trades, metrics = run_backtest(config, symbols, features_dir, DB_PATH)
        
        result = {
            'config': desc,
            'threshold': config.funding_threshold,
            'sl': config.stop_loss_pct,
            'tp': config.take_profit_pct,
            'longs_only': config.longs_only,
            **metrics
        }
        results.append(result)
        
        status = "✅" if metrics.get('profit_factor', 0) >= 1.0 else "❌"
        print(f"{status} {metrics['n_trades']} trades, {metrics['trades_per_day']:.2f}/day, "
              f"WR {metrics['win_rate']:.1%}, PF {metrics.get('profit_factor', 0):.2f}, "
              f"Return {metrics['total_return']:.1f}%")
    
    # Sort by profit factor
    results.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)
    
    print("\n" + "=" * 90)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 90)
    
    print(f"\n{'Config':<50} {'Trades/Day':<12} {'Win Rate':<10} {'R:R':<8} {'PF':<8} {'Return':<10} {'EV/Trade':<10}")
    print("-" * 108)
    
    for r in results[:5]:
        print(f"{r['config']:<50} {r['trades_per_day']:<12.2f} {r['win_rate']:<10.1%} "
              f"{r.get('realized_rr', 0):<8.2f} {r.get('profit_factor', 0):<8.2f} "
              f"{r['total_return']:<10.1f}% {r.get('ev_per_trade', 0):<10.3f}%")
    
    # Find best that meets activity target
    print("\n" + "=" * 90)
    print("BEST CONFIG FOR TARGET ACTIVITY (0.3-0.5 trades/day)")
    print("=" * 90)
    
    target_results = [r for r in results if 0.2 <= r['trades_per_day'] <= 0.6]
    if target_results:
        target_results.sort(key=lambda x: x.get('profit_factor', 0), reverse=True)
        best = target_results[0]
        
        print(f"\n🎯 BEST CONFIG: {best['config']}")
        print(f"   Trades/Day: {best['trades_per_day']:.2f} ({best['trades_per_day']*7:.1f}/week)")
        print(f"   Win Rate: {best['win_rate']:.1%}")
        print(f"   Avg Win: +{best['avg_win']:.2f}%")
        print(f"   Avg Loss: {best['avg_loss']:.2f}%")
        print(f"   R:R: {best.get('realized_rr', 0):.2f}:1")
        print(f"   Profit Factor: {best.get('profit_factor', 0):.2f}")
        print(f"   Total Return: {best['total_return']:.1f}%")
        print(f"   EV per trade: {best.get('ev_per_trade', 0):.3f}%")
        print(f"   TP exits: {best.get('pct_take_profit', 0):.1f}%")
        print(f"   SL exits: {best.get('pct_stop_loss', 0):.1f}%")
        
        if best.get('profit_factor', 0) >= 1.0:
            print(f"\n   ✅ VIABLE FOR PAPER TRADING")
        else:
            print(f"\n   ⚠️ Still not profitable - may need further optimization")
    else:
        print("\n   No configs in target activity range. Try adjusting parameters.")
    
    # Save results
    output_dir = Path("./results/sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Full results saved to {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()