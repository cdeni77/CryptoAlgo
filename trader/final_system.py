"""
FINAL PRODUCTION TRADING SYSTEM
================================

Based on parameter sweep results:
- Best config: z>2.0 SL10%/TP10% !bull !bear
- Trades/Day: 0.34 (2.4/week, ~124/year)
- Win Rate: 51.4%
- Profit Factor: 1.14
- Total Return: 287.7% over backtest period
- EV per trade: +0.386%

Key success factors:
1. 10% stop/10% TP - gives trades room to work
2. Block shorts in bull regimes (they fail)
3. Block longs in bear regimes (they fail)
4. z > 2.0 threshold - only high-conviction signals

Usage:
    # Backtest
    python final_system.py --backtest
    
    # Generate current signals
    python final_system.py --signals
    
    # Paper trade mode (checks every hour)
    python final_system.py --paper-trade
"""

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"


# =============================================================================
# WINNING CONFIGURATION
# =============================================================================

@dataclass
class ProductionConfig:
    """Production configuration - DO NOT CHANGE without re-testing."""
    
    # Signal threshold
    funding_zscore_threshold: float = 2.0
    
    # Risk management - EQUAL stop and TP
    stop_loss_pct: float = 0.10   # 10%
    take_profit_pct: float = 0.10  # 10%
    
    # Regime filters - CRITICAL for profitability
    block_bull_shorts: bool = True   # Don't short in bull markets
    block_bear_longs: bool = True    # Don't long in bear markets
    
    # Position sizing
    position_size: float = 0.15      # 15% of portfolio per trade
    max_positions: int = 3           # Max concurrent positions
    
    # Timing
    max_hold_hours: int = 96         # 4 days max
    cooldown_hours: int = 12         # 12 hours between trades per coin
    
    # Trailing stop - DISABLED (was cutting winners short)
    use_trailing_stop: bool = False
    trailing_activation_pct: float = 0.08  # Activate after 8% profit
    trailing_distance_pct: float = 0.03    # 3% trailing distance


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Signal:
    """Trading signal."""
    timestamp: datetime
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    stop_loss: float
    take_profit: float
    funding_zscore: float
    regime: str
    position_size: float


@dataclass
class Position:
    """Open position."""
    entry_time: datetime
    symbol: str
    direction: int
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    initial_stop: float
    highest_price: float
    lowest_price: float
    trailing_active: bool
    regime: str
    funding_zscore: float


@dataclass
class Trade:
    """Completed trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl_pct: float
    pnl_usd: float
    exit_reason: str
    regime: str
    funding_zscore: float
    hold_hours: float


# =============================================================================
# TRADING SYSTEM
# =============================================================================

class ProductionTradingSystem:
    """Production-ready trading system."""
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()
    
    # -------------------------------------------------------------------------
    # SIGNAL GENERATION
    # -------------------------------------------------------------------------
    
    def check_for_signal(
        self,
        timestamp: datetime,
        symbol: str,
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
    ) -> Optional[Signal]:
        """Check if there's a valid signal at this timestamp."""
        
        if timestamp not in features.index or timestamp not in ohlcv.index:
            return None
        
        row = features.loc[timestamp]
        price = ohlcv.loc[timestamp, 'close']
        
        # Get funding z-score
        funding_z = self._get_val(row, 'funding_rate_zscore', 0)
        
        # Check threshold
        if abs(funding_z) < self.config.funding_zscore_threshold:
            return None
        
        # Determine direction
        # High positive funding = short (get paid to short)
        # High negative funding = long (get paid to long)
        direction = -1 if funding_z > 0 else 1
        
        # Get regime
        regime = self._get_regime(ohlcv, timestamp)
        
        # CRITICAL REGIME FILTERS
        if self.config.block_bull_shorts:
            if direction == -1 and regime in ['bull', 'strong_bull']:
                return None
        
        if self.config.block_bear_longs:
            if direction == 1 and regime in ['bear', 'strong_bear']:
                return None
        
        # Calculate stops
        if direction == 1:  # Long
            stop_loss = price * (1 - self.config.stop_loss_pct)
            take_profit = price * (1 + self.config.take_profit_pct)
        else:  # Short
            stop_loss = price * (1 + self.config.stop_loss_pct)
            take_profit = price * (1 - self.config.take_profit_pct)
        
        return Signal(
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            funding_zscore=funding_z,
            regime=regime,
            position_size=self.config.position_size,
        )
    
    def _get_regime(self, ohlcv: pd.DataFrame, timestamp: datetime) -> str:
        """Determine market regime."""
        if timestamp not in ohlcv.index:
            return 'neutral'
        
        loc = ohlcv.index.get_loc(timestamp)
        if loc < 168:
            return 'neutral'
        
        hist = ohlcv.iloc[max(0, loc-168):loc+1]
        current = hist['close'].iloc[-1]
        ma_168 = hist['close'].mean()
        
        trend = (current - ma_168) / ma_168
        
        if trend > 0.12:
            return 'strong_bull'
        elif trend > 0.04:
            return 'bull'
        elif trend < -0.12:
            return 'strong_bear'
        elif trend < -0.04:
            return 'bear'
        return 'neutral'
    
    def _get_val(self, row, key, default=0):
        """Safely get value from row."""
        if key in row.index:
            val = row[key]
            return float(val) if pd.notna(val) else default
        return default
    
    # -------------------------------------------------------------------------
    # POSITION MANAGEMENT
    # -------------------------------------------------------------------------
    
    def check_exit(
        self,
        position: Position,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
    ) -> Tuple[Optional[str], float]:
        """Check if position should be exited."""
        
        # Update trailing stop first
        if self.config.use_trailing_stop:
            self._update_trailing_stop(position, high, low)
        
        # Check stop loss
        if position.direction == 1:  # Long
            if low <= position.stop_loss:
                return 'stop_loss', position.stop_loss
            if high >= position.take_profit:
                return 'take_profit', position.take_profit
        else:  # Short
            if high >= position.stop_loss:
                return 'stop_loss', position.stop_loss
            if low <= position.take_profit:
                return 'take_profit', position.take_profit
        
        # Max hold time
        hold_hours = (timestamp - position.entry_time).total_seconds() / 3600
        if hold_hours >= self.config.max_hold_hours:
            return 'max_hold', close
        
        return None, close
    
    def _update_trailing_stop(self, position: Position, high: float, low: float):
        """Update trailing stop if activated."""
        if position.direction == 1:  # Long
            position.highest_price = max(position.highest_price, high)
            profit_pct = (position.highest_price - position.entry_price) / position.entry_price
            
            if profit_pct >= self.config.trailing_activation_pct:
                position.trailing_active = True
                new_stop = position.highest_price * (1 - self.config.trailing_distance_pct)
                position.stop_loss = max(position.stop_loss, new_stop)
        
        else:  # Short
            position.lowest_price = min(position.lowest_price, low)
            profit_pct = (position.entry_price - position.lowest_price) / position.entry_price
            
            if profit_pct >= self.config.trailing_activation_pct:
                position.trailing_active = True
                new_stop = position.lowest_price * (1 + self.config.trailing_distance_pct)
                position.stop_loss = min(position.stop_loss, new_stop)
    
    # -------------------------------------------------------------------------
    # BACKTESTING
    # -------------------------------------------------------------------------
    
    def backtest(
        self,
        symbols: List[str],
        features_dir: Path,
        db_path: str,
    ) -> Tuple[List[Trade], Dict]:
        """Run backtest."""
        
        # Load data
        data = {}
        for symbol in symbols:
            features = self._load_features(features_dir, symbol)
            ohlcv = self._load_ohlcv(db_path, symbol)
            
            if features is not None and ohlcv is not None:
                common = features.index.intersection(ohlcv.index)
                if len(common) > 1000:
                    data[symbol] = {
                        'features': features.loc[common],
                        'ohlcv': ohlcv.loc[common]
                    }
        
        if not data:
            logger.error("No data loaded!")
            return [], {}
        
        # Get all timestamps
        all_ts = sorted(set().union(*[set(d['ohlcv'].index) for d in data.values()]))
        logger.info(f"Backtesting {len(all_ts)} bars across {len(data)} symbols")
        
        # State
        positions: Dict[str, Position] = {}
        cooldowns: Dict[str, datetime] = {}
        trades: List[Trade] = []
        initial_capital = 100000
        capital = initial_capital
        
        for ts in all_ts:
            for symbol, d in data.items():
                if ts not in d['ohlcv'].index:
                    continue
                
                ohlcv = d['ohlcv']
                features = d['features']
                
                price = ohlcv.loc[ts, 'close']
                high = ohlcv.loc[ts, 'high']
                low = ohlcv.loc[ts, 'low']
                
                # === Manage existing position ===
                if symbol in positions:
                    pos = positions[symbol]
                    exit_reason, exit_price = self.check_exit(pos, high, low, price, ts)
                    
                    if exit_reason:
                        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * pos.direction
                        pnl_usd = pnl_pct * pos.size * capital
                        hold_hours = (ts - pos.entry_time).total_seconds() / 3600
                        
                        # Subtract fees (0.1% each way)
                        fees = 0.002 * pos.size * capital
                        pnl_usd -= fees
                        
                        trades.append(Trade(
                            entry_time=pos.entry_time,
                            exit_time=ts,
                            symbol=symbol,
                            direction=pos.direction,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            size=pos.size,
                            pnl_pct=pnl_pct,
                            pnl_usd=pnl_usd,
                            exit_reason=exit_reason,
                            regime=pos.regime,
                            funding_zscore=pos.funding_zscore,
                            hold_hours=hold_hours,
                        ))
                        
                        capital += pnl_usd
                        del positions[symbol]
                        cooldowns[symbol] = ts
                        continue
                
                # === Check for new entry ===
                if symbol not in positions:
                    # Cooldown check
                    if symbol in cooldowns:
                        hours_since = (ts - cooldowns[symbol]).total_seconds() / 3600
                        if hours_since < self.config.cooldown_hours:
                            continue
                    
                    # Max positions check
                    if len(positions) >= self.config.max_positions:
                        continue
                    
                    # Check for signal
                    signal = self.check_for_signal(ts, symbol, features, ohlcv)
                    
                    if signal:
                        positions[symbol] = Position(
                            entry_time=ts,
                            symbol=symbol,
                            direction=signal.direction,
                            entry_price=signal.entry_price,
                            size=signal.position_size,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            initial_stop=signal.stop_loss,
                            highest_price=signal.entry_price,
                            lowest_price=signal.entry_price,
                            trailing_active=False,
                            regime=signal.regime,
                            funding_zscore=signal.funding_zscore,
                        )
        
        # Close remaining positions
        for symbol, pos in positions.items():
            if symbol in data:
                last_ts = data[symbol]['ohlcv'].index[-1]
                last_price = data[symbol]['ohlcv'].loc[last_ts, 'close']
                pnl_pct = (last_price - pos.entry_price) / pos.entry_price * pos.direction
                hold_hours = (last_ts - pos.entry_time).total_seconds() / 3600
                
                trades.append(Trade(
                    entry_time=pos.entry_time,
                    exit_time=last_ts,
                    symbol=symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=last_price,
                    size=pos.size,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_pct * pos.size * capital,
                    exit_reason='end_of_data',
                    regime=pos.regime,
                    funding_zscore=pos.funding_zscore,
                    hold_hours=hold_hours,
                ))
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, all_ts, initial_capital, capital)
        
        return trades, metrics
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        timestamps: List,
        initial_capital: float,
        final_capital: float,
    ) -> Dict:
        """Calculate performance metrics."""
        
        if not trades:
            return {'n_trades': 0}
        
        n_days = len(timestamps) / 24
        n_years = n_days / 365
        
        pnls = [t.pnl_pct for t in trades]
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        
        metrics = {
            'n_trades': len(trades),
            'trades_per_day': len(trades) / n_days,
            'trades_per_week': len(trades) / n_days * 7,
            'trades_per_year': len(trades) / n_years,
            'win_rate': len(wins) / len(trades),
            'avg_win_pct': np.mean([t.pnl_pct * 100 for t in wins]) if wins else 0,
            'avg_loss_pct': np.mean([t.pnl_pct * 100 for t in losses]) if losses else 0,
            'total_return_pct': (final_capital - initial_capital) / initial_capital * 100,
            'total_pnl_usd': final_capital - initial_capital,
            'avg_pnl_pct': np.mean(pnls) * 100,
            'avg_hold_hours': np.mean([t.hold_hours for t in trades]),
        }
        
        # R:R
        if wins and losses:
            metrics['realized_rr'] = abs(metrics['avg_win_pct'] / metrics['avg_loss_pct'])
        else:
            metrics['realized_rr'] = 0
        
        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe
        if np.std(pnls) > 0 and metrics['avg_hold_hours'] > 0:
            trades_per_year = 365 * 24 / metrics['avg_hold_hours']
            metrics['sharpe'] = np.mean(pnls) / np.std(pnls) * np.sqrt(trades_per_year)
        else:
            metrics['sharpe'] = 0
        
        # EV per trade
        metrics['ev_per_trade'] = (
            metrics['win_rate'] * metrics['avg_win_pct'] +
            (1 - metrics['win_rate']) * metrics['avg_loss_pct']
        )
        
        # Exit analysis
        for reason in ['take_profit', 'stop_loss', 'max_hold', 'end_of_data']:
            r_trades = [t for t in trades if t.exit_reason == reason]
            metrics[f'n_{reason}'] = len(r_trades)
            metrics[f'pct_{reason}'] = len(r_trades) / len(trades) * 100
            if r_trades:
                metrics[f'avg_pnl_{reason}'] = np.mean([t.pnl_pct * 100 for t in r_trades])
        
        # By direction
        longs = [t for t in trades if t.direction == 1]
        shorts = [t for t in trades if t.direction == -1]
        
        if longs:
            metrics['long_trades'] = len(longs)
            metrics['long_winrate'] = len([t for t in longs if t.pnl_usd > 0]) / len(longs)
        if shorts:
            metrics['short_trades'] = len(shorts)
            metrics['short_winrate'] = len([t for t in shorts if t.pnl_usd > 0]) / len(shorts)
        
        # By symbol
        for symbol in set(t.symbol for t in trades):
            s_trades = [t for t in trades if t.symbol == symbol]
            metrics[f'{symbol}_trades'] = len(s_trades)
            metrics[f'{symbol}_winrate'] = len([t for t in s_trades if t.pnl_usd > 0]) / len(s_trades)
            metrics[f'{symbol}_pnl'] = sum(t.pnl_usd for t in s_trades)
        
        return metrics
    
    # -------------------------------------------------------------------------
    # SIGNAL GENERATION (Live)
    # -------------------------------------------------------------------------
    
    def generate_current_signals(
        self,
        symbols: List[str],
        features_dir: Path,
        db_path: str,
    ) -> List[Signal]:
        """Generate signals for current timestamp."""
        
        signals = []
        
        for symbol in symbols:
            features = self._load_features(features_dir, symbol)
            ohlcv = self._load_ohlcv(db_path, symbol)
            
            if features is None or ohlcv is None:
                continue
            
            # Get latest timestamp
            common = features.index.intersection(ohlcv.index)
            if len(common) == 0:
                continue
            
            latest_ts = max(common)
            
            signal = self.check_for_signal(latest_ts, symbol, features, ohlcv)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------
    
    def _load_features(self, features_dir: Path, symbol: str) -> Optional[pd.DataFrame]:
        """Load features."""
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
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
        return None
    
    def _load_ohlcv(self, db_path: str, symbol: str) -> Optional[pd.DataFrame]:
        """Load OHLCV."""
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                "SELECT event_time, open, high, low, close, volume FROM ohlcv "
                "WHERE symbol = ? AND timeframe = '1h' ORDER BY event_time",
                conn, params=(symbol,), parse_dates=['event_time']
            )
            conn.close()
            df.set_index('event_time', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading OHLCV for {symbol}: {e}")
            return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Production Trading System")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--signals", action="store_true", help="Generate current signals")
    parser.add_argument("--paper-trade", action="store_true", help="Paper trade mode")
    parser.add_argument("--features-dir", type=str, default=str(FEATURES_DIR))
    parser.add_argument("--db-path", type=str, default=DB_PATH)
    parser.add_argument("--output", type=str, default="./results/production")
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find symbols
    symbols = []
    for f in features_dir.glob("*_features.csv"):
        symbols.append(f.stem.replace("_features", "").replace("_", "-"))
    
    system = ProductionTradingSystem()
    config = system.config
    
    if args.backtest:
        print("=" * 80)
        print("PRODUCTION SYSTEM - BACKTEST")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Funding threshold: z > {config.funding_zscore_threshold}")
        print(f"  Stop Loss: {config.stop_loss_pct:.0%}")
        print(f"  Take Profit: {config.take_profit_pct:.0%}")
        print(f"  Block bull shorts: {config.block_bull_shorts}")
        print(f"  Block bear longs: {config.block_bear_longs}")
        print(f"  Max hold: {config.max_hold_hours}h")
        print(f"  Symbols: {symbols}")
        print()
        
        trades, metrics = system.backtest(symbols, features_dir, args.db_path)
        
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Activity:")
        print(f"   Total Trades: {metrics['n_trades']}")
        print(f"   Trades/Day: {metrics['trades_per_day']:.2f}")
        print(f"   Trades/Week: {metrics['trades_per_week']:.1f}")
        print(f"   Trades/Year: {metrics['trades_per_year']:.0f}")
        print(f"   Avg Hold: {metrics['avg_hold_hours']:.1f}h")
        
        print(f"\n📈 Performance:")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Avg Win: +{metrics['avg_win_pct']:.2f}%")
        print(f"   Avg Loss: {metrics['avg_loss_pct']:.2f}%")
        print(f"   Realized R:R: {metrics['realized_rr']:.2f}:1")
        print(f"   Total Return: {metrics['total_return_pct']:.1f}%")
        print(f"   Total PnL: ${metrics['total_pnl_usd']:,.0f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"   EV per Trade: {metrics['ev_per_trade']:.3f}%")
        
        print(f"\n📊 Exit Analysis:")
        for reason in ['take_profit', 'stop_loss', 'max_hold']:
            n = metrics.get(f'n_{reason}', 0)
            pct = metrics.get(f'pct_{reason}', 0)
            avg = metrics.get(f'avg_pnl_{reason}', 0)
            print(f"   {reason}: {n} ({pct:.1f}%) avg: {avg:+.2f}%")
        
        print(f"\n📊 By Direction:")
        if 'long_trades' in metrics:
            print(f"   Long: {metrics['long_trades']} trades, {metrics['long_winrate']:.1%} win")
        if 'short_trades' in metrics:
            print(f"   Short: {metrics['short_trades']} trades, {metrics['short_winrate']:.1%} win")
        
        print(f"\n📊 By Symbol:")
        for symbol in symbols:
            if f'{symbol}_trades' in metrics:
                print(f"   {symbol}: {metrics[f'{symbol}_trades']} trades, "
                      f"{metrics[f'{symbol}_winrate']:.1%} win, "
                      f"${metrics[f'{symbol}_pnl']:,.0f}")
        
        # Save results
        trades_data = [asdict(t) for t in trades]
        for td in trades_data:
            td['entry_time'] = td['entry_time'].isoformat()
            td['exit_time'] = td['exit_time'].isoformat()
        
        with open(output_dir / 'trades.json', 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        print(f"\n✅ Results saved to {output_dir}")
        
        print("\n" + "=" * 80)
        if metrics['profit_factor'] >= 1.1 and metrics['ev_per_trade'] > 0:
            print("✅ SYSTEM VERIFIED - Ready for paper trading!")
        else:
            print("⚠️ Performance below expectations - review before trading")
        print("=" * 80)
    
    elif args.signals:
        print("=" * 80)
        print("CURRENT SIGNALS")
        print("=" * 80)
        
        signals = system.generate_current_signals(symbols, features_dir, args.db_path)
        
        if not signals:
            print("\n   No signals at current time")
        else:
            for sig in signals:
                direction = "LONG" if sig.direction == 1 else "SHORT"
                print(f"\n   📊 {sig.symbol}")
                print(f"      Direction: {direction}")
                print(f"      Entry: ${sig.entry_price:.2f}")
                print(f"      Stop Loss: ${sig.stop_loss:.2f} ({config.stop_loss_pct:.0%})")
                print(f"      Take Profit: ${sig.take_profit:.2f} ({config.take_profit_pct:.0%})")
                print(f"      Funding Z-Score: {sig.funding_zscore:.2f}")
                print(f"      Regime: {sig.regime}")
                print(f"      Position Size: {sig.position_size:.0%}")
        
        # Save signals
        signals_data = [asdict(s) for s in signals]
        for sd in signals_data:
            sd['timestamp'] = sd['timestamp'].isoformat()
        
        with open(output_dir / 'current_signals.json', 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'signals': signals_data
            }, f, indent=2)
        
        print(f"\n✅ Signals saved to {output_dir / 'current_signals.json'}")
    
    elif args.paper_trade:
        print("=" * 80)
        print("PAPER TRADING MODE")
        print("=" * 80)
        print("Checking for signals every hour...")
        print("Press Ctrl+C to stop")
        print()
        
        while True:
            signals = system.generate_current_signals(symbols, features_dir, args.db_path)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            
            if signals:
                for sig in signals:
                    direction = "LONG" if sig.direction == 1 else "SHORT"
                    print(f"[{timestamp}] 🚨 SIGNAL: {sig.symbol} {direction} @ ${sig.entry_price:.2f} "
                          f"(z={sig.funding_zscore:.2f}, regime={sig.regime})")
            else:
                print(f"[{timestamp}] No signals")
            
            time.sleep(3600)  # Check every hour
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()