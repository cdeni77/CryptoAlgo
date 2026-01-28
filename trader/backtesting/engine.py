"""
Backtesting Engine for Crypto Perpetual Futures Trading System.

Implements event-driven backtesting with realistic execution simulation.
"""

import logging
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Side(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """Trading signal from strategy."""
    timestamp: datetime
    symbol: str
    direction: Side
    confidence: float = 1.0
    target_weight: float = 1.0
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Current position in an instrument."""
    symbol: str
    side: Side
    size: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Executed trade record."""
    timestamp: datetime
    symbol: str
    side: Side
    size: float
    price: float
    commission: float
    slippage: float
    signal_price: float
    pnl: float = 0.0
    reason: str = ""


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    def to_dict(self) -> Dict:
        return {
            'total_return': f"{self.total_return*100:.2f}%",
            'annualized_return': f"{self.annualized_return*100:.2f}%",
            'annualized_volatility': f"{self.annualized_volatility*100:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'sortino_ratio': f"{self.sortino_ratio:.2f}",
            'max_drawdown': f"{self.max_drawdown*100:.2f}%",
            'calmar_ratio': f"{self.calmar_ratio:.2f}",
            'win_rate': f"{self.win_rate*100:.1f}%",
            'profit_factor': f"{self.profit_factor:.2f}",
            'num_trades': self.num_trades,
            'avg_trade_return': f"{self.avg_trade_return*100:.4f}%",
        }
    
    def __str__(self):
        return (
            f"Return: {self.total_return*100:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown*100:.2f}% | "
            f"Trades: {self.num_trades} | "
            f"WinRate: {self.win_rate*100:.1f}%"
        )


@dataclass
class CostModel:
    """
    Realistic cost model for perpetual futures trading.
    
    Coinbase US Perps fees (as of Jan 2026):
    - Trading fee: 0.1% per trade (10 bps)
    - Round-trip: ~0.2% (20 bps)
    """
    maker_fee_bps: float = 10.0    # 0.10%
    taker_fee_bps: float = 10.0    # 0.10%
    base_slippage_bps: float = 2.0  # ~0.02%
    volatility_slippage_multiplier: float = 1.5
    size_impact_coefficient: float = 0.02
    
    def calculate_cost(
        self,
        size_usd: float,
        is_maker: bool = False,
        volatility_ratio: float = 1.0,
        daily_volume: float = 1e9,
    ) -> Tuple[float, float]:
        """Returns (fee_usd, slippage_usd)."""
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee_usd = size_usd * fee_bps / 10000
        
        if is_maker:
            slippage_usd = 0.0
        else:
            vol_adjusted = self.base_slippage_bps * max(1.0, volatility_ratio)
            size_impact = self.size_impact_coefficient * (size_usd / daily_volume) * 10000
            total_slippage_bps = vol_adjusted + size_impact
            slippage_usd = size_usd * total_slippage_bps / 10000
        
        return fee_usd, slippage_usd


class Portfolio:
    """Portfolio tracker with position management."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.2,
        max_leverage: float = 3.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.closed_trades: List[Trade] = []
    
    @property
    def equity(self) -> float:
        position_value = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.size) for p in self.positions.values())
    
    def update_position(self, symbol: str, current_price: float, timestamp: datetime):
        """Update position mark-to-market."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.side == Side.LONG:
                pos.unrealized_pnl = pos.size * (current_price - pos.entry_price) / pos.entry_price
            else:
                pos.unrealized_pnl = pos.size * (pos.entry_price - current_price) / pos.entry_price
    
    def execute_trade(
        self,
        signal: Signal,
        current_price: float,
        cost_model: CostModel,
        volatility_ratio: float = 1.0,
    ) -> Optional[Trade]:
        """Execute a trade based on signal."""
        
        current_pos = self.positions.get(signal.symbol)
        current_side = current_pos.side if current_pos else Side.FLAT
        current_size = current_pos.size if current_pos else 0
        
        # Calculate target size
        if signal.direction == Side.FLAT:
            target_size = 0
        else:
            base_size = self.equity * signal.target_weight * signal.confidence
            max_position = self.equity * self.max_position_pct
            target_size = min(base_size, max_position)
        
        # Determine trade
        if signal.direction == Side.FLAT and current_pos:
            trade_size = current_size
            trade_side = Side.SHORT if current_side == Side.LONG else Side.LONG
        elif signal.direction == current_side:
            return None
        elif current_pos and signal.direction != Side.FLAT:
            trade_size = current_size + target_size
            trade_side = signal.direction
        else:
            trade_size = target_size
            trade_side = signal.direction
        
        if trade_size < 100:
            return None
        
        # Calculate costs
        fee, slippage = cost_model.calculate_cost(
            trade_size, is_maker=False, volatility_ratio=volatility_ratio
        )
        
        # Fill price
        slippage_pct = slippage / trade_size if trade_size > 0 else 0
        if trade_side == Side.LONG:
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)
        
        # Calculate PnL if closing
        trade_pnl = 0.0
        if current_pos:
            if current_side == Side.LONG:
                trade_pnl = current_size * (current_price - current_pos.entry_price) / current_pos.entry_price
            else:
                trade_pnl = current_size * (current_pos.entry_price - current_price) / current_pos.entry_price
            trade_pnl -= (fee + slippage)
            self.cash += trade_pnl
            del self.positions[signal.symbol]
        else:
            self.cash -= (fee + slippage)
        
        trade = Trade(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=trade_side,
            size=trade_size,
            price=fill_price,
            commission=fee,
            slippage=slippage,
            signal_price=current_price,
            pnl=trade_pnl,
            reason=getattr(signal, 'reason', ''),
        )
        self.trades.append(trade)
        
        if current_pos:
            self.closed_trades.append(trade)
        
        # Open new position
        if signal.direction != Side.FLAT:
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                side=signal.direction,
                size=target_size,
                entry_price=fill_price,
                entry_time=signal.timestamp,
            )
        
        return trade
    
    def record_equity(self, timestamp: datetime):
        self.equity_curve.append((timestamp, self.equity))


class Backtester:
    """Event-driven backtester."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: Optional[CostModel] = None,
        max_position_pct: float = 0.2,
        max_leverage: float = 3.0,
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
    
    def run(
        self,
        strategy,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple['Portfolio', PerformanceMetrics]:
        """Run backtest."""
        
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_position_pct=self.max_position_pct,
            max_leverage=self.max_leverage,
        )
        
        # Get timestamps
        all_timestamps = set()
        for df in ohlcv_data.values():
            all_timestamps.update(df.index.tolist())
        timestamps = sorted(all_timestamps)
        
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]
        
        if not timestamps:
            return portfolio, self._empty_metrics()
        
        # Event loop
        for timestamp in timestamps:
            # Update positions
            for symbol, df in ohlcv_data.items():
                if timestamp in df.index:
                    price = df.loc[timestamp, 'close']
                    portfolio.update_position(symbol, price, timestamp)
            
            # Generate signals
            signals = strategy.generate_signals(
                timestamp=timestamp,
                ohlcv_data=ohlcv_data,
                features=features,
                portfolio=portfolio,
            )
            
            # Execute
            for signal in signals:
                if signal.symbol in ohlcv_data:
                    df = ohlcv_data[signal.symbol]
                    if timestamp in df.index:
                        price = df.loc[timestamp, 'close']
                        vol_ratio = self._get_vol_ratio(features, signal.symbol, timestamp)
                        portfolio.execute_trade(signal, price, self.cost_model, vol_ratio)
            
            portfolio.record_equity(timestamp)
        
        return portfolio, self._calculate_metrics(portfolio)
    
    def _get_vol_ratio(self, features, symbol, timestamp) -> float:
        if symbol in features and 'vol_regime_ratio' in features[symbol].columns:
            if timestamp in features[symbol].index:
                val = features[symbol].loc[timestamp, 'vol_regime_ratio']
                return val if not pd.isna(val) else 1.0
        return 1.0
    
    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_return=0, annualized_return=0, annualized_volatility=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, calmar_ratio=0,
            win_rate=0, profit_factor=0, num_trades=0, avg_trade_return=0,
            avg_win=0, avg_loss=0, max_consecutive_wins=0, max_consecutive_losses=0
        )
    
    def _calculate_metrics(self, portfolio: Portfolio) -> PerformanceMetrics:
        """Calculate performance metrics."""
        
        if len(portfolio.equity_curve) < 2:
            return self._empty_metrics()
        
        equity_df = pd.DataFrame(portfolio.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['equity'].pct_change().dropna()
        
        if len(returns) < 2:
            return self._empty_metrics()
        
        total_return = (portfolio.equity / portfolio.initial_capital) - 1
        
        # Annualize (hourly data)
        periods_per_year = 365 * 24
        n_periods = len(returns)
        years = n_periods / periods_per_year
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        annualized_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 1
        sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        # Trade stats
        trade_pnls = [t.pnl for t in portfolio.closed_trades]
        num_trades = len(trade_pnls)
        
        if num_trades > 0:
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            
            win_rate = len(wins) / num_trades
            avg_trade = np.mean(trade_pnls) / portfolio.initial_capital
            avg_win = np.mean(wins) / portfolio.initial_capital if wins else 0
            avg_loss = np.mean(losses) / portfolio.initial_capital if losses else 0
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            max_wins = max_losses = curr_wins = curr_losses = 0
            for pnl in trade_pnls:
                if pnl > 0:
                    curr_wins += 1
                    curr_losses = 0
                    max_wins = max(max_wins, curr_wins)
                else:
                    curr_losses += 1
                    curr_wins = 0
                    max_losses = max(max_losses, curr_losses)
        else:
            win_rate = avg_trade = avg_win = avg_loss = 0
            profit_factor = 0
            max_wins = max_losses = 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_return=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
        )