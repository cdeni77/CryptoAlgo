"""
Backtesting Engine for Crypto Perpetual Futures Trading System.
FIXED VERSION - Addresses:
1. Proper leverage enforcement
2. Margin call / liquidation simulation  
3. Stop-loss at position level
4. Prevents equity from going deeply negative
5. Correct Coinbase US Perps fee structure (Jan 2026)
6. Funding rate application to positions
"""

import logging
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
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
    size: float  # Notional size in USD
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss_pct: float = 0.20  # Max loss as % of position size (20% default)
    highest_price: float = 0.0  # For trailing stop (optional)
    lowest_price: float = float('inf')  # For trailing stop (optional)
    cumulative_funding: float = 0.0  # Track funding paid/received
    
    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
    
    @property
    def current_return_pct(self) -> float:
        """Current return as a percentage of position size."""
        if self.size == 0:
            return 0.0
        return self.unrealized_pnl / self.size
    
    def get_stop_loss_price(self) -> float:
        """Calculate stop loss price based on max loss percentage."""
        if self.side == Side.LONG:
            return self.entry_price * (1 - self.stop_loss_pct)
        else:  # SHORT
            return self.entry_price * (1 + self.stop_loss_pct)
    
    def is_stopped_out(self, current_price: float) -> bool:
        """Check if position should be stopped out."""
        if self.side == Side.LONG:
            return current_price <= self.get_stop_loss_price()
        else:  # SHORT
            return current_price >= self.get_stop_loss_price()


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
    exit_type: str = ""  # 'signal', 'stop_loss', 'liquidation', 'max_hold'


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
    # Additional metrics
    num_liquidations: int = 0
    num_stop_losses: int = 0
    total_funding_paid: float = 0.0
    total_fees_paid: float = 0.0
    
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
            'num_liquidations': self.num_liquidations,
            'num_stop_losses': self.num_stop_losses,
            'total_funding_paid': f"${self.total_funding_paid:.2f}",
            'total_fees_paid': f"${self.total_fees_paid:.2f}",
        }
    
    def __str__(self):
        return (
            f"Return: {self.total_return*100:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown*100:.2f}% | "
            f"Trades: {self.num_trades} | "
            f"WinRate: {self.win_rate*100:.1f}% | "
            f"StopLosses: {self.num_stop_losses} | "
            f"Liquidations: {self.num_liquidations}"
        )


@dataclass
class CostModel:
    """
    Realistic cost model for Coinbase US Perpetual Futures.
    
    Coinbase US Perps fees (as of Jan 2026) from design.md:
    - Trading fee: 0.10% per trade (10 bps)
    - Round-trip: ~0.20% (20 bps)
    - Slippage (estimated): ~0.02% (2 bps)
    - Total round-trip: ~0.22%
    
    Funding Rates (Variable):
    - Neutral market: 0.001%/hour (~0.024%/day)
    - Bullish market: 0.003%/hour (~0.072%/day)
    - Extreme bullish: 0.005%+/hour (~0.12%+/day)
    """
    maker_fee_bps: float = 10.0    # 0.10%
    taker_fee_bps: float = 10.0    # 0.10%
    base_slippage_bps: float = 2.0  # ~0.02%
    volatility_slippage_multiplier: float = 1.5
    size_impact_coefficient: float = 0.02
    
    # Liquidation parameters
    maintenance_margin_pct: float = 0.05  # 5% maintenance margin
    liquidation_fee_bps: float = 50.0  # 0.50% liquidation penalty
    
    def calculate_cost(
        self,
        size_usd: float,
        is_maker: bool = False,
        volatility_ratio: float = 1.0,
        daily_volume: float = 1e9,
    ) -> Tuple[float, float]:
        """
        Calculate trading costs.
        
        Returns: (fee_usd, slippage_usd)
        """
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee_usd = size_usd * fee_bps / 10000
        
        if is_maker:
            slippage_usd = 0.0
        else:
            # Volatility-adjusted slippage
            vol_adjusted = self.base_slippage_bps * max(1.0, volatility_ratio)
            # Size impact (market impact)
            size_impact = self.size_impact_coefficient * (size_usd / daily_volume) * 10000
            total_slippage_bps = vol_adjusted + size_impact
            slippage_usd = size_usd * total_slippage_bps / 10000
        
        return fee_usd, slippage_usd
    
    def calculate_liquidation_cost(self, position_size: float) -> float:
        """Calculate cost of liquidation."""
        return position_size * self.liquidation_fee_bps / 10000
    
    def calculate_funding_payment(
        self,
        position_size: float,
        position_side: Side,
        funding_rate: float,
        mark_price: float,
    ) -> float:
        """
        Calculate funding payment for a position.
        
        Positive funding rate: Longs pay shorts
        Negative funding rate: Shorts pay longs
        
        Returns: Amount to ADD to position PnL (negative = payment, positive = receipt)
        """
        if position_side == Side.FLAT:
            return 0.0
        
        # Payment = Position_Size * Funding_Rate
        # Note: funding_rate is already a decimal (e.g., 0.0001 = 0.01%)
        payment = position_size * abs(funding_rate)
        
        if funding_rate > 0:
            # Positive funding: longs pay, shorts receive
            if position_side == Side.LONG:
                return -payment  # Longs pay
            else:
                return payment   # Shorts receive
        else:
            # Negative funding: shorts pay, longs receive
            if position_side == Side.SHORT:
                return -payment  # Shorts pay
            else:
                return payment   # Longs receive


class Portfolio:
    """Portfolio tracker with position management, leverage limits, and risk controls."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.3,  # Max 30% of equity per position
        max_leverage: float = 3.0,       # Max 3x gross leverage
        default_stop_loss_pct: float = 0.15,  # 15% stop loss per position
        maintenance_margin_pct: float = 0.05,  # 5% maintenance margin
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.default_stop_loss_pct = default_stop_loss_pct
        self.maintenance_margin_pct = maintenance_margin_pct
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.closed_trades: List[Trade] = []
        
        # Risk tracking
        self.num_liquidations: int = 0
        self.num_stop_losses: int = 0
        self.total_funding_paid: float = 0.0
        self.total_fees_paid: float = 0.0
        self.peak_equity: float = initial_capital
    
    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        position_value = sum(p.unrealized_pnl for p in self.positions.values())
        return max(0, self.cash + position_value)  # Floor at 0
    
    @property
    def gross_exposure(self) -> float:
        """Total absolute position size."""
        return sum(abs(p.size) for p in self.positions.values())
    
    @property
    def current_leverage(self) -> float:
        """Current leverage ratio."""
        if self.equity <= 0:
            return float('inf')
        return self.gross_exposure / self.equity
    
    @property
    def available_margin(self) -> float:
        """Available margin for new positions."""
        max_exposure = self.equity * self.max_leverage
        return max(0, max_exposure - self.gross_exposure)
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity
    
    def update_peak_equity(self):
        """Update peak equity for drawdown tracking."""
        self.peak_equity = max(self.peak_equity, self.equity)
    
    def update_position_mtm(self, symbol: str, current_price: float):
        """Update position mark-to-market."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # Update high/low water marks
        pos.highest_price = max(pos.highest_price, current_price)
        pos.lowest_price = min(pos.lowest_price, current_price)
        
        # Calculate unrealized PnL
        if pos.side == Side.LONG:
            price_return = (current_price - pos.entry_price) / pos.entry_price
        else:  # SHORT
            price_return = (pos.entry_price - current_price) / pos.entry_price
        
        pos.unrealized_pnl = pos.size * price_return
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Check if position should be stopped out."""
        if symbol not in self.positions:
            return False
        return self.positions[symbol].is_stopped_out(current_price)
    
    def check_liquidation(self, symbol: str) -> bool:
        """
        Check if position should be liquidated.
        
        Liquidation occurs when:
        - Position loss exceeds (1 - maintenance_margin) of position size
        - For 5% maintenance margin, liquidation at ~95% loss
        
        In practice, we'll liquidate earlier to prevent negative equity.
        """
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        
        # Liquidate if loss exceeds threshold
        loss_pct = -pos.current_return_pct if pos.current_return_pct < 0 else 0
        liquidation_threshold = 1 - self.maintenance_margin_pct
        
        return loss_pct >= liquidation_threshold
    
    def apply_funding(
        self,
        symbol: str,
        funding_rate: float,
        mark_price: float,
        cost_model: CostModel,
    ) -> float:
        """
        Apply funding rate to position.
        
        Returns: Funding payment amount (negative = paid, positive = received)
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        payment = cost_model.calculate_funding_payment(
            pos.size, pos.side, funding_rate, mark_price
        )
        
        # Apply to cash and track
        self.cash += payment
        pos.cumulative_funding += payment
        self.total_funding_paid -= payment  # Track total paid (flip sign)
        
        return payment
    
    def close_position(
        self,
        symbol: str,
        current_price: float,
        cost_model: CostModel,
        exit_type: str = "signal",
        reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        timestamp = timestamp or datetime.now()
        
        # Calculate final PnL
        if pos.side == Side.LONG:
            price_return = (current_price - pos.entry_price) / pos.entry_price
            trade_side = Side.SHORT  # Sell to close
        else:
            price_return = (pos.entry_price - current_price) / pos.entry_price
            trade_side = Side.LONG  # Buy to close
        
        gross_pnl = pos.size * price_return
        
        # Calculate costs
        if exit_type == "liquidation":
            fee = cost_model.calculate_liquidation_cost(pos.size)
            slippage = pos.size * 0.005  # 0.5% liquidation slippage
            self.num_liquidations += 1
        else:
            fee, slippage = cost_model.calculate_cost(pos.size, is_maker=False)
            if exit_type == "stop_loss":
                self.num_stop_losses += 1
        
        net_pnl = gross_pnl - fee - slippage
        
        # Update portfolio
        self.cash += net_pnl
        self.total_fees_paid += fee + slippage
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=trade_side,
            size=pos.size,
            price=current_price,
            commission=fee,
            slippage=slippage,
            signal_price=current_price,
            pnl=net_pnl,
            reason=reason,
            exit_type=exit_type,
        )
        
        self.trades.append(trade)
        self.closed_trades.append(trade)
        
        # Remove position
        del self.positions[symbol]
        
        return trade
    
    def open_position(
        self,
        signal: Signal,
        current_price: float,
        cost_model: CostModel,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Trade]:
        """Open a new position."""
        if signal.direction == Side.FLAT:
            return None
        
        timestamp = timestamp or signal.timestamp
        
        # Calculate position size with constraints
        base_size = self.equity * signal.target_weight * signal.confidence
        max_by_position = self.equity * self.max_position_pct
        max_by_leverage = self.available_margin
        
        target_size = min(base_size, max_by_position, max_by_leverage)
        
        # Minimum position size check
        if target_size < 100:
            return None
        
        # Check if we have enough margin
        if self.equity <= 0 or self.current_leverage >= self.max_leverage:
            logger.warning(f"Cannot open position: insufficient margin")
            return None
        
        # Calculate costs
        fee, slippage = cost_model.calculate_cost(target_size, is_maker=False)
        
        # Adjust fill price for slippage
        slippage_pct = slippage / target_size if target_size > 0 else 0
        if signal.direction == Side.LONG:
            fill_price = current_price * (1 + slippage_pct)
        else:
            fill_price = current_price * (1 - slippage_pct)
        
        # Deduct costs from cash
        self.cash -= (fee + slippage)
        self.total_fees_paid += fee + slippage
        
        # Create position
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            side=signal.direction,
            size=target_size,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_loss_pct=self.default_stop_loss_pct,
        )
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=signal.symbol,
            side=signal.direction,
            size=target_size,
            price=fill_price,
            commission=fee,
            slippage=slippage,
            signal_price=current_price,
            pnl=0.0,
            reason=getattr(signal, 'reason', ''),
            exit_type="",
        )
        self.trades.append(trade)
        
        return trade
    
    def execute_signal(
        self,
        signal: Signal,
        current_price: float,
        cost_model: CostModel,
    ) -> Optional[Trade]:
        """Execute a trading signal (open, close, or flip position)."""
        current_pos = self.positions.get(signal.symbol)
        
        # Case 1: Close existing position
        if signal.direction == Side.FLAT and current_pos:
            return self.close_position(
                signal.symbol,
                current_price,
                cost_model,
                exit_type="signal",
                reason=signal.reason,
                timestamp=signal.timestamp,
            )
        
        # Case 2: Already in same direction - do nothing
        if current_pos and current_pos.side == signal.direction:
            return None
        
        # Case 3: Flip position (close then open)
        if current_pos and signal.direction != Side.FLAT:
            self.close_position(
                signal.symbol,
                current_price,
                cost_model,
                exit_type="signal",
                reason="Flipping position",
                timestamp=signal.timestamp,
            )
        
        # Case 4: Open new position
        if signal.direction != Side.FLAT:
            return self.open_position(signal, current_price, cost_model, signal.timestamp)
        
        return None
    
    def record_equity(self, timestamp: datetime):
        """Record current equity for equity curve."""
        self.update_peak_equity()
        self.equity_curve.append((timestamp, self.equity))


class Backtester:
    """
    Event-driven backtester with proper risk management.
    
    Key features:
    - Stop-loss enforcement
    - Liquidation simulation
    - Leverage limits
    - Funding rate application
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        cost_model: Optional[CostModel] = None,
        max_position_pct: float = 0.3,
        max_leverage: float = 3.0,
        default_stop_loss_pct: float = 0.15,
        apply_funding: bool = True,
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.max_position_pct = max_position_pct
        self.max_leverage = max_leverage
        self.default_stop_loss_pct = default_stop_loss_pct
        self.apply_funding = apply_funding
    
    def run(
        self,
        strategy,
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[Portfolio, PerformanceMetrics]:
        """Run backtest with full risk management."""
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_position_pct=self.max_position_pct,
            max_leverage=self.max_leverage,
            default_stop_loss_pct=self.default_stop_loss_pct,
        )
        
        # Get combined timeline
        all_timestamps = set()
        for symbol, ohlcv in ohlcv_data.items():
            all_timestamps.update(ohlcv.index.tolist())
        
        timestamps = sorted(all_timestamps)
        
        if start_date:
            timestamps = [t for t in timestamps if t >= start_date]
        if end_date:
            timestamps = [t for t in timestamps if t <= end_date]
        
        if not timestamps:
            return portfolio, self._empty_metrics()
        
        # Main simulation loop
        for timestamp in timestamps:
            # Skip if portfolio is blown up
            if portfolio.equity <= 0:
                logger.warning(f"Portfolio equity depleted at {timestamp}")
                break
            
            # 1. Update all position mark-to-market
            for symbol in list(portfolio.positions.keys()):
                if symbol in ohlcv_data and timestamp in ohlcv_data[symbol].index:
                    current_price = ohlcv_data[symbol].loc[timestamp, 'close']
                    portfolio.update_position_mtm(symbol, current_price)
            
            # 2. Check for liquidations (before stop-losses)
            for symbol in list(portfolio.positions.keys()):
                if portfolio.check_liquidation(symbol):
                    if symbol in ohlcv_data and timestamp in ohlcv_data[symbol].index:
                        current_price = ohlcv_data[symbol].loc[timestamp, 'close']
                        portfolio.close_position(
                            symbol,
                            current_price,
                            self.cost_model,
                            exit_type="liquidation",
                            reason="Margin call - position liquidated",
                            timestamp=timestamp,
                        )
            
            # 3. Check for stop-losses
            for symbol in list(portfolio.positions.keys()):
                if symbol in ohlcv_data and timestamp in ohlcv_data[symbol].index:
                    current_price = ohlcv_data[symbol].loc[timestamp, 'close']
                    
                    # Also check against high/low of the bar for more realistic stop execution
                    high_price = ohlcv_data[symbol].loc[timestamp, 'high']
                    low_price = ohlcv_data[symbol].loc[timestamp, 'low']
                    
                    pos = portfolio.positions[symbol]
                    stop_price = pos.get_stop_loss_price()
                    
                    triggered = False
                    if pos.side == Side.LONG and low_price <= stop_price:
                        triggered = True
                        exit_price = stop_price  # Assume stop executed at stop price
                    elif pos.side == Side.SHORT and high_price >= stop_price:
                        triggered = True
                        exit_price = stop_price
                    
                    if triggered:
                        portfolio.close_position(
                            symbol,
                            exit_price,
                            self.cost_model,
                            exit_type="stop_loss",
                            reason=f"Stop loss triggered at {stop_price:.2f}",
                            timestamp=timestamp,
                        )
                        logger.debug(f"STOP LOSS: {symbol} at {timestamp}, price={exit_price:.2f}")
            
            # 4. Apply funding rates (hourly)
            if self.apply_funding:
                for symbol in list(portfolio.positions.keys()):
                    if symbol in features and timestamp in features[symbol].index:
                        feat = features[symbol].loc[timestamp]
                        if 'funding_rate' in feat.index and not pd.isna(feat['funding_rate']):
                            funding_rate = feat['funding_rate']
                            if symbol in ohlcv_data and timestamp in ohlcv_data[symbol].index:
                                mark_price = ohlcv_data[symbol].loc[timestamp, 'close']
                                portfolio.apply_funding(
                                    symbol, funding_rate, mark_price, self.cost_model
                                )
            
            # 5. Generate strategy signals
            try:
                signals = strategy.generate_signals(
                    timestamp, ohlcv_data, features, portfolio
                )
            except Exception as e:
                logger.warning(f"Signal generation error at {timestamp}: {e}")
                signals = []
            
            # 6. Execute signals
            for signal in signals:
                if signal.symbol in ohlcv_data and timestamp in ohlcv_data[signal.symbol].index:
                    current_price = ohlcv_data[signal.symbol].loc[timestamp, 'close']
                    portfolio.execute_signal(signal, current_price, self.cost_model)
            
            # 7. Record equity
            portfolio.record_equity(timestamp)
        
        # Calculate final metrics
        metrics = self._calculate_metrics(portfolio)
        
        return portfolio, metrics
    
    def _calculate_metrics(self, portfolio: Portfolio) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(portfolio.equity_curve) < 2:
            return self._empty_metrics()
        
        equity_df = pd.DataFrame(portfolio.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Handle zero/negative equity
        equity_df['equity'] = equity_df['equity'].clip(lower=0.01)
        
        returns = equity_df['equity'].pct_change().dropna()
        
        # Cap extreme returns to avoid inf issues
        returns = returns.clip(-0.5, 0.5)
        
        if len(returns) < 2:
            return self._empty_metrics()
        
        total_return = (portfolio.equity / portfolio.initial_capital) - 1
        total_return = max(-0.9999, min(total_return, 100))  # Cap at -99.99% to +10000%
        
        # Annualize (hourly data)
        periods_per_year = 365 * 24
        n_periods = len(returns)
        years = n_periods / periods_per_year
        
        if years > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = -1.0
        
        annualized_vol = returns.std() * np.sqrt(periods_per_year)
        
        if annualized_vol > 0 and not np.isnan(annualized_vol):
            sharpe = annualized_return / annualized_vol
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_vol = downside.std() * np.sqrt(periods_per_year)
            sortino = annualized_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino = 0.0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        max_dd = min(max_dd, 1.0)  # Cap at 100%
        
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        # Trade statistics
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
            
            # Consecutive wins/losses
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
            num_liquidations=portfolio.num_liquidations,
            num_stop_losses=portfolio.num_stop_losses,
            total_funding_paid=portfolio.total_funding_paid,
            total_fees_paid=portfolio.total_fees_paid,
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for failed backtests."""
        return PerformanceMetrics(
            total_return=0,
            annualized_return=0,
            annualized_volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            win_rate=0,
            profit_factor=0,
            num_trades=0,
            avg_trade_return=0,
            avg_win=0,
            avg_loss=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            num_liquidations=0,
            num_stop_losses=0,
            total_funding_paid=0,
            total_fees_paid=0,
        )