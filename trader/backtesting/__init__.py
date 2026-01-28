"""
Backtesting module for Crypto Trading System.
"""

from .engine import (
    Backtester,
    Portfolio,
    CostModel,
    PerformanceMetrics,
    Side,
    Signal,
    Position,
    Trade,
)

from .validation import (
    WalkForwardValidator,
    WalkForwardResult,
)

from .strategies import (
    BaseStrategy,
    MeanReversionStrategy,
    MomentumReversalStrategy,
)

__all__ = [
    'Backtester',
    'Portfolio',
    'CostModel',
    'PerformanceMetrics',
    'Side',
    'Signal',
    'Position',
    'Trade',
    'WalkForwardValidator',
    'WalkForwardResult',
    'BaseStrategy',
    'MeanReversionStrategy',
    'MomentumReversalStrategy',
]