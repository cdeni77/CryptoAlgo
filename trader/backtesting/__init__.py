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

from .strategies import (
    FundingArbitrageStrategy,
    OIDivergenceStrategy,
    CombinedOIFundingStrategy,
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
    'FundingArbitrageStrategy',
    'CombinedFundingPriceStrategy',
    'OIDivergenceStrategy',
    'CombinedOIFundingStrategy',
]