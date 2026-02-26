from .base import StrategyContext, StrategyDecision
from .momentum import MomentumTrendStrategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .vol_overlay import VolatilityOverlayStrategy

STRATEGY_FAMILY_REGISTRY = {
    MomentumTrendStrategy.name: MomentumTrendStrategy(),
    BreakoutStrategy.name: BreakoutStrategy(),
    MeanReversionStrategy.name: MeanReversionStrategy(),
    VolatilityOverlayStrategy.name: VolatilityOverlayStrategy(),
}


def get_strategy_family(name: str | None):
    normalized = (name or MomentumTrendStrategy.name).strip().lower()
    return STRATEGY_FAMILY_REGISTRY.get(normalized, STRATEGY_FAMILY_REGISTRY[MomentumTrendStrategy.name])
