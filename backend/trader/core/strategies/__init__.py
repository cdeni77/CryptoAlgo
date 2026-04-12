from .base import StrategyContext, StrategyDecision
from .momentum import MomentumTrendStrategy
from .breakout import BreakoutStrategy
from .mean_reversion import MeanReversionStrategy
from .vol_overlay import VolatilityOverlayStrategy
from .trend_pullback import TrendPullbackStrategy
from .breakout_expansion import BreakoutExpansionStrategy
from .funding_carry import FundingCarryStrategy
from .squeeze_breakout import SqueezeBreakoutStrategy
from .oi_divergence import OIDivergenceStrategy
from .btc_lead import BtcLeadStrategy
from .autocorr_regime import AutocorrRegimeStrategy

STRATEGY_FAMILY_REGISTRY = {
    MomentumTrendStrategy.name: MomentumTrendStrategy(),
    BreakoutStrategy.name: BreakoutStrategy(),
    MeanReversionStrategy.name: MeanReversionStrategy(),
    VolatilityOverlayStrategy.name: VolatilityOverlayStrategy(),
    TrendPullbackStrategy.name: TrendPullbackStrategy(),
    BreakoutExpansionStrategy.name: BreakoutExpansionStrategy(),
    FundingCarryStrategy.name: FundingCarryStrategy(),
    SqueezeBreakoutStrategy.name: SqueezeBreakoutStrategy(),
    OIDivergenceStrategy.name: OIDivergenceStrategy(),
    BtcLeadStrategy.name: BtcLeadStrategy(),
    AutocorrRegimeStrategy.name: AutocorrRegimeStrategy(),
}


def get_strategy_family(name: str | None):
    normalized = (name or MomentumTrendStrategy.name).strip().lower()
    return STRATEGY_FAMILY_REGISTRY.get(normalized, STRATEGY_FAMILY_REGISTRY[MomentumTrendStrategy.name])
