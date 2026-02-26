from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol


@dataclass(frozen=True)
class StrategyContext:
    ret_24h: float
    ret_72h: float
    price: float
    sma_50: float
    sma_200: float
    funding_z: float = 0.0
    vol_24h: Optional[float] = None
    features: Optional[Dict[str, float]] = None

    def feature(self, name: str, default: float = 0.0) -> float:
        if not self.features:
            return float(default)
        value = self.features.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


@dataclass(frozen=True)
class StrategyDecision:
    direction: int
    rank_modifier: float
    gate_contributions: Dict[str, bool]


class StrategyFamily(Protocol):
    name: str

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool) -> StrategyDecision:
        """Return direction plus rank/gating contributions for candidate selection."""
