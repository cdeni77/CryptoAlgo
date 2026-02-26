from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class MeanReversionStrategy:
    name = 'mean_reversion'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool) -> StrategyDecision:
        zscore = context.feature('zscore_72h', context.feature('zscore_48h', 0.0))
        rsi = context.feature('rsi_14', 50.0)

        direction = 0
        if zscore >= 0.8 or rsi >= 65:
            direction = -1
        elif zscore <= -0.8 or rsi <= 35:
            direction = 1

        stretch = max(abs(zscore), abs(context.ret_24h) / max(min_momentum_magnitude, 1e-4))
        stretch_gate = stretch >= 1.0

        return StrategyDecision(
            direction=direction if stretch_gate else 0,
            rank_modifier=0.15 * min(3.0, stretch),
            gate_contributions={
                'momentum_magnitude': stretch_gate,
                'momentum_dir_agreement': direction != 0,
            },
        )
