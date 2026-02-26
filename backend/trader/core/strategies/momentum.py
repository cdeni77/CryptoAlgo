from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class MomentumTrendStrategy:
    name = 'momentum_trend'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool) -> StrategyDecision:
        if strict_mode and context.ret_24h * context.ret_72h < 0:
            return StrategyDecision(direction=0, rank_modifier=0.0, gate_contributions={'momentum_dir_agreement': False})

        momentum_score = (
            (1 if context.ret_24h > 0 else -1)
            + (1 if context.ret_72h > 0 else -1)
            + (1 if context.price > context.sma_50 else -1)
        )

        if momentum_score >= score_threshold:
            direction = 1
        elif momentum_score <= -score_threshold:
            direction = -1
        else:
            direction = 0

        momentum_pass = abs(context.ret_72h) >= min_momentum_magnitude
        return StrategyDecision(
            direction=direction,
            rank_modifier=0.2 * abs(context.ret_72h),
            gate_contributions={
                'momentum_magnitude': momentum_pass,
                'momentum_dir_agreement': direction != 0,
            },
        )
