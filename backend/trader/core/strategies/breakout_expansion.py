from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class BreakoutExpansionStrategy:
    name = 'breakout_expansion'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool, family_params=None) -> StrategyDecision:
        params = family_params or {}
        breakout_buffer = float(params.get('breakout_buffer', 0.003))
        expansion_confirm = float(params.get('expansion_confirm_threshold', 0.004))

        breakout_strength = context.feature('breakout_strength_24h', 0.0)
        range_expansion = context.feature('range_expansion', 0.0)
        volume_confirm = context.feature('volume_ratio_24h', 1.0)
        trend_strength = context.feature('trend_strength_24h', context.ret_72h)

        impulse = breakout_strength + range_expansion + context.ret_24h
        direction = 1 if impulse > 0 else -1

        breakout_gate = abs(impulse) >= max(min_momentum_magnitude * 0.7, breakout_buffer)
        expansion_gate = abs(range_expansion) >= expansion_confirm
        trend_gate = trend_strength * direction > 0
        volume_gate = volume_confirm >= 1.0

        if strict_mode and context.ret_24h * context.ret_72h < 0:
            direction = 0

        passed = breakout_gate and expansion_gate and trend_gate and volume_gate and direction != 0
        return StrategyDecision(
            direction=direction if passed else 0,
            rank_modifier=0.22 * abs(impulse) + 0.08 * max(0.0, volume_confirm - 1.0),
            gate_contributions={
                'momentum_magnitude': breakout_gate and expansion_gate,
                'momentum_dir_agreement': trend_gate and volume_gate and direction != 0,
            },
        )
