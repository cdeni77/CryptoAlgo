from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class BreakoutStrategy:
    name = 'breakout'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool, family_params=None) -> StrategyDecision:
        breakout_strength = context.feature('breakout_strength_24h', 0.0)
        range_position = context.feature('range_position_72h', 0.5)
        raw_impulse = breakout_strength + context.ret_24h

        direction = 1 if raw_impulse >= 0 else -1
        if strict_mode and context.ret_24h * context.ret_72h < 0:
            direction = 0

        breakout_gate = abs(raw_impulse) >= (min_momentum_magnitude * 0.7)
        agreement_gate = abs(range_position - 0.5) >= 0.05 or abs(context.ret_72h) >= min_momentum_magnitude

        norm_impulse = abs(raw_impulse) / max(min_momentum_magnitude * 0.7, 1e-6)
        range_factor = max(0.0, abs(range_position - 0.5)) / 0.5  # 0–1
        return StrategyDecision(
            direction=direction if breakout_gate and agreement_gate else 0,
            rank_modifier=0.20 * min(2.0, norm_impulse) + 0.05 * range_factor,
            gate_contributions={
                'momentum_magnitude': breakout_gate,
                'momentum_dir_agreement': agreement_gate,
            },
        )
