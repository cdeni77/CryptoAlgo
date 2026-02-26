from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class VolatilityOverlayStrategy:
    name = 'vol_overlay'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool) -> StrategyDecision:
        vol = context.vol_24h if context.vol_24h is not None else abs(context.feature('volatility_24h', 0.02))
        base_direction = 1 if context.ret_72h >= 0 else -1
        if strict_mode and context.ret_24h * context.ret_72h < 0:
            base_direction = 0

        momentum_gate = abs(context.ret_72h) >= min_momentum_magnitude * 0.8
        vol_gate = 0.004 <= vol <= 0.12
        gate_pass = momentum_gate and vol_gate
        vol_bonus = 0.0 if vol <= 0 else min(0.3, 0.05 / vol)

        return StrategyDecision(
            direction=base_direction if gate_pass else 0,
            rank_modifier=0.12 * abs(context.ret_72h) + vol_bonus,
            gate_contributions={
                'momentum_magnitude': momentum_gate,
                'momentum_dir_agreement': gate_pass,
            },
        )
