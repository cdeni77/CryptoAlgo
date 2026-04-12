from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class VolatilityOverlayStrategy:
    name = 'vol_overlay'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool, family_params=None) -> StrategyDecision:
        vol = context.vol_24h if context.vol_24h is not None else abs(context.feature('volatility_24h', 0.02))
        base_direction = 1 if context.ret_72h >= 0 else -1
        if strict_mode and context.ret_24h * context.ret_72h < 0:
            base_direction = 0

        momentum_gate = abs(context.ret_72h) >= min_momentum_magnitude * 0.8
        vol_gate = 0.004 <= vol <= 0.12
        gate_pass = momentum_gate and vol_gate
        # Normalized momentum strength: 0.20 at gate threshold, scales up to 0.40
        norm_mom = abs(context.ret_72h) / max(min_momentum_magnitude * 0.8, 1e-6)
        mom_rank = 0.20 * min(2.0, norm_mom)
        # Vol quality: peaks when vol is in the ideal 0.01–0.04 range, tapers at extremes
        vol_quality = max(0.0, 1.0 - abs(vol - 0.025) / 0.10)
        vol_rank = 0.10 * vol_quality

        return StrategyDecision(
            direction=base_direction if gate_pass else 0,
            rank_modifier=mom_rank + vol_rank,
            gate_contributions={
                'momentum_magnitude': momentum_gate,
                'momentum_dir_agreement': gate_pass,
            },
        )
