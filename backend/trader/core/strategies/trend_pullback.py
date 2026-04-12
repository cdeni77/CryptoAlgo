from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class TrendPullbackStrategy:
    name = 'trend_pullback'

    def evaluate(self, context: StrategyContext, *, min_momentum_magnitude: float, score_threshold: float, strict_mode: bool, family_params=None) -> StrategyDecision:
        params = family_params or {}
        pullback_depth = float(params.get('pullback_depth_threshold', 0.02))
        rebound_threshold = float(params.get('rebound_confirmation_threshold', 0.004))
        trend_strength_min = float(params.get('trend_strength_min', 0.002))

        trend_strength = context.feature('trend_strength_24h', context.ret_72h)
        trend_sign = 1 if trend_strength >= trend_strength_min else -1 if trend_strength <= -trend_strength_min else 0

        pullback_metric = context.feature('range_position_24h', 0.5) - 0.5
        price_vs_sma50 = 0.0 if context.sma_50 == 0 else (context.price / context.sma_50 - 1.0)

        long_pullback = trend_sign > 0 and pullback_metric <= -pullback_depth and price_vs_sma50 <= -pullback_depth * 0.5
        short_pullback = trend_sign < 0 and pullback_metric >= pullback_depth and price_vs_sma50 >= pullback_depth * 0.5

        rebound_signal = context.ret_24h if trend_sign > 0 else -context.ret_24h
        rebound_pass = rebound_signal >= rebound_threshold

        direction = 0
        if long_pullback and rebound_pass:
            direction = 1
        elif short_pullback and rebound_pass:
            direction = -1

        if strict_mode and context.ret_24h * trend_strength < 0:
            direction = 0

        trend_gate = abs(trend_strength) >= trend_strength_min
        pullback_gate = long_pullback or short_pullback

        norm_trend = abs(trend_strength) / max(trend_strength_min, 1e-6)
        norm_rebound = abs(rebound_signal) / max(rebound_threshold, 1e-6)
        return StrategyDecision(
            direction=direction,
            rank_modifier=0.18 * min(2.0, norm_trend) + 0.07 * min(1.0, norm_rebound),
            gate_contributions={
                'momentum_magnitude': trend_gate and rebound_pass,
                'momentum_dir_agreement': direction != 0 and pullback_gate,
            },
        )
