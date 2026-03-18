from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class SqueezeBreakoutStrategy:
    """Bollinger Band squeeze → directional breakout.

    When implied volatility (bb_width) is compressed into a low percentile rank, the
    market is coiling for a move.  The first strong directional impulse out of the
    squeeze is traded with momentum.

    Alpha source: bb_width_pct_rank (new rolling-pct-rank feature added to PriceFeatures)
    + ret_24h for direction.  Avoids trading when volatility is already expanded.
    """

    name = 'squeeze_breakout'

    def evaluate(
        self,
        context: StrategyContext,
        *,
        min_momentum_magnitude: float,
        score_threshold: float,
        strict_mode: bool,
        family_params=None,
    ) -> StrategyDecision:
        params = family_params or {}
        squeeze_pct_threshold = float(params.get('squeeze_pct_threshold', 0.20))

        # bb_width_pct_rank: rolling percentile rank [0, 1] of bb_width over past 168 bars
        # 0 = minimum observed volatility; 1 = maximum; squeeze when near 0
        bb_pct_rank = context.feature('bb_width_pct_rank', 0.5)
        bb_squeeze_flag = context.feature('bb_squeeze', 0)  # binary, BTC only; 0 for others

        # Squeeze is active when either the continuous rank is low or the flag is set
        squeeze_active = (bb_pct_rank <= squeeze_pct_threshold) or (bb_squeeze_flag == 1)

        # Directional breakout: use 24h return for primary direction
        breakout_impulse = context.ret_24h
        if abs(breakout_impulse) < min_momentum_magnitude:
            # Try 72h for slower-developing breakouts
            breakout_impulse = context.ret_72h

        direction = 1 if breakout_impulse > 0 else -1 if breakout_impulse < 0 else 0

        if strict_mode and context.ret_24h * context.ret_72h < 0:
            # Conflicting signals — skip
            direction = 0

        magnitude_gate = abs(breakout_impulse) >= min_momentum_magnitude
        squeeze_gate = squeeze_active

        rank_mod = (
            0.30 * max(0.0, squeeze_pct_threshold - bb_pct_rank) / max(squeeze_pct_threshold, 1e-4)
            + 0.20 * abs(breakout_impulse) / max(min_momentum_magnitude, 1e-4)
        )

        return StrategyDecision(
            direction=direction if (magnitude_gate and squeeze_gate) else 0,
            rank_modifier=min(rank_mod, 1.0),
            gate_contributions={
                'momentum_magnitude': magnitude_gate,
                'momentum_dir_agreement': squeeze_gate and direction != 0,
            },
        )
