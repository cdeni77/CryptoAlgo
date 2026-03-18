from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class OIDivergenceStrategy:
    """Open Interest vs Price divergence / confirmation signal.

    Two modes:
    1. **Divergence (reversal)**: Price drops while OI rises sharply → crowded short
       position building → potential short squeeze → go long.  Conversely, price
       rallies while OI drops (long liquidation) → go short.
       Triggered when liquidation_cascade_score is elevated.

    2. **Alignment (momentum)**: OI and price move in the same direction with extreme
       OI z-score → institutional accumulation / distribution confirmed → follow trend.

    Alpha source: oi_change_4h, oi_zscore, liquidation_cascade_score (all T-1 lagged).
    """

    name = 'oi_divergence'

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
        liq_threshold = float(params.get('liq_threshold', 0.30))
        oi_z_threshold = float(params.get('oi_z_threshold', 1.0))

        oi_change_4h = context.feature('oi_change_4h', 0.0)
        oi_zscore = context.feature('oi_zscore', 0.0)
        liq_score = context.feature('liquidation_cascade_score', 0.0)
        oi_change_24h = context.feature('oi_change_24h', 0.0)

        price_dir = 1 if context.ret_24h > 0 else -1 if context.ret_24h < 0 else 0
        oi_dir_4h = 1 if oi_change_4h > 0 else -1 if oi_change_4h < 0 else 0

        direction = 0
        mode = 'none'

        # Mode 1: Divergence + liquidation cascade → reversal trade
        divergence = (price_dir != 0 and oi_dir_4h != 0 and price_dir != oi_dir_4h)
        if divergence and liq_score >= liq_threshold:
            # OI rising while price falling → shorts piling in → fade the shorts
            # OI falling while price rising → longs liquidating → fade the rally
            direction = -price_dir
            mode = 'divergence'

        # Mode 2: OI + price alignment with extreme OI z-score → trend follow
        if direction == 0:
            aligned = (price_dir != 0 and oi_dir_4h == price_dir)
            if aligned and abs(oi_zscore) >= oi_z_threshold and abs(context.ret_24h) >= min_momentum_magnitude:
                direction = price_dir
                mode = 'alignment'

        if strict_mode and mode == 'alignment' and context.ret_24h * context.ret_72h < 0:
            direction = 0

        momentum_gate = abs(oi_zscore) >= oi_z_threshold or liq_score >= liq_threshold
        direction_gate = direction != 0

        oi_strength = abs(oi_zscore) / max(oi_z_threshold, 1e-4)
        liq_strength = liq_score / max(liq_threshold, 1e-4)
        rank_mod = 0.15 * min(2.0, oi_strength) + 0.15 * min(2.0, liq_strength)

        return StrategyDecision(
            direction=direction,
            rank_modifier=rank_mod,
            gate_contributions={
                'momentum_magnitude': momentum_gate,
                'momentum_dir_agreement': direction_gate,
            },
        )
