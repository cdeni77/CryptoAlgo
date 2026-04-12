from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class BtcLeadStrategy:
    """BTC lead-lag momentum for altcoins.

    Alpha hypothesis: BTC moves first; correlated altcoins follow with a 4–24h lag.
    When an altcoin has significantly underperformed BTC's recent move, bet on catch-up.
    When an altcoin has significantly outperformed BTC, fade the divergence.

    Uses btc_rel_return_4h and btc_rel_return_24h features:
        btc_rel = coin_return - btc_return  (negative = coin lagged BTC's rally)

    NOT applicable to BTC itself (btc_rel will be ~0 by construction).
    Strongest for high-correlation coins: ETH, LINK, LTC, SOL, AVAX.
    """

    name = 'btc_lead'

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
        # 4h threshold is smaller since 4h moves are naturally smaller than 24h
        threshold_4h_scale = float(params.get('threshold_4h_scale', 0.40))

        btc_rel_4h  = context.feature('btc_rel_return_4h',  0.0)
        btc_rel_24h = context.feature('btc_rel_return_24h', 0.0)
        btc_rel_72h = context.feature('btc_rel_return_72h', 0.0)

        # Primary gate: 24h lag must be significant
        threshold_24h = min_momentum_magnitude
        threshold_4h  = min_momentum_magnitude * threshold_4h_scale

        # Catch-up long: coin substantially lagged BTC's 24h rally (btc_rel < 0)
        # Fade short: coin substantially outran BTC's 24h move (btc_rel > 0)
        lag_long  = btc_rel_24h < -threshold_24h
        lag_short = btc_rel_24h >  threshold_24h

        # 4h confirmation: lag is still persisting (not already being closed)
        confirm_long  = btc_rel_4h < -threshold_4h
        confirm_short = btc_rel_4h >  threshold_4h

        # 72h context: longer-term drift shouldn't be strongly against the trade
        # If coin has been lagging for 72h in same direction, catch-up is more likely
        context_long  = btc_rel_72h <= 0           # coin hasn't been systematically outrunning
        context_short = btc_rel_72h >= 0

        if lag_long and confirm_long and context_long:
            direction = 1
        elif lag_short and confirm_short and context_short:
            direction = -1
        else:
            direction = 0

        # strict_mode: don't trade if coin's own momentum strongly opposes catch-up direction
        if strict_mode and direction == 1 and context.ret_24h < -min_momentum_magnitude * 0.5:
            # Coin is falling hard in absolute terms — may be coin-specific problem, not BTC lag
            direction = 0
        if strict_mode and direction == -1 and context.ret_24h > min_momentum_magnitude * 0.5:
            direction = 0

        gate = abs(btc_rel_24h) >= threshold_24h
        strength_24h = abs(btc_rel_24h) / max(threshold_24h, 1e-6)
        strength_4h  = abs(btc_rel_4h)  / max(threshold_4h,  1e-6)

        return StrategyDecision(
            direction=direction if gate else 0,
            rank_modifier=0.18 * min(2.0, strength_24h) + 0.07 * min(1.0, strength_4h),
            gate_contributions={
                'momentum_magnitude':    gate,
                'momentum_dir_agreement': direction != 0,
            },
        )
