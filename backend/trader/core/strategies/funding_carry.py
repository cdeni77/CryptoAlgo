from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class FundingCarryStrategy:
    """Contrarian carry trade: fade extreme funding rates.

    When perpetual funding is extremely positive (longs paying), open interest is biased
    long and the crowded trade unwinds when momentum reverses.  Going against the carry
    direction after extreme readings captures the mean-reversion premium embedded in
    perpetual funding.

    Alpha source: funding_rate_zscore + funding_persistence_24h (both already T-1 lagged).
    """

    name = 'funding_carry'

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
        funding_z_threshold = float(params.get('funding_z_threshold', 2.5))

        # Funding z-score: positive → longs paying, negative → shorts paying
        funding_z = context.funding_z
        # Fall back to the feature column if context.funding_z is not populated
        if funding_z == 0.0:
            funding_z = context.feature('funding_rate_zscore', 0.0)

        persistence = context.feature('funding_persistence_24h', 0.5)
        cum_funding = context.feature('cumulative_funding_72h', 0.0)

        # Gate: funding must be extreme enough
        funding_gate = abs(funding_z) >= funding_z_threshold

        # Direction: contra to funding bias
        # Extreme positive funding → shorts are cheap → go short
        # Extreme negative funding → longs are cheap → go long
        if funding_z >= funding_z_threshold:
            direction = -1
            # Confirm: persistence > 0.5 means longs have been paying consistently
            persistence_gate = persistence >= 0.5
        elif funding_z <= -funding_z_threshold:
            direction = 1
            persistence_gate = persistence <= 0.5
        else:
            direction = 0
            persistence_gate = False

        if not funding_gate or not persistence_gate:
            direction = 0

        strength = abs(funding_z) / max(funding_z_threshold, 1e-4)
        rank_mod = 0.20 * min(3.0, strength) + 0.05 * abs(cum_funding)

        return StrategyDecision(
            direction=direction,
            rank_modifier=rank_mod,
            gate_contributions={
                'momentum_magnitude': funding_gate,
                'momentum_dir_agreement': persistence_gate and direction != 0,
            },
        )
