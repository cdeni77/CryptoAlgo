from __future__ import annotations

from .base import StrategyContext, StrategyDecision


class AutocorrRegimeStrategy:
    """Autocorrelation-adaptive regime strategy.

    Alpha hypothesis: crypto alternates between trending (positive autocorrelation)
    and mean-reverting (negative autocorrelation) regimes. Detecting the current
    regime and applying the appropriate signal type improves directional accuracy.

    Uses ret_autocorr_lag1 (48h rolling 1-bar autocorrelation):
        > +0.08  → trending regime → trade momentum direction
        < -0.08  → mean-reverting regime → fade recent moves (use zscore / RSI)
        [-0.08, +0.08] → ambiguous → no trade

    Feature availability: ret_autocorr_lag1/lag2 computed for AVAX, SOL.
    Falls back to z-score + momentum heuristic for coins without the feature.
    """

    name = 'autocorr_regime'

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
        trend_threshold  = float(params.get('trend_threshold',  0.08))
        revert_threshold = float(params.get('revert_threshold', 0.08))
        zscore_gate      = float(params.get('zscore_gate',      0.70))

        autocorr_1 = context.feature('ret_autocorr_lag1', 999.0)  # sentinel for missing
        autocorr_2 = context.feature('ret_autocorr_lag2', 0.0)

        zscore = context.feature('zscore_72h', context.feature('zscore_48h', 0.0))
        rsi    = context.feature('rsi_14', 50.0)

        direction     = 0
        magnitude_ok  = False
        regime_conf   = 0.0

        if autocorr_1 == 999.0:
            # Feature not available for this coin — skip (don't default silently)
            return StrategyDecision(
                direction=0,
                rank_modifier=0.0,
                gate_contributions={
                    'momentum_magnitude': False,
                    'momentum_dir_agreement': False,
                },
            )

        if autocorr_1 >= trend_threshold:
            # ── Trending regime: follow momentum ──────────────────────────
            regime_conf = (autocorr_1 - trend_threshold) / max(trend_threshold, 1e-6)
            if context.ret_72h >= min_momentum_magnitude:
                direction = 1
            elif context.ret_72h <= -min_momentum_magnitude:
                direction = -1
            magnitude_ok = abs(context.ret_72h) >= min_momentum_magnitude

            # 2-period lag confirmation: lag2 should not strongly oppose lag1 regime
            if autocorr_2 < -0.20:
                direction = 0  # conflicting regime signals

        elif autocorr_1 <= -revert_threshold:
            # ── Mean-reverting regime: fade extremes ──────────────────────
            regime_conf = (abs(autocorr_1) - revert_threshold) / max(revert_threshold, 1e-6)
            stretched   = abs(zscore) >= zscore_gate or rsi >= 68 or rsi <= 32

            if zscore >= zscore_gate or rsi >= 68:
                direction = -1  # overbought in mean-reversion regime → short
            elif zscore <= -zscore_gate or rsi <= 32:
                direction = 1   # oversold in mean-reversion regime → long

            magnitude_ok = stretched
            if strict_mode and abs(context.ret_24h) < min_momentum_magnitude * 0.3:
                direction = 0   # no recent catalyst, skip

        # else: ambiguous regime — direction stays 0

        if strict_mode and direction == 1 and context.ret_24h < -min_momentum_magnitude:
            direction = 0
        if strict_mode and direction == -1 and context.ret_24h > min_momentum_magnitude:
            direction = 0

        mom_strength = abs(context.ret_72h) / max(min_momentum_magnitude, 1e-6)
        return StrategyDecision(
            direction=direction if magnitude_ok else 0,
            rank_modifier=0.12 * min(2.0, mom_strength) + 0.13 * min(1.0, regime_conf),
            gate_contributions={
                'momentum_magnitude':    magnitude_ok,
                'momentum_dir_agreement': direction != 0,
            },
        )
