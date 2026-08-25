"""The significance machinery, validated against planted truth.

Written before any real number was computed, which is the point: this code judges
a result, and code authored while someone is looking at the result it judges is
not a test, it is a rationalisation.

Two established methods are unavailable here and both rejections are load-bearing.
The breadth formula `N/(1+(N-1)rho)` is rejected by name in `core/metrics.py:10`.
Fold dispersion is rejected by span: over 69 days six folds are ~11.4 days each
and "5 of 6 positive" is a 34.6% event at rho = 0.7. So the block bootstrap has
to actually work, and "works" means two things that are tested separately:

* it finds an effect that is really there, and
* it does NOT find one that is not there, *even when the data is correlated* —
  which is exactly where an iid bootstrap fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.inference import (
    MIN_USABLE_DAYS, circular_block_bootstrap, governing, model_minus_market,
    pnl_against_market_null,
)


def synthetic(n_days: int = 69, per_day: int = 96 * 3, *, edge: float = 0.0,
              day_edge_sd: float = 0.0, persistence: float = 0.0,
              noise: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Windows over `n_days`, with optional per-day correlation.

    `edge` moves the model toward the truth relative to the market, so edge > 0 is
    a model that genuinely forecasts better and edge < 0 one that is worse.

    `noise` perturbs the model away from the market in a direction uncorrelated
    with the outcome — no edge, but a statistic that actually varies. Without it
    `edge=0` makes the model bit-identical to the market and every resample
    returns exactly 0.0, which silently passes a width comparison by making both
    widths zero. That is how the first version of this file fooled itself.

    `day_edge_sd` makes the model's ADVANTAGE vary by day, and `persistence` makes
    that advantage carry over from one day to the next as an AR(1). Both were
    needed, and getting there took two wrong turns worth recording:

    1. Biasing the market on whole days does nothing, because it shifts the
       model's and the market's log loss by the same amount and cancels in the
       difference. A block bootstrap protects against day-correlation *in the
       statistic being bootstrapped*, not in the raw data generally.
    2. A day-varying edge drawn INDEPENDENTLY each day also does nothing, because
       the blocks exist to capture serial dependence between ADJACENT days. With
       iid days a 5-day block is no more informative than a 1-day block.

    So the thing that makes blocks matter is persistence — which is also the real
    phenomenon they are there for. A regime is exactly an edge that carries over.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp('2026-06-18', tz='UTC')
    drift = 0.0
    for d in range(n_days):
        if day_edge_sd:
            # AR(1): today's edge remembers yesterday's, which is what a regime is.
            drift = persistence * drift + rng.normal(0, day_edge_sd)
        edge_today = edge + drift
        for i in range(per_day):
            market = float(np.clip(rng.beta(2, 2), 0.02, 0.98))
            outcome = float(rng.random() < market)
            toward = (1.0 - market) if outcome else (0.0 - market)
            model = market + edge_today * toward
            if noise:
                model += rng.normal(0.0, noise)
            rows.append({
                'window_open': start + pd.Timedelta(days=d, minutes=15 * (i % 96)),
                'outcome': outcome,
                'model_probability': float(np.clip(model, 0.01, 0.99)),
                'market_probability': market})
    return pd.DataFrame(rows)


class TestTheBlockBootstrapRecoversPlantedTruth:
    def test_a_real_edge_is_detected(self):
        """Planted: the model is meaningfully closer to the truth than the price."""
        r = model_minus_market(synthetic(edge=0.25, seed=2), n_resamples=2000)
        g = governing(r)
        assert g.point > 0
        assert g.p_value < 0.05, f'a planted edge was missed: p = {g.p_value:.4f}'

    def test_no_edge_is_not_detected(self):
        """Planted: the model IS the market. p must be nowhere near significant."""
        r = model_minus_market(synthetic(edge=0.0, noise=0.05, seed=3),
                               n_resamples=2000)
        assert governing(r).p_value > 0.10

    def test_a_negative_edge_reads_negative(self):
        r = model_minus_market(synthetic(edge=-0.25, seed=4), n_resamples=2000)
        g = governing(r)
        assert g.point < 0 and g.p_value > 0.5

    def test_correlated_days_widen_the_interval(self):
        """The property the whole design rests on.

        With a persistent (AR(1)) day-level edge, ADJACENT days are dependent. A
        5-day block absorbs more of that than a 1-day block, so its interval must
        be WIDER. If it is not, the blocks are inert and the method is an iid
        bootstrap wearing a hat — which is what two earlier versions of this
        fixture actually measured.
        """
        frame = synthetic(edge=0.0, day_edge_sd=0.10, persistence=0.9, noise=0.02, seed=5)
        r = model_minus_market(frame, block_days=(1, 5), n_resamples=2000)
        width1 = r[1].hi - r[1].lo
        width5 = r[5].hi - r[5].lo
        assert width5 > width1, (
            f'5-day blocks ({width5:.6f}) were not wider than 1-day ({width1:.6f}); '
            f'the block structure is inert'
        )

    def test_the_conservative_block_length_governs(self):
        r = model_minus_market(synthetic(edge=0.05, day_edge_sd=0.08, persistence=0.9,
                                         noise=0.02, seed=6), n_resamples=2000)
        assert governing(r).p_value == max(x.p_value for x in r.values())


class TestTheBootstrapMechanics:
    def test_blocks_wrap_so_every_day_is_equally_likely(self):
        """Circular, not moving. Under a moving scheme the first and last days
        appear in fewer blocks and are undersampled."""
        groups = [np.array([i]) for i in range(50)]
        draws = circular_block_bootstrap(
            groups, lambda idx: float(idx.mean()), block_days=5,
            n_resamples=4000, seed=11)
        # the mean of a uniform resample of 0..49 is 24.5; a moving-block scheme
        # biases this toward the interior.
        assert abs(float(np.mean(draws)) - 24.5) < 0.5

    def test_each_resample_keeps_the_original_day_count(self):
        groups = [np.arange(i * 10, i * 10 + 10) for i in range(37)]
        seen = {}
        def statistic(idx):
            seen['n'] = len(idx)
            return 0.0
        circular_block_bootstrap(groups, statistic, block_days=5,
                                 n_resamples=1, seed=1)
        assert seen['n'] == 37 * 10

    def test_an_empty_frame_does_not_raise(self):
        assert len(circular_block_bootstrap([], lambda i: 0.0)) == 0

    def test_the_minimum_day_count_is_the_documented_one(self):
        assert MIN_USABLE_DAYS == 30


class TestTheEconomicNull:
    def _trades(self, n=160, p_win=0.35, seed=7, edge=0.0):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n):
            price = float(rng.uniform(0.15, 0.6))
            contracts = int(rng.integers(3, 15))
            fee = 0.07 * price * (1 - price) * contracts
            outlay = price * contracts
            won = rng.random() < (price + edge)
            rows.append({'window_open': f'w{i // 2}', 'contracts': contracts,
                         'outlay': outlay, 'fee': fee,
                         'p_win_market': price,
                         'pnl': (contracts - outlay - fee) if won else -(outlay + fee)})
        return pd.DataFrame(rows)

    def test_a_fairly_priced_book_is_not_significant(self):
        """Trades won at exactly the market's implied rate. p should be ordinary."""
        r = pnl_against_market_null(self._trades(edge=0.0), n_resamples=2000)
        assert 0.05 < r.p_value < 0.95

    def test_a_planted_edge_is_significant(self):
        r = pnl_against_market_null(self._trades(edge=0.20, seed=8), n_resamples=2000)
        assert r.p_value < 0.05, f'planted +20pp edge missed: p = {r.p_value:.4f}'

    def test_the_expected_pnl_under_the_null_is_negative(self):
        """Because every trade pays the fee and crosses the spread. If this comes
        out positive the null is mis-specified and every p is wrong."""
        r = pnl_against_market_null(self._trades(), n_resamples=2000)
        assert r.expected < 0

    def test_higher_correlation_widens_the_distribution(self):
        """The reason rho is in the test at all: three symbols in one window are
        not three independent bets."""
        t = self._trades()
        lo = pnl_against_market_null(t, rho=0.0, n_resamples=2000)
        hi = pnl_against_market_null(t, rho=0.9, n_resamples=2000)
        assert hi.sd > lo.sd

    def test_the_default_rho_is_the_documented_one(self):
        """DECISION_RULE.md fixes 0.7, above the measured +0.618."""
        assert pnl_against_market_null(self._trades(), n_resamples=200).rho == 0.7
