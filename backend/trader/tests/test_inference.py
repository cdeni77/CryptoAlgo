"""Tests for the significance tests in `core.inference`, checked against planted truth.

Fold dispersion is too coarse over a short span, so the block bootstrap must
work. It must find an effect that exists, and must not find one that does not,
even when days are correlated.
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
    """Synthetic windows over `n_days`, with optional day-level correlation.

    `edge` moves the model toward the truth relative to the market. `noise` moves it
    away from the market without edge, so the statistic varies. `day_edge_sd` with
    `persistence` makes the model's advantage an AR(1) across days. Persistence is
    what makes blocks matter, since blocks capture dependence between adjacent days.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp('2026-06-18', tz='UTC')
    drift = 0.0
    for d in range(n_days):
        if day_edge_sd:
            # AR(1): today's edge carries part of yesterday's, which is what a regime
            # is.
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
        """A model meaningfully closer to the truth than the price is detected."""
        r = model_minus_market(synthetic(edge=0.25, seed=2), n_resamples=2000)
        g = governing(r)
        assert g.point > 0
        assert g.p_value < 0.05, f'a planted edge was missed: p = {g.p_value:.4f}'

    def test_no_edge_is_not_detected(self):
        """A model equal to the market (plus noise) is not significant."""
        r = model_minus_market(synthetic(edge=0.0, noise=0.05, seed=3),
                               n_resamples=2000)
        assert governing(r).p_value > 0.10

    def test_a_negative_edge_reads_negative(self):
        r = model_minus_market(synthetic(edge=-0.25, seed=4), n_resamples=2000)
        g = governing(r)
        assert g.point < 0 and g.p_value > 0.5

    def test_correlated_days_widen_the_interval(self):
        """With a persistent day-level edge, 5-day blocks give a wider interval than 1-day
        blocks.
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
        """Blocks wrap circularly, so every day is equally likely to be drawn."""
        groups = [np.array([i]) for i in range(50)]
        draws = circular_block_bootstrap(
            groups, lambda idx: float(idx.mean()), block_days=5,
            n_resamples=4000, seed=11)
        # The mean of a uniform resample of 0..49 is 24.5; a moving-block scheme biases
        # it
        # toward the interior.
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
            # As `Book` records it: the outlay already includes the fee.
            outlay = price * contracts + fee
            won = rng.random() < (price + edge)
            rows.append({'window_open': f'w{i // 2}', 'contracts': contracts,
                         'outlay': outlay, 'fee': fee,
                         'p_win_market': price,
                         'pnl': (contracts - outlay) if won else -outlay})
        return pd.DataFrame(rows)

    def test_the_fee_is_charged_once(self):
        """The fee is charged once: `outlay` already includes it."""
        t = self._trades()
        r = pnl_against_market_null(t, n_resamples=20_000)
        analytic = float((t['p_win_market'] * t['contracts'] - t['outlay']).sum())
        assert r.expected == pytest.approx(analytic, abs=4 * r.sd / np.sqrt(20_000))
        assert abs(r.expected - (analytic - t['fee'].sum())) > 0.5 * t['fee'].sum()

    def test_a_real_book_trade_ledger_is_accepted_as_is(self):
        """The ledger from `Book.trades()` is accepted unchanged."""
        from core.book import Book
        from core.config import Config
        from core.decide import Decision, Reason, Side

        config = Config()
        book = Book(config=config)
        window = pd.Timestamp('2026-01-01', tz='UTC')
        decision = Decision(
            symbol='BTC-USD', window_open=window,
            settle_time=window + pd.Timedelta(minutes=15), offset=12,
            reason=Reason.TRADED, model_probability=0.6, baseline_probability=0.5,
            side=Side.UP, price=0.5, contracts=10, stake=5.0 + 0.175, fee=0.175)
        book.record(decision)
        book.settle({('BTC-USD', window): True})
        trades = book.trades().assign(p_win_market=0.5)
        assert trades['outlay'].iloc[0] == pytest.approx(5.175)
        r = pnl_against_market_null(trades, n_resamples=4_000)
        assert r.expected == pytest.approx(0.5 * 10 - 5.175, abs=0.3)

    def test_a_fairly_priced_book_is_not_significant(self):
        """Trades won at the market's implied rate are not significant."""
        r = pnl_against_market_null(self._trades(edge=0.0), n_resamples=2000)
        assert 0.05 < r.p_value < 0.95

    def test_a_planted_edge_is_significant(self):
        r = pnl_against_market_null(self._trades(edge=0.20, seed=8), n_resamples=2000)
        assert r.p_value < 0.05, f'planted +20pp edge missed: p = {r.p_value:.4f}'

    def test_the_expected_pnl_under_the_null_is_negative(self):
        """Expected P&L under the null is negative, since every trade pays the fee and the
        spread.
        """
        r = pnl_against_market_null(self._trades(), n_resamples=2000)
        assert r.expected < 0

    def test_higher_correlation_widens_the_distribution(self):
        """Higher rho widens the distribution: symbols in one window are not independent
        bets.
        """
        t = self._trades()
        lo = pnl_against_market_null(t, rho=0.0, n_resamples=2000)
        hi = pnl_against_market_null(t, rho=0.9, n_resamples=2000)
        assert hi.sd > lo.sd

    def test_the_default_rho_is_the_documented_one(self):
        """The default rho is 0.7, above the measured cross-symbol correlation."""
        assert pnl_against_market_null(self._trades(), n_resamples=200).rho == 0.7
