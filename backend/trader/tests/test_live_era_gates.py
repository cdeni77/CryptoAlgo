"""The live-era gates: market significance, fold count, edge bound, overconfidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.metrics import (
    DEFAULT_GATES, block_bootstrap_means, bootstrap_interval, market_gate_values,
    market_rows_from_scored, trade_gate_values,
)

START = pd.Timestamp('2026-08-03', tz='UTC')


def scored(weeks: int, *, model_edge: float, venue: bool = True, seed: int = 3,
           per_week: int = 672) -> pd.DataFrame:
    """Backtest rows, one fold per week of 15-minute windows.

    `model_edge` moves the model toward the truth (positive) or away from it (negative).
    """
    rng = np.random.default_rng(seed)
    n = weeks * per_week
    outcome = rng.integers(0, 2, n).astype(float)
    market = np.clip(0.5 + np.where(outcome == 1, 0.15, -0.15)
                     + rng.normal(0, 0.12, n), 0.03, 0.97)
    toward = np.where(outcome == 1, 1.0 - market, -market)
    model = np.clip(market + model_edge * toward + rng.normal(0, 0.01, n), 0.02, 0.98)
    opens = START + pd.to_timedelta(np.arange(n) * 15, unit='min')
    frame = pd.DataFrame({
        'symbol': 'BTC-USD', 'window_open': opens, 'offset': 3,
        'market_probability': market, 'baseline_probability': market,
        'model_probability': model, 'outcome': outcome,
        'fold': (np.arange(n) // per_week),
    })
    if venue:
        frame['venue_outcome'] = outcome
    return frame


def gates(values: dict) -> dict:
    return {name: (values[name] <= bar if way == 'max' else values[name] >= bar)
            for name, (bar, way) in DEFAULT_GATES.items() if name in values}


# ------------------------------------------------------------ market gates

def test_a_model_that_beats_the_price_in_every_fold_is_significant():
    values = market_gate_values(market_rows_from_scored(scored(7, model_edge=0.2)))
    assert values['min_folds'] == 7
    assert values['model_minus_market_p'] < 0.01
    assert gates(values)['model_minus_market_p'] and gates(values)['min_folds']


def test_a_model_that_matches_the_price_is_not_significant():
    values = market_gate_values(market_rows_from_scored(scored(8, model_edge=0.0)))
    assert values['model_minus_market_p'] > 0.05
    assert not gates(values)['model_minus_market_p']


def test_five_weekly_folds_fail_the_sample_size_gate():
    values = market_gate_values(market_rows_from_scored(scored(5, model_edge=0.2)))
    assert values['min_folds'] == 5
    assert not gates(values)['min_folds']


def test_only_venue_labelled_rows_enter_the_significance_test():
    frame = scored(7, model_edge=0.2, venue=False)
    values = market_gate_values(market_rows_from_scored(frame))
    assert np.isfinite(values['model_minus_market'])
    assert values['min_folds'] == 0
    assert np.isnan(values['model_minus_market_p'])


def test_rows_without_a_fold_take_weekly_blocks():
    rows = [r[:8] for r in market_rows_from_scored(scored(7, model_edge=0.2))]
    values = market_gate_values(rows)
    assert values['min_folds'] in (7, 8)  # calendar blocks, not folds
    assert values['model_minus_market_p'] < 0.01


def test_no_rows_fail_both_new_gates():
    values = market_gate_values([])
    assert values['min_folds'] == 0 and np.isnan(values['model_minus_market_p'])


# ------------------------------------------------------------ trade gates

def trades(n_windows: int, *, win_rate: float, cost: float, predicted: float,
           per_window: int = 3, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opens = np.repeat(START + pd.to_timedelta(np.arange(n_windows) * 15, unit='min'),
                      per_window)
    # Symbols in one window settle together, as on the venue.
    won = np.repeat(rng.random(n_windows) < win_rate, per_window)
    contracts = np.full(len(opens), 10)
    return pd.DataFrame({
        'window_open': opens, 'contracts': contracts,
        'outlay': contracts * cost, 'payout': np.where(won, contracts, 0.0),
        'model_probability': np.full(len(opens), predicted),
    })


def test_a_real_edge_has_a_positive_lower_bound():
    values = trade_gate_values(trades(3000, win_rate=0.62, cost=0.55, predicted=0.625))
    assert values['realised_edge_lcb'] > 0
    assert values['overconfidence_abs'] < 0.015


def test_no_edge_has_a_lower_bound_below_zero():
    values = trade_gate_values(trades(3000, win_rate=0.55, cost=0.55, predicted=0.60))
    assert values['realised_edge_lcb'] < 0
    assert values['overconfidence_abs'] == pytest.approx(0.05, abs=0.02)


def test_resampling_whole_windows_widens_the_interval():
    """Resampling whole windows widens the interval: trades that settle together are one
    observation.
    """
    frame = trades(400, win_rate=0.6, cost=0.55, predicted=0.6)
    won = (frame['payout'] > 0).astype(float).to_numpy()
    by_window = bootstrap_interval(block_bootstrap_means(won, frame['window_open']))
    by_row = bootstrap_interval(block_bootstrap_means(won, np.arange(len(frame))))
    assert (by_window[1] - by_window[0]) > 1.4 * (by_row[1] - by_row[0])


def test_the_bound_is_deterministic():
    frame = trades(500, win_rate=0.6, cost=0.55, predicted=0.6)
    assert trade_gate_values(frame) == trade_gate_values(frame)


def test_no_trades_fail_both_trade_gates():
    values = trade_gate_values(pd.DataFrame())
    assert np.isnan(values['realised_edge_lcb'])
    assert np.isnan(values['overconfidence_abs'])
