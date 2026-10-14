"""Tests for the ensemble gates EDGE_UNCERTAIN and MEMBERS_DISAGREE.

Both are off by default and pass a row without ensemble columns, so a single
model decides as before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import run_book
from core.config import Config
from core.decide import Reason, Side, decide, decide_window, stateless_screen

W = pd.Timestamp('2026-01-01 00:00', tz='UTC')


def row(**over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=9,
                baseline_probability=0.88, model_probability=0.95)
    base.update(over)
    return base


def test_without_the_columns_nothing_changes():
    strict = Config(edge_lower_bound_z=2.0, require_member_agreement=True)
    plain = decide(row(), Config(), bankroll=100.0)
    gated = decide(row(), strict, bankroll=100.0)
    assert plain.traded and gated == plain


def test_with_the_gates_off_the_columns_are_ignored():
    noisy = row(model_probability_sd=0.2, model_probability_min=0.5,
                model_probability_max=0.99)
    assert decide(noisy, Config(), bankroll=100.0).traded


def test_a_wide_spread_is_refused_as_uncertain():
    config = Config(edge_lower_bound_z=1.0)
    wide = decide(row(model_probability_sd=0.05), config, bankroll=100.0)
    assert wide.reason is Reason.EDGE_UNCERTAIN
    assert wide.side is Side.UP and np.isfinite(wide.edge)
    assert decide(row(model_probability_sd=0.001), config, bankroll=100.0).traded
    # sd = 0 is a measurement (the members agree), not a missing value.
    assert decide(row(model_probability_sd=0.0), config, bankroll=100.0).traded
    assert decide(row(model_probability_sd=np.nan), config, bankroll=100.0).traded


def test_the_lower_bound_reads_the_chosen_side():
    """For a DOWN trade the probability is 1 - q; the spread is the same either way."""
    config = Config(edge_lower_bound_z=1.0)
    down = row(model_probability=0.70)
    assert decide(down, config, bankroll=100.0).traded
    refused = decide({**down, 'model_probability_sd': 0.20}, config, bankroll=100.0)
    assert refused.side is Side.DOWN and refused.reason is Reason.EDGE_UNCERTAIN


def test_a_member_that_does_not_beat_the_cost_refuses_the_trade():
    config = Config(require_member_agreement=True)
    split = row(model_probability_min=0.85, model_probability_max=0.99)
    assert decide(split, config, bankroll=100.0).reason is Reason.MEMBERS_DISAGREE
    agree = row(model_probability_min=0.93, model_probability_max=0.97)
    assert decide(agree, config, bankroll=100.0).traded


def test_member_agreement_on_a_down_trade_reads_the_highest_member():
    config = Config(require_member_agreement=True)
    # Buying DOWN at 0.12: the worst member is the one with the HIGHEST P(up).
    split = row(model_probability=0.70, model_probability_min=0.60,
                model_probability_max=0.90)
    decision = decide(split, config, bankroll=100.0)
    assert decision.side is Side.DOWN
    assert decision.reason is Reason.MEMBERS_DISAGREE
    agree = row(model_probability=0.70, model_probability_min=0.65,
                model_probability_max=0.75)
    assert decide(agree, config, bankroll=100.0).traded


def test_the_edge_gate_still_fires_first():
    config = Config(edge_lower_bound_z=1.0, require_member_agreement=True)
    weak = row(model_probability=0.885, model_probability_sd=0.2,
               model_probability_min=0.0, model_probability_max=1.0)
    assert decide(weak, config, bankroll=100.0).reason is Reason.EDGE_BELOW_GATE


def _frame():
    rng = np.random.default_rng(0)
    n = 400
    q = rng.uniform(0.3, 0.98, n)
    sd = rng.uniform(0.0, 0.06, n)
    return pd.DataFrame({
        'symbol': 'BTC-USD', 'window_open': W + pd.to_timedelta(np.arange(n) * 15, 'min'),
        'settle_time': W + pd.to_timedelta(np.arange(n) * 15 + 15, 'min'),
        'offset': 9, 'baseline_probability': 0.88, 'model_probability': q,
        'model_probability_sd': sd,
        'model_probability_min': np.clip(q - 2 * sd, 0, 1),
        'model_probability_max': np.clip(q + 2 * sd, 0, 1),
        'outcome': (rng.random(n) < q).astype(float),
    })


@pytest.mark.parametrize('config', [
    Config(edge_lower_bound_z=1.0),
    Config(require_member_agreement=True),
    Config(edge_lower_bound_z=0.5, require_member_agreement=True),
])
def test_the_vectorised_screen_agrees_with_decide(config):
    frame = _frame()
    survivors, counts = stateless_screen(frame, config)
    scalar = [decide(r, config, bankroll=1e9) for _, r in frame.iterrows()]
    stateless = {Reason.NOT_FINITE, Reason.PRICE_OUT_OF_BAND,
                 Reason.DISAGREEMENT_IMPLAUSIBLE, Reason.EDGE_BELOW_GATE,
                 Reason.EDGE_UNCERTAIN, Reason.MEMBERS_DISAGREE}
    refused = [d.reason for d in scalar if d.reason in stateless]
    assert len(survivors) == len(frame) - len(refused)
    for reason in stateless:
        assert counts[reason.value] == sum(r is reason for r in refused)
    assert counts[Reason.EDGE_UNCERTAIN.value] + counts[Reason.MEMBERS_DISAGREE.value] > 0


def test_a_frame_without_the_columns_screens_as_before():
    frame = _frame().drop(columns=['model_probability_sd', 'model_probability_min',
                                   'model_probability_max'])
    strict = Config(edge_lower_bound_z=2.0, require_member_agreement=True)
    a, counts_a = stateless_screen(frame, Config())
    b, counts_b = stateless_screen(frame, strict)
    assert a.index.equals(b.index)
    assert counts_b[Reason.EDGE_UNCERTAIN.value] == 0
    assert counts_b[Reason.MEMBERS_DISAGREE.value] == 0
    assert decide_window(frame.head(1), strict, bankroll=100.0) == \
        decide_window(frame.head(1), Config(), bankroll=100.0)


def test_the_backtest_book_honours_the_gates():
    frame = _frame()
    loose, _ = run_book(frame, Config())
    strict, _ = run_book(frame, Config(edge_lower_bound_z=3.0))
    assert len(strict.trades()) < len(loose.trades())
    assert strict.stateless_rejections[Reason.EDGE_UNCERTAIN.value] > 0
