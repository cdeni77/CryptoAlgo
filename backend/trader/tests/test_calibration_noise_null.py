"""`calibration_vs_market` must be judged against chance, not against zero.

**A zero bar is unreachable by construction.** ECE is built on absolute
deviations, so perturbing a well-calibrated forecast inflates measured ECE even
when the perturbation carries no bias at all. The model is market-initialised —
an untrained one reproduces the price exactly — so it starts at 0 and any
movement scores positive in expectation.

Measured on 7,611 live rows, adding PURE NOISE to the market logit:

    noise sd 0.05  ->  +0.00027   passes a zero bar 28% of the time
    noise sd 0.10  ->  +0.00053                     20%
    noise sd 0.20  ->  +0.00116                     15%
    noise sd 0.40  ->  +0.00178                      8%

So the old gate charged a model for moving, regardless of whether it moved
well. The null is now subtracted, which asks the question worth asking: is this
correction better calibrated than a RANDOM one of the same size?

**This does not let the current model through**, which is what makes it safe to
change while it is the only failing gate. The model reads +0.00359 against a
+0.00125 null — +0.00234 worse than its own noise, beaten by a random
perturbation in 98% of draws. The errors are systematic, matching the 6.7pp
overconfidence measured on the rows it trades.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import calibration_noise_null, market_gate_values


def _market_rows(n=4000, seed=0):
    """A well-calibrated market: P(up) drawn, outcomes drawn from it."""
    rng = np.random.default_rng(seed)
    k = np.clip(rng.normal(0.5, 0.18, n), 0.05, 0.95)
    y = (rng.random(n) < k).astype(float)
    return k, y


def test_the_null_is_positive_for_any_real_correction():
    """The whole point: noise alone scores above zero, so zero is the wrong
    bar."""
    k, y = _market_rows()
    null = calibration_noise_null(k, y, 0.25, draws=60)
    assert null > 0, 'if noise scored zero there would be nothing to correct'


def test_a_bigger_correction_has_a_bigger_null():
    """The bias scales with how far the correction travels, which is why the
    null is computed at the candidate's OWN magnitude rather than fixed."""
    k, y = _market_rows()
    small = calibration_noise_null(k, y, 0.10, draws=60)
    large = calibration_noise_null(k, y, 0.50, draws=60)
    assert large > small


def test_no_correction_means_no_null():
    k, y = _market_rows()
    assert calibration_noise_null(k, y, 0.0, draws=20) == 0.0


def test_the_null_is_deterministic_for_a_given_seed():
    """A gate that moved between two runs of the same candidate would be
    unusable."""
    k, y = _market_rows()
    a = calibration_noise_null(k, y, 0.25, draws=40)
    b = calibration_noise_null(k, y, 0.25, draws=40)
    assert a == b


def test_a_model_that_merely_perturbs_the_market_scores_near_zero():
    """The behaviour the rebase buys: a correction that is pure noise should
    land around the null, not be penalised for existing."""
    k, y = _market_rows(n=6000)
    rng = np.random.default_rng(7)
    lg = np.log(k/(1-k))
    noisy = 1/(1+np.exp(-(lg + rng.normal(0, 0.25, len(k)))))
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, mm, yy)
            for i, (kk, mm, yy) in enumerate(zip(k, noisy, y))]
    v = market_gate_values(rows)
    assert abs(v['calibration_vs_market']) < 0.002, (
        f"pure noise should sit near the null, got "
        f"{v['calibration_vs_market']:+.5f}")


def test_a_systematically_overconfident_model_still_fails():
    """The guard that makes this not a loosening: real miscalibration must
    still be caught, which is the case the live model is in."""
    k, y = _market_rows(n=6000)
    lg = np.log(k/(1-k))
    overconfident = 1/(1+np.exp(-lg*1.6))      # pushed away from 0.5, no noise
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, mm, yy)
            for i, (kk, mm, yy) in enumerate(zip(k, overconfident, y))]
    v = market_gate_values(rows)
    assert v['calibration_vs_market'] > 0, 'systematic overconfidence must fail'


def test_the_null_is_reported_alongside_the_gate():
    """So a reader can see the margin over chance rather than a bare number."""
    k, y = _market_rows(n=1000)
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, kk, yy)
            for i, (kk, yy) in enumerate(zip(k, y))]
    assert 'calibration_noise_null' in market_gate_values(rows)
