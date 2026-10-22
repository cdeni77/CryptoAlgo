"""A gate records the spread of its own null, where one is measured.

`calibration_vs_market` subtracts the MEAN of a 200-draw no-signal null and
then gates at zero. Under that null the observed value lands above the mean
half the time, so a model with no signal fails about half the time -- and it
failed 12 of the first 21 promotions. The spread was computed and discarded;
now it is reported, so a threshold can be read in units of it.

Reporting only. Changing a threshold needs `--accept-gate-change`.
"""

from __future__ import annotations

import numpy as np

from core.metrics import Gate, calibration_noise_null, evaluate_gates


def calibrated(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    market = rng.uniform(0.1, 0.9, n)
    outcome = (rng.uniform(size=n) < market).astype(float)
    return market, outcome


def test_the_null_reports_mean_and_spread():
    market, outcome = calibrated()
    null = calibration_noise_null(market, outcome, 0.25)
    assert null.draws == 200
    assert np.isfinite(null.sd) and null.sd > 0, 'the spread must be measured'
    assert null.mean > 0, 'any unbiased perturbation raises ECE'


def test_the_null_still_behaves_as_a_number_for_existing_arithmetic():
    """`model_ece - market_ece - null` must keep working."""
    market, outcome = calibrated()
    null = calibration_noise_null(market, outcome, 0.25)
    assert float(null) == null.mean


def test_a_degenerate_input_gives_no_spread_but_a_usable_centre():
    market, outcome = calibrated()
    null = calibration_noise_null(market, outcome, 0.0)
    assert null.mean == 0.0 and null.draws == 0
    assert not np.isfinite(null.sd), 'no draws means no measured spread'


def test_sigmas_expresses_the_breach_in_units_of_the_null():
    g = Gate('calibration_vs_market', value=0.0006, threshold=0.0,
             direction='max', sd=0.0033)
    assert not g.passed
    assert abs(g.sigmas - 0.1818) < 0.001, 'a fifth of a standard deviation'
    assert '[+0.18 sd of its null]' in g.line()


def test_a_gate_without_a_measured_null_reports_nothing_extra():
    g = Gate('max_drawdown', value=0.24, threshold=0.35, direction='max')
    assert not np.isfinite(g.sd) and not np.isfinite(g.sigmas)
    assert 'sd of its null' not in g.line()


def test_evaluate_gates_attaches_the_spread_to_the_right_gate():
    gates = evaluate_gates(
        _Report(), gates={'calibration_vs_market': (0.0, 'max'),
                          'max_drawdown': (0.35, 'max')},
        extra={'calibration_vs_market': 0.0006,
               'calibration_noise_sd': 0.0033, 'max_drawdown': 0.24})
    by_name = {g.name: g for g in gates}
    assert by_name['calibration_vs_market'].sd == 0.0033
    assert not np.isfinite(by_name['max_drawdown'].sd)


class _Report:
    def gate_values(self):
        return {}
