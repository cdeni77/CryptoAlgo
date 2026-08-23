"""Gates, and the ledger that makes a trial count possible.

Two invariants. A blocked candidate must not reach `models/forecast.joblib` —
that path is what the live signal writer loads by name, so it is the single place
a model becomes real. And every attempt is recorded, passed or blocked, because a
project that deletes its failures cannot compute its own multiple-testing
correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.book import BookStats
from core.config import Config
from core.metrics import (
    DEFAULT_GATES, GATE_NOTES, IMPLAUSIBLE_SHARPE, EvaluationReport, FoldEvaluation,
    Gate, brier_skill, evaluate_gates, gate_report, gates_passed, log_loss_skill,
)
from core.promotion import (
    LIVE_MODEL, evaluate_candidate, history, promote, trial_count, version_stamp,
)


def stats(**over) -> BookStats:
    base = dict(n_trades=2_000, n_windows_available=100_000, coverage=0.02,
                starting_bankroll=100.0, ending_equity=118.0, total_return=0.18,
                total_pnl=18.0, total_fees=9.0, win_rate=0.88,
                mean_edge_pp=1.4, realised_edge_pp=0.6,
                mean_return_on_stake=0.01, sd_return_on_stake=0.2,
                trades_per_year=30_000.0, sharpe=1.4, max_drawdown=0.2,
                halted=False)
    base.update(over)
    return BookStats(**base)


def fold(index=0, skill=0.002, ece=0.009, alpha=0.7, control=0.18,
         deviation=0.02, non_finite=0, **over):
    return FoldEvaluation(
        index=index,
        test_start=pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(days=60 * index),
        test_end=pd.Timestamp('2025-03-01', tz='UTC') + pd.Timedelta(days=60 * index),
        n_rows=80_000, n_windows=20_000,
        model_log_loss=0.51 - skill, baseline_log_loss=0.51,
        model_brier=0.17, baseline_brier=0.175,
        model_ece=ece, baseline_ece=0.021,
        residual_scale=alpha, control_gain_share=control,
        # A fold that does not report these is a fold whose calibration was not
        # measured, and both gates fail closed on a missing value — so the
        # fixture has to supply them like any other measurement.
        model_max_deviation=deviation, n_non_finite=non_finite,
        stats=stats(**over))


def report(skills=(0.002,) * 6, *, deviation=0.02, non_finite=0,
           **over) -> EvaluationReport:
    """`deviation` and `non_finite` are per-fold; everything else sizes the book."""
    return EvaluationReport(
        folds=[fold(i, skill=s, deviation=deviation, non_finite=non_finite)
               for i, s in enumerate(skills)],
        continuous=stats(**over), config_provenance=Config().provenance())


class FakeModel:
    """Enough of a model to be promoted, without fitting one."""

    def __init__(self, deployable=True):
        self.scoring = object() if deployable else None
        self.residual_scale = 0.7
        self.features = ['z_score']
        self.n_train_windows = 50_000

    @property
    def deployable(self):
        return self.scoring is not None

    def provenance(self):
        return {'features': self.features, 'deployable': self.deployable}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fake-booster')
        path.with_suffix('.provenance.json').write_text(json.dumps(self.provenance()))
        return path


# ------------------------------------------------------------------ metrics

def test_skill_is_the_difference_not_the_level():
    y = np.array([1, 1, 0, 1, 0] * 200, dtype=float)
    baseline = np.where(y == 1, 0.7, 0.3)
    better = np.where(y == 1, 0.8, 0.2)
    worse = np.where(y == 1, 0.6, 0.4)
    assert log_loss_skill(y, better, baseline) > 0
    assert log_loss_skill(y, worse, baseline) < 0
    assert brier_skill(y, better, baseline) > 0


def test_the_standard_error_comes_from_fold_dispersion():
    """Not from a breadth formula.

    Four offsets share a label and the three symbols are ~0.7 correlated within a
    window, so `N/(1+(N-1)rho)` on this structure is not merely optimistic but
    degenerate — cross-sectional structure can drive the denominator negative.
    """
    tight = report(skills=(0.002, 0.0021, 0.0019, 0.002, 0.0022, 0.0018))
    loose = report(skills=(0.01, -0.008, 0.012, -0.006, 0.009, -0.005))
    assert tight.skill_standard_error < loose.skill_standard_error
    assert tight.skill_t > loose.skill_t
    assert tight.folds_total == 6


def test_five_of_six_is_reported_with_its_p_value():
    """Because it happens 10.9% of the time under no skill."""
    r = report(skills=(0.002, 0.002, 0.002, 0.002, 0.002, -0.001))
    assert r.folds_positive == 5
    assert r.sign_agreement_p_value == pytest.approx(7 / 64, abs=1e-6)


def test_a_gate_with_no_measurement_fails():
    """"Not measured" fails like every other gate here."""
    gate = Gate(name='x', value=float('nan'), threshold=0.0, direction='min')
    assert not gate.passed


def test_every_gate_carries_a_note_explaining_it():
    for name in DEFAULT_GATES:
        assert GATE_NOTES.get(name), f'{name} has no explanation'


def test_the_forecast_gates_come_before_the_money_gates():
    """A candidate that fails on skill should not have its Sharpe discussed."""
    names = list(DEFAULT_GATES)
    assert names.index('log_loss_skill') < names.index('sharpe')
    assert names.index('calibration_error') < names.index('total_return')


def test_an_implausible_sharpe_fails_even_though_it_is_high():
    """Every other gate asks whether the number is good; this asks if it is possible.

    The first full run of this stack reported +12.6 and passed `>= 0.5`.
    """
    good = evaluate_gates(report(sharpe=1.4))
    assert {g.name: g.passed for g in good}['sharpe_implausible']
    absurd = evaluate_gates(report(sharpe=IMPLAUSIBLE_SHARPE + 7))
    flags = {g.name: g.passed for g in absurd}
    assert flags['sharpe'] is True
    assert flags['sharpe_implausible'] is False
    assert not gates_passed(absurd)


def test_abstaining_on_everything_does_not_pass():
    gates = evaluate_gates(report(n_trades=0, coverage=0.0))
    assert not gates_passed(gates)
    failed = {g.name for g in gates if not g.passed}
    assert 'trades' in failed and 'coverage' in failed


def test_a_clock_driven_model_fails_the_control_gate():
    r = EvaluationReport(
        folds=[fold(i, control=0.62) for i in range(6)], continuous=stats())
    gates = {g.name: g.passed for g in evaluate_gates(r)}
    assert gates['control_gain_share'] is False


def test_the_gate_report_names_what_failed():
    text = gate_report(evaluate_gates(report(sharpe=-1.0)))
    assert 'sharpe' in text and 'FAIL' in text


def test_per_offset_skill_is_aggregated_across_folds():
    folds = []
    for i in range(3):
        f = fold(i)
        f.per_offset = pd.DataFrame({'offset': [3, 12], 'n': [100, 100],
                                     'skill': [0.003, 0.001],
                                     'mean_abs_correction_pp': [2.0, 1.0]})
        folds.append(f)
    table = EvaluationReport(folds=folds, continuous=stats()).per_offset()
    assert set(table['offset']) == {3, 12}
    assert table.loc[table['offset'] == 3, 'folds_positive'].iloc[0] == 3


# ---------------------------------------------------------------- promotion

def test_a_blocked_candidate_does_not_reach_the_live_path(tmp_path):
    attempt = promote(FakeModel(), report(sharpe=-2.0), root=tmp_path)
    assert not attempt.passed
    assert not attempt.installed
    assert not (tmp_path / LIVE_MODEL).exists(), (
        'a blocked candidate wrote the artifact the live path loads by name'
    )


def test_a_passing_candidate_installs_atomically(tmp_path):
    attempt = promote(FakeModel(), report(), root=tmp_path)
    assert attempt.passed and attempt.installed
    assert (tmp_path / LIVE_MODEL).exists()
    assert not list(tmp_path.glob(f'.{LIVE_MODEL}.incoming')), (
        'a temporary file was left where the live path could load it'
    )


def test_forcing_requires_a_written_reason(tmp_path):
    with pytest.raises(ValueError, match='written reason'):
        promote(FakeModel(), report(sharpe=-2.0), root=tmp_path, force=True)


def test_a_forced_install_records_the_reason_with_the_artifact(tmp_path):
    reason = 'skill is on the >0.9 tail; the gates read averages'
    attempt = promote(FakeModel(), report(sharpe=-2.0), root=tmp_path,
                      force=True, force_reason=reason)
    assert attempt.installed and attempt.forced
    stored = json.loads(next((tmp_path / 'promotions').glob('*.json')).read_text())
    assert stored['force_reason'] == reason
    assert stored['forced'] is True
    assert stored['passed'] is False


def test_every_attempt_is_recorded_even_when_blocked(tmp_path):
    promote(FakeModel(), report(sharpe=-2.0), root=tmp_path)
    promote(FakeModel(), report(sharpe=-3.0), root=tmp_path)
    promote(FakeModel(), report(), root=tmp_path)
    frame = history(tmp_path)
    assert len(frame) == 3
    assert trial_count(tmp_path) == 3
    assert frame['installed'].sum() == 1, 'the ledger lost a blocked attempt'


def test_the_ledger_survives_an_unreadable_entry(tmp_path):
    promote(FakeModel(), report(), root=tmp_path)
    (tmp_path / 'promotions' / 'corrupt.json').write_text('{not json')
    frame = history(tmp_path)
    assert len(frame) == 1


def test_evaluate_candidate_touches_no_filesystem(tmp_path):
    attempt = evaluate_candidate(FakeModel(), report())
    assert attempt.passed
    assert not attempt.installed
    assert not any(tmp_path.iterdir())


def test_the_version_stamp_sorts_chronologically():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 8, 23, 1, 2, 3, tzinfo=timezone.utc)
    first = version_stamp(now)
    second = version_stamp(now + timedelta(seconds=1))
    assert first < second
    assert first == '20260823T010203Z'


def test_a_handful_of_unscoreable_rows_is_not_a_failure():
    """An outage is not a defect, so the gate is a share and not a count.

    A row with no volatility estimate has no forecast. Measured on real bars: one
    6.5-hour Coinbase outage leaves ~86 rows in 372,532 unscoreable in the two
    hours afterwards, because the 240-minute lookback cannot be filled. Refusing
    to evaluate at all because the venue went down in May is not a judgement about
    the model — and my first version of this gate did exactly that, and killed a
    whole `scripts.evaluate` run.
    """
    gates = {g.name: g for g in evaluate_gates(report(non_finite=4))}
    assert gates['non_finite_share'].passed, (
        f"24 unscoreable rows in ~480,000 scored {gates['non_finite_share'].value:.6f}; "
        f'an outage has to pass'
    )
    assert gates_passed(evaluate_gates(report(non_finite=4)))


def test_a_large_share_of_unscoreable_rows_does_fail():
    """Because that is a lookback or an embargo, not the venue.

    What must never happen again is the silence: `np.mean` propagated the NaN into
    every fold statistic while `np.digitize` filed those rows in the 0.95-1.00
    reliability bin — the band this system trades — and `scripts/baseline.py` then
    printed "gate passed: worst-fold calibration error 0.01516 <= 0.02" with five
    of six folds reading NaN, because `nan > 0.02` is False.
    """
    # 8,000 unscoreable per fold against 80,000 rows is 9%, well over the 0.1%.
    gates = {g.name: g for g in evaluate_gates(report(non_finite=8_000))}
    assert not gates['non_finite_share'].passed
    assert not gates_passed(evaluate_gates(report(non_finite=8_000)))


def test_a_nan_calibration_in_one_fold_reaches_the_gate():
    """The aggregation must not skip the fold it could not measure.

    `max_ece` used the builtin `max`, which is order-dependent with NaN:
    `max([0.015, nan])` is 0.015 and `max([nan, 0.015])` is nan. So whether an
    unmeasurable fold was noticed depended on fold ordering. `Gate.passed`
    already fails closed on a non-finite value; the aggregation has to let it
    get there.
    """
    folds = [fold(0, ece=float('nan')), fold(1), fold(2), fold(3), fold(4), fold(5)]
    subject = EvaluationReport(folds=folds, continuous=stats(),
                               config_provenance=Config().provenance())
    assert not np.isfinite(subject.max_ece), 'the NaN fold was skipped'
    assert not {g.name: g for g in evaluate_gates(subject)}['calibration_error'].passed


def test_a_model_miscalibrated_only_where_it_trades_is_refused():
    """The mean ECE averages away the band the money is in.

    Constructed and measured: perfectly calibrated on 190,000 pinned rows and
    5pp overconfident on the 10,000 it trades gives an aggregate ECE of 0.0044 —
    a pass with 78% of the budget unused — because those rows contribute
    0.05 * 10/200 = 0.0025. `calibration_max_deviation` is the gate that sees it.
    """
    gates = {g.name: g for g in evaluate_gates(report(deviation=0.05))}
    assert gates['calibration_error'].passed, (
        'the mean ECE is meant to be blind to this; if it is not, the fixture '
        'no longer demonstrates the problem'
    )
    assert not gates['calibration_max_deviation'].passed

