"""Tests for the promotion gates and the ledger of attempts.

A blocked candidate never reaches `models/forecast.joblib`, the file live loads.
Every attempt is recorded, passed or blocked, because the ledger is the trial
count for multiple-testing correction.
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
    DEFAULT_GATES, GATE_NOTES, IMPLAUSIBLE_SHARPE, MIN_MARKET_WINDOWS,
    EvaluationReport, FoldEvaluation, market_gate_values,
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
        # A fold without these values has no measured calibration, and both gates fail
        # closed on a missing value, so the fixture supplies them.
        model_max_deviation=deviation, n_non_finite=non_finite,
        stats=stats(**over))


def report(skills=(0.002,) * 6, *, deviation=0.02, non_finite=0,
           trade_metrics=None, **over) -> EvaluationReport:
    """A report: `deviation` and `non_finite` are per fold, `trade_metrics` stands in for
    traded rows, the rest sizes the book.
    """
    return EvaluationReport(
        folds=[fold(i, skill=s, deviation=deviation, non_finite=non_finite)
               for i, s in enumerate(skills)],
        continuous=stats(**over), config_provenance=Config().provenance(),
        trade_metrics=(trade_metrics if trade_metrics is not None else
                       {'realised_edge_lcb': 0.3, 'overconfidence_abs': 0.005}))


def beats_the_market() -> dict:
    """The market measurement a fully passing candidate must carry.

    A gate reads a measurement, so a fixture that omits it describes an unmeasured
    candidate. `market_gate_values` is a pure function of rows.
    """
    return market_gate_values(market_rows(4_000, model_edge=+0.35))


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
    """The standard error comes from fold dispersion, not a breadth formula.

    Offsets share a label and symbols are correlated, so `N/(1+(N-1)rho)` is
    degenerate here.
    """
    tight = report(skills=(0.002, 0.0021, 0.0019, 0.002, 0.0022, 0.0018))
    loose = report(skills=(0.01, -0.008, 0.012, -0.006, 0.009, -0.005))
    assert tight.skill_standard_error < loose.skill_standard_error
    assert tight.skill_t > loose.skill_t
    assert tight.folds_total == 6


def test_five_of_six_is_reported_with_its_p_value():
    """Five of six positive is reported with its p-value (7/64 under no skill)."""
    r = report(skills=(0.002, 0.002, 0.002, 0.002, 0.002, -0.001))
    assert r.folds_positive == 5
    assert r.sign_agreement_p_value == pytest.approx(7 / 64, abs=1e-6)


def test_a_gate_with_no_measurement_fails():
    """A gate with no measurement fails."""
    gate = Gate(name='x', value=float('nan'), threshold=0.0, direction='min')
    assert not gate.passed


def test_every_gate_carries_a_note_explaining_it():
    for name in DEFAULT_GATES:
        assert GATE_NOTES.get(name), f'{name} has no explanation'


def test_the_forecast_gates_come_before_the_money_gates():
    """The forecast gates come before the money gates."""
    names = list(DEFAULT_GATES)
    assert names.index('log_loss_skill') < names.index('realised_edge_lcb')
    assert names.index('calibration_error') < names.index('trades')


def test_an_implausible_sharpe_fails_even_though_it_is_high():
    """A Sharpe above the plausible limit fails, since it signals a bug."""
    good = evaluate_gates(report(sharpe=1.4))
    assert {g.name: g.passed for g in good}['sharpe_implausible']
    absurd = evaluate_gates(report(sharpe=IMPLAUSIBLE_SHARPE + 7))
    flags = {g.name: g.passed for g in absurd}
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
    attempt = promote(FakeModel(), report(), root=tmp_path,
                      extra=beats_the_market())
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
    promote(FakeModel(), report(), root=tmp_path, extra=beats_the_market())
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
    attempt = evaluate_candidate(FakeModel(), report(),
                                 extra=beats_the_market())
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
    """A few unscoreable rows (as from an outage) pass, because the gate is a share, not a
    count.
    """
    gates = {g.name: g for g in evaluate_gates(report(non_finite=4))}
    assert gates['non_finite_share'].passed, (
        f"24 unscoreable rows in ~480,000 scored {gates['non_finite_share'].value:.6f}; "
        f'an outage has to pass'
    )
    assert gates_passed(evaluate_gates(report(non_finite=4),
                                       extra=beats_the_market()))


def test_a_large_share_of_unscoreable_rows_does_fail():
    """A large share of unscoreable rows fails, since that points to a lookback or embargo
    problem.
    """
    # 8,000 unscoreable per fold against 80,000 rows is 9%, well over the 0.1% limit.
    gates = {g.name: g for g in evaluate_gates(report(non_finite=8_000))}
    assert not gates['non_finite_share'].passed
    assert not gates_passed(evaluate_gates(report(non_finite=8_000)))


def test_a_nan_calibration_in_one_fold_reaches_the_gate():
    """A NaN calibration in one fold reaches the gate, whatever the fold order.

    The builtin `max` is order-dependent with NaN, so the aggregation must not use it.
    """
    folds = [fold(0, ece=float('nan')), fold(1), fold(2), fold(3), fold(4), fold(5)]
    subject = EvaluationReport(folds=folds, continuous=stats(),
                               config_provenance=Config().provenance())
    assert not np.isfinite(subject.max_ece), 'the NaN fold was skipped'
    assert not {g.name: g for g in evaluate_gates(subject)}['calibration_error'].passed


def test_a_model_miscalibrated_only_where_it_trades_is_refused():
    """A model miscalibrated only where it trades is refused.

    The mean ECE averages that band away, so `calibration_vs_market` is the gate that
    catches it.
    """
    gates = {g.name: g for g in evaluate_gates(report(deviation=0.05))}
    assert gates['calibration_error'].passed, (
        'the mean ECE is meant to be blind to this; if it is not, the fixture '
        'no longer demonstrates the problem'
    )
    # The protection is `calibration_vs_market`. The market itself misses its worst
    # bin by several points, so an absolute bar tight enough to catch this would also
    # reject every model that tracks the price.
    extra = {'market_windows': 5_000.0, 'model_minus_market': 0.001,
             'baseline_minus_market': 0.0,
             'market_max_deviation': 0.01, 'calibration_vs_market': 0.04}
    relative = {g.name: g for g in evaluate_gates(report(deviation=0.05),
                                                 extra=extra)}
    assert not relative['calibration_vs_market'].passed, (
        'a model 5pp off where the market is not must still be refused'
    )
    # The loose absolute bar still catches gross miscalibration.
    gross = {g.name: g for g in evaluate_gates(report(deviation=0.15))}
    assert not gross['calibration_max_deviation'].passed


# ---- the market as the benchmark -----------------------------------------

def market_rows(n: int, *, model_edge: float = 0.0, windows: int = 4_000):
    """Synthetic `scored_against_market()` rows.

    `model_edge` moves the model toward the truth relative to the price: positive
    is a better forecaster than the quote, negative a worse one.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        outcome = float(rng.integers(0, 2))
        market = 0.5 + (0.18 if outcome else -0.18) + rng.normal(0, 0.04)
        market = float(np.clip(market, 0.02, 0.98))
        toward = (1.0 - market) if outcome else (0.0 - market)
        model = float(np.clip(market + model_edge * toward, 0.01, 0.99))
        rows.append(('BTC-USD', pd.Timestamp('2026-08-01', tz='UTC')
                     + pd.Timedelta(minutes=15 * (i % windows)),
                     3, market, market, model, outcome))
    return rows


class TestTheMarketGates:
    """Tests the market gates: the counterparty is the price, not `F(x/sigma)`."""

    def test_a_model_that_loses_to_the_price_fails_even_passing_everything_else(self):
        """A model that loses to the price fails even when every other gate passes."""
        others = {g.name: g.passed for g in evaluate_gates(
            report(), extra=market_gate_values(market_rows(4_000, model_edge=-0.35)))}
        assert others['log_loss_skill'], 'the fixture must otherwise be a passing one'
        assert others['realised_edge_lcb']
        assert not others['model_minus_market'], (
            'a model worse than the quote it must trade against was promotable'
        )

    def test_a_model_that_beats_the_price_passes(self):
        gates = {g.name: g.passed for g in evaluate_gates(
            report(), extra=market_gate_values(market_rows(4_000, model_edge=+0.35)))}
        assert gates['model_minus_market'] and gates['market_windows']

    def test_too_few_windows_fails_however_good_the_comparison_looks(self):
        """Too few market windows fails, however good the comparison looks."""
        values = market_gate_values(market_rows(300, model_edge=+0.35, windows=100))
        gates = {g.name: g.passed for g in evaluate_gates(report(), extra=values)}
        assert values['model_minus_market'] > 0
        assert not gates['market_windows']

    def test_no_quotes_at_all_fails_rather_than_passes(self):
        """No quotes at all fails: not measured is not a pass."""
        values = market_gate_values([])
        assert values['market_windows'] == 0
        assert not np.isfinite(values['model_minus_market'])
        gates = {g.name: g.passed for g in evaluate_gates(report(), extra=values)}
        assert not gates['market_windows'] and not gates['model_minus_market']

    def test_forgetting_to_pass_the_measurement_cannot_promote(self):
        """Omitting `extra` fails closed."""
        gates = {g.name: g.passed for g in evaluate_gates(report())}
        assert not gates['model_minus_market']
        assert not gates['market_windows']

    def test_the_market_gates_are_read_before_the_baseline_ones(self):
        """The market gates come before the baseline gates."""
        names = list(DEFAULT_GATES)
        assert names.index('model_minus_market') < names.index('log_loss_skill')
        assert names.index('market_windows') < names.index('trades')

    def test_the_threshold_is_defined_once(self):
        """`scripts.market_benchmark` and the gate share one window threshold."""
        from scripts.market_benchmark import MIN_WINDOWS

        assert MIN_WINDOWS == MIN_MARKET_WINDOWS
        assert DEFAULT_GATES['market_windows'][0] == float(MIN_MARKET_WINDOWS)

    def test_a_backtest_alone_can_never_satisfy_these(self):
        """A backtest report alone cannot supply the market gates, since it has no book.
        """
        assert 'model_minus_market' not in report().gate_values()
        assert 'market_windows' not in report().gate_values()


# --- the ledger carries the reliability table -------------------------------
#
# The ledger is the record of account, so it must hold the reliability table
# itself and not depend on the Postgres mirror.

def test_the_ledger_carries_the_deployed_folds_reliability_table():
    import pandas as pd
    from core.promotion import reliability_rows

    class Table:
        def frame(self):
            return pd.DataFrame({
                'bin_low': [0.0, 0.4, 0.8], 'bin_high': [0.2, 0.6, 1.0],
                'predicted': [0.10, 0.50, 0.90],
                'observed': [0.12, 0.48, 0.93], 'count': [40, 0, 25]})

    early = type('F', (), {'reliability_table': None})()
    deployed = type('F', (), {'reliability_table': Table()})()
    rows = reliability_rows(type('R', (), {'folds': [early, deployed]})())

    # The last fold, trained on the most history, is the deployed one.
    # `publish_to_serving`
    # makes the same choice.
    assert len(rows) == 2, 'empty bins are dropped, populated ones kept'
    assert rows[0] == {'bin_low': 0.0, 'bin_high': 0.2, 'predicted': 0.10,
                       'observed': 0.12, 'count': 40}


def test_reliability_rows_are_strict_json():
    """Reliability rows are strict JSON: no bare NaN, which strict parsers reject."""
    import json
    import pandas as pd
    from core.promotion import reliability_rows

    class Table:
        def frame(self):
            return pd.DataFrame({
                'bin_low': [0.0], 'bin_high': [0.2],
                'predicted': [float('nan')], 'observed': [0.12], 'count': [5]})

    rows = reliability_rows(
        type('R', (), {'folds': [type('F', (), {'reliability_table': Table()})()]})())
    assert rows[0]['predicted'] is None
    assert 'NaN' not in json.dumps(rows)


def test_a_report_with_no_folds_does_not_raise():
    """A report with no folds returns no rows instead of raising, so the ledger entry is
    still written.
    """
    from core.promotion import reliability_rows
    assert reliability_rows(type('R', (), {'folds': []})()) == []
    assert reliability_rows(type('R', (), {})()) == []


def test_report_provenance_actually_CARRIES_the_reliability_rows():
    """`report_provenance` actually includes the reliability rows."""
    import pandas as pd
    from core.promotion import report_provenance

    class Table:
        def frame(self):
            return pd.DataFrame({
                'bin_low': [0.0], 'bin_high': [0.2], 'predicted': [0.10],
                'observed': [0.12], 'count': [40]})

    report = type('R', (), {
        'folds': [type('F', (), {'reliability_table': Table()})()],
        'folds_total': 6, 'total_windows': 1000, 'mean_skill': 0.001,
        'skill_standard_error': 0.0002, 'skill_t': 5.0, 'folds_positive': 6,
        'sign_agreement_p_value': 0.03, 'max_ece': 0.019,
        'mean_residual_scale': 1.1, 'max_control_gain_share': 0.0,
        'gate_values': staticmethod(lambda: {}),
        'config_provenance': {}, 'notes': [],
    })()
    payload = report_provenance(report)
    assert 'reliability' in payload, 'the ledger must carry the table'
    assert payload['reliability'] == [
        {'bin_low': 0.0, 'bin_high': 0.2, 'predicted': 0.10,
         'observed': 0.12, 'count': 40}]
