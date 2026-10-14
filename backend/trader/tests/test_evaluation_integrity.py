"""What the evaluation records about itself: weighting, provenance, trials,
the deflated Sharpe, the sealed holdout, the gate lock, one staleness limit."""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.promotion as promotion
from core.config import DEFAULT_CONFIG, Config
from core.metrics import (
    EvaluationReport, deflated_sharpe_ratio, evaluate_gates, gate_spec_hash,
    weighted_fold_mean,
)
from tests.test_promotion_and_metrics import (
    FakeModel, beats_the_market, fold, report, stats,
)


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Every write under a temporary models root, never the repository's."""
    monkeypatch.setattr(promotion, 'MODELS_ROOT', tmp_path)
    return tmp_path


def _sized_report(pairs) -> EvaluationReport:
    folds = []
    for i, (skill, windows) in enumerate(pairs):
        f = fold(i, skill=skill)
        f.n_windows = windows
        folds.append(f)
    return EvaluationReport(folds=folds, continuous=stats(),
                            config_provenance=Config().provenance())


# ---- fold weighting -------------------------------------------------------

class TestFoldWeighting:
    def test_equal_weights_reproduce_the_ordinary_mean_and_error(self):
        v = np.array([0.002, -0.001, 0.003, 0.0005, 0.001])
        w = weighted_fold_mean(v, np.full(5, 7.0))
        assert w.mean == pytest.approx(v.mean())
        assert w.standard_error == pytest.approx(v.std(ddof=1) / np.sqrt(5))
        assert w.effective_n == pytest.approx(5.0)

    def test_a_thin_fold_moves_the_mean_less_than_a_thick_one(self):
        r = _sized_report([(0.004, 5_000), (0.004, 5_000), (-0.010, 500)])
        assert r.mean_skill == pytest.approx((0.004 * 10_000 - 0.010 * 500) / 10_500)
        assert r.unweighted_skill.mean == pytest.approx((0.008 - 0.010) / 3)
        assert r.mean_skill > r.unweighted_skill.mean

    def test_unequal_folds_buy_fewer_effective_degrees_of_freedom(self):
        r = _sized_report([(0.002, 10_000), (0.001, 500), (0.003, 500)])
        assert r.effective_folds == pytest.approx(11_000 ** 2 / (10_000 ** 2 + 2 * 500 ** 2))
        assert r.effective_folds < 3

    def test_both_figures_reach_the_summary_and_the_ledger(self):
        r = _sized_report([(0.004, 5_000), (-0.001, 500), (0.002, 2_000)])
        assert 'unweighted' in r.summary()
        payload = promotion.report_provenance(r)
        assert payload['log_loss_skill'] == pytest.approx(r.mean_skill)
        assert payload['log_loss_skill_unweighted'] == pytest.approx(r.unweighted_skill.mean)
        assert payload['skill_weighting'] == 'windows'


class TestSkillSignificance:
    def test_noise_around_zero_fails_the_significance_gate(self):
        r = _sized_report([(0.0010, 2016), (-0.0002, 2016), (0.0, 2016)])
        gates = {g.name: g for g in evaluate_gates(r)}
        assert gates['log_loss_skill'].passed, 'the point estimate alone is fooled'
        assert not gates['log_loss_skill_p'].passed

    def test_a_consistent_skill_passes_it(self):
        rng = np.random.default_rng(3)
        skills = 0.0027 + rng.normal(0, 0.003, 28)
        r = _sized_report([(s, int(n)) for s, n in zip(skills, rng.integers(300, 3000, 28))])
        assert r.skill_p_value < 0.05

    def test_identical_positive_folds_are_certain_of_the_sign(self):
        assert report(skills=(0.002,) * 6).skill_p_value < 1e-9
        exact = weighted_fold_mean([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        assert exact.standard_error == 0.0 and exact.p_value == 0.0


# ---- provenance -----------------------------------------------------------

class TestProvenance:
    def test_every_field_is_recorded_and_json_ready(self):
        from dataclasses import fields

        p = Config().provenance()
        assert set(p) == {f.name for f in fields(Config)}
        json.dumps(p)
        assert p['symbols'] == list(Config().symbols)
        assert p['cli_overrides'] == []

    def test_any_knob_distinguishes_two_entries(self):
        a, b = Config(), Config(fold_block_days=21.0)
        assert a.provenance() != b.provenance()
        assert a.provenance_hash() != b.provenance_hash()
        assert Config(num_leaves=31).provenance_hash() != a.provenance_hash()
        assert Config(max_residual_scale=2.0).provenance_hash() != a.provenance_hash()

    def test_how_a_value_was_reached_is_not_what_it_is(self):
        moved = Config().with_overrides(kelly_fraction=Config().kelly_fraction)
        assert moved.cli_overrides and moved.provenance_hash() == Config().provenance_hash()

    def test_verify_now_sees_every_material_field(self):
        """`ForecastModel.verify` compares these only when both sides record them;
        four were never recorded, so its check on them was silently skipped."""
        p = Config().provenance()
        for key in ('maker_fee_rate', 'assume_maker', 'min_traded_price',
                    'max_traded_price', 'window_minutes', 'embargo_minutes'):
            assert key in p

    def test_live_adopts_nothing_from_a_default_artifact(self):
        """Real values recorded for the economics keys must not change what
        `config_for_artifact` adopts when the artifact matches the defaults."""
        from scripts.live import config_for_artifact

        model = types.SimpleNamespace(init_score_source='baseline',
                                      config_provenance=Config().provenance(),
                                      groups=())
        assert config_for_artifact(Config(), model, mode='paper') == Config()


# ---- the trial log --------------------------------------------------------

class TestTrialLog:
    def test_a_trial_records_what_was_run(self, models):
        record = promotion.record_trial(
            'evaluate', argv=['--groups', 'clock'], config=Config(),
            groups=['clock'], data_span={'windows': 10},
            metrics=promotion.trial_metrics(report()))
        line = json.loads((models / promotion.TRIALS).read_text().splitlines()[0])
        assert line['trial_id'] == record['trial_id']
        assert line['script'] == 'evaluate'
        assert line['argv'] == ['--groups', 'clock']
        assert line['config_hash'] == Config().provenance_hash()
        assert line['gate_spec_hash'] == gate_spec_hash()
        assert line['metrics']['log_loss_skill'] == pytest.approx(0.002)
        assert line['metrics']['sharpe'] == pytest.approx(1.4)
        assert 'git_commit' in line and 'timestamp' in line

    def test_the_log_is_strict_json(self, models):
        promotion.record_trial('ablate', argv=[], metrics={'x': float('nan')})
        assert 'NaN' not in (models / promotion.TRIALS).read_text()

    def test_a_promotion_counts_once_across_the_log_and_the_ledger(self, models):
        trial = promotion.new_trial_id()
        promotion.promote(FakeModel(), report(sharpe=-2.0), trial_id=trial)
        promotion.record_trial('promote', argv=[], trial_id=trial)
        promotion.record_trial('evaluate', argv=[])
        assert promotion.trial_count() == 2

    def test_an_old_ledger_entry_without_an_id_still_counts(self, models):
        (models / promotion.LEDGER).mkdir()
        (models / promotion.LEDGER / '20260101T000000Z.json').write_text(
            json.dumps({'version': '20260101T000000Z', 'report': {}}))
        promotion.record_trial('evaluate', argv=[])
        assert promotion.trial_count() == 2

    def test_a_run_refused_before_evaluating_is_not_a_trial(self, models):
        trial = promotion.new_trial_id()
        promotion.record_refusal(refused='spec changed', trial_id=trial)
        promotion.record_trial('promote', argv=[], trial_id=trial,
                               extra={'evaluated': False})
        assert promotion.trial_count() == 0

    def test_the_ledger_records_the_count_and_the_deflated_sharpe(self, models):
        for sharpe in (0.3, -1.0, 2.2):
            promotion.record_trial('evaluate', argv=[], metrics={'sharpe': sharpe})
        r = report()
        r.sharpe_moments = {'n_obs': 300, 'skew': 0.1, 'kurtosis': 4.0}
        attempt = promotion.promote(FakeModel(), r, extra=beats_the_market())
        stored = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert stored['trial_count'] == 4 == attempt.trial_count
        assert stored['trial_id'] == attempt.trial_id
        dsr = stored['deflated_sharpe']
        assert dsr['n_trials'] == 4
        assert 0.0 <= dsr['deflated_sharpe'] <= dsr['probabilistic_sharpe'] <= 1.0


# ---- the deflated Sharpe --------------------------------------------------

class TestDeflatedSharpe:
    KW = dict(n_obs=365, skew=0.0, kurtosis=3.0)

    def test_one_trial_is_the_probabilistic_sharpe(self):
        out = deflated_sharpe_ratio(2.0, trial_sharpes=[2.0], n_trials=1, **self.KW)
        assert out['deflated_sharpe'] == pytest.approx(out['probabilistic_sharpe'])
        assert out['expected_max_sharpe'] == 0.0

    def test_more_trials_deflate_more(self):
        trials = [0.0, 1.0, -1.0, 2.0]
        few = deflated_sharpe_ratio(2.0, trial_sharpes=trials, n_trials=4, **self.KW)
        many = deflated_sharpe_ratio(2.0, trial_sharpes=trials, n_trials=100, **self.KW)
        assert many['deflated_sharpe'] < few['deflated_sharpe'] < few['probabilistic_sharpe']

    def test_wider_trial_dispersion_deflates_more(self):
        tight = deflated_sharpe_ratio(2.0, trial_sharpes=[1.9, 2.0, 2.1], n_trials=3, **self.KW)
        wide = deflated_sharpe_ratio(2.0, trial_sharpes=[-2.0, 2.0, 6.0], n_trials=3, **self.KW)
        assert wide['deflated_sharpe'] < tight['deflated_sharpe']

    def test_the_bailey_lopez_de_prado_formula(self):
        from scipy.stats import norm

        trials, n, T, skew, kurt, sr_a = [0.5, 1.0, 1.5, 2.5], 4, 400, -0.3, 5.0, 2.5
        out = deflated_sharpe_ratio(sr_a, n_obs=T, skew=skew, kurtosis=kurt,
                                    trial_sharpes=trials, n_trials=n)
        k = np.sqrt(365.25)
        sr, v = sr_a / k, np.var(trials, ddof=1) / k ** 2
        g = 0.5772156649015329
        sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
        z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
        assert out['deflated_sharpe'] == pytest.approx(norm.cdf(z))

    def test_missing_moments_report_nan_rather_than_a_guess(self):
        out = deflated_sharpe_ratio(2.0, n_obs=0, skew=float('nan'), kurtosis=3.0,
                                    trial_sharpes=[1.0, 2.0], n_trials=2)
        assert np.isnan(out['deflated_sharpe'])


# ---- the gate lock at promotion -------------------------------------------

def _stale_lock(tmp_path) -> Path:
    from core.metrics import gate_spec
    rows = [dict(r, threshold=0.02) if r['name'] == 'calibration_error' else r
            for r in gate_spec()]
    path = tmp_path / 'stale.lock.json'
    path.write_text(json.dumps({'gate_spec_hash': 'not-the-current-hash', 'gates': rows}))
    return path


class TestGateLockAtPromotion:
    def test_every_ledger_entry_records_the_spec_hash(self, models):
        promotion.promote(FakeModel(), report(sharpe=-2.0))
        stored = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert stored['gate_spec_hash'] == gate_spec_hash()

    def test_a_changed_spec_is_refused_and_recorded(self, models, tmp_path):
        attempt = promotion.promote(FakeModel(), report(), extra=beats_the_market(),
                                    lock_path=_stale_lock(tmp_path))
        assert attempt.refused and not attempt.installed and not attempt.passed
        assert not (models / promotion.LIVE_MODEL).exists()
        stored = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert stored['refused']
        assert 'calibration_error' in stored['gate_change']['diff']['changed']

    def test_force_does_not_reach_past_a_changed_spec(self, models, tmp_path):
        attempt = promotion.promote(FakeModel(), report(sharpe=-2.0), force=True,
                                    force_reason='tail skill',
                                    lock_path=_stale_lock(tmp_path))
        assert attempt.refused and not attempt.installed

    def test_an_accepted_change_installs_and_carries_its_reason(self, models, tmp_path):
        attempt = promotion.promote(FakeModel(), report(), extra=beats_the_market(),
                                    lock_path=_stale_lock(tmp_path),
                                    accept_gate_change='re-derived under 7-day blocks')
        assert attempt.installed and not attempt.refused
        stored = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert stored['gate_change']['reason'] == 're-derived under 7-day blocks'
        assert stored['gate_change']['locked_hash'] == 'not-the-current-hash'
        assert stored['gate_change']['current_hash'] == gate_spec_hash()


# ---- the scripts log their trials -----------------------------------------

def _fake_dataset():
    index = pd.date_range('2026-01-01', periods=4, freq='15min', tz='UTC')
    return types.SimpleNamespace(
        window_index=index, windows=pd.DataFrame({'window_open': index}),
        symbols=['BTC-USD'], config=Config())


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, 'argv', [module.__name__] + argv)
    return module.main()


class TestScriptsRecordTrials:
    def _promote_stubs(self, monkeypatch, *, model=None):
        import scripts.promote as promote_script
        result = types.SimpleNamespace(
            report=report(), scored=None, models=[model or FakeModel()],
            trades=lambda: pd.DataFrame())
        monkeypatch.setattr(promote_script, 'load_dataset', lambda a, c: _fake_dataset())
        monkeypatch.setattr(promote_script, 'walk_forward', lambda *a, **k: result)
        monkeypatch.setattr(promote_script, 'market_measurement',
                            lambda *a, **k: beats_the_market())
        monkeypatch.setattr(promote_script, 'load_live', lambda *a, **k: None)
        monkeypatch.setattr(promote_script, 'refit_on_all', lambda *a, **k: None)
        monkeypatch.delenv('DATABASE_URL', raising=False)
        return promote_script

    def test_a_dry_run_is_a_trial_and_not_a_ledger_entry(self, models, monkeypatch):
        script = self._promote_stubs(monkeypatch)
        assert _run(script, ['--dry-run'], monkeypatch) == 0
        rows = promotion.read_trials()
        assert len(rows) == 1 and rows[0]['script'] == 'promote'
        assert rows[0]['promotion'] == {'dry_run': True}
        assert rows[0]['argv'] == ['--dry-run']
        assert not (models / promotion.LEDGER).exists() or not list(
            (models / promotion.LEDGER).glob('*.json'))

    def test_a_promotion_writes_both_under_one_trial_id(self, models, monkeypatch):
        script = self._promote_stubs(monkeypatch)
        assert _run(script, [], monkeypatch) == 0
        rows = promotion.read_trials()
        ledger = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert len(rows) == 1 and rows[0]['trial_id'] == ledger['trial_id']
        assert rows[0]['promotion']['installed'] is True
        assert promotion.trial_count() == 1

    def test_a_changed_spec_stops_promote_before_any_data_is_read(self, models, monkeypatch):
        script = self._promote_stubs(monkeypatch)
        monkeypatch.setattr(script, 'load_dataset', lambda *a: pytest.fail('read data'))
        monkeypatch.setattr(script, 'gate_lock_status', lambda: {
            'matches': False, 'current_hash': 'a' * 64, 'locked_hash': 'b' * 64,
            'diff': {'added': [], 'removed': [],
                     'changed': {'sharpe': {'from': {}, 'to': {}}}}})
        assert _run(script, [], monkeypatch) == 1
        ledger = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert ledger['refused'] and ledger['gate_change']['locked_hash'] == 'b' * 64
        assert promotion.read_trials()[0]['evaluated'] is False

    def test_evaluate_is_a_trial(self, models, monkeypatch):
        import scripts.evaluate as evaluate_script
        import scripts.promote as promote_script
        result = types.SimpleNamespace(report=report(), scored=None,
                                       trades=lambda: pd.DataFrame())
        monkeypatch.setattr(evaluate_script, 'load_dataset', lambda a, c: _fake_dataset())
        monkeypatch.setattr(evaluate_script, 'walk_forward', lambda *a, **k: result)
        monkeypatch.setattr(promote_script, 'market_measurement',
                            lambda *a, **k: beats_the_market())
        _run(evaluate_script, ['--no-trade', '--no-stress'], monkeypatch)
        rows = promotion.read_trials()
        assert len(rows) == 1 and rows[0]['script'] == 'evaluate'
        assert rows[0]['metrics']['model_minus_market'] is not None
        assert rows[0]['data_span']['windows'] == 4


# ---- the sealed holdout ---------------------------------------------------

class _NoStore:
    def read(self, *args, **kwargs):
        raise LookupError('no such table')


@pytest.fixture
def bars_store(monkeypatch):
    """`load_dataset` against synthetic bars, recording the `end` it asked for."""
    import scripts._common as common
    from tests.conftest import make_bars

    seen = {}
    bars = make_bars(days=4, start='2025-01-01')

    def load(config, *, store=None, start=None, end=None, symbols=None):
        seen['end'] = end
        return {s: (b[b['event_time'] < end] if end is not None else b).reset_index(drop=True)
                for s, b in bars.items()}

    monkeypatch.setattr(common, 'ResearchStore', _NoStore)
    monkeypatch.setattr(common, 'load_minute_bars', load)
    return seen


def _args(argv):
    from scripts._common import add_data_arguments
    return add_data_arguments(argparse.ArgumentParser()).parse_args(argv)


class TestSealedHoldout:
    def test_the_config_carries_it_and_refuses_a_bad_date(self):
        assert Config(holdout_start='2026-09-01').holdout_start == '2026-09-01'
        with pytest.raises(ValueError):
            Config(holdout_start='last tuesday')

    def test_sealed_windows_are_never_read(self, models, bars_store):
        from scripts._common import config_from_args, load_dataset

        args = _args(['--holdout-start', '2025-01-03'])
        config = config_from_args(args)
        dataset = load_dataset(args, config)
        cutoff = pd.Timestamp('2025-01-03', tz='UTC')
        assert bars_store['end'] == cutoff
        assert len(dataset.windows)
        assert (pd.to_datetime(dataset.windows['window_open'], utc=True) < cutoff).all()
        assert args.holdout_state == {'start': '2025-01-03', 'sealed': True}
        assert not (models / promotion.HOLDOUT_MARKER).exists()

    def test_an_earlier_end_still_wins(self, models, bars_store):
        from scripts._common import config_from_args, load_dataset

        args = _args(['--holdout-start', '2025-01-03', '--end', '2025-01-02'])
        load_dataset(args, config_from_args(args))
        assert bars_store['end'] == pd.Timestamp('2025-01-02', tz='UTC')

    def test_unsealing_reads_it_once_and_records_why(self, models, bars_store):
        from scripts._common import config_from_args, load_dataset

        argv = ['--holdout-start', '2025-01-03', '--unseal-holdout', 'final look']
        args = _args(argv)
        dataset = load_dataset(args, config_from_args(args))
        assert bars_store['end'] is None
        assert (pd.to_datetime(dataset.windows['window_open'], utc=True)
                >= pd.Timestamp('2025-01-03', tz='UTC')).any()
        assert args.holdout_state['sealed'] is False
        marker = [json.loads(l) for l in
                  (models / promotion.HOLDOUT_MARKER).read_text().splitlines()]
        assert marker[0]['holdout_start'] == '2025-01-03'
        assert marker[0]['reason'] == 'final look' and marker[0]['unsealed_at']

        again = _args(argv)
        with pytest.raises(SystemExit, match='already unsealed'):
            load_dataset(again, config_from_args(again))

        forced = _args(argv + ['--force-unseal'])
        load_dataset(forced, config_from_args(forced))
        lines = (models / promotion.HOLDOUT_MARKER).read_text().splitlines()
        assert len(lines) == 2 and json.loads(lines[1])['forced'] is True

    def test_a_different_holdout_date_is_a_new_seal(self, models):
        promotion.unseal_holdout('2025-01-03', 'first')
        promotion.unseal_holdout('2025-02-01', 'a fresh holdout')
        with pytest.raises(promotion.HoldoutError):
            promotion.unseal_holdout('2025-01-03', 'again')

    def test_unsealing_without_a_seal_is_a_usage_error(self, models, bars_store):
        from scripts._common import config_from_args, load_dataset

        args = _args(['--unseal-holdout', 'why not'])
        with pytest.raises(SystemExit, match='no seal'):
            load_dataset(args, config_from_args(args))

    def test_the_unseal_reaches_the_trial_log(self, models, bars_store):
        from scripts._common import config_from_args, load_dataset, log_trial

        args = _args(['--holdout-start', '2025-01-03', '--unseal-holdout', 'final'])
        config = config_from_args(args)
        dataset = load_dataset(args, config)
        row = log_trial('evaluate', args, config, dataset=dataset)
        assert row['holdout']['sealed'] is False
        assert row['holdout']['unsealed']['reason'] == 'final'
        assert row['config']['holdout_start'] == '2025-01-03'

    def test_the_holdout_reaches_the_ledger(self, models):
        state = {'start': '2025-01-03', 'sealed': True}
        promotion.promote(FakeModel(), report(sharpe=-2.0), holdout=state)
        stored = json.loads(next((models / promotion.LEDGER).glob('*.json')).read_text())
        assert stored['holdout'] == state


# ---- walk_forward cuts folds one way ---------------------------------------

def test_walk_forward_builds_its_folds_from_the_config(monkeypatch):
    import core.backtest as backtest

    class Called(Exception):
        pass

    def spy(index, config):
        raise Called(config.fold_block_days)

    monkeypatch.setattr(backtest, 'folds_for_config', spy)
    dataset = types.SimpleNamespace(window_index=pd.DatetimeIndex([]),
                                    config=Config(fold_block_days=3.0))
    with pytest.raises(Called) as excinfo:
        backtest.walk_forward(dataset)
    assert excinfo.value.args == (3.0,)


# ---- one staleness limit ---------------------------------------------------

def test_one_quote_staleness_limit_everywhere():
    from core.metrics import MAX_QUOTE_AGE_SECONDS
    from core.quotes import DEFAULT_MAX_AGE

    assert DEFAULT_CONFIG.max_quote_age_seconds == 30
    assert DEFAULT_MAX_AGE == MAX_QUOTE_AGE_SECONDS == DEFAULT_CONFIG.max_quote_age_seconds
