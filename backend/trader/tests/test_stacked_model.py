"""The two-stage stack: spot -> probability on all bars, the market-relative
part on quoted rows only, with stage 1 seen out of sample by stage 2.

`b` is the headline: how much of the spot model's disagreement with the price
survives. The tests that matter most are the ones that could make it lie — a
stage-1 overfit read in-sample inflates b toward its ceiling, and the cross-fit
exists so it cannot.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import pytest

from core.baseline import expit, logit
from core.config import Config
from core.dataset import Dataset, apply_fold, fit_fold
from core.model import (
    MARKET_LOGIT, ForecastModel, StackedForecastModel, _fit_stack,
    crossfit_spot, fit_for_config, fit_stacked_model, stack_feature_groups,
    table_groups,
)
from tests.conftest import make_bars
from tests.stack_fixtures import make_depth

FAST = Config(n_estimators=40, early_stopping_rounds=8, n_folds=2,
              seasonality_min_days=5, stack_mode='spot_then_market',
              quote_sources=('live_touch',))


# ---- stage 2 alone ------------------------------------------------------------

def _planted(n=40_000, a=0.9, b=0.6, c=0.0, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1.2, n)
    s = m + rng.normal(0, 0.6, n)
    y = (rng.random(n) < expit(a * m + b * (s - m) + c)).astype(float)
    return m, s, y


class TestStageTwo:
    def test_it_recovers_a_planted_a_and_b(self):
        m, s, y = _planted()
        a, b, c, coef = _fit_stack(m, s, np.zeros((len(y), 0)), y)
        assert a == pytest.approx(0.9, abs=0.05)
        assert b == pytest.approx(0.6, abs=0.06)
        assert c == 0.0 and coef.shape == (0,)

    def test_it_recovers_a_planted_intercept_when_allowed(self):
        m, s, y = _planted(c=0.2)
        a, b, c, _ = _fit_stack(m, s, np.zeros((len(y), 0)), y, intercept=True)
        assert c == pytest.approx(0.2, abs=0.05)
        assert b == pytest.approx(0.6, abs=0.06)

    def test_b_is_zero_when_the_spot_model_knows_nothing(self):
        m, s, y = _planted(b=0.0)
        _, b, _, _ = _fit_stack(m, s, np.zeros((len(y), 0)), y)
        assert b < 0.05

    def test_b_respects_its_ceiling(self):
        m, s, y = _planted(b=1.6)
        _, b, _, _ = _fit_stack(m, s, np.zeros((len(y), 0)), y, max_b=1.0)
        assert b == pytest.approx(1.0)

    def test_a_book_term_is_recovered_and_shrunk_by_l2(self):
        # A different seed from `_planted`'s, or z would replay m's draws.
        rng = np.random.default_rng(99)
        m, s, _ = _planted(seed=3)
        z = rng.normal(0, 1, (len(m), 1))
        y = (rng.random(len(m)) < expit(m + 0.3 * z[:, 0])).astype(float)
        _, _, _, loose = _fit_stack(m, s, z, y, l2=0.0)
        _, _, _, tight = _fit_stack(m, s, z, y, l2=1e6)
        assert loose[0] == pytest.approx(0.3, abs=0.05)
        assert abs(tight[0]) < abs(loose[0])


# ---- the cross-fit -------------------------------------------------------------

class Memorizer:
    """A stage 1 that memorises its training rows' outcomes and knows nothing
    about anything else: the purest possible overfit."""

    def __init__(self, rows: pd.DataFrame):
        self.seen = dict(zip(rows['key'], rows['outcome']))

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        known = table['key'].map(self.seen)
        base = expit(table['baseline_probability_logit'].to_numpy(dtype=float))
        return np.where(known.notna(), np.where(known == 1, 0.98, 0.02), base)


def _overfit_table(windows=3000, seed=7):
    rng = np.random.default_rng(seed)
    opens = pd.date_range('2026-01-01', periods=windows, freq='15min', tz='UTC')
    m = rng.normal(0, 1, windows)
    return pd.DataFrame({
        'key': np.arange(windows), 'window_open': opens,
        MARKET_LOGIT: m, 'baseline_probability_logit': m,
        'outcome': (rng.random(windows) < expit(m)).astype(float)})


class TestCrossFit:
    def test_a_planted_overfit_does_not_inflate_b(self):
        table = _overfit_table()
        y, m = table['outcome'].to_numpy(), table[MARKET_LOGIT].to_numpy()
        everything = np.ones(len(table), bool)
        none = np.zeros((len(table), 0))

        in_sample = Memorizer(table).predict(table)
        _, b_naive, _, _ = _fit_stack(m, logit(in_sample), none, y)
        oof, records = crossfit_spot(table, everything, Memorizer,
                                     n_blocks=4, embargo_minutes=1440)
        _, b_oof, _, _ = _fit_stack(m, logit(oof), none, y)

        assert b_naive > 0.9, 'the planted overfit should fool an in-sample fit'
        assert b_oof < 0.05, f'cross-fit b {b_oof:.3f}: stage 1 leaked into stage 2'
        assert len(records) == 4 and np.isfinite(oof).all()

    def test_each_block_is_predicted_by_a_fit_that_never_saw_it(self):
        table = _overfit_table()
        seen = []

        def fitter(rows):
            seen.append(pd.DatetimeIndex(rows['window_open']))
            return Memorizer(rows)

        stage2 = np.zeros(len(table), bool)
        stage2[1500:] = True
        _, records = crossfit_spot(table, stage2, fitter, n_blocks=3,
                                   embargo_minutes=1440)
        embargo = pd.Timedelta(minutes=1440)
        for record, fitted in zip(records, seen):
            lo, hi = pd.Timestamp(record['start']), pd.Timestamp(record['end'])
            inside = (fitted >= lo - embargo) & (fitted <= hi + embargo)
            assert not inside.any()
            # Rows before stage 2 begins are always available to every fit.
            assert (fitted < table['window_open'].iloc[1500]).any()


# ---- the packaged model -------------------------------------------------------

@pytest.fixture(scope='module')
def stacked():
    """One fold's stacked fit on 18 days of bars, quotes on the last 9."""
    bars = make_bars(days=18, lead=0.9, seed=11)
    depth = make_depth(bars, FAST, quoted_days=9)
    dataset = Dataset.build(bars, FAST, depth=depth)
    opens = dataset.window_index
    cut = opens[int(len(opens) * 0.8)]
    train = opens[opens < cut - pd.Timedelta(days=1)]
    test = opens[opens >= cut]
    groups = table_groups(FAST)
    fit, train_table = fit_fold(dataset, train, FAST, groups=groups)
    model = fit_for_config(train_table, fit.baseline, FAST, groups=groups,
                           scoring=fit.bundle(FAST))
    test_table = apply_fold(dataset, fit, test, FAST, groups=groups)
    return model, train_table, test_table


@pytest.mark.slow
class TestPackaged:
    def test_fit_for_config_returns_the_stack_in_stack_mode(self, stacked):
        model, _, _ = stacked
        assert isinstance(model, StackedForecastModel)
        assert isinstance(model.spot, ForecastModel)
        assert model.spot.init_score_source == 'baseline'
        assert model.init_score_source == 'market'
        assert model.stack_fitted and model.n_stack_rows >= 500
        assert 0.5 <= model.init_coefficient <= 1.5
        assert 0.0 <= model.residual_scale <= FAST.max_residual_scale
        assert len(model.crossfit) == FAST.stack_crossfit_blocks

    def test_unquoted_rows_predict_nan(self, stacked):
        model, _, test = stacked
        p = model.predict(test)
        quoted = np.isfinite(test[MARKET_LOGIT].to_numpy(dtype=float))
        assert np.isnan(p[~quoted]).all()
        assert np.isfinite(p[quoted]).all() and quoted.any()

    def test_score_carries_the_probability_and_stage_one(self, stacked):
        model, _, test = stacked
        scored = model.score(test)
        assert {'model_probability', 'spot_probability'} <= set(scored.columns)
        np.testing.assert_allclose(scored['model_probability'], model.predict(test))

    def test_an_untrained_stage_two_reproduces_the_market(self, stacked):
        model, _, test = stacked
        import copy
        null = copy.copy(model)
        null.init_coefficient, null.residual_scale, null.intercept = 1.0, 0.0, 0.0
        quoted = np.isfinite(test[MARKET_LOGIT].to_numpy(dtype=float))
        np.testing.assert_allclose(
            null.predict(test)[quoted],
            np.clip(expit(test[MARKET_LOGIT].to_numpy()[quoted]), 1e-6, 1 - 1e-6))

    def test_joblib_round_trip_scores_identically_and_verifies(self, stacked, tmp_path):
        model, _, test = stacked
        path = model.save(tmp_path / 'stack.joblib')
        loaded = ForecastModel.load(path)
        assert isinstance(loaded, StackedForecastModel)
        np.testing.assert_array_equal(loaded.predict(test), model.predict(test))
        loaded.verify(FAST)
        # Live adopts the artifact's init source, and a market config verifies.
        loaded.verify(Config(init_score_source='market'))
        with pytest.raises(ValueError):
            loaded.verify(Config())

    def test_provenance_records_the_stage_two_coefficients(self, stacked, tmp_path):
        import json
        model, _, _ = stacked
        path = model.save(tmp_path / 'stack.joblib')
        record = json.loads(path.with_suffix('.provenance.json').read_text())
        assert record['stack']['b'] == pytest.approx(model.residual_scale)
        assert record['stack']['a'] == pytest.approx(model.init_coefficient)
        assert record['stack']['n_rows'] == model.n_stack_rows
        assert record['init_score_source'] == 'market'
        assert record['spot']['init_score_source'] == 'baseline'
        assert record['config']['stack_mode'] == 'spot_then_market'

    def test_a_book_term_enters_stage_two_and_its_group_is_built(self):
        config = FAST.with_overrides(stack_book_features=('spread',))
        assert 'market_state' in stack_feature_groups(config)

    def test_a_too_short_quote_history_leaves_stage_two_at_the_market(self, stacked):
        model, train, _ = stacked
        thin = train.copy()
        quoted = np.flatnonzero(np.isfinite(thin[MARKET_LOGIT].to_numpy(dtype=float)))
        thin.loc[thin.index[quoted[100:]], MARKET_LOGIT] = np.nan
        fitted = fit_stacked_model(thin, model.baseline, FAST)
        assert not fitted.stack_fitted
        assert (fitted.init_coefficient, fitted.residual_scale) == (1.0, 0.0)


# ---- end to end ------------------------------------------------------------------

@pytest.mark.slow
def test_walk_forward_in_stack_mode():
    from core.backtest import walk_forward

    bars = make_bars(days=26, lead=0.9, seed=5)
    dataset = Dataset.build(bars, FAST, depth=make_depth(bars, FAST, quoted_days=15))
    result = walk_forward(dataset, FAST, trade=True)
    assert result.models and all(isinstance(m, StackedForecastModel)
                                 for m in result.models)
    scored = result.scored
    quoted = scored['market_probability'].notna()
    assert scored.loc[quoted, 'model_probability'].notna().all()
    assert scored.loc[~quoted, 'model_probability'].isna().all()
    # Every tested window carries a quote; training reached back before them.
    first_quote = dataset.windows.loc[
        dataset.windows['market_probability'].notna(), 'window_open'].min()
    assert result.models[0].n_train_windows > 0
    assert scored['window_open'].min() >= first_quote
    fold = result.report.folds[0]
    assert fold.residual_scale == pytest.approx(result.models[0].residual_scale)


@pytest.mark.slow
def test_scripts_evaluate_runs_end_to_end_in_stack_mode(tmp_path, monkeypatch, capsys):
    """`python -m scripts.evaluate --stack-mode spot_then_market --quote-sources
    live_touch` against a store holding bars and recorded quotes."""
    from core import promotion
    from core.datastore import ResearchStore
    import scripts._common as common
    import scripts.evaluate as evaluate

    store = ResearchStore(tmp_path / 'store')
    bars = make_bars(days=26, lead=0.9, seed=5)
    for symbol, frame in bars.items():
        store.write('minute_bars', frame.assign(
            venue='coinbase_spot', symbol=symbol, available_time=frame['event_time'],
            quality='valid'))
    depth = make_depth(bars, FAST, quoted_days=15)
    store.write('venue_depth', depth.assign(quality='valid'))
    # A backfill row on every quoted minute, which the filter must keep out.
    store.write('venue_depth', depth.assign(quality='valid', source='backfill',
                                            yes_bid=0.01, yes_ask=0.99))

    monkeypatch.setenv('RESEARCH_STORE', str(tmp_path / 'store'))
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.setattr(promotion, 'MODELS_ROOT', tmp_path / 'models')
    monkeypatch.setattr(common, 'DEFAULT_CONFIG',
                        Config(n_estimators=40, early_stopping_rounds=8,
                               n_folds=2, seasonality_min_days=5))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'evaluate', '--stack-mode', 'spot_then_market',
        '--quote-sources', 'live_touch', '--null-quote-age', 'stale',
        '--no-stress', '--out', str(out)])
    evaluate.main()
    printed = capsys.readouterr().out
    assert 'stage 2 per fold' in printed
    import json
    report = json.loads((out / 'report.json').read_text())
    assert report['stack'] and 0.0 <= report['stack'][0]['b'] <= 1.0
    assert report['config']['quote_sources'] == ['live_touch']
    scored = pd.read_parquet(out / 'scored.parquet')
    assert set(scored['quote_observer'].dropna()) == {'live_touch'}


def test_complete_cases_is_refused_in_stack_mode():
    from scripts._common import add_data_arguments, config_from_args

    parser = add_data_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--stack-mode', 'spot_then_market',
                              '--stack-book-features', 'spread,imbalance_5c',
                              '--stack-l2', '2.5', '--stack-crossfit-blocks', '3',
                              '--stack-intercept'])
    config = config_from_args(args)
    assert config.stack_mode == 'spot_then_market'
    assert config.stack_book_features == ('spread', 'imbalance_5c')
    assert config.stack_l2 == 2.5 and config.stack_crossfit_blocks == 3
    assert config.stack_intercept is True
    assert {'stack_mode', 'stack_book_features'} <= config.cli_overrides
    with pytest.raises(ValueError):
        Config(stack_mode='stacked')
    with pytest.raises(ValueError):
        Config(stack_crossfit_blocks=1)
