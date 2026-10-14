"""Tests for `init_score_source`: the correction sits on the baseline or market logit.

An untrained market-init model reproduces the price; every misuse is refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.model import (BASELINE_LOGIT, INIT_SCORE_COLUMNS, MARKET_LOGIT,
                        ForecastModel, attach_market_logit)


def table(n: int = 6, *, market=None) -> pd.DataFrame:
    frame = pd.DataFrame({
        'z_score': np.linspace(-1.0, 1.0, n),
        'baseline_probability': np.linspace(0.3, 0.7, n),
        BASELINE_LOGIT: np.linspace(-0.8, 0.8, n),
    })
    if market is not None:
        frame['market_probability'] = market
    return frame


class Booster:
    """A booster whose correction is a known constant, so the arithmetic can be checked."""

    def __init__(self, correction: float = 0.5):
        self.correction = correction

    def feature_name(self):
        return ['z_score']

    def feature_importance(self, importance_type='gain'):
        return np.array([1.0])

    def predict(self, matrix, raw_score=False):
        return np.full(len(matrix), self.correction, dtype=float)


class Baseline:
    def provenance(self):
        return {'distribution': 'student_t'}


def model(source: str = 'baseline', correction: float = 0.5) -> ForecastModel:
    return ForecastModel(booster=Booster(correction), features=['z_score'],
                         baseline=Baseline(), residual_scale=1.0,
                         init_score_source=source)


class TestTheConfigField:
    def test_the_default_is_the_baseline_and_nothing_else_changes(self):
        assert Config().init_score_source == 'baseline'

    def test_a_typo_is_refused_at_construction(self):
        """A typo is refused at Config construction, before any fitting."""
        with pytest.raises(ValueError, match="expected 'baseline' or 'market'"):
            Config(init_score_source='markets')

    def test_it_reaches_the_provenance_record(self):
        """The source is recorded in provenance, so ledger entries show which benchmark
        was used.
        """
        assert Config(init_score_source='market').provenance()[
            'init_score_source'] == 'market'


class TestPrediction:
    def test_the_baseline_source_reads_the_baseline_logit(self):
        out = model('baseline').predict(table())
        expected = 1.0 / (1.0 + np.exp(-(np.linspace(-0.8, 0.8, 6) + 0.5)))
        assert out == pytest.approx(expected, abs=1e-9)

    def test_the_market_source_reads_the_market_logit(self):
        frame = attach_market_logit(table(market=0.6))
        out = model('market').predict(frame)
        base = float(np.log(0.6 / 0.4))
        assert out == pytest.approx(1.0 / (1.0 + np.exp(-(base + 0.5))), abs=1e-6)

    def test_the_two_sources_disagree_on_the_same_row(self):
        """The two sources give different predictions on the same row."""
        frame = attach_market_logit(table(market=0.6))
        assert model('baseline').predict(frame)[0] != pytest.approx(
            model('market').predict(frame)[0])

    def test_a_market_model_refuses_a_table_with_no_market_column(self):
        """A market model refuses a table with no market column instead of using the
        baseline logit.
        """
        with pytest.raises(ValueError, match=MARKET_LOGIT):
            model('market').predict(table())

    def test_a_missing_quote_scores_NaN_and_not_the_baseline(self):
        """A missing quote scores NaN, so `decide()` abstains."""
        frame = attach_market_logit(table(market=[0.6, np.nan, 0.4,
                                                 np.nan, 0.55, 0.45]))
        out = model('market').predict(frame)
        assert np.isnan(out[[1, 3]]).all()
        assert np.isfinite(out[[0, 2, 4, 5]]).all()

    def test_clip_prob_propagates_nan_which_this_relies_on(self):
        """`clip_prob` and `expit` propagate NaN, which the missing-quote behaviour
        relies on.

        A `clip_prob` that filled NaN (with 0.5, say) would turn "no quote" into a
        confident coin flip, and the other tests would still pass.
        """
        from core.baseline import clip_prob, expit

        assert np.isnan(expit(np.array([np.nan]))[0])
        assert np.isnan(clip_prob(np.array([np.nan]))[0])

    def test_an_untrained_market_model_reproduces_the_price_exactly(self):
        prices = np.array([0.05, 0.2, 0.5, 0.83, 0.95])
        frame = attach_market_logit(table(len(prices), market=prices))
        assert model('market', correction=0.0).predict(frame) == pytest.approx(
            prices, abs=1e-6)


class TestTheArtifactCannotBeUsedAgainstTheWrongBenchmark:
    def test_a_mismatch_between_artifact_and_config_refuses_to_load(self):
        with pytest.raises(ValueError, match='answers a different question'):
            model('market').verify(Config(init_score_source='baseline'))

    def test_the_reverse_mismatch_also_refuses(self):
        with pytest.raises(ValueError, match='answers a different question'):
            model('baseline').verify(Config(init_score_source='market'))

    def test_a_matching_pair_verifies(self):
        model('market').verify(Config(init_score_source='market'))
        model('baseline').verify(Config(init_score_source='baseline'))

    def test_an_unscoreable_source_on_the_artifact_refuses(self):
        with pytest.raises(ValueError, match='nothing can score'):
            model('nonsense').verify(None)

    def test_the_source_is_in_the_artifacts_provenance(self):
        prov = model('market').provenance()
        assert prov['init_score_source'] == 'market'


class TestAttachMarketLogit:
    def test_a_missing_column_names_the_backtest_as_the_reason(self):
        """The error names the backtest as the reason, so nobody substitutes the
        baseline column.
        """
        with pytest.raises(ValueError, match='backtest has no book'):
            attach_market_logit(table())

    def test_extreme_prices_stay_finite(self):
        """Prices of 0 and 1 still give a finite logit."""
        out = attach_market_logit(table(3, market=[0.0, 0.5, 1.0]))
        assert np.isfinite(out[MARKET_LOGIT].to_numpy()).all()

    def test_the_columns_map_is_the_single_source_of_truth(self):
        assert INIT_SCORE_COLUMNS == {'baseline': BASELINE_LOGIT,
                                      'market': MARKET_LOGIT}
