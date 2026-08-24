"""The correction can sit on the baseline's logit or on the market's.

Measured over 1,109 live-recorded rows and 285 symbol-windows: log loss 0.331 for
the market's de-spread mid, 0.428 for `F(x/sigma)`, 0.430 for the model. Same sign
on all three symbols and all four offsets, and on the 108 rows actually traded the
model came in *worse* than its own baseline. So a baseline-initialised model spends
its capacity correcting a forecaster already 0.10 nats behind the price it has to
trade against, and `log_loss_skill` beating `F(x/sigma)` stops implying anything
about whether the trade pays.

Initialising on the market makes the fitted residual `logit(truth) - logit(price)`
— how the price is wrong, which is the quantity the money depends on — and inverts
the null in the right direction. An untrained baseline-init model reproduces
`F(x/sigma)`, which disagrees with the price by 5.79pp on average and trades on
it. An untrained market-init model reproduces the price, so the edge is identically
zero and nothing trades.

**It is not trainable yet and these tests do not pretend otherwise.** 285
symbol-windows of quotes exist against a `windows_evaluated >= 20,000` gate. What
is tested here is that the mechanism is correct and that every way of getting it
wrong is a refusal rather than a plausible-looking number.
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
    """A booster whose correction is a known constant, so the arithmetic is
    checkable rather than merely plausible."""

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
        """Not deep inside the fit, after the fold's baseline, volatility models
        and seasonality have already been fitted."""
        with pytest.raises(ValueError, match="expected 'baseline' or 'market'"):
            Config(init_score_source='markets')

    def test_it_reaches_the_provenance_record(self):
        """Two attempts with the same `log_loss_skill` over two different
        benchmarks must be distinguishable in the ledger."""
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
        """If they agreed, none of this would matter — and a mix-up would be
        undetectable."""
        frame = attach_market_logit(table(market=0.6))
        assert model('baseline').predict(frame)[0] != pytest.approx(
            model('market').predict(frame)[0])

    def test_a_market_model_refuses_a_table_with_no_market_column(self):
        """The backtest case. It must not fall through to the baseline logit,
        which is sitting right there on the same table."""
        with pytest.raises(ValueError, match=MARKET_LOGIT):
            model('market').predict(table())

    def test_a_missing_quote_scores_NaN_and_not_the_baseline(self):
        """A one-sided book gives `Quote.mid = None`. The row must come back NaN
        so `decide()` abstains, rather than a well-formed probability computed
        from the wrong forecaster under the market artifact's provenance."""
        frame = attach_market_logit(table(market=[0.6, np.nan, 0.4,
                                                 np.nan, 0.55, 0.45]))
        out = model('market').predict(frame)
        assert np.isnan(out[[1, 3]]).all()
        assert np.isfinite(out[[0, 2, 4, 5]]).all()

    def test_clip_prob_propagates_nan_which_this_relies_on(self):
        """The actual load-bearing fact behind "a missing quote scores NaN".

        `ForecastModel.predict` also masks explicitly, but that mask is redundant
        while NaN propagates, so no mutation of it can fail a test. This pins the
        dependency instead: a `clip_prob` that filled NaN — with 0.5, say — would
        turn "there was no quote" into "a coin flip, confidently asserted" on a
        real-money path, and every other test here would still pass.
        """
        from core.baseline import clip_prob, expit

        assert np.isnan(expit(np.array([np.nan]))[0])
        assert np.isnan(clip_prob(np.array([np.nan]))[0])

    def test_an_untrained_market_model_reproduces_the_price_exactly(self):
        """The null that makes this worth doing: zero correction means zero edge
        means no trade. The baseline-init null instead disagrees with the price by
        5.79pp on average and trades on it."""
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
        """The error has to say why, or someone attaches the baseline column to
        make it go away and the measurement quietly becomes the old one."""
        with pytest.raises(ValueError, match='backtest has no book'):
            attach_market_logit(table())

    def test_extreme_prices_stay_finite(self):
        """1c and 99c are real quotes on a tapered deci-cent ladder, and
        `logit(0)` is not a number."""
        out = attach_market_logit(table(3, market=[0.0, 0.5, 1.0]))
        assert np.isfinite(out[MARKET_LOGIT].to_numpy()).all()

    def test_the_columns_map_is_the_single_source_of_truth(self):
        assert INIT_SCORE_COLUMNS == {'baseline': BASELINE_LOGIT,
                                      'market': MARKET_LOGIT}
