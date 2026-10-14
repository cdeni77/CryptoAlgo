"""Tests for `ForecastModel.verify`: an artifact is not scored under a config it was not
fitted for.

A booster trained on a different feature list still returns numbers on a
well-formed matrix of the wrong columns, so the feature check must raise.
"""

from __future__ import annotations

import pytest

from core.config import Config
from core.model import ForecastModel


class FakeBooster:
    def __init__(self, names):
        self._names = list(names)

    def feature_name(self):
        return list(self._names)

    def num_feature(self):
        return len(self._names)


def model(*, booster_names, listed, provenance=None) -> ForecastModel:
    return ForecastModel(
        booster=FakeBooster(booster_names),
        features=list(listed),
        baseline=None,
        config_provenance=provenance if provenance is not None else Config().provenance(),
    )


def test_a_matching_artifact_verifies():
    names = ['z_score', 'log_rv_15', 'hour_sin']
    model(booster_names=names, listed=names).verify(Config())


def test_a_feature_list_the_booster_disagrees_with_is_refused():
    with pytest.raises(ValueError, match='booster was trained on'):
        model(booster_names=['z_score', 'log_rv_15'],
              listed=['z_score', 'log_rv_15', 'hour_sin']).verify(Config())


def test_a_reordered_feature_list_is_refused():
    """A reordered list is refused: `booster.predict` checks only width, so it would score
    the wrong columns.
    """
    names = ['z_score', 'log_rv_15', 'hour_sin']
    with pytest.raises(ValueError, match='booster was trained on'):
        model(booster_names=names, listed=[names[1], names[0], names[2]]).verify(Config())


@pytest.mark.parametrize('field,value', [
    ('window_minutes', 30),
    ('decision_offsets', [1, 2]),
    ('vol_lookbacks_minutes', [15, 60]),
    ('embargo_minutes', 2880),
    ('fee_rate', 0.05),
    ('half_spread_cents', 2.0),
])
def test_a_config_that_changes_an_answer_is_refused(field, value):
    names = ['z_score']
    fitted = Config().provenance()
    fitted[field] = value
    with pytest.raises(ValueError, match='different configuration'):
        model(booster_names=names, listed=names, provenance=fitted).verify(Config())


def test_an_artifact_without_provenance_is_allowed_with_a_warning(caplog):
    """An artifact with no provenance loads with a warning instead of a refusal."""
    names = ['z_score']
    with caplog.at_level('WARNING'):
        model(booster_names=names, listed=names, provenance={}).verify(Config())
    assert any('provenance' in r.message for r in caplog.records)


def test_verify_without_a_config_still_checks_the_features():
    with pytest.raises(ValueError, match='booster was trained on'):
        model(booster_names=['a'], listed=['a', 'b']).verify(None)


def test_integrity_reports_the_booster_and_listed_features():
    """`integrity()` is a read-only snapshot; `verify()` is the gate. This pins the
    snapshot's shape.
    """
    names = ['z_score', 'log_rv_15', 'hour_sin']
    report = model(booster_names=names, listed=names).integrity()
    assert report['features'] == names
    assert report['booster_features'] == names
    assert report['n_features'] == 3


def test_integrity_surfaces_a_mismatch_verify_would_reject():
    """It reports the disagreement instead of raising."""
    report = model(booster_names=['a', 'b'], listed=['a', 'c']).integrity()
    assert report['features'] == ['a', 'c']
    assert report['booster_features'] == ['a', 'b']
