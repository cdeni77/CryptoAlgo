"""An artifact must not be scored under a configuration it was not fitted for.

`ForecastModel.load` was a bare `joblib.load`: nothing checked the feature list,
nothing checked the config, and nothing checked that the booster inside agreed
with the column names beside it. `scripts/live.py` never read
`config_provenance` at all, so a model promoted under one set of economics traded
under whatever the defaults happened to be — worth up to the whole `min_edge_pp`
gate in probability terms, silently.

The feature mismatch is the worst of the three. `_feature_matrix` selects by name
from `model.features`, so a booster trained on a different list is handed a
well-formed matrix of the wrong columns and returns numbers rather than raising.
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
    """Same names, same count, different order.

    `booster.predict` on a numpy matrix validates only the width, so a reordered
    list is accepted and silently scores the wrong column against each split.
    Measured elsewhere in this audit: swapping the two highest-gain entries moved
    probabilities by up to 0.0174 with no exception.
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
    """Old artifacts predate the field. Refusing them outright would be a
    different kind of surprise, so say so loudly and continue."""
    names = ['z_score']
    with caplog.at_level('WARNING'):
        model(booster_names=names, listed=names, provenance={}).verify(Config())
    assert any('provenance' in r.message for r in caplog.records)


def test_verify_without_a_config_still_checks_the_features():
    """The structural check does not need a config to be worth doing."""
    with pytest.raises(ValueError, match='booster was trained on'):
        model(booster_names=['a'], listed=['a', 'b']).verify(None)


def test_integrity_reports_the_booster_and_listed_features():
    """`integrity()` is introspection, not the gate — `verify()` is the gate.

    Its docstring used to describe the feature-mismatch failure mode as the
    reason it exists, which reads as "this is the check." It is not: `verify()`
    (above) already raises on exactly that mismatch and is what `load()` calls.
    `integrity()` has no caller and never raised anything — it returns a
    snapshot for a human or a script to read, and this pins that shape so it
    does not silently drift from what `verify()` actually checks.
    """
    names = ['z_score', 'log_rv_15', 'hour_sin']
    report = model(booster_names=names, listed=names).integrity()
    assert report['features'] == names
    assert report['booster_features'] == names
    assert report['n_features'] == 3


def test_integrity_surfaces_a_mismatch_verify_would_reject():
    """It reports the disagreement rather than raising — `verify()` raises."""
    report = model(booster_names=['a', 'b'], listed=['a', 'c']).integrity()
    assert report['features'] == ['a', 'c']
    assert report['booster_features'] == ['a', 'b']
