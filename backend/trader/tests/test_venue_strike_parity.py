"""Tests that the venue strike never enters the feature matrix.

It would leave the derived features in a state never seen in training; it is used for
settlement only.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from scripts import live as live_mod


def test_live_does_not_assign_strike_or_displacement_into_the_scored_frame():
    source = inspect.getsource(live_mod)
    for column in ("'strike'", "'displacement'"):
        assert f"scored.loc[index, {column}]" not in source, (
            f'live still writes {column} into the feature matrix after '
            f'build_features has produced it')


def test_the_venue_strike_is_still_observed_just_not_fed_to_the_model():
    """The venue strike is still recorded (`venue_strike`, `strike_source`), just not
    fed to the model.
    """
    source = inspect.getsource(live_mod)
    assert 'venue_strike' in source
    assert 'strike_source' in source
