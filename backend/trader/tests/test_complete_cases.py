"""Tests for `complete_cases`: keep only rows where every input the model uses exists.

LightGBM learns a default for missing values, so mixed rows would fit a blend of two
models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.dataset import complete_cases


def _panel(**over):
    base = {
        'symbol': ['BTC-USD'] * 4,
        'window_open': pd.to_datetime(['2026-07-01T12:00Z'] * 4),
        'offset': [3, 6, 9, 12],
        'ask_up': [0.46, 0.46, 0.46, 0.46],
        'market_probability': [0.45, 0.45, 0.45, 0.45],
        'bid_at_touch': [10.0, 10.0, 10.0, 10.0],
        'pm_market_probability': [0.40, 0.40, 0.40, 0.40],
        'implied_sigma_per_min': [0.0006] * 4,
        'venue_outcome': [1.0, 1.0, 1.0, 1.0],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_a_row_missing_the_venue_book_is_dropped():
    got = complete_cases(_panel(ask_up=[0.46, np.nan, 0.46, 0.46]))
    assert len(got) == 3


def test_a_row_missing_the_other_venue_is_dropped():
    got = complete_cases(_panel(pm_market_probability=[0.4, 0.4, np.nan, 0.4]))
    assert len(got) == 3


def test_a_row_missing_a_ladder_fit_is_dropped():
    got = complete_cases(_panel(implied_sigma_per_min=[np.nan] * 4))
    assert got.empty


def test_a_row_missing_the_venues_settlement_is_dropped():
    """A row without the venue's settlement is dropped, so it is not graded on our
    Coinbase label.
    """
    got = complete_cases(_panel(venue_outcome=[1.0, np.nan, 1.0, np.nan]))
    assert len(got) == 2


def test_everything_present_keeps_every_row():
    assert len(complete_cases(_panel())) == 4


def test_the_requirement_can_be_narrowed_to_named_groups():
    """Requiring only `market_state` does not drop rows for a missing ladder fit."""
    got = complete_cases(_panel(implied_sigma_per_min=[np.nan] * 4),
                         groups=('market_state',))
    assert len(got) == 4


def test_an_empty_panel_returns_empty_rather_than_raising():
    assert complete_cases(_panel().iloc[0:0]).empty


# --- the CLI must pass the groups through ----------------------------------
#
# These test the wiring in `scripts._common`, not just the function.

def test_the_cli_forwards_its_groups_to_the_filter():
    """`--groups` reaches `complete_cases`."""
    import argparse
    from scripts._common import apply_complete_cases

    class Ds:
        def __init__(self, windows):
            self.windows = windows

    # A panel whose only gap is a missing ladder fit.
    ds = Ds(_panel(implied_sigma_per_min=[np.nan] * 4))
    args = argparse.Namespace(
        complete_cases=True,
        groups='cross_venue,geometry,vol_state,cross_asset,market_state')
    apply_complete_cases(args, ds)
    assert len(ds.windows) == 4, (
        'a config without implied_vol must not be denied rows for want of a '
        'ladder fit')


def test_a_config_that_uses_implied_vol_still_requires_it():
    """A config that uses `implied_vol` still requires a ladder fit."""
    import argparse
    import pytest as _pytest
    from core.dataset import DatasetError
    from scripts._common import apply_complete_cases

    class Ds:
        def __init__(self, windows):
            self.windows = windows

    ds = Ds(_panel(implied_sigma_per_min=[np.nan] * 4))
    args = argparse.Namespace(complete_cases=True,
                              groups='cross_venue,implied_vol,market_state')
    with _pytest.raises(DatasetError):
        apply_complete_cases(args, ds)


def test_no_groups_still_requires_everything():
    """`--complete-cases` with no `--groups` requires every input."""
    import argparse
    from scripts._common import apply_complete_cases

    class Ds:
        def __init__(self, windows):
            self.windows = windows

    ds = Ds(_panel(implied_sigma_per_min=[0.001, np.nan, 0.001, 0.001]))
    args = argparse.Namespace(complete_cases=True, groups=None)
    apply_complete_cases(args, ds)
    assert len(ds.windows) == 3
