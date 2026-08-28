"""Testing only on rows where every field the model uses actually exists.

Every run so far trained on a mixture: 15 of 57 features NaN for most rows,
because the book starts 2026-01-08 against five years of bars. LightGBM does
not skip those — it learns a DEFAULT DIRECTION for each missing feature, so the
fit is really two models blended, and the evaluation measures the blend. That
is not a test of whether the book helps; it is a test of a chimera.

Complete cases are the honest version: keep a row only if the venue's book, the
other venue's book, a ladder fit and the venue's own settlement are all present
for it. Fewer rows, but each one is a row the live path could actually have
traded with everything it claims to use.
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
    """Without it the row would be graded on our Coinbase label, which is the
    leak this whole exercise just spent a day removing."""
    got = complete_cases(_panel(venue_outcome=[1.0, np.nan, 1.0, np.nan]))
    assert len(got) == 2


def test_everything_present_keeps_every_row():
    assert len(complete_cases(_panel())) == 4


def test_the_requirement_can_be_narrowed_to_named_groups():
    """market_state alone is a weaker requirement than all three, and asking
    for it should not discard rows for want of a ladder fit."""
    got = complete_cases(_panel(implied_sigma_per_min=[np.nan] * 4),
                         groups=('market_state',))
    assert len(got) == 4


def test_an_empty_panel_returns_empty_rather_than_raising():
    assert complete_cases(_panel().iloc[0:0]).empty
