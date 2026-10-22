"""The baseline must grade on the label the CONFIG asks for.

`outcome` is `settle_price >= strike`, so an arm given the venue's strike is
otherwise scored against a label derived from that same strike -- and a
strike comparison then measures the label, not the strike. The campaign
ledger records the identical trap for the Coinbase label against BRTI.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from scripts.baseline import _graded_label

PROXY = [1.0, 0.0, 1.0, 0.0]
VENUE = [0.0, 0.0, np.nan, 1.0]


def _frame(with_venue=True):
    d = {'outcome': PROXY}
    if with_venue:
        d['venue_outcome'] = VENUE
    return pd.DataFrame(d)


def test_the_default_grades_on_the_proxy():
    got = _graded_label(_frame(), DEFAULT_CONFIG)
    assert got.tolist() == PROXY


def test_venue_source_grades_on_the_venue_label():
    cfg = dataclasses.replace(DEFAULT_CONFIG, label_source='venue')
    got = _graded_label(_frame(), cfg)
    assert got[0] == 0.0 and got[1] == 0.0 and got[3] == 1.0


def test_it_falls_back_row_by_row_where_the_venue_has_no_settlement():
    """A partial span must still be scored, not dropped."""
    cfg = dataclasses.replace(DEFAULT_CONFIG, label_source='venue')
    got = _graded_label(_frame(), cfg)
    assert got[2] == PROXY[2]


def test_a_missing_column_falls_back_rather_than_crashing():
    cfg = dataclasses.replace(DEFAULT_CONFIG, label_source='venue')
    assert _graded_label(_frame(with_venue=False), cfg).tolist() == PROXY
