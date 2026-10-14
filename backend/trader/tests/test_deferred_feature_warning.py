"""Tests that `build_features` does not warn about columns the caller will attach later.

The live path fills the book groups after building features from bars. Only
columns nobody promised to attach are warned about.
"""
from __future__ import annotations

import logging

import pandas as pd

from core.features import reindex_to_features


def _table():
    return pd.DataFrame({'symbol': ['BTC-USD'], 'sigma_per_min': [0.0005]})


def test_a_deferred_column_is_created_but_not_warned_about(caplog):
    with caplog.at_level(logging.WARNING, logger='core.features'):
        out = reindex_to_features(_table(), ('spread', 'imbalance_touch'),
                                  deferred=('spread', 'imbalance_touch'))
    assert 'spread' in out.columns and out['spread'].isna().all(), (
        'the column must still exist — the matrix is built by name and a '
        'missing column is a different failure from an empty one')
    assert 'was not produced' not in caplog.text


def test_a_column_nobody_promised_to_attach_is_still_warned_about(caplog):
    """A column not listed as deferred is still warned about."""
    with caplog.at_level(logging.WARNING, logger='core.features'):
        reindex_to_features(_table(), ('spread', 'venue_prob_gap'),
                            deferred=('spread',))
    assert 'venue_prob_gap' in caplog.text
    assert 'spread was not produced' not in caplog.text


def test_a_column_already_present_is_left_alone(caplog):
    table = _table()
    table['spread'] = 2.0
    with caplog.at_level(logging.WARNING, logger='core.features'):
        out = reindex_to_features(table, ('spread',), deferred=())
    assert out['spread'].iloc[0] == 2.0
    assert caplog.text == ''
