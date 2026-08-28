"""`build_features` warned about columns the caller was about to attach.

The live path builds features from bars first and fills the book groups
afterwards from the in-process recorder caches — it cannot do otherwise, since
`iv_minus_realised` needs the FITTED `sigma_per_min` and the book is read at the
decision instant. So every cycle logged nine warnings about features that were
populated a moment later, and a real gap looked exactly like the noise.
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
    """Silencing everything would hide the real gap the warning exists for."""
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
