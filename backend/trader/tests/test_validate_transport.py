"""The transport comparison must not invent disagreement.

`nan == nan` is False, so comparing top-of-book directly scored every
empty-vs-empty minute as a disagreement. It reported 33% agreement on the NO
side of a live sample where the two transports had in fact never differed — the
worst direction for this particular check to be wrong in, because it is the
evidence the migration is gated on.
"""
from __future__ import annotations

import pandas as pd

from core.datastore import ResearchStore
from research.validate._validate_transport import compare


def _row(transport, **over):
    base = dict(venue='kalshi', symbol='BTC-USD',
                event_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                available_time=pd.Timestamp('2026-08-26 12:00:05', tz='UTC'),
                quality='valid', market_ticker='K',
                window_open=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                minute_into_window=0.0,
                yes_levels='[[0.3, 10.0]]', no_levels='[[0.65, 4.0]]',
                yes_total=10.0, no_total=4.0,
                transport=transport, book_age_ms=0.0 if transport == 'rest' else 90.0)
    base.update(over)
    return base


def _store(tmp_path, rows):
    store = ResearchStore(tmp_path)
    store.write('venue_ladder', pd.DataFrame(rows))
    return store


def test_two_empty_sides_agree_rather_than_disagree(tmp_path):
    store = _store(tmp_path, [_row('rest', no_levels='[]', no_total=0.0),
                              _row('ws', no_levels='[]', no_total=0.0)])
    both = compare(store)
    assert len(both) == 1
    assert bool(both['top_no_same'].iloc[0]), 'both empty is agreement'
    assert bool(both['no_both_empty'].iloc[0])


def test_a_genuine_top_of_book_difference_is_still_reported(tmp_path):
    store = _store(tmp_path, [_row('rest'),
                              _row('ws', yes_levels='[[0.31, 10.0]]')])
    both = compare(store)
    assert not bool(both['top_yes_same'].iloc[0])


def test_identical_ladders_agree_with_zero_drift(tmp_path):
    both = compare(_store(tmp_path, [_row('rest'), _row('ws')]))
    assert bool(both['top_yes_same'].iloc[0]) and both['drift_yes'].iloc[0] == 0
    assert both['size_ratio'].iloc[0] == 1.0


def test_only_one_transport_present_yields_nothing_to_compare(tmp_path):
    assert compare(_store(tmp_path, [_row('rest')])).empty
