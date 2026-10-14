"""Tests that `_record_touch` archives the ladder the decision scored, not a post-order fetch."""

from __future__ import annotations

import inspect

import livetrade.quotes as live_quotes


def test_the_ladder_comes_from_the_cache_scoring_used():
    src = inspect.getsource(live_quotes._record_touch)
    assert 'ladder_from_cache(' in src, (
        'the archive must record the book the decision saw, not one fetched '
        'after the order'
    )


def test_the_cache_is_tried_before_the_rest_fetch():
    src = inspect.getsource(live_quotes._record_touch)
    assert src.index('ladder_from_cache(') < src.index("'/markets/{ticker}/orderbook'")


def test_the_rest_fallback_is_flagged_on_the_row():
    """A REST fallback is flagged in `transport`, so it is distinguishable from an observation."""
    src = inspect.getsource(live_quotes._record_touch)
    assert "ladder_source = 'stream'" in src
    assert "ladder_source = 'rest_after_order'" in src
    assert "'transport': ladder_source" in src


def test_the_schema_carries_the_column():
    from core.datastore import SCHEMAS

    assert 'transport' in SCHEMAS['venue_depth'], (
        'a key absent from the schema is dropped by _prepare without warning'
    )
