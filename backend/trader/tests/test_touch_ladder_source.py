"""The archived ladder must be the one the decision scored.

`_record_touch` runs AFTER `act_on`, deliberately: it costs REST calls and a
synchronous Parquet write, and every millisecond of that was staleness the
order paid. The call site defends the placement with "moving it changes only
WHEN the archive is written, never what it contains: `quotes` is the same
object the decision priced against."

That holds for the four touch fields. It did **not** hold for the six
ladder-derived depth columns, which were re-fetched over REST at that point —
after our own fill had removed resting size from the side we bought. Those feed
`imbalance_5c`, `depth_ratio` and `book_convexity`: three of the five
`market_state` features, systematically understated on the traded side, in rows
`QUOTE_SOURCE_PRIORITY` ranks FIRST.

Reading the stream cache instead costs nothing — it is the same in-process
book scoring already used — and removes three REST round trips from the
post-order path. The REST fetch survives only as a fallback, flagged on the row,
because a reconstruction and an observation are different claims.
"""

from __future__ import annotations

import inspect

import scripts.live as live


def test_the_ladder_comes_from_the_cache_scoring_used():
    src = inspect.getsource(live._record_touch)
    assert 'ladder_from_cache(' in src, (
        'the archive must record the book the decision saw, not one fetched '
        'after the order'
    )


def test_the_cache_is_tried_before_the_rest_fetch():
    src = inspect.getsource(live._record_touch)
    assert src.index('ladder_from_cache(') < src.index("'/markets/{ticker}/orderbook'")


def test_the_rest_fallback_is_flagged_on_the_row():
    """A post-order reconstruction must be distinguishable from an
    observation — the column that could not tell them apart is how this went
    unnoticed."""
    src = inspect.getsource(live._record_touch)
    assert "ladder_source = 'stream'" in src
    assert "ladder_source = 'rest_after_order'" in src
    assert "'transport': ladder_source" in src


def test_the_schema_carries_the_column():
    from core.datastore import SCHEMAS

    assert 'transport' in SCHEMAS['venue_depth'], (
        'a key absent from the schema is dropped by _prepare without warning'
    )
