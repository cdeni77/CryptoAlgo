"""Which observer wins when several recorded the same minute.

**`quote_age_seconds` is not comparable across sources.** For `backfill` it is
time since the last book CHANGE — Predexon serves changes, so a quiet book reads
~0.5s — while for the live recorders it is the sampler's own lag, and
`record_ladder` samples at +25s past the minute. `live_touch` carries no age at
all. Measured at offset 12 on the real store: backfill 0.52s median, live and
live_ws 32.7s, live_touch NaN.

Ranked on that number, backfill wins essentially every contested row, and it
wins for the wrong reason: not because it is fresher, but because its clock
starts later. `live_touch` — the quote `scripts/live.py` actually decided on —
sorted LAST of the four, because nulls sort last.

That matters more here than anywhere else the sources are mixed. The model is
fitted to correct `market_probability` (`init_score_source=market`) and is then
traded against the book the live loop reads at the decision instant. Fitting to
a reconstruction and grading against an observation makes those two different
objects, and the two disagree by ~10c at +12m.

Same class of defect as the `levels_bid`/`levels_ask` cross-source ratio of
0.579 already recorded in CLAUDE.md, except used as a SELECTOR rather than a
feature — so it silently chose a provenance while appearing to choose a
freshness, and nothing raised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.quotes import QUOTE_SOURCE_PRIORITY, attach_quotes

OPEN = pd.Timestamp('2026-07-01T12:00Z')


def _windows(offset=12):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': OPEN,
                          'offset': offset, 'baseline_probability': 0.40}])


def _row(source, bid, ask, age, offset=12):
    return {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
            'offset_minutes': offset, 'yes_bid': bid, 'yes_ask': ask,
            'quote_age_seconds': age, 'source': source}


def test_the_decision_instant_quote_beats_the_reconstruction():
    """The whole point. `live_touch` has no age and backfill reads 0.5s, so on
    the old key the reconstruction won every contested row."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live_touch', 0.60, 0.62, np.nan),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61
    assert out['ask_up'].iloc[0] == 0.62


def test_the_recorders_beat_the_reconstruction_despite_a_worse_age():
    """`record_ladder` samples at +25s, so a live row's age is ~33s against
    backfill's 0.5s. It is still the better evidence of what was quotable."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live', 0.60, 0.62, 29.0),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61


def test_the_priority_order_is_honoured_between_the_two_live_recorders():
    depth = pd.DataFrame([
        _row('live', 0.30, 0.32, 1.0),
        _row('live_ws', 0.60, 0.62, 29.0),
    ])
    out = attach_quotes(_windows(), depth)
    assert QUOTE_SOURCE_PRIORITY.index('live_ws') < \
        QUOTE_SOURCE_PRIORITY.index('live')
    assert out['market_probability'].iloc[0] == 0.61


def test_an_unusable_preferred_row_does_not_displace_a_usable_one():
    """The hazard the fix introduces, and the reason `_unusable` leads the sort.

    The old key got this for free: a row failing `sane`/`fresh` had its age
    NaN-ed and sorted last. Rank by source first and a crossed `live_touch`
    would win the row and price it NaN — turning a preference into data loss.
    """
    depth = pd.DataFrame([
        _row('live_touch', 0.80, 0.20, np.nan),   # crossed: ask below bid
        _row('backfill', 0.30, 0.32, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_a_stale_preferred_row_does_not_displace_a_fresh_one():
    depth = pd.DataFrame([
        _row('live_ws', 0.80, 0.82, 4000.0),      # far past max_age_seconds
        _row('backfill', 0.30, 0.32, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_age_still_breaks_ties_within_one_source():
    """Age is not meaningless — it is meaningless ACROSS sources. Within one
    producer it means one thing and remains the right tie-break."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 20.0),
        _row('backfill', 0.60, 0.62, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61


def test_an_unknown_source_ranks_behind_every_known_one():
    depth = pd.DataFrame([
        _row('some_future_recorder', 0.30, 0.32, 0.0),
        _row('backfill', 0.60, 0.62, 900.0 - 1),
    ])
    out = attach_quotes(_windows(), depth, max_age_seconds=1000.0)
    assert out['market_probability'].iloc[0] == 0.61


def test_a_depth_frame_with_no_source_column_still_prices():
    """`source` is not in every caller's frame, and a KeyError here would take
    out the backtest rather than degrade it."""
    depth = pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)]).drop(
        columns=['source'])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_the_winning_observer_is_recorded_on_the_row():
    """`quote_source` has always held the VENUE despite its name, so no frame
    downstream could tell a reconstruction from an observation. That is why the
    ranking defect above survived for months: the mixture was invisible in
    every table that carried it."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live_touch', 0.60, 0.62, np.nan),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['quote_source'].iloc[0] == 'kalshi'      # the venue, unchanged
    assert out['quote_observer'].iloc[0] == 'live_touch'  # the observer


def test_an_unpriced_window_names_no_observer():
    out = attach_quotes(_windows(offset=3),
                        pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)]))
    assert pd.isna(out['market_probability'].iloc[0])
    assert out['quote_observer'].iloc[0] is None or \
        pd.isna(out['quote_observer'].iloc[0])
