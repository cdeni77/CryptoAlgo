"""Inside one second, prefer the quote that carries depth.

`live_touch` writes two rows per decision -- one with sizes, one without --
and the unsized one is fresher by a median 140ms. Ranking on raw age gave it
the win in 8,013 of 8,013 groups, so only 53.2% of live-era decision rows
kept `yes_bid_size`, `depth_*` and `bid_at_touch`. Half the live era lost its
depth for a seventh of a second.
"""

from __future__ import annotations

import pandas as pd

from core.quotes import attach_quotes

WINDOW = pd.Timestamp('2026-10-20 12:00', tz='UTC')


def _windows():
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'window_open': [WINDOW],
        'offset': [3],
        'decision_time': [WINDOW + pd.Timedelta(minutes=3)],
    })


def _book(rows):
    out = []
    for age, bid_size in rows:
        out.append({
            'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': WINDOW,
            'offset_minutes': 3, 'source': 'live_touch',
            'event_time': WINDOW + pd.Timedelta(minutes=3),
            'available_time': WINDOW + pd.Timedelta(minutes=3),
            'yes_bid': 0.60, 'yes_ask': 0.61,
            'yes_bid_size': bid_size, 'yes_ask_size': bid_size,
            'quote_age_seconds': age,
        })
    return pd.DataFrame(out)


def _attached(book):
    return attach_quotes(_windows(), book, max_age_seconds=5.0,
                         decision_latency_seconds=3.0, null_age='stale')


def test_the_sized_row_wins_when_the_age_difference_is_sub_second():
    """The real pattern: unsized 0.285s against sized 0.423s."""
    got = _attached(_book([(0.285, None), (0.423, 12530.0)]))
    assert got['bid_at_touch'].notna().all(), (
        'the unsized row won on 140ms and took the depth with it')
    assert float(got['bid_at_touch'].iloc[0]) == 12530.0


def test_a_materially_fresher_row_still_wins_even_without_sizes():
    """Across whole seconds the staleness is real, not noise."""
    got = _attached(_book([(0.2, None), (4.4, 999.0)]))
    assert got['bid_at_touch'].isna().all() or \
        float(got['bid_at_touch'].iloc[0]) != 999.0


def test_one_row_with_sizes_is_unchanged():
    got = _attached(_book([(0.3, 500.0)]))
    assert float(got['bid_at_touch'].iloc[0]) == 500.0
