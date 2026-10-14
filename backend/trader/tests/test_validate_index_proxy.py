"""The index-proxy validation scores both labels on the same Kalshi windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.index_features import minute_summary
from research.validate._validate_index_proxy import (
    LABELS, agreement_table, coinbase_ends, compare, proxy_ends, render,
)

T0 = pd.Timestamp('2026-10-14 12:00', tz='UTC')
WINDOWS = 8


def _scenario(seed=0):
    """Each window ends a known distance from its strike; Coinbase sees the same path
    plus a basis that flips near-ties.
    """
    rng = np.random.default_rng(seed)
    seconds, bars, settled = [], [], []
    # Moves (bp) chosen to land one window in each bucket, twice over.
    moves = [0.5, 1.5, 3.0, 7.0, 20.0, -0.5, -1.5, -20.0]
    level = 100.0
    for w in range(WINDOWS):
        open_ = T0 + pd.Timedelta(minutes=15 * w)
        strike = level
        settle = strike * (1 + moves[w] / 1e4)
        # A window's strike minute is the previous window's settlement minute, so only
        # the first window writes one.
        ends = [(open_ + pd.Timedelta(minutes=14), settle)]
        if w == 0:
            ends.insert(0, (open_ - pd.Timedelta(minutes=1), strike))
        # Proxy: flat at the strike in the strike minute, flat at the settle in the last
        # minute.
        for minute, value in ends:
            times = pd.date_range(minute, minute + pd.Timedelta(minutes=1),
                                  freq='1s', inclusive='left')
            seconds.append(pd.DataFrame({'symbol': 'BTC-USD', 'event_time': times,
                                         'proxy': value, 'n_exchanges': 3}))
        # Coinbase: a +1bp basis in window 5's settlement minute, which flips that
        # sub-1bp
        # DOWN window (and only moves window 6's strike, whose -1.5bp move survives).
        basis = 1.0001 if w == 5 else 1.0
        for minute, value in [(m, v * (basis if m > open_ else 1.0))
                              for m, v in ends]:
            bars.append({'event_time': minute, 'open': value, 'high': value,
                         'low': value, 'close': value})
        settled.append({'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': open_,
                        'settled_up': float(settle >= strike)})
        level = settle
    # A Polymarket row that must be ignored.
    settled.append({'venue': 'polymarket', 'symbol': 'BTC-USD', 'window_open': T0,
                    'settled_up': 0.0})
    return (pd.concat(seconds, ignore_index=True), pd.DataFrame(bars),
            pd.DataFrame(settled))


def test_both_labels_on_identical_windows_and_the_proxy_wins_the_near_tie():
    seconds, bars, settled = _scenario()
    joined = compare(settled, coinbase_ends({'BTC-USD': bars}),
                     proxy_ends(minute_summary(seconds)))
    assert len(joined) == WINDOWS
    assert joined['px_up'].eq(joined['kalshi_up']).all()
    assert (joined['cb_up'] != joined['kalshi_up']).sum() == 1
    table = agreement_table(joined, 'px_move_bps').set_index('move')
    assert list(table.index) == list(LABELS) + ['all']
    assert table.loc['<1bp', 'windows'] == 2
    assert table.loc['<1bp', 'proxy_agree'] == 1.0
    assert table.loc['<1bp', 'coinbase_agree'] == 0.5
    assert table.loc['all', 'proxy_minus_coinbase'] == pytest.approx(1 / WINDOWS)
    assert table.loc['>10bp', 'windows'] == 2
    text = render(table.reset_index(), 'title')
    assert '<1bp' in text and '+50.00pp' in text


def test_a_window_without_a_proxy_is_dropped_from_both_columns():
    seconds, bars, settled = _scenario()
    # Remove the proxy's settlement minute for the first window.
    cut = T0 + pd.Timedelta(minutes=14)
    seconds = seconds[(seconds['event_time'] < cut)
                      | (seconds['event_time'] >= cut + pd.Timedelta(minutes=1))]
    joined = compare(settled, coinbase_ends({'BTC-USD': bars}),
                     proxy_ends(minute_summary(seconds)))
    # That minute settles window 0 and is the strike of window 1.
    assert len(joined) == WINDOWS - 2
    assert not {T0, T0 + pd.Timedelta(minutes=15)} & set(joined['window_open'])


def test_a_thin_proxy_minute_does_not_make_a_label():
    seconds, bars, settled = _scenario()
    thin = seconds.groupby(seconds['event_time'].dt.floor('min')).head(10)
    joined = compare(settled, coinbase_ends({'BTC-USD': bars}),
                     proxy_ends(minute_summary(thin)))
    assert joined.empty
    assert agreement_table(joined).loc[5, 'windows'] == 0
