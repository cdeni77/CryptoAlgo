"""The lead-lag study finds a planted lag, and finds none where none is planted.

Synthetic world: a one-second spot random walk, and a Kalshi book whose mid is
the barrier probability of spot as it stood `lag` seconds earlier — so Kalshi
is a pure delayed copy. The book reaches the study as raw `venue_book_events`
rows (a snapshot, then signed deltas), replayed through `BookCache` exactly as
the archive would be.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from research.analysis import lead_lag as ll

SIGMA_1S = 2.5e-5
OPEN0 = pd.Timestamp('2026-10-01T00:00Z')


def _world(lag: int, markets: int = 24, seed: int = 3):
    rng = np.random.default_rng(seed)
    lead_in = ll.SIGMA_LOOKBACK_SECONDS + 120
    total = lead_in + markets * ll.WINDOW_SECONDS
    times = pd.date_range(OPEN0 - pd.Timedelta(seconds=lead_in), periods=total, freq='s')
    price = 60_000 * np.exp(np.cumsum(rng.normal(0, SIGMA_1S, total)))
    spot = pd.Series(price, index=times)
    remaining = ll.remaining_variance_seconds()

    trades, events, depth, pm = [], [], [], []
    seq = 0
    for k in range(markets):
        open_ = OPEN0 + pd.Timedelta(seconds=k * ll.WINDOW_SECONDS)
        ticker = f'KXBTC15M-T{k:03d}'
        depth.append({'market_ticker': ticker, 'window_open': open_})
        strike = spot[open_]
        grid = pd.date_range(open_, periods=ll.WINDOW_SECONDS, freq='s')
        delayed = spot.reindex(grid - pd.Timedelta(seconds=lag)).to_numpy()
        z = np.log(delayed / strike) / (SIGMA_1S * np.sqrt(np.maximum(remaining, 1.0)))
        mid = np.clip(norm.cdf(z), 0.03, 0.97)
        prev = None
        for i, t in enumerate(grid):
            seq += 1
            at = t + pd.Timedelta(milliseconds=200)
            bid, no_bid = mid[i] - 0.005, 1.0 - (mid[i] + 0.005)
            base = {'venue': 'kalshi', 'symbol': 'BTC-USD', 'event_time': at,
                    'market_ticker': ticker, 'seq': seq}
            if prev is None:
                events += [{**base, 'kind': 'snapshot', 'side': 'yes', 'price': bid,
                            'size': 100.0, 'absolute': True},
                           {**base, 'kind': 'snapshot', 'side': 'no', 'price': no_bid,
                            'size': 100.0, 'absolute': True}]
            else:
                for side, old, new in (('yes', prev[0], bid), ('no', prev[1], no_bid)):
                    events += [{**base, 'kind': 'delta', 'side': side, 'price': old,
                                'size': -100.0, 'absolute': False},
                               {**base, 'kind': 'delta', 'side': side, 'price': new,
                                'size': 100.0, 'absolute': False}]
            prev = (bid, no_bid)
            # Polymarket: Kalshi's own mid, five seconds LATE.
            j = max(i - 5, 0)
            pm.append({'symbol': 'BTC-USD', 'window_open': open_,
                       'available_time': at,
                       'yes_levels': json.dumps([[mid[j] - 0.005, 50]]),
                       'no_levels': json.dumps([[1 - mid[j] - 0.005, 50]])})
    for t, p in spot.items():
        trades.append({'symbol': 'BTC-USD', 'trade_id': len(trades),
                       'time': t + pd.Timedelta(milliseconds=500), 'price': p,
                       'size': 0.01})
    return {'events': pd.DataFrame(events), 'trades': pd.DataFrame(trades),
            'bars': pd.DataFrame(), 'pm': pd.DataFrame(pm),
            'depth': pd.DataFrame(depth)}


@pytest.fixture(scope='module')
def lagged():
    return ll.run_study(_world(lag=5), max_lag=20)


@pytest.fixture(scope='module')
def simultaneous():
    return ll.run_study(_world(lag=0), max_lag=20)


def _row(result, bucket='all'):
    summary = result.summary
    return summary[(summary['symbol'] == 'BTC-USD') & (summary['bucket'] == bucket)].iloc[0]


def test_the_replay_rebuilds_the_mid_every_second():
    world = _world(lag=0, markets=2)
    mids = ll.kalshi_mid_series(world['events'])
    assert mids['market_ticker'].nunique() == 2
    assert len(mids) == 2 * ll.WINDOW_SECONDS
    assert mids['mid'].between(0.0, 1.0).all()


def test_a_planted_five_second_lag_is_found(lagged):
    spot = lagged[0]
    assert spot.pair == 'kalshi vs spot'
    row = _row(spot)
    assert row['peak_lag'] == 5
    assert row['peak_corr'] > 0.8
    assert abs(row['corr_at_0']) < 0.2
    assert row['lead_share'] > 0.9
    assert row['delayed_share'] > 0.8


def test_no_lag_is_found_where_none_is_planted(simultaneous):
    row = _row(simultaneous[0])
    assert row['peak_lag'] == 0
    assert row['peak_corr'] > 0.8
    assert row['delayed_share'] < 0.2


def test_every_bucket_is_reported(lagged):
    buckets = set(lagged[0].summary['bucket'])
    assert buckets == {'all', *ll.bucket_labels(3)}
    # The planted lag holds in each bucket, not only on average.
    for label in ll.bucket_labels(3):
        assert _row(lagged[0], label)['peak_lag'] == 5


def test_polymarket_lagging_kalshi_reads_as_a_negative_lag(lagged):
    pm = lagged[1]
    assert pm.pair == 'kalshi vs polymarket'
    assert _row(pm)['peak_lag'] == -5
    assert _row(pm)['lead_share'] < 0.1


def test_the_report_formats(lagged):
    text = ll.format_result(lagged[0])
    assert 'kalshi vs spot' in text and 'LEADS Kalshi' in text
    assert 'correlation profile' in text


def test_without_trades_it_falls_back_to_bars_and_says_so():
    world = _world(lag=5, markets=4)
    trades = world.pop('trades')
    minute = trades.set_index('time')['price'].resample('1min').last()
    world['trades'] = pd.DataFrame()
    world['bars'] = pd.DataFrame({'symbol': 'BTC-USD',
                                  'event_time': minute.index - pd.Timedelta(minutes=1),
                                  'close': minute.to_numpy()})
    result = ll.run_study(world, max_lag=10)[0]
    assert 'ONE-MINUTE' in result.resolution


def test_market_windows_prefers_the_recorded_map_and_falls_back_to_the_frames():
    world = _world(lag=0, markets=2)
    mids = ll.kalshi_mid_series(world['events'])
    from_map = ll.market_windows(mids, world['depth'])
    inferred = ll.market_windows(mids, None)
    assert from_map == inferred
    assert from_map['KXBTC15M-T001'] == OPEN0 + pd.Timedelta(minutes=15)
