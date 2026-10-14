"""Tests that the cycle records the peer book and the implied-vol fit it decided on.

These rows record what the decision read, so the point-in-time joins accept them.
"""
from __future__ import annotations

import asyncio
import types

import numpy as np
import pandas as pd
import pytest

from core.book_features import attach_implied_vol
from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore
from core.quotes import attach_quotes, decision_age
from livetrade import features
from livetrade.features import (
    DECISION_INPUTS, _attach_book_features, fit_row, peer_depth_row,
)
from livetrade.quotes import _record_touch

WINDOW = pd.Timestamp('2026-10-13 12:00', tz='UTC')
OFFSET = 3
MINUTE = WINDOW + pd.Timedelta(minutes=OFFSET)
DECIDED = MINUTE + pd.Timedelta(seconds=2)
PEER_AT = MINUTE + pd.Timedelta(seconds=1.5)
FIT_AT = MINUTE - pd.Timedelta(seconds=29)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.setenv('RESEARCH_STORE', str(tmp_path / 'store'))
    DECISION_INPUTS.clear()
    features._FITS_WRITTEN.clear()
    features.reset_peer_tracking()
    yield
    DECISION_INPUTS.clear()
    features._FITS_WRITTEN.clear()
    features.reset_peer_tracking()


def _peer(at=PEER_AT, window=WINDOW, bid=47.0, ask=49.0, transport='stream'):
    return {'best_bid': bid, 'best_ask': ask, 'at': at, 'window': window,
            'transport': transport}


def _fit(at=FIT_AT, source='recorder'):
    return {'implied_sigma_per_min': 0.0006, 'r2': 0.99, 'n_strikes': 6.0,
            'at': at, 'minutes_to_close': 57.0, 'event_ticker': 'KXBTCD-X',
            'close_time': WINDOW + pd.Timedelta(hours=1),
            'implied_spot': 100_000.0, 'atm_strike': 100_000.0,
            'source': source}


def _inputs(peer=None, refusal=None, fit=None):
    return {'window_open': WINDOW, 'offset': OFFSET, 'peer': peer,
            'refusal': refusal, 'fit': fit}


def _quote():
    return types.SimpleNamespace(
        yes_bid=0.50, yes_ask=0.52, yes_bid_size=100.0, yes_ask_size=80.0,
        book_source='stream', book_time=(MINUTE + pd.Timedelta(seconds=1)).timestamp(),
        yes_levels=[[0.50, 100.0]], no_levels=[[0.48, 80.0]])


def _windows():
    return pd.DataFrame({'symbol': ['BTC-USD'], 'window_open': [WINDOW],
                         'offset': [OFFSET], 'decision_time': [MINUTE],
                         'sigma_per_min': [0.0005]})


def _record(inputs):
    DECISION_INPUTS['BTC-USD'] = inputs
    scored = pd.DataFrame({'symbol': ['BTC-USD']})
    quotes = {'BTC-USD': (_quote(), 'KXBTC15M-26OCT141215-15')}
    asyncio.run(_record_touch(scored, quotes, WINDOW, OFFSET, DEFAULT_CONFIG,
                              None, decided_at=DECIDED))
    return ResearchStore()


# ---- the peer row ------------------------------------------------------------

def test_the_used_peer_row():
    row = peer_depth_row('BTC-USD', _inputs(_peer()), decided_at=DECIDED)
    assert row['venue'] == 'polymarket' and row['source'] == 'live_touch'
    assert row['event_time'] == MINUTE
    assert row['offset_minutes'] == OFFSET and row['window_open'] == WINDOW
    assert row['yes_bid'] == pytest.approx(0.47)
    assert row['yes_ask'] == pytest.approx(0.49)
    assert row['quote_age_seconds'] == pytest.approx(0.5)
    assert row['transport'] == 'stream'
    assert row['quality'] == 'valid'


@pytest.mark.parametrize('refusal, quality', [
    ('too_old', 'suspicious'), ('future', 'suspicious'),
    ('one_sided', 'suspicious'), ('wrong_window', 'invalid'),
    ('no_book', 'invalid')])
def test_a_refused_peer_row_is_kept_out_of_the_default_read(refusal, quality):
    peer = None if refusal == 'no_book' else _peer()
    row = peer_depth_row('BTC-USD', _inputs(peer, refusal), decided_at=DECIDED)
    assert row['quality'] == quality


def test_a_future_stamp_records_age_zero():
    row = peer_depth_row('BTC-USD', _inputs(
        _peer(at=DECIDED + pd.Timedelta(seconds=1)), 'future'), decided_at=DECIDED)
    assert row['quote_age_seconds'] == 0.0


def test_the_peer_row_passes_decision_age_and_the_minute_grid_row_does_not():
    ours = peer_depth_row('BTC-USD', _inputs(_peer()), decided_at=DECIDED)
    grid = dict(ours, source='live_ws', quote_age_seconds=29.0)
    age = decision_age(pd.DataFrame([ours, grid]), 'fresh', 3.0)
    assert age.iloc[0] >= 0
    assert age.iloc[1] < 0


def test_attach_quotes_takes_the_recorded_peer_for_its_decision():
    store = _record(_inputs(_peer(), fit=_fit()))
    depth = store.read('venue_depth')
    assert set(depth['venue']) == {'kalshi', 'polymarket'}
    priced = attach_quotes(_windows(), depth)
    assert priced['pm_market_probability'].iloc[0] == pytest.approx(0.48)


def test_attach_quotes_refuses_the_minute_grid_peer():
    store = _record(_inputs(None, 'no_book'))
    grid = peer_depth_row('BTC-USD', _inputs(_peer(
        at=MINUTE + pd.Timedelta(seconds=29))), decided_at=MINUTE)
    grid.update(source='live_ws', quote_age_seconds=29.0,
                available_time=MINUTE + pd.Timedelta(seconds=29))
    store.write('venue_depth', pd.DataFrame([grid]))
    priced = attach_quotes(_windows(), store.read('venue_depth'))
    assert np.isnan(priced['pm_market_probability'].iloc[0])


def test_a_refused_peer_is_written_but_not_read_by_default():
    store = _record(_inputs(_peer(at=MINUTE - pd.Timedelta(minutes=5)), 'too_old'))
    every = store.read('venue_depth', min_quality=None)
    assert (every['venue'] == 'polymarket').sum() == 1
    assert not (store.read('venue_depth')['venue'] == 'polymarket').any()


# ---- the implied-vol row -------------------------------------------------------

def test_the_fit_row_is_stamped_at_its_capture():
    row = fit_row('BTC-USD', _fit())
    assert row['event_time'] == FIT_AT and row['available_time'] == FIT_AT
    assert row['minutes_to_close'] == 57.0


def test_a_store_fit_is_not_written_again():
    assert fit_row('BTC-USD', _fit(source='store')) is None


def test_attach_implied_vol_takes_the_recorded_fit_for_its_decision():
    store = _record(_inputs(_peer(), fit=_fit()))
    fits = store.read('venue_implied_vol')
    assert len(fits) == 1
    joined = attach_implied_vol(_windows(), fits)
    assert joined['implied_sigma_per_min'].iloc[0] == pytest.approx(0.0006)
    # The age live used: decision instant (plus latency) minus capture time.
    assert joined['iv_staleness_minutes'].iloc[0] == pytest.approx(32 / 60)


def test_attach_implied_vol_refuses_a_fit_captured_after_the_decision():
    late = dict(fit_row('BTC-USD', _fit()), event_time=MINUTE,
                available_time=MINUTE + pd.Timedelta(seconds=29))
    joined = attach_implied_vol(_windows(), pd.DataFrame([late]))
    assert np.isnan(joined['implied_sigma_per_min'].iloc[0])


def test_the_same_fit_is_written_once():
    _record(_inputs(_peer(), fit=_fit()))
    store = _record(_inputs(_peer(), fit=_fit()))
    assert len(store.read('venue_implied_vol')) == 1


def test_the_nearest_close_wins_among_fits_seen_at_one_instant():
    near = fit_row('BTC-USD', _fit())
    far = dict(near, event_time=FIT_AT.floor('min'), minutes_to_close=117.0,
               implied_sigma_per_min=0.0009)
    for order in ([near, far], [far, near]):
        joined = attach_implied_vol(_windows(), pd.DataFrame(order))
        assert joined['implied_sigma_per_min'].iloc[0] == pytest.approx(0.0006)


# ---- what _attach_book_features remembers ---------------------------------------

def test_attach_remembers_the_peer_and_the_fit(monkeypatch):
    monkeypatch.setattr(features, '_venue_caches', lambda: (
        {'BTC-USD': _peer(at=pd.Timestamp.now(tz='UTC'),
                          window=WINDOW)}, {}))
    monkeypatch.setattr(features, 'latest_fit',
                        lambda symbol, now, cache: _fit())
    scored = pd.DataFrame({'symbol': ['BTC-USD'], 'window_open': [WINDOW],
                           'offset': [OFFSET], 'baseline_probability': [0.5],
                           'sigma_per_min': [0.0005]})
    _attach_book_features(scored, {'BTC-USD': (_quote(), 'T')})
    inputs = DECISION_INPUTS['BTC-USD']
    assert inputs['window_open'] == WINDOW and inputs['offset'] == OFFSET
    assert inputs['peer']['transport'] == 'stream'
    assert inputs['refusal'] is None
    assert inputs['fit']['source'] == 'recorder'
