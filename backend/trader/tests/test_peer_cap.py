"""The peer book's age cap is `peer_max_quote_age_seconds`, as in the backtest."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from livetrade.features import (
    cross_venue_row, peer_max_age_seconds, warn_if_peer_cap_tight,
)

NOW = pd.Timestamp('2026-10-14 12:12:02', tz='UTC')


class _Quote:
    yes_bid, yes_ask = 0.50, 0.52


def _pm(age):
    return {'best_bid': 40.0, 'best_ask': 42.0,
            'at': NOW - pd.Timedelta(seconds=age)}


def test_the_peer_cap_falls_back_to_the_quote_cap():
    config = DEFAULT_CONFIG.with_overrides(peer_max_quote_age_seconds=None)
    assert peer_max_age_seconds(config) == config.max_quote_age_seconds


def test_the_peer_cap_is_read_when_set():
    config = DEFAULT_CONFIG.with_overrides(
        max_quote_age_seconds=30.0, peer_max_quote_age_seconds=60.0)
    assert peer_max_age_seconds(config) == 60.0
    row = cross_venue_row(_pm(45), _Quote(), now=NOW, config=config)
    assert not np.isnan(row['venue_prob_gap'])
    row = cross_venue_row(_pm(61), _Quote(), now=NOW, config=config)
    assert np.isnan(row['venue_prob_gap'])


def test_a_tight_peer_cap_is_refused_at_its_own_bound():
    config = DEFAULT_CONFIG.with_overrides(
        max_quote_age_seconds=30.0, peer_max_quote_age_seconds=2.0)
    assert np.isnan(cross_venue_row(_pm(3), _Quote(), now=NOW,
                                    config=config)['venue_prob_gap'])


def test_a_tight_peer_cap_warns(caplog):
    config = DEFAULT_CONFIG.with_overrides(peer_max_quote_age_seconds=2.0)
    with caplog.at_level(logging.WARNING, logger='live'):
        warn_if_peer_cap_tight(config)
    assert any('would void most peer books' in r.message for r in caplog.records)


def test_a_normal_peer_cap_does_not_warn(caplog):
    config = DEFAULT_CONFIG.with_overrides(peer_max_quote_age_seconds=None)
    with caplog.at_level(logging.WARNING, logger='live'):
        warn_if_peer_cap_tight(config)
    assert not caplog.records
