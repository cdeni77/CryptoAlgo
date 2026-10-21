"""The peer venue gets its OWN source list.

`quote_sources` exists to exclude a producer whose KALSHI prices are not
executable -- Predexon, which quantises a deci-cent grid to whole cents.
`filter_sources` matches on the bare source name, and Polymarket's observers
are named `backfill`, `live` and `live_touch` too. So naming Kalshi's live
sources also discarded Polymarket's `backfill`, which is the gamma API and has
nothing to do with that defect.

Measured 2026-10-21 it cost 77,759 peer windows, and under `--complete-cases`
every row carrying a `cross_venue` feature went with them: 6,465 candidate
windows became 956 and `scripts.evaluate` could not form a single fold.

Nothing raised. Both venues have a source column, the filter matched, and the
loss showed up only as a window count nobody had a prior for.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from core.config import Config
from core.dataset import peer_quote_limits
from core.quotes import attach_quotes

OPEN = pd.Timestamp('2026-10-01T12:00Z')


def _windows(offset=12):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': OPEN,
                          'offset': offset, 'baseline_probability': 0.4}])


def _row(source, bid, ask, *, venue, offset=12):
    event = OPEN + pd.Timedelta(minutes=offset)
    return {'venue': venue, 'symbol': 'BTC-USD', 'window_open': OPEN,
            'offset_minutes': offset, 'yes_bid': bid, 'yes_ask': ask,
            'quote_age_seconds': 0.5, 'source': source, 'event_time': event,
            'available_time': event + pd.Timedelta(seconds=1)}


def _depth():
    return pd.DataFrame([
        _row('live_touch', 0.30, 0.32, venue='kalshi'),
        _row('backfill', 0.40, 0.42, venue='polymarket'),
    ])


def test_the_peer_is_lost_when_the_filter_is_shared():
    # The behaviour this exists to make visible, kept as the DEFAULT so the
    # change cannot silently alter an existing caller.
    out = attach_quotes(_windows(), _depth(), sources=('live_touch',))
    assert out['market_probability'].iloc[0] == pytest.approx(0.31)
    assert pd.isna(out['pm_market_probability'].iloc[0])


def test_an_explicit_empty_peer_list_frees_the_peer():
    out = attach_quotes(_windows(), _depth(), sources=('live_touch',),
                        peer_sources=())
    assert out['market_probability'].iloc[0] == pytest.approx(0.31)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.41)


def test_the_peer_list_is_independent_of_the_venue_list():
    out = attach_quotes(_windows(), _depth(), sources=('live_touch',),
                        peer_sources=('backfill',))
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.41)


def test_a_peer_list_that_excludes_everything_still_drops_the_peer():
    out = attach_quotes(_windows(), _depth(), sources=('live_touch',),
                        peer_sources=('live_ws',))
    assert pd.isna(out['pm_market_probability'].iloc[0])


def test_none_follows_the_venue_list_exactly():
    shared = attach_quotes(_windows(), _depth(), sources=('live_touch',))
    explicit = attach_quotes(_windows(), _depth(), sources=('live_touch',),
                             peer_sources=None)
    assert shared['pm_market_probability'].isna().equals(
        explicit['pm_market_probability'].isna())


class TestConfigWiring:
    def test_the_default_config_follows_the_venue_list(self):
        assert peer_quote_limits(Config())['peer_sources'] is None

    def test_an_empty_tuple_reaches_attach_quotes_as_empty(self):
        cfg = Config(quote_sources=('live_touch',), peer_quote_sources=())
        assert peer_quote_limits(cfg)['peer_sources'] == ()

    def test_an_explicit_list_reaches_attach_quotes(self):
        cfg = Config(peer_quote_sources=('backfill', 'live'))
        assert peer_quote_limits(cfg)['peer_sources'] == ('backfill', 'live')


class TestCli:
    def _config(self, argv):
        from scripts._common import add_data_arguments, config_from_args
        parser = argparse.ArgumentParser()
        add_data_arguments(parser)
        return config_from_args(parser.parse_args(argv))

    def test_absent_follows_the_venue_list(self):
        assert self._config([]).peer_quote_sources is None

    def test_all_frees_the_peer(self):
        cfg = self._config(['--quote-sources', 'live_touch',
                            '--peer-quote-sources', 'all'])
        assert cfg.peer_quote_sources == ()
        assert cfg.quote_sources == ('live_touch',)

    def test_a_list_is_parsed(self):
        assert self._config(['--peer-quote-sources', 'backfill,live']
                            ).peer_quote_sources == ('backfill', 'live')
