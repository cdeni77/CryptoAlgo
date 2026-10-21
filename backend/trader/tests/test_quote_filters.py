"""`quote_sources` and `null_quote_age`: research may price only on quotes
somebody watched, and may refuse a quote whose age nobody recorded.

The backfill is often stale (>30s for 12-29% of rows) and its stale asks sit
~13-16c from the live one; `live_touch` rows carried no age at all until
2026-10-14. Both defaults keep the historical behaviour exactly.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.quotes import (
    attach_quotes, derived_quote_age, quote_coverage, source_age_table,
)

OPEN = pd.Timestamp('2026-10-01T12:00Z')


def _windows(offsets=(12,)):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': OPEN, 'offset': o,
                          'baseline_probability': 0.4} for o in offsets])


def _row(source, bid, ask, age, offset=12, *, venue='kalshi', written_after=None):
    event = OPEN + pd.Timedelta(minutes=offset)
    row = {'venue': venue, 'symbol': 'BTC-USD', 'window_open': OPEN,
           'offset_minutes': offset, 'yes_bid': bid, 'yes_ask': ask,
           'quote_age_seconds': age, 'source': source, 'event_time': event,
           'available_time': event + pd.Timedelta(seconds=written_after or 600)}
    return row


class TestSources:
    def test_empty_keeps_every_source(self):
        depth = pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)])
        out = attach_quotes(_windows(), depth)
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)

    def test_an_excluded_source_does_not_price_the_row(self):
        depth = pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)])
        out = attach_quotes(_windows(), depth, sources=('live_touch', 'live'))
        assert np.isnan(out['market_probability'].iloc[0])
        assert np.isnan(out['ask_up'].iloc[0])

    def test_an_allowed_source_wins_even_where_priority_ranks_it_lower(self):
        depth = pd.DataFrame([_row('live_touch', 0.60, 0.62, np.nan),
                              _row('live', 0.30, 0.32, 1.0)])
        out = attach_quotes(_windows(), depth, sources=('live',))
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)
        assert out['quote_observer'].iloc[0] == 'live'

    def test_the_peer_venue_is_filtered_too(self):
        depth = pd.DataFrame([_row('live', 0.30, 0.32, 1.0),
                              _row('backfill', 0.50, 0.52, 1.0, venue='polymarket')])
        out = attach_quotes(_windows(), depth, sources=('live',))
        assert np.isnan(out['pm_market_probability'].iloc[0])
        kept = attach_quotes(_windows(), depth)
        assert kept['pm_market_probability'].iloc[0] == pytest.approx(0.51)

    def test_no_source_column_admits_nothing_when_sources_are_named(self):
        depth = pd.DataFrame([_row('live', 0.30, 0.32, 1.0)]).drop(columns='source')
        out = attach_quotes(_windows(), depth, sources=('live',))
        assert np.isnan(out['market_probability'].iloc[0])


class TestNullAge:
    def test_fresh_keeps_a_null_age(self):
        depth = pd.DataFrame([_row('live_ws', 0.30, 0.32, np.nan)])
        out = attach_quotes(_windows(), depth, null_age='fresh')
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)

    def test_stale_drops_a_null_age_that_cannot_be_bounded(self):
        depth = pd.DataFrame([_row('live_ws', 0.30, 0.32, np.nan)])
        out = attach_quotes(_windows(), depth, null_age='stale')
        assert np.isnan(out['market_probability'].iloc[0])

    def test_stale_bounds_a_live_touch_age_from_its_own_timestamps(self):
        """Written 4s after the decision instant, so the book it priced was read
        at most 4s from it: kept, and the bound is what the row now carries."""
        depth = pd.DataFrame([_row('live_touch', 0.30, 0.32, np.nan, written_after=4)])
        out = attach_quotes(_windows(), depth, null_age='stale')
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)
        assert out['quote_age_seconds'].iloc[0] == pytest.approx(4.0)

    def test_a_live_touch_bound_past_the_limit_is_dropped(self):
        depth = pd.DataFrame([_row('live_touch', 0.30, 0.32, np.nan, written_after=95)])
        out = attach_quotes(_windows(), depth, null_age='stale', max_age_seconds=30)
        assert np.isnan(out['market_probability'].iloc[0])

    def test_a_recorded_age_is_never_replaced_by_the_bound(self):
        depth = pd.DataFrame([_row('live_touch', 0.30, 0.32, 1.5, written_after=95)])
        out = attach_quotes(_windows(), depth, null_age='stale')
        assert out['quote_age_seconds'].iloc[0] == pytest.approx(1.5)

    def test_the_bound_is_only_derived_for_live_touch(self):
        """Every other observer's available_time is when IT ran, not when the
        quote was read — the backfill's is weeks later."""
        frame = pd.DataFrame([_row('live_touch', 0.3, 0.32, np.nan, written_after=7),
                              _row('backfill', 0.3, 0.32, np.nan, written_after=7)])
        bound = derived_quote_age(frame)
        assert bound.iloc[0] == pytest.approx(7.0)
        assert np.isnan(bound.iloc[1])

    def test_the_default_bounds_a_missing_age_instead_of_assuming_it(self):
        """The default is 'stale', and an unboundable-or-too-old row is DROPPED.

        'fresh' was the historical default and it is an assumption rather than a
        measurement: it admitted 53,721 rows -- 30% of every priced row in a
        Jan-Oct walk-forward -- whose staleness was simply unknown. Tightening a
        July run from ('fresh', 30s) to ('stale', 5s) took one fold from 881
        trades and +19.25% to 70 trades and -15.59%, so this is not a tolerance
        dial. The live loop had no way to set it at all, which meant a model
        validated under bounded ages would trade under assumed-fresh ones.
        """
        assert Config().null_quote_age == 'stale'
        assert Config().quote_sources == ()
        # 95s between the quote being read and the row being written: bounded,
        # and far past the 30s cap, so it goes.
        depth = pd.DataFrame([_row('live_touch', 0.30, 0.32, np.nan, written_after=95)])
        out = attach_quotes(_windows(), depth)
        assert np.isnan(out['market_probability'].iloc[0])

    def test_a_missing_age_that_bounds_INSIDE_the_cap_is_still_kept(self):
        """'stale' bounds, it does not blanket-drop. The live recorder reads the
        book at the decision instant and stamps `available_time` after the fetch
        returns, so its bound is ~3s and overstates the true age -- dropping
        those would discard the live era for a stopwatch artifact."""
        depth = pd.DataFrame([_row('live_touch', 0.30, 0.32, np.nan, written_after=3)])
        out = attach_quotes(_windows(), depth)
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)

    def test_the_default_is_env_readable_so_the_live_path_can_set_it(self, monkeypatch):
        """`scripts/live.py` carries no freshness flag, so without this the live
        loop is pinned to whatever the dataclass default happens to be."""
        monkeypatch.setenv('QUOTE_NULL_AGE', 'fresh')
        import importlib
        import core.config as cfg
        importlib.reload(cfg)
        try:
            assert cfg.Config().null_quote_age == 'fresh'
        finally:
            monkeypatch.delenv('QUOTE_NULL_AGE', raising=False)
            importlib.reload(cfg)
        assert cfg.Config().null_quote_age == 'stale'


class TestPeerCap:
    """The peer's staleness cap is its own: only ~5% of live Polymarket rows are
    5s fresh, so a tight Kalshi cap applied to the peer voided `venue_prob_gap`
    nearly everywhere and `--complete-cases` kept 0.1% of rows."""

    def _depth(self, kalshi_age, peer_age):
        # Backfill rows: their age counts back from the minute, so each cap
        # sees age + the 3s decision latency.
        return pd.DataFrame([_row('backfill', 0.30, 0.32, kalshi_age),
                             _row('backfill', 0.50, 0.52, peer_age,
                                  venue='polymarket')])

    def test_unset_follows_the_kalshi_cap_as_before(self):
        out = attach_quotes(_windows(), self._depth(1.0, 20.0), max_age_seconds=5)
        assert np.isnan(out['pm_market_probability'].iloc[0])

    def test_tightening_the_kalshi_cap_no_longer_touches_the_peer(self):
        out = attach_quotes(_windows(), self._depth(20.0, 20.0), max_age_seconds=5,
                            peer_max_age_seconds=30)
        assert np.isnan(out['market_probability'].iloc[0])
        assert out['pm_market_probability'].iloc[0] == pytest.approx(0.51)

    def test_the_peer_cap_binds_on_the_peer_alone(self):
        out = attach_quotes(_windows(), self._depth(20.0, 20.0), max_age_seconds=30,
                            peer_max_age_seconds=5)
        assert out['market_probability'].iloc[0] == pytest.approx(0.31)
        assert np.isnan(out['pm_market_probability'].iloc[0])

    def test_stale_null_age_reaches_the_peer_only_when_its_cap_asks(self):
        from core.dataset import peer_quote_limits

        depth = self._depth(1.0, np.nan)
        loose = Config(null_quote_age='stale')
        out = attach_quotes(_windows(), depth, null_age='stale',
                            **peer_quote_limits(loose))
        assert out['pm_market_probability'].iloc[0] == pytest.approx(0.51)
        strict = Config(null_quote_age='stale', peer_max_quote_age_seconds=30)
        out = attach_quotes(_windows(), depth, null_age='stale',
                            **peer_quote_limits(strict))
        assert np.isnan(out['pm_market_probability'].iloc[0])

    def test_dataset_build_reads_the_peer_cap_from_config(self):
        from core.dataset import Dataset
        from tests.conftest import make_bars
        from tests.stack_fixtures import make_depth

        bars = make_bars(days=3)
        kalshi = make_depth(bars, Config(), quoted_days=1, source='backfill',
                            age=20.0)
        depth = pd.concat([kalshi, kalshi.assign(venue='polymarket')],
                          ignore_index=True)
        tight = Dataset.build(bars, Config(max_quote_age_seconds=5,
                                           peer_max_quote_age_seconds=30), depth=depth)
        assert tight.windows['market_probability'].notna().sum() == 0
        assert tight.windows['pm_market_probability'].notna().sum() > 0

    def test_the_flag_reaches_config_and_provenance(self):
        from scripts._common import add_data_arguments, config_from_args

        parser = add_data_arguments(argparse.ArgumentParser())
        config = config_from_args(parser.parse_args(
            ['--peer-max-quote-age-seconds', '30']))
        assert config.peer_max_quote_age_seconds == 30.0
        assert 'peer_max_quote_age_seconds' in config.cli_overrides
        assert config.provenance()['peer_max_quote_age_seconds'] == 30.0
        assert Config().provenance()['peer_max_quote_age_seconds'] is None
        with pytest.raises(ValueError):
            Config(peer_max_quote_age_seconds=0)


def test_an_unlisted_source_is_excluded_not_merely_ranked_last():
    """QUOTE_SOURCE_PRIORITY only RANKS: with no live row for the key, a backfill
    row still wins under it. `quote_sources` must EXCLUDE it."""
    depth = pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5, offset=12),
                          _row('live_touch', 0.60, 0.62, np.nan, offset=6)])
    # null_age pinned to 'fresh': the subject here is `sources`, and the
    # default is now 'stale', which would drop the unaged live_touch row before
    # source selection ever ran.
    ranked = attach_quotes(_windows((6, 12)), depth, null_age='fresh')
    assert ranked['market_probability'].notna().all()
    filtered = attach_quotes(_windows((6, 12)), depth, null_age='fresh',
                             sources=('live_touch', 'live_ws', 'live'))
    by_offset = filtered.set_index('offset')
    assert by_offset.loc[6, 'market_probability'] == pytest.approx(0.61)
    assert np.isnan(by_offset.loc[12, 'market_probability'])
    assert np.isnan(by_offset.loc[12, 'ask_up'])
    assert by_offset.loc[12, 'quote_observer'] is None or \
        pd.isna(by_offset.loc[12, 'quote_observer'])


class TestCoverage:
    def test_quote_coverage_breaks_the_priced_rows_down_by_observer_and_age(self):
        depth = pd.DataFrame([_row('live_touch', 0.3, 0.32, np.nan, offset=3),
                              _row('live', 0.3, 0.32, 0.0, offset=6),
                              _row('backfill', 0.3, 0.32, 12.0, offset=9)])
        # null_age pinned to 'fresh': this asserts on the 'null' age bucket,
        # which is only populated when an unaged row is admitted at all.
        out = attach_quotes(_windows((3, 6, 9, 12)), depth, null_age='fresh')
        cov = quote_coverage(out)
        assert cov['rows'] == 4 and cov['quoted'] == 3
        table = pd.DataFrame(cov['by_source_age']).set_index('source')
        assert table.loc['live_touch', 'null'] == 1
        assert table.loc['live', '2-5s'] == 1
        assert table.loc['backfill', '5-30s'] == 1

    def test_the_raw_table_counts_every_bucket(self):
        frame = pd.DataFrame({'source': ['a', 'a', 'a', 'b'],
                              'quote_age_seconds': [1.0, 40.0, np.nan, 3.0]})
        table = source_age_table(frame).set_index('source')
        assert table.loc['a', '<=2s'] == 1
        assert table.loc['a', '>30s'] == 1
        assert table.loc['a', 'null'] == 1
        assert table.loc['a', 'total'] == 3
        assert table.loc['b', '2-5s'] == 1


class TestConfigAndCli:
    def test_a_typo_in_null_quote_age_is_refused(self):
        with pytest.raises(ValueError):
            Config(null_quote_age='old')

    def test_the_flags_land_in_the_config_and_its_overrides(self):
        from scripts._common import add_data_arguments, config_from_args

        parser = add_data_arguments(argparse.ArgumentParser())
        args = parser.parse_args(['--quote-sources', 'live_touch,live_ws',
                                  '--null-quote-age', 'stale'])
        config = config_from_args(args)
        assert config.quote_sources == ('live_touch', 'live_ws')
        assert config.null_quote_age == 'stale'
        assert {'quote_sources', 'null_quote_age'} <= config.cli_overrides
        assert config.provenance()['quote_sources'] == ['live_touch', 'live_ws']

    def test_dataset_build_applies_the_config(self):
        from core.dataset import Dataset
        from tests.conftest import make_bars
        from tests.stack_fixtures import make_depth

        config = Config(quote_sources=('live',))
        bars = make_bars(days=3)
        depth = make_depth(bars, config, quoted_days=1, source='live_touch')
        built = Dataset.build(bars, config, depth=depth)
        assert built.windows['market_probability'].notna().sum() == 0
        built = Dataset.build(bars, Config(), depth=depth)
        assert built.windows['market_probability'].notna().sum() > 0
