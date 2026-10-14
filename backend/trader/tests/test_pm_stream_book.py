"""Tests for `PmBookCache`, which folds the Polymarket CLOB socket into a book.

Sizes are absolute (0 deletes a level), the touch is the last entry, one message can
carry several assets, and asks are returned NO-denominated.
"""
from __future__ import annotations

import pytest

from core.pm_stream_book import PmBookCache

TOK = '4821508186813518184469108506593773244337390855623517830490998736267099599'
OTHER = '1005822971762365026661997344369430112508643108390338588467720408281579546'


def _book(bids, asks, asset=TOK, ts='1788487941413'):
    return {'event_type': 'book', 'asset_id': asset, 'timestamp': ts,
            'bids': [{'price': p, 'size': s} for p, s in bids],
            'asks': [{'price': p, 'size': s} for p, s in asks]}


def _change(rows, ts='1788487941435'):
    return {'event_type': 'price_change', 'timestamp': ts,
            'price_changes': [{'asset_id': a, 'price': p, 'size': s, 'side': side}
                              for a, p, s, side in rows]}


def test_a_snapshot_gives_the_touch_from_the_LAST_entry():
    """Bids ascend and asks descend on this venue, so the touch is the last entry."""
    c = PmBookCache()
    c.apply(_book([('0.40', '100'), ('0.44', '250')],
                  [('0.60', '80'), ('0.46', '150')]))
    assert c.best_bid(TOK) == pytest.approx(0.44)
    assert c.best_ask(TOK) == pytest.approx(0.46)


def test_price_change_size_is_absolute_not_an_increment():
    c = PmBookCache()
    c.apply(_book([('0.44', '250')], [('0.46', '150')]))
    c.apply(_change([(TOK, '0.44', '120', 'BUY')]))
    assert c.size_at(TOK, 'bid', 0.44) == pytest.approx(120.0), (
        'the message size replaces the level; treating it as an increment '
        'would give 370'
    )


def test_a_zero_size_removes_the_level_outright():
    c = PmBookCache()
    c.apply(_book([('0.40', '100'), ('0.44', '250')], [('0.46', '150')]))
    c.apply(_change([(TOK, '0.44', '0.00', 'BUY')]))
    assert c.best_bid(TOK) == pytest.approx(0.40), (
        'a zeroed level must be deleted, not kept at size 0 above the touch'
    )


def test_one_message_can_carry_two_markets_and_they_do_not_mix():
    c = PmBookCache()
    c.apply(_book([('0.44', '250')], [('0.46', '150')], asset=TOK))
    c.apply(_book([('0.10', '999')], [('0.90', '999')], asset=OTHER))
    c.apply(_change([(OTHER, '0.10', '1', 'BUY'), (TOK, '0.44', '77', 'BUY')]))
    assert c.size_at(TOK, 'bid', 0.44) == pytest.approx(77.0)
    assert c.size_at(OTHER, 'bid', 0.10) == pytest.approx(1.0)


def test_the_ladder_is_returned_in_kalshis_denomination():
    """The YES ask 0.46 is returned as the NO level 0.54, matching Kalshi's `no_levels`."""
    c = PmBookCache()
    c.apply(_book([('0.44', '250')], [('0.46', '150')]))
    yes, no = c.ladder(TOK)
    assert yes[0] == pytest.approx([0.44, 250.0])
    assert no[0] == pytest.approx([0.54, 150.0])


def test_a_snapshot_replaces_the_book_rather_than_merging():
    """A later `book` message replaces the whole book; merging would keep dropped levels."""
    c = PmBookCache()
    c.apply(_book([('0.40', '100'), ('0.44', '250')], [('0.46', '150')]))
    c.apply(_book([('0.42', '10')], [('0.48', '20')]))
    assert c.best_bid(TOK) == pytest.approx(0.42)
    assert c.size_at(TOK, 'bid', 0.44) is None


def test_staleness_is_reported_so_a_decision_can_refuse_it():
    import pandas as pd
    c = PmBookCache()
    c.apply(_book([('0.44', '250')], [('0.46', '150')]))
    age = c.age_seconds(TOK, now=pd.Timestamp(1788487941413 + 5000, unit='ms', tz='UTC'))
    assert age == pytest.approx(5.0, abs=0.01)


def test_an_unknown_asset_is_absent_not_empty():
    c = PmBookCache()
    assert c.ladder('nope') == ([], [])
    assert c.best_bid('nope') is None
