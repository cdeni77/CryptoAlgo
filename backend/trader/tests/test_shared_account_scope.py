"""Tests for `_is_ours`, which limits reconciliation and the ledger to traded series.

Other markets in the account must not raise false alarms or enter this strategy's P&L.
"""
from __future__ import annotations

from livetrade.ledger import _is_ours


# --- what the loop trades --------------------------------------------------

def test_the_three_traded_series_are_ours():
    for ticker in ('KXBTC15M-26SEP041130-30',
                   'KXETH15M-26SEP041000-00',
                   'KXSOL15M-26SEP040945-45'):
        assert _is_ours(ticker), ticker


def test_case_and_whitespace_do_not_decide_it():
    assert _is_ours('  kxbtc15m-26sep041130-30  ')


# --- what it does not ------------------------------------------------------

def test_the_football_position_that_caused_this_is_not_ours():
    assert not _is_ours(
        'KXMVECROSSCATEGORY-SHARD1-S20265462F3E4B88-FE8833EE6EC')
    assert not _is_ours('KXNFLGAME-25SEP07DALPHI-DAL')
    assert not _is_ours('KXNCAAFGAME-25SEP06ALAWIS-ALA')


def test_the_HOURLY_crypto_ladder_is_not_ours_either():
    """The hourly `KXBTCD` threshold ladder is not one of our series."""
    assert not _is_ours('KXBTCD-26AUG2317-T86749.99')


def test_a_prefix_near_miss_does_not_match():
    """The hyphen after the series is required, so `KXBTC15MINI` does not match."""
    assert not _is_ours('KXBTC15MINI-26SEP04')
    assert not _is_ours('KXBTC15M')  # a series with no market suffix


def test_empty_and_missing_are_not_ours():
    assert not _is_ours('')
    assert not _is_ours(None)  # type: ignore[arg-type]


# --- the alarm must still work ---------------------------------------------

def test_scoping_does_not_silence_a_REAL_unbooked_fill():
    """Scoping still raises the unbooked-fill alarm for our own series."""
    venue_positions = [
        'KXBTC15M-26SEP041130-30',  # ours, real
        'KXMVECROSSCATEGORY-SHARD1-S20265462F3E4B88-FE8833EE6EC',  # a foreign market
    ]
    ours_held: set[str] = set()  # we booked nothing
    would_alarm = {t for t in venue_positions if _is_ours(t)} - ours_held
    assert would_alarm == {'KXBTC15M-26SEP041130-30'}, (
        'an unbooked fill in a traded series must still raise; only the '
        'foreign market is filtered')


def test_an_env_override_of_a_series_is_respected():
    """An env override of a series is respected, since the scope derives from
    `SERIES_BY_SYMBOL`.
    """
    import scripts.live as live
    original = dict(live.SERIES_BY_SYMBOL)
    try:
        live.SERIES_BY_SYMBOL.clear()
        live.SERIES_BY_SYMBOL.update({'BTC-USD': 'KXTESTSERIES'})
        assert _is_ours('KXTESTSERIES-26SEP041130-30')
        assert not _is_ours('KXBTC15M-26SEP041130-30')
    finally:
        live.SERIES_BY_SYMBOL.clear()
        live.SERIES_BY_SYMBOL.update(original)
