"""The Kalshi account is a person's, and may hold markets this system never traded.

On 2026-09-10 the operator took an NFL and a college football position in the
same account. Reconciliation reported the open position it did not recognise as

    ERROR the venue reports an open position in KXMVECROSSCATEGORY-... that we
    have no record of. An order was filled and not booked — most likely a POST
    that timed out after being accepted. Reconcile by hand before trading again.

once per cycle, for hours. That is the alarm that cries wolf, and this codebase
has already paid for one: `adopt_venue_balance` records a drift alarm firing on
every settlement, which hid the drift it existed to catch. An operator who
learns to scroll past a red ERROR line will scroll past the real one.

**The quieter half is worse.** `core/venue_ledger.py` has no series filter, so a
settled football market would have added its revenue and fees to
`venue_settlements` and appeared in the realised P&L the dashboard reports for
THIS strategy. Two strategies summed into one equity curve is the same class of
error as building the curve from balance differences, where the first deposit
reads as the best day the strategy ever had.

Scoped by series prefix from `SERIES_BY_SYMBOL`, so a `KALSHI_SERIES_*`
override reaches it and there is no second hardcoded list to drift out of step.
"""
from __future__ import annotations

from scripts.live import _is_ours


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
    """`KXBTCD` was tried and every window abstained: it closes on the hour and
    carries an explicit strike, so it is a threshold ladder rather than an
    up/down market. A position there is not one of ours."""
    assert not _is_ours('KXBTCD-26AUG2317-T86749.99')


def test_a_prefix_near_miss_does_not_match():
    """The hyphen is load-bearing. Without it a hypothetical `KXBTC15MINI`
    would be adopted as ours, which is how a scope check quietly widens."""
    assert not _is_ours('KXBTC15MINI-26SEP04')
    assert not _is_ours('KXBTC15M')          # a series with no market suffix


def test_empty_and_missing_are_not_ours():
    """A position with no ticker cannot be attributed, and guessing that it is
    ours would resurrect the false alarm from the other direction."""
    assert not _is_ours('')
    assert not _is_ours(None)  # type: ignore[arg-type]


# --- the alarm must still work ---------------------------------------------

def test_scoping_does_not_silence_a_REAL_unbooked_fill():
    """The whole risk of this change.

    The reverse-direction check — a position the venue holds and we do not — is
    what the audit called the one discrepancy that costs money silently. It
    must still fire for our own series, or muting the noise has muted the
    signal too.
    """
    venue_positions = [
        'KXBTC15M-26SEP041130-30',                                # ours, real
        'KXMVECROSSCATEGORY-SHARD1-S20265462F3E4B88-FE8833EE6EC',  # the punt
    ]
    ours_held: set[str] = set()          # we booked nothing
    would_alarm = {t for t in venue_positions if _is_ours(t)} - ours_held
    assert would_alarm == {'KXBTC15M-26SEP041130-30'}, (
        'an unbooked fill in a traded series must still raise; only the '
        'foreign market is filtered')


def test_an_env_override_of_a_series_is_respected():
    """`SERIES_BY_SYMBOL` reads `KALSHI_SERIES_*`, and this derives from it
    rather than hardcoding a second copy — the trap `series_to_symbol` exists
    to avoid."""
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
