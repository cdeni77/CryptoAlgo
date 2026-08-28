"""Features computed from the venue's own order book.

Everything here answers one question: **in what way is `F(x/sigma)` wrong?**
A column that cannot answer it does not get built, which is the rule that kept
the previous incarnation of this project from a 27-cell grid whose best cell was
its own control.

Three properties run through all of them:

  * **Cents, not dollars.** Both venues store integer cents in these fields
    (measured: Kalshi best_bid median 55, Polymarket 47). A probability is
    therefore price/100, and getting that wrong scales every edge by 100.
  * **A missing side is NaN, never zero.** A one-sided book has no mid, and
    inventing one fabricates a probability. This is the same rule `_price` in
    the live client already follows for quotes.
  * **No lookahead.** A feature at offset m may only see snapshots at or before
    that decision instant — a book is a step function, so the state at T is the
    last tick AT OR BEFORE T, never the nearest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.book_features import (
    MARKET_STATE, CROSS_VENUE, IMPLIED_VOL,
    book_at_decision, cross_venue_features, implied_vol_features,
    market_state_features,
)


def _snap(**over):
    row = {'best_bid': 44.0, 'best_ask': 46.0, 'bid_at_touch': 250.0,
           'ask_at_touch': 150.0, 'bid_1c': 400.0, 'ask_1c': 300.0,
           'bid_5c': 900.0, 'ask_5c': 600.0, 'bid_vol': 2000.0,
           'ask_vol': 1000.0, 'baseline_probability': 0.40}
    row.update(over)
    return pd.DataFrame([row])


# --- market_state ----------------------------------------------------------

def test_the_market_probability_is_the_mid_in_cents_converted_to_a_fraction():
    got = market_state_features(_snap())
    assert abs(got['market_prob'].iloc[0] - 0.45) < 1e-9


def test_market_minus_baseline_is_the_markets_correction_to_the_arithmetic():
    """The single most informative column: where the market disagrees with
    F(x/sigma), which is exactly what the model is trying to learn."""
    got = market_state_features(_snap(baseline_probability=0.40))
    assert abs(got['market_minus_baseline'].iloc[0] - 0.05) < 1e-9


def test_a_one_sided_book_has_no_probability_rather_than_a_guessed_one():
    got = market_state_features(_snap(best_ask=np.nan))
    assert pd.isna(got['market_prob'].iloc[0])
    assert pd.isna(got['market_minus_baseline'].iloc[0])


def test_the_spread_is_in_probability_units_not_cents():
    """Everything downstream — the edge gate, the fee model — is in probability
    units. A spread left in cents is a hundredfold error in the same direction
    every time."""
    got = market_state_features(_snap(best_bid=44.0, best_ask=46.0))
    assert abs(got['spread'].iloc[0] - 0.02) < 1e-9


def test_imbalance_is_signed_and_bounded():
    got = market_state_features(_snap(bid_at_touch=300.0, ask_at_touch=100.0))
    assert abs(got['imbalance_touch'].iloc[0] - 0.5) < 1e-9
    got = market_state_features(_snap(bid_at_touch=100.0, ask_at_touch=300.0))
    assert abs(got['imbalance_touch'].iloc[0] + 0.5) < 1e-9


def test_an_empty_book_gives_no_imbalance_rather_than_zero():
    """Zero size on both sides means nothing is resting there. Reporting a
    balanced book would claim knowledge of a book that does not exist."""
    got = market_state_features(_snap(bid_at_touch=0.0, ask_at_touch=0.0))
    assert pd.isna(got['imbalance_touch'].iloc[0])


def test_depth_ratio_is_logged_so_it_is_symmetric():
    """A 2:1 bid-heavy book and a 1:2 ask-heavy one should be equal and
    opposite, which a raw ratio is not."""
    a = market_state_features(_snap(bid_vol=2000.0, ask_vol=1000.0))['depth_ratio'].iloc[0]
    b = market_state_features(_snap(bid_vol=1000.0, ask_vol=2000.0))['depth_ratio'].iloc[0]
    assert abs(a + b) < 1e-9 and a > 0


def test_level_counts_are_not_features():
    """Measured ratio 0.579 between backfilled and live level counts. A feature
    built on them measures which pipe the row arrived through."""
    assert not any('levels' in c for c in MARKET_STATE)


def test_the_price_free_variant_cannot_echo_the_market():
    """Structure-only exists so an edge can be claimed that provably is not a
    copy of a well-calibrated quote."""
    structural = market_state_features(_snap(), include_price=False)
    assert 'market_prob' not in structural.columns
    assert 'market_minus_baseline' not in structural.columns
    assert 'imbalance_touch' in structural.columns


# --- cross_venue -----------------------------------------------------------

def test_the_venue_gap_is_kalshi_minus_polymarket_in_probability_units():
    k = pd.DataFrame([{'best_bid': 44.0, 'best_ask': 46.0}])
    p = pd.DataFrame([{'best_bid': 40.0, 'best_ask': 42.0}])
    got = cross_venue_features(k, p)
    assert abs(got['venue_prob_gap'].iloc[0] - 0.04) < 1e-9
    assert got['pm_available'].iloc[0] == 1.0


def test_a_missing_polymarket_book_is_flagged_not_zeroed():
    """Polymarket coverage differs from Kalshi's, and absence is not agreement."""
    k = pd.DataFrame([{'best_bid': 44.0, 'best_ask': 46.0}])
    p = pd.DataFrame([{'best_bid': np.nan, 'best_ask': np.nan}])
    got = cross_venue_features(k, p)
    assert pd.isna(got['venue_prob_gap'].iloc[0])
    assert got['pm_available'].iloc[0] == 0.0


def test_volumes_never_cross_the_venue_boundary():
    """Kalshi is integer contracts, Polymarket fractional shares. A ratio of the
    two would measure the unit, not the market."""
    assert not any(c.startswith('venue_vol') or c.endswith('_vol_ratio')
                   for c in CROSS_VENUE)


# --- implied_vol -----------------------------------------------------------

def test_iv_minus_realised_is_the_mechanism_and_is_logged():
    """The baseline's sigma is a BACKWARD-looking realised-vol forecast. Where
    the market's forward-looking sigma disagrees, the baseline is wrong in a
    knowable direction — that disagreement is the feature, not the level."""
    table = pd.DataFrame([{'sigma_per_min': 0.0005}])
    fits = pd.DataFrame([{'implied_sigma_per_min': 0.0010, 'r2': 0.98,
                          'n_strikes': 9, 'staleness_minutes': 12.0}])
    got = implied_vol_features(table, fits)
    assert abs(got['iv_minus_realised'].iloc[0] - np.log(2.0)) < 1e-9


def test_a_missing_fit_is_nan_and_its_staleness_says_why():
    table = pd.DataFrame([{'sigma_per_min': 0.0005}])
    fits = pd.DataFrame([{'implied_sigma_per_min': np.nan, 'r2': np.nan,
                          'n_strikes': np.nan, 'staleness_minutes': np.nan}])
    got = implied_vol_features(table, fits)
    assert pd.isna(got['iv_minus_realised'].iloc[0])
    assert 'iv_staleness_minutes' in got.columns


def test_staleness_is_carried_because_coverage_is_15_percent():
    """60% of ladders yield no fit at all, leaving a 5-hour mean gap. A sigma
    forward-filled from three hours ago is a different claim from a fresh one,
    and the model has to be able to tell them apart."""
    assert 'iv_staleness_minutes' in IMPLIED_VOL


# --- the as-of join --------------------------------------------------------

def test_the_book_state_is_the_last_snapshot_at_or_before_the_decision():
    """A nearest-match would let a quote from AFTER the decision inform it,
    which reads exactly like skill."""
    books = pd.DataFrame({
        'symbol': ['BTC-USD'] * 3,
        'window_open': pd.to_datetime(['2026-07-01T12:00Z'] * 3),
        'event_time': pd.to_datetime(['2026-07-01T12:02Z', '2026-07-01T12:05Z',
                                      '2026-07-01T12:09Z']),
        'best_bid': [40.0, 44.0, 48.0], 'best_ask': [42.0, 46.0, 50.0]})
    table = pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': pd.to_datetime(['2026-07-01T12:00Z']),
        'decision_time': pd.to_datetime(['2026-07-01T12:06Z'])})
    got = book_at_decision(books, table)
    assert got['best_bid'].iloc[0] == 44.0, 'must take 12:05, not 12:09'


def test_a_decision_before_any_snapshot_gets_nothing_rather_than_the_first():
    books = pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': pd.to_datetime(['2026-07-01T12:00Z']),
        'event_time': pd.to_datetime(['2026-07-01T12:09Z']),
        'best_bid': [48.0], 'best_ask': [50.0]})
    table = pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': pd.to_datetime(['2026-07-01T12:00Z']),
        'decision_time': pd.to_datetime(['2026-07-01T12:03Z'])})
    got = book_at_decision(books, table)
    assert pd.isna(got['best_bid'].iloc[0])


def test_the_join_does_not_leak_across_windows():
    """Consecutive windows chain — one window's strike is the previous one's
    settlement — so a join that spilled across the boundary would look correct
    and be wrong."""
    books = pd.DataFrame({
        'symbol': ['BTC-USD'] * 2,
        'window_open': pd.to_datetime(['2026-07-01T12:00Z', '2026-07-01T12:15Z']),
        'event_time': pd.to_datetime(['2026-07-01T12:05Z', '2026-07-01T12:20Z']),
        'best_bid': [44.0, 60.0], 'best_ask': [46.0, 62.0]})
    table = pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': pd.to_datetime(['2026-07-01T12:15Z']),
        'decision_time': pd.to_datetime(['2026-07-01T12:18Z'])})
    got = book_at_decision(books, table)
    assert pd.isna(got['best_bid'].iloc[0]), 'the 12:05 book belongs to another window'
