"""Features built from the venues' own order books and strike ladders.

Every column names a way `F(x/sigma)` is wrong. Three families:

    market_state   Kalshi's book at the decision instant.
    cross_venue    Kalshi against Polymarket: two books on nearly the same outcome.
    implied_vol    the strike ladder inverted to a forward-looking sigma.

Invariants:

* Book prices are in integer cents. A probability is price / 100.
* A missing side is NaN, never zero. A one-sided book has no mid.
* No lookahead. A book's state at T is the last tick at or before T, never the nearest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Not features: `bid_levels` / `ask_levels`. The backfilled and live books use
# different price grids, so a model would learn which source a row came from.
PRICE_COLUMNS = ('market_prob', 'market_minus_baseline')
# Structure only: it cannot echo the market, because the price is not in it.
# This keeps a structure edge separable from a copy of the quoted price.
MARKET_STATE = (
    'spread', 'imbalance_touch', 'imbalance_5c', 'depth_ratio',
    'book_convexity',
)
# The columns that can echo. Selected explicitly, never by default.
MARKET_PRICE = PRICE_COLUMNS
# No volume ratio across venues: Kalshi counts integer contracts, Polymarket
# fractional shares, so the quotient measures the unit.
CROSS_VENUE = ('venue_prob_gap', 'venue_gap_change_5', 'venue_spread_ratio',
               'pm_available')
IMPLIED_VOL = ('iv_minus_realised', 'implied_sigma_per_min', 'iv_r2',
               'iv_n_strikes', 'iv_staleness_minutes')

CENTS = 100.0


def _safe_div(num, den):
    den = pd.Series(den).astype(float).replace(0.0, np.nan)
    return pd.Series(num).astype(float).values / den.values


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A numeric column, or all-NaN if it is absent.

    `sigma_per_min` is fitted per fold, so it is absent on a fresh panel.
    """
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)


def _mid(frame: pd.DataFrame) -> np.ndarray:
    """The two-sided mid as a probability, or NaN if either side is missing."""
    bid = pd.Series(_column(frame, 'best_bid'), index=frame.index)
    ask = pd.Series(_column(frame, 'best_ask'), index=frame.index)
    mid = (bid + ask) / 2.0 / CENTS
    return mid.where(bid.notna() & ask.notna()).values


def market_state_features(snap: pd.DataFrame, *,
                          include_price: bool = True) -> pd.DataFrame:
    """Kalshi's book at the decision instant.

    `include_price=False` is the structure-only variant. It cannot copy the
    quoted price, so an edge it finds is not an echo of the market.
    """
    out = pd.DataFrame(index=snap.index)
    mid = _mid(snap)
    bid = pd.Series(_column(snap, 'best_bid'), index=snap.index)
    ask = pd.Series(_column(snap, 'best_ask'), index=snap.index)

    if include_price:
        out['market_prob'] = mid
        # Absent when features are built: `attach_baseline` runs afterwards.
        out['market_minus_baseline'] = mid - _column(snap, 'baseline_probability')

    # Probability units: the edge gate, the fee model and the Kelly stake use them.
    out['spread'] = ((ask - bid) / CENTS).where(bid.notna() & ask.notna()).values

    for name, lo, hi in (('imbalance_touch', 'bid_at_touch', 'ask_at_touch'),
                         ('imbalance_5c', 'bid_5c', 'ask_5c')):
        a = pd.Series(_column(snap, lo), index=snap.index)
        b = pd.Series(_column(snap, hi), index=snap.index)
        # Both sides empty means nothing rests there: NaN, not a balanced book.
        out[name] = _safe_div(a - b, a + b)

    bid_vol = pd.Series(_column(snap, 'bid_vol'), index=snap.index)
    ask_vol = pd.Series(_column(snap, 'ask_vol'), index=snap.index)
    # Log ratio, so 2:1 and 1:2 books are equal and opposite.
    ratio = _safe_div(bid_vol, ask_vol)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['depth_ratio'] = np.where(ratio > 0, np.log(ratio), np.nan)

    bid_1c = pd.Series(_column(snap, 'bid_1c'), index=snap.index)
    ask_1c = pd.Series(_column(snap, 'ask_1c'), index=snap.index)
    bid_5c = pd.Series(_column(snap, 'bid_5c'), index=snap.index)
    ask_5c = pd.Series(_column(snap, 'ask_5c'), index=snap.index)
    out['book_convexity'] = _safe_div(bid_5c, bid_1c) - _safe_div(ask_5c, ask_1c)
    return out


# A ladder fit older than this is not a forward-looking estimate.
# Generous, because `iv_staleness_minutes` is a feature and most ladders yield no fit;
# a tight filter would discard most of the dataset.
MAX_FIT_AGE_MINUTES = 360.0


def attach_implied_vol(table: pd.DataFrame, fits: pd.DataFrame, *,
                       max_age_minutes: float = MAX_FIT_AGE_MINUTES,
                       decision_latency_seconds: float = 3.0) -> pd.DataFrame:
    """The last ladder fit at or before each decision, plus its staleness.

    As-of, never nearest: a fit from after the decision is lookahead.
    Staleness is a feature, not a filter, so the model can tell a fresh fit from a stale one.
    """
    out = table.copy()
    for column in IMPLIED_VOL:
        out[column] = np.nan
    if fits is None or not len(fits) or not len(out):
        return out

    left = pd.DataFrame({
        '_order': np.arange(len(out)),
        'symbol': out['symbol'].values,
        '_at': (pd.to_datetime(out['decision_time'], utc=True)
                + pd.Timedelta(seconds=float(decision_latency_seconds))).values,
    })
    # Use the capture time, not the stamp. The live recorder floors `event_time`
    # to the minute and stores the real capture in `available_time`.
    # The backfill sets both to the fit instant.
    seen = (fits['available_time'] if 'available_time' in fits.columns
            else fits['event_time'])
    seen = pd.to_datetime(seen, utc=True).fillna(
        pd.to_datetime(fits['event_time'], utc=True))
    right = pd.DataFrame({
        'symbol': fits['symbol'].values,
        '_at': seen.values,
        'implied_sigma_per_min': _column(fits, 'implied_sigma_per_min'),
        'iv_r2': _column(fits, 'r2'),
        'iv_n_strikes': _column(fits, 'n_strikes'),
        '_close': _column(fits, 'minutes_to_close'),
    }).dropna(subset=['_at'])
    # Keep the fit time through the join: merge_asof consumes the right key,
    # and staleness is the gap to the fit actually used.
    right['_fit_at'] = right['_at']
    # Among fits seen at the same instant, live uses the nearest close
    # (`record_implied_vol.fits_for`). merge_asof takes the last match, so it sorts last.
    right = right.sort_values(['_at', '_close'], ascending=[True, False],
                              na_position='first', kind='mergesort')

    merged = pd.merge_asof(
        left.sort_values('_at'), right.drop(columns=['_close']),
        on='_at', by='symbol', direction='backward',
    ).sort_values('_order')

    age = ((merged['_at'] - merged['_fit_at']).dt.total_seconds() / 60.0).values
    fresh = np.isfinite(age) & (age >= 0) & (age <= max_age_minutes)

    implied = np.where(fresh, merged['implied_sigma_per_min'].values, np.nan)
    realised = _column(out, 'sigma_per_min')
    ratio = _safe_div(implied, realised)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['iv_minus_realised'] = np.where(ratio > 0, np.log(ratio), np.nan)
    out['implied_sigma_per_min'] = implied
    out['iv_r2'] = np.where(fresh, merged['iv_r2'].values, np.nan)
    out['iv_n_strikes'] = np.where(fresh, merged['iv_n_strikes'].values, np.nan)
    out['iv_staleness_minutes'] = np.where(fresh, age, np.nan)
    return out
