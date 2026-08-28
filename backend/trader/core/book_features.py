"""Features built from the venues' own order books and strike ladders.

Every column here names a way `F(x/sigma)` is wrong. A feature that cannot
answer that question does not get built — the rule that saved the previous
incarnation of this project, whose 27-cell survey found its best cell was its
own control.

Three families, in descending order of prior support:

    market_state   Kalshi's book at the decision instant. The market's own
                   correction to the arithmetic, plus the shape of the book
                   producing it.
    cross_venue    Kalshi against Polymarket — two independent books on very
                   nearly the same random variable. Their settlements agree
                   99.52%, so a gap in PRICE is information or liquidity, not
                   a different question.
    implied_vol    the strike ladder inverted to a FORWARD-looking sigma. The
                   barrier framing says sigma_remaining is the only quantity
                   requiring a forecast, and every volatility feature in
                   `core/features.py` is backward-looking realised vol.

Three invariants run through all of it:

* **Cents, not dollars.** Both venues store integer cents in these fields
  (measured: Kalshi best_bid median 55, Polymarket 47). A probability is
  price/100, and getting that wrong scales every edge by one hundred in the
  same direction every time.
* **A missing side is NaN, never zero.** A one-sided book has no mid, and an
  invented one fabricates a probability. `_price` in the live client already
  follows this rule for quotes; `_money` deliberately does not, because a
  settlement of zero is a real observation. Books are the first case.
* **No lookahead.** A book is a step function, so its state at T is the last
  tick AT OR BEFORE T — never the nearest. A one-minute leak in a
  fifteen-minute window is 7% of the whole question and reads exactly like
  skill.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Deliberately NOT features: `bid_levels` / `ask_levels`. Measured ratio 0.579
# between the backfilled book and the live one, unchanged by any time filter —
# a model trained on them learns which pipe a row arrived through.
PRICE_COLUMNS = ('market_prob', 'market_minus_baseline')
MARKET_STATE = PRICE_COLUMNS + (
    'spread', 'imbalance_touch', 'imbalance_5c', 'depth_ratio',
    'book_convexity', 'quote_intensity',
)
# No volume ratio across venues: Kalshi is integer contracts, Polymarket
# fractional shares, so the quotient would measure the unit.
CROSS_VENUE = ('venue_prob_gap', 'venue_gap_change_5', 'venue_spread_ratio',
               'pm_available')
IMPLIED_VOL = ('iv_minus_realised', 'implied_sigma_per_min', 'iv_r2',
               'iv_n_strikes', 'iv_staleness_minutes')

CENTS = 100.0


def _safe_div(num, den):
    den = pd.Series(den).astype(float).replace(0.0, np.nan)
    return pd.Series(num).astype(float).values / den.values


def _mid(frame: pd.DataFrame) -> np.ndarray:
    """The two-sided mid as a probability, or NaN.

    NaN rather than a fallback to whichever side exists: a lone bid says the
    probability is *at least* something, which is not a probability.
    """
    bid = pd.to_numeric(frame.get('best_bid'), errors='coerce')
    ask = pd.to_numeric(frame.get('best_ask'), errors='coerce')
    mid = (bid + ask) / 2.0 / CENTS
    return mid.where(bid.notna() & ask.notna()).values


def market_state_features(snap: pd.DataFrame, *,
                          include_price: bool = True) -> pd.DataFrame:
    """Kalshi's book at the decision instant.

    `include_price=False` is the structure-only variant, and it exists for a
    specific reason: given the quoted probability, the cheapest path to a low
    log loss is to copy a well-calibrated market. That scores beautifully on
    `log_loss_skill` and then reads ~0 on `model_minus_market`, which is the
    gate that pays. Structure-only physically cannot do that, so an edge it
    finds is provably not an echo.
    """
    out = pd.DataFrame(index=snap.index)
    mid = _mid(snap)
    bid = pd.to_numeric(snap.get('best_bid'), errors='coerce')
    ask = pd.to_numeric(snap.get('best_ask'), errors='coerce')

    if include_price:
        out['market_prob'] = mid
        base = pd.to_numeric(snap.get('baseline_probability'), errors='coerce')
        out['market_minus_baseline'] = mid - base.values

    # Probability units throughout: everything downstream — the edge gate, the
    # fee model, the Kelly stake — is denominated that way.
    out['spread'] = ((ask - bid) / CENTS).where(bid.notna() & ask.notna()).values

    for name, lo, hi in (('imbalance_touch', 'bid_at_touch', 'ask_at_touch'),
                         ('imbalance_5c', 'bid_5c', 'ask_5c')):
        a = pd.to_numeric(snap.get(lo), errors='coerce')
        b = pd.to_numeric(snap.get(hi), errors='coerce')
        # Both sides empty means nothing is resting there. Reporting a balanced
        # book would claim knowledge of a book that does not exist.
        out[name] = _safe_div(a - b, a + b)

    bid_vol = pd.to_numeric(snap.get('bid_vol'), errors='coerce')
    ask_vol = pd.to_numeric(snap.get('ask_vol'), errors='coerce')
    # Logged so a 2:1 bid-heavy book and a 1:2 ask-heavy one are equal and
    # opposite, which a raw ratio is not.
    ratio = _safe_div(bid_vol, ask_vol)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['depth_ratio'] = np.where(ratio > 0, np.log(ratio), np.nan)

    bid_1c = pd.to_numeric(snap.get('bid_1c'), errors='coerce')
    ask_1c = pd.to_numeric(snap.get('ask_1c'), errors='coerce')
    bid_5c = pd.to_numeric(snap.get('bid_5c'), errors='coerce')
    ask_5c = pd.to_numeric(snap.get('ask_5c'), errors='coerce')
    out['book_convexity'] = _safe_div(bid_5c, bid_1c) - _safe_div(ask_5c, ask_1c)

    # Produced by `book_at_decision`; carried as NaN when the caller has not
    # joined a book, so the matrix shape never depends on availability.
    out['quote_intensity'] = pd.to_numeric(
        snap.get('n_snapshots', pd.Series(np.nan, index=snap.index)),
        errors='coerce').values
    return out


def cross_venue_features(kalshi: pd.DataFrame, pm: pd.DataFrame, *,
                         prev_gap: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Two independent books on the same fifteen minutes.

    The venues settle on different oracles — CF Benchmarks BRTI against
    Chainlink's BTC-USD TWAP-60s — but agree on 99.52% of shared windows, so a
    price gap is information or liquidity rather than a different question.
    """
    out = pd.DataFrame(index=kalshi.index)
    k_mid, p_mid = _mid(kalshi), _mid(pm.set_index(kalshi.index))
    out['venue_prob_gap'] = k_mid - p_mid
    # Absence is not agreement: Polymarket coverage differs from Kalshi's, and
    # a zeroed gap would read as two venues concurring.
    out['pm_available'] = np.where(np.isnan(p_mid), 0.0, 1.0)
    out['venue_gap_change_5'] = (
        out['venue_prob_gap'].values - prev_gap if prev_gap is not None
        else np.nan)

    def _spread(frame):
        bid = pd.to_numeric(frame.get('best_bid'), errors='coerce')
        ask = pd.to_numeric(frame.get('best_ask'), errors='coerce')
        return ((ask - bid) / CENTS).where(bid.notna() & ask.notna()).values

    ratio = _safe_div(_spread(kalshi), _spread(pm.set_index(kalshi.index)))
    with np.errstate(divide='ignore', invalid='ignore'):
        out['venue_spread_ratio'] = np.where(ratio > 0, np.log(ratio), np.nan)
    return out


def implied_vol_features(table: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """The ladder's forward-looking sigma against the baseline's backward one.

    `iv_minus_realised` is the mechanism and the rest is context. The baseline
    scales a realised-vol forecast; where the market's implied sigma disagrees,
    the baseline's `sigma_remaining` is wrong in a knowable direction, and that
    is the only quantity the barrier framing says needs forecasting at all.
    """
    out = pd.DataFrame(index=table.index)
    implied = pd.to_numeric(fits.get('implied_sigma_per_min'), errors='coerce').values
    realised = pd.to_numeric(table.get('sigma_per_min'), errors='coerce').values
    ratio = _safe_div(implied, realised)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['iv_minus_realised'] = np.where(ratio > 0, np.log(ratio), np.nan)
    out['implied_sigma_per_min'] = implied
    out['iv_r2'] = pd.to_numeric(fits.get('r2'), errors='coerce').values
    out['iv_n_strikes'] = pd.to_numeric(fits.get('n_strikes'), errors='coerce').values
    # Carried because coverage is ~15% of the timeline with a five-hour mean
    # gap: a sigma forward-filled from three hours ago is a different claim
    # from a fresh one, and the model has to be able to tell them apart.
    out['iv_staleness_minutes'] = pd.to_numeric(
        fits.get('staleness_minutes'), errors='coerce').values
    return out


def book_at_decision(books: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """The last book snapshot AT OR BEFORE each decision instant.

    An as-of join rather than a nearest one, and keyed on the window as well as
    the symbol. Both matter:

      * nearest would let a quote from *after* the decision inform it, which is
        a leak that reads exactly like skill;
      * consecutive windows chain — one window's strike is the previous one's
        settlement value — so a join spilling across the boundary would look
        entirely correct and be wrong.
    """
    if not len(books) or not len(table):
        return pd.DataFrame(index=table.index)
    left = table.sort_values('decision_time')
    right = books.sort_values('event_time')
    joined = pd.merge_asof(
        left, right,
        left_on='decision_time', right_on='event_time',
        by=['symbol', 'window_open'], direction='backward',
        suffixes=('', '_book'))
    return joined.sort_index().reindex(table.index)


# A ladder fit older than this is not a forward-looking estimate of anything.
# Generous, because `iv_staleness_minutes` rides along as a feature and a study
# can tighten it without re-joining: 60% of ladders yield no fit at all, leaving
# a ~5-hour mean gap, so a hard filter here would discard most of the dataset.
MAX_FIT_AGE_MINUTES = 360.0


def attach_implied_vol(table: pd.DataFrame, fits: pd.DataFrame, *,
                       max_age_minutes: float = MAX_FIT_AGE_MINUTES) -> pd.DataFrame:
    """The last ladder fit AT OR BEFORE each decision, plus its staleness.

    As-of, never nearest: a sigma inverted from a ladder that closed after the
    decision instant is a forecast made with tomorrow's newspaper, and it would
    read as skill.

    Staleness is a FEATURE rather than a filter. The fits are irregular — 60% of
    ladders yield nothing — so almost every decision is priced against a sigma
    measured minutes or hours earlier. That is usable, because volatility is
    persistent, but only if the model can tell a fresh estimate from a stale
    one. Filtering everything stale would discard most of the dataset to avoid a
    problem the model can simply be told about.
    """
    out = table.copy()
    for column in IMPLIED_VOL:
        out[column] = np.nan
    if fits is None or not len(fits) or not len(out):
        return out

    left = pd.DataFrame({
        '_order': np.arange(len(out)),
        'symbol': out['symbol'].values,
        '_at': pd.to_datetime(out['decision_time'], utc=True).values,
    })
    right = pd.DataFrame({
        'symbol': fits['symbol'].values,
        '_at': pd.to_datetime(fits['event_time'], utc=True).values,
        'implied_sigma_per_min': pd.to_numeric(
            fits.get('implied_sigma_per_min'), errors='coerce').values,
        'iv_r2': pd.to_numeric(fits.get('r2'), errors='coerce').values,
        'iv_n_strikes': pd.to_numeric(fits.get('n_strikes'), errors='coerce').values,
    }).dropna(subset=['_at'])
    # Carried through the join so staleness is the gap to the fit ACTUALLY used;
    # merge_asof consumes the right key, so without this there is nothing to
    # measure the age against.
    right['_fit_at'] = right['_at']

    merged = pd.merge_asof(
        left.sort_values('_at'), right.sort_values('_at'),
        on='_at', by='symbol', direction='backward',
    ).sort_values('_order')

    age = ((merged['_at'] - merged['_fit_at']).dt.total_seconds() / 60.0).values
    fresh = np.isfinite(age) & (age >= 0) & (age <= max_age_minutes)

    implied = np.where(fresh, merged['implied_sigma_per_min'].values, np.nan)
    realised = pd.to_numeric(out.get('sigma_per_min'), errors='coerce').values
    ratio = _safe_div(implied, realised)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['iv_minus_realised'] = np.where(ratio > 0, np.log(ratio), np.nan)
    out['implied_sigma_per_min'] = implied
    out['iv_r2'] = np.where(fresh, merged['iv_r2'].values, np.nan)
    out['iv_n_strikes'] = np.where(fresh, merged['iv_n_strikes'].values, np.nan)
    out['iv_staleness_minutes'] = np.where(fresh, age, np.nan)
    return out
