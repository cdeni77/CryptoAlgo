"""Computing the market comparison from a backtest that finally has quotes.

`market_gate_values` says "this cannot come from the backtest, and that is the
point" — a backtest had no order book, so `price_source` stood the calibrated
baseline in for the market and "beat the market" collapsed into "beat the
baseline". That was true when written and is now false: eight months of book are
collected, validated to 0.70c against the live recording with a resting-size
ratio of 1.000, and 17,078 windows can be priced.

Two things have to be right or the number is worse than useless.

**The market's forecast is the MID, not the ask.** `model_minus_market` compares
LOG LOSSES — whose probability is better — and the ask is what a trade costs,
which is the mid plus half the spread. Using the ask would hand the model a free
half-spread of apparent skill on every row, in its own favour, and the gate
exists precisely to stop that kind of self-flattery.

**A backfilled quote is a different claim from a recorded one.** They agree
closely, but the row has to say which it is, because the gate is the one thing
standing between a promising number and real money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.metrics import MARKET_COLUMNS, market_gate_values, market_rows_from_scored


def _scored(n=4, **over):
    frame = pd.DataFrame({
        'symbol': ['BTC-USD'] * n,
        'window_open': pd.to_datetime(['2026-07-01T12:00Z'] * n)
                       + pd.to_timedelta(np.arange(n) * 15, unit='m'),
        'offset': [12] * n,
        'decision_time': pd.to_datetime(['2026-07-01T12:12Z'] * n),
        'baseline_probability': [0.40] * n,
        'model_probability': [0.45] * n,
        'outcome': [1, 0, 1, 0],
        'market_probability': [0.50] * n,
    })
    for k, v in over.items():
        frame[k] = v
    return frame


def test_the_market_column_is_the_mid_not_the_ask():
    """Using the ask would credit the model with half the spread as skill."""
    rows = market_rows_from_scored(_scored(market_probability=[0.50] * 4))
    frame = pd.DataFrame(rows, columns=list(MARKET_COLUMNS))
    assert (frame['market'] == 0.50).all()


def test_rows_without_a_quote_are_dropped_not_defaulted():
    """A row priced against the baseline is exactly the circularity this
    replaces. Included, it would report the baseline as the market."""
    frame = _scored()
    frame.loc[[1, 3], 'market_probability'] = np.nan
    rows = market_rows_from_scored(frame)
    assert len(rows) == 2


def test_rows_without_a_settled_outcome_are_dropped():
    frame = _scored()
    frame.loc[[0], 'outcome'] = np.nan
    assert len(market_rows_from_scored(frame)) == 3


def test_the_rows_are_shaped_for_market_gate_values():
    """The contract is positional; a column out of order silently swaps the
    model's probability with the baseline's."""
    rows = market_rows_from_scored(_scored())
    values = market_gate_values(rows)
    assert values['market_windows'] == 4
    assert np.isfinite(values['model_minus_market'])


def test_a_model_that_matches_the_market_scores_zero():
    """The anchor: identical forecasts must give exactly zero, so the sign of
    this number always means what it says."""
    frame = _scored(model_probability=[0.50] * 4)
    values = market_gate_values(market_rows_from_scored(frame))
    assert values['model_minus_market'] == pytest.approx(0.0, abs=1e-12)


def test_a_worse_model_scores_negative():
    """The measured live result was market 0.333 against model 0.430 — a
    candidate can pass every other gate while being a worse forecaster than the
    price it must trade against."""
    frame = _scored(model_probability=[0.05, 0.95, 0.05, 0.95])
    values = market_gate_values(market_rows_from_scored(frame))
    assert values['model_minus_market'] < 0


def test_an_empty_frame_fails_rather_than_passes():
    """Not measured is not measured good."""
    values = market_gate_values(market_rows_from_scored(_scored().iloc[0:0]))
    assert values['market_windows'] == 0.0
    assert np.isnan(values['model_minus_market'])


# --- staleness is not skill ------------------------------------------------
#
# The first run of this reported model_minus_market +0.0371 over 39,740 windows
# and passed the gate. Binned by how stale the quote was, on 132,250 rows:
#
#     <=5s    +0.0041      <=180s   +0.0130
#     <=30s   +0.0052      <=900s   +0.0371
#
# Two effects, both flattering: market_ll WORSENS with age (0.4605 -> 0.4785,
# because a stale price is a bad forecast) while model_ll IMPROVES (0.4564 ->
# 0.4414, because the rows carrying stale quotes are easier ones). Nine tenths
# of the headline was the model beating a price nobody was quoting any more.

def test_a_stale_quote_is_not_the_market():
    """The gate asks whether our probability beats the PRICE. A quote from ten
    minutes earlier is not the price at the decision instant."""
    frame = _scored()
    frame['quote_age_seconds'] = [1.0, 1.0, 600.0, 600.0]
    assert len(market_rows_from_scored(frame)) == 2


def test_the_freshness_bar_is_explicit_and_tunable():
    frame = _scored()
    frame['quote_age_seconds'] = [1.0, 45.0, 100.0, 600.0]
    assert len(market_rows_from_scored(frame, max_quote_age_seconds=120.0)) == 3


def test_rows_without_an_age_are_kept():
    """Live-recorded rows carry no `quote_age_seconds`; dropping them would
    discard the one source that needs no reconstruction at all."""
    frame = _scored()
    assert 'quote_age_seconds' not in frame.columns
    assert len(market_rows_from_scored(frame)) == 4


def test_the_default_bar_is_tight_enough_to_matter():
    from core.metrics import MAX_QUOTE_AGE_SECONDS
    assert MAX_QUOTE_AGE_SECONDS <= 60.0, (
        'at 900s the measured model_minus_market was 9x its fresh-quote value')


# --- the label must be the one the MARKET was priced against ---------------
#
# The measurement this guards against, and it reversed a headline result:
#
#   scored against our Coinbase label      base-mkt  +0.00382
#   scored against the venue's settlement  base-mkt  -0.00245
#   on the 96.8% where the two agree       base-mkt  +0.00101
#
# The baseline is computed from Coinbase bars and our outcome label is computed
# from Coinbase bars; the market prices on CF Benchmarks BRTI. Grading the two
# forecasters on a label that shares a source with one of them hands that one a
# free advantage worth the label noise -- measured at 2.99% sign flips,
# concentrated in the near-ties, which is exactly where the whole effect lived.
#
# It looked like "a lognormal formula beats a liquid market", which should have
# been suspicious on its face. Against the venue's own settlement the baseline
# LOSES, reproducing the live measurement that had the market at 0.333 log loss
# against the model's 0.430.

def test_the_venue_label_is_preferred_when_present():
    """`venue_outcome` is what the market settled on. When we hold it, it is
    the only fair grade for a comparison between our forecast and the price."""
    frame = _scored()
    frame['venue_outcome'] = [0, 1, 0, 1]        # deliberately the opposite
    rows = market_rows_from_scored(frame)
    outcomes = [r[6] for r in rows]
    assert outcomes == [0, 1, 0, 1], 'must grade on the venue label, not ours'


def test_our_label_is_used_only_where_the_venue_has_none():
    """The venue's settlements reach back only so far; our label still has to
    cover the rest, and dropping those rows would discard most of the sample."""
    frame = _scored()
    frame['venue_outcome'] = [np.nan, np.nan, 0, 1]
    rows = market_rows_from_scored(frame)
    assert [r[6] for r in rows] == [1, 0, 0, 1]


def test_a_row_with_neither_label_is_dropped():
    frame = _scored()
    frame['outcome'] = [1, np.nan, 1, 0]
    frame['venue_outcome'] = [np.nan, np.nan, 0, np.nan]
    assert len(market_rows_from_scored(frame)) == 3
