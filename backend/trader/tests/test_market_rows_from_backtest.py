"""Tests for `market_rows_from_scored`: the market comparison computed from a backtest with
quotes.

The market's forecast is the mid, not the ask: `model_minus_market` compares log
losses, and the ask would give the model half the spread as free skill. A row
needs a real quote, a settled label, and a fresh quote age.
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
    """The market column is the mid; the ask would credit the model with half the spread.
    """
    rows = market_rows_from_scored(_scored(market_probability=[0.50] * 4))
    frame = pd.DataFrame(rows, columns=list(MARKET_COLUMNS))
    assert (frame['market'] == 0.50).all()


def test_rows_without_a_quote_are_dropped_not_defaulted():
    """Rows without a quote are dropped, not priced at the baseline."""
    frame = _scored()
    frame.loc[[1, 3], 'market_probability'] = np.nan
    rows = market_rows_from_scored(frame)
    assert len(rows) == 2


def test_rows_without_a_settled_outcome_are_dropped():
    frame = _scored()
    frame.loc[[0], 'outcome'] = np.nan
    assert len(market_rows_from_scored(frame)) == 3


def test_the_rows_are_shaped_for_market_gate_values():
    """Rows match the positional layout `market_gate_values` expects."""
    rows = market_rows_from_scored(_scored())
    values = market_gate_values(rows)
    assert values['market_windows'] == 4
    assert np.isfinite(values['model_minus_market'])


def test_a_model_that_matches_the_market_scores_zero():
    """Identical model and market forecasts score exactly zero."""
    frame = _scored(model_probability=[0.50] * 4)
    values = market_gate_values(market_rows_from_scored(frame))
    assert values['model_minus_market'] == pytest.approx(0.0, abs=1e-12)


def test_a_worse_model_scores_negative():
    """A worse forecaster than the price scores negative."""
    frame = _scored(model_probability=[0.05, 0.95, 0.05, 0.95])
    values = market_gate_values(market_rows_from_scored(frame))
    assert values['model_minus_market'] < 0


def test_an_empty_frame_fails_rather_than_passes():
    """No rows means not measured: NaN, not a pass."""
    values = market_gate_values(market_rows_from_scored(_scored().iloc[0:0]))
    assert values['market_windows'] == 0.0
    assert np.isnan(values['model_minus_market'])


# --- staleness is not skill ------------------------------------------------
#
# A stale price is a bad forecast, so beating it is not skill. Rows whose quote
# is older than the age limit are dropped.

def test_a_stale_quote_is_not_the_market():
    """A quote from minutes earlier is not the price at the decision instant, so it is
    dropped.
    """
    frame = _scored()
    frame['quote_age_seconds'] = [1.0, 1.0, 600.0, 600.0]
    assert len(market_rows_from_scored(frame)) == 2


def test_the_freshness_bar_is_explicit_and_tunable():
    frame = _scored()
    frame['quote_age_seconds'] = [1.0, 45.0, 100.0, 600.0]
    assert len(market_rows_from_scored(frame, max_quote_age_seconds=120.0)) == 3


def test_rows_without_an_age_are_kept():
    """Live-recorded rows carry no `quote_age_seconds` and are kept."""
    frame = _scored()
    assert 'quote_age_seconds' not in frame.columns
    assert len(market_rows_from_scored(frame)) == 4


def test_the_default_bar_is_tight_enough_to_matter():
    from core.metrics import MAX_QUOTE_AGE_SECONDS
    assert MAX_QUOTE_AGE_SECONDS <= 60.0, (
        'at 900s the measured model_minus_market was 9x its fresh-quote value')


# --- the label must be the one the market settled on ------------------------
#
# Our label and the baseline both come from Coinbase bars, while the market
# prices on CF Benchmarks BRTI. Grading on our label gives the baseline a free
# advantage in the near-ties. So the venue's own settlement is used when present.

def test_the_venue_label_is_preferred_when_present():
    """`venue_outcome` is preferred when present."""
    frame = _scored()
    frame['venue_outcome'] = [0, 1, 0, 1]  # deliberately the opposite
    rows = market_rows_from_scored(frame)
    outcomes = [r[6] for r in rows]
    assert outcomes == [0, 1, 0, 1], 'must grade on the venue label, not ours'


def test_our_label_is_used_only_where_the_venue_has_none():
    """Our label fills only the rows where the venue has none."""
    frame = _scored()
    frame['venue_outcome'] = [np.nan, np.nan, 0, 1]
    rows = market_rows_from_scored(frame)
    assert [r[6] for r in rows] == [1, 0, 0, 1]


def test_a_row_with_neither_label_is_dropped():
    frame = _scored()
    frame['outcome'] = [1, np.nan, 1, 0]
    frame['venue_outcome'] = [np.nan, np.nan, 0, np.nan]
    assert len(market_rows_from_scored(frame)) == 3
