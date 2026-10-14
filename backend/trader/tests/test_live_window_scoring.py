"""Tests that `score_live` scores the window it is inside, from bars up to the decision only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.dataset import Dataset, DatasetError, fit_fold, score_live
from tests.conftest import make_bars

FAST = Config(n_estimators=40, early_stopping_rounds=10, n_folds=3,
              seasonality_min_days=5)


@pytest.fixture(scope='module')
def fitted():
    bars = make_bars(days=40, lead=0.3)
    dataset = Dataset.build(bars, FAST)
    index = dataset.window_index
    fit, _ = fit_fold(dataset, index[:int(len(index) * 0.8)], FAST)
    return bars, fit.bundle(FAST), index


def truncate(bars: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict:
    """The feed as it exists at `cutoff`: every bar that has closed, none after."""
    return {symbol: frame.loc[frame['event_time'] < cutoff].reset_index(drop=True)
            for symbol, frame in bars.items()}


@pytest.mark.parametrize('offset', (3, 6, 9, 12))
def test_the_window_being_decided_can_be_scored(fitted, offset):
    """A live cycle at +offset has bars up to offset-1 and gets a row per symbol."""
    bars, bundle, index = fitted
    window = index[-6]
    feed = truncate(bars, window + pd.Timedelta(minutes=offset))

    scored = score_live(feed, bundle, FAST, window_open=window, offset=offset)

    assert len(scored) == len(bars), 'one row per symbol'
    assert (scored['window_open'] == window).all()
    assert (scored['offset'] == offset).all()
    for column in ('strike', 'last_price', 'displacement', 'sigma_remaining',
                   'baseline_probability'):
        assert np.isfinite(scored[column]).all(), f'{column} is not finite'


@pytest.mark.parametrize('offset', (3, 6, 9, 12))
def test_the_decision_reads_the_close_of_the_bar_before_it(fitted, offset):
    """`last_price` is exactly `close(window_open + offset - 1)`; a one-minute slip
    reads like skill.
    """
    bars, bundle, index = fitted
    window = index[-6]
    feed = truncate(bars, window + pd.Timedelta(minutes=offset))
    scored = score_live(feed, bundle, FAST, window_open=window, offset=offset)

    for row in scored.itertuples():
        frame = bars[row.symbol]
        expected = frame.loc[
            frame['event_time'] == window + pd.Timedelta(minutes=offset - 1),
            'close'].iloc[0]
        assert row.last_price == pytest.approx(float(expected), abs=1e-9), (
            f'{row.symbol} at +{offset}m read {row.last_price}, not the close of '
            f'the bar covering [+{offset-1}m, +{offset}m)'
        )


def test_an_unsettled_window_reports_no_outcome(fitted):
    """An unsettled window reports NaN outcome, not zero (which would book an open bet
    as a loss).
    """
    bars, bundle, index = fitted
    window = index[-6]
    feed = truncate(bars, window + pd.Timedelta(minutes=3))
    scored = score_live(feed, bundle, FAST, window_open=window, offset=3)

    assert scored['outcome'].isna().all()
    assert scored['settle_price'].isna().all()
    assert scored['settle_return'].isna().all()
    # The strike is real: it is the previous window's settlement mean, already known.
    assert np.isfinite(scored['strike']).all()


def test_a_feed_that_has_not_reached_the_decision_minute_abstains(fitted):
    """A feed that has not reached the decision minute raises instead of forward-filling
    a price.
    """
    bars, bundle, index = fitted
    window = index[-6]
    # Bars only to +3m, but asked to decide at +12m.
    feed = truncate(bars, window + pd.Timedelta(minutes=3))
    with pytest.raises(DatasetError):
        score_live(feed, bundle, FAST, window_open=window, offset=12)


def test_the_backtest_never_sees_an_unsettled_window(fitted):
    """The measured panel never contains an unsettled window."""
    bars, _, _ = fitted
    dataset = Dataset.build(bars, FAST)
    assert dataset.windows['outcome'].notna().all(), (
        'the measured panel contains an unsettled window'
    )
    assert dataset.windows['settle_price'].notna().all()
