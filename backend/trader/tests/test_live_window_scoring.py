"""`score_live` must score the window it is inside, not the one before it.

This is the defect that meant live and paper trading had never worked. `score_live`
asks `build_windows` for the window currently being decided; `build_windows`
trimmed the grid to whole windows (`// window`) and read the settlement from
`means[:, window - 1]`, so a window three minutes old was absent twice over. The
slice came back empty, `DatasetError` was raised, and `scripts/live.py`'s loop
catches only `KeyboardInterrupt` — so the process exited on its first cycle,
every cycle, under `restart: unless-stopped`.

The two tests that named this case could not fail for it: both
`test_score_live_*` in `test_features_and_model.py` pass `window_index[-3]`, a
window that settled long ago, with bars running well past it. One of them is
literally called `test_score_live_reports_no_outcome_for_an_unsettled_window`.

So these feed exactly what a live cycle has: bars ending at the decision minute
and nothing after.
"""

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
    """A live cycle at +offset has bars up to offset-1 and must get a row."""
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
    """`last_price` is `close(window_open + offset - 1)`, exactly.

    The whole barrier rests on this one number, and a one-minute slip is 7% of a
    fifteen-minute question — which reads exactly like skill.
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
    """The label must be absent, not zero.

    `(nan >= x)` is False, so an in-progress window filed as `outcome=0` is an
    unresolved bet recorded as a loss.
    """
    bars, bundle, index = fitted
    window = index[-6]
    feed = truncate(bars, window + pd.Timedelta(minutes=3))
    scored = score_live(feed, bundle, FAST, window_open=window, offset=3)

    assert scored['outcome'].isna().all()
    assert scored['settle_price'].isna().all()
    assert scored['settle_return'].isna().all()
    # The strike, by contrast, is real: it is the previous window's settlement
    # average, which has already happened.
    assert np.isfinite(scored['strike']).all()


def test_a_feed_that_has_not_reached_the_decision_minute_abstains(fitted):
    """Withheld, not forward-filled.

    The interior forward-fill is right for a minute in which nothing traded, but
    it cannot tell that from a minute not yet fetched — and ffilling a stale feed
    invents a `last_price` and hands it to the barrier as a measurement. The
    honest answer is to abstain, so this must raise rather than return a row.
    """
    bars, bundle, index = fitted
    window = index[-6]
    # Bars only to +3m, but asked to decide at +12m.
    feed = truncate(bars, window + pd.Timedelta(minutes=3))
    with pytest.raises(DatasetError):
        score_live(feed, bundle, FAST, window_open=window, offset=12)


def test_the_backtest_never_sees_an_unsettled_window(fitted):
    """`include_unsettled` must stay off everywhere the label is used.

    An unsettled row carries `outcome=NaN`. In a training or scoring frame that
    is either a crash or, worse, a row silently dropped from a metric's
    denominator.
    """
    bars, _, _ = fitted
    dataset = Dataset.build(bars, FAST)
    assert dataset.windows['outcome'].notna().all(), (
        'the measured panel contains an unsettled window'
    )
    assert dataset.windows['settle_price'].notna().all()
