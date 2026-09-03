"""Cross-validation at the window level, because the row is not the unit.

Four decision offsets share one settlement. They are four rows with different
features and the *same* label, so a split that puts offset 3 in train and
offset 12 in test has leaked the answer — not subtly, but completely: the two
rows describe the same fifteen minutes and one of them is nine minutes closer
to knowing. Every split here is therefore on `window_open`, and the row-level
frames are selected by membership rather than sliced.

**The embargo is a day, and it is not about the label.** A fifteen-minute label
needs a fifteen-minute purge. What needs twenty-four hours is the *features*: a
training row immediately after a test block computes `log_rv_1440` from bars
inside that block. Purging for the label and forgetting the feature lookback is
the standard version of this mistake, and it leaks in the direction that
inflates measured skill.

**Folds are expanding, not rolling.** Each fold trains on everything before its
test block. That matches how the thing would actually be deployed — you never
throw away history you have — and it means train is always entirely before
test, so only the gap immediately preceding the test block needs purging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LeakageError(AssertionError):
    """A fold's train and test sets are not separated as claimed."""


@dataclass(frozen=True)
class WindowFold:
    """One expanding-window fold, addressed by window-open timestamps."""

    index: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    embargo_minutes: int

    @property
    def train_end(self) -> Optional[pd.Timestamp]:
        return self.train.max() if len(self.train) else None

    @property
    def test_start(self) -> Optional[pd.Timestamp]:
        return self.test.min() if len(self.test) else None

    @property
    def test_end(self) -> Optional[pd.Timestamp]:
        return self.test.max() if len(self.test) else None

    @property
    def gap_minutes(self) -> float:
        if self.train_end is None or self.test_start is None:
            return float('nan')
        return (self.test_start - self.train_end).total_seconds() / 60.0

    def label(self) -> str:
        if self.test_start is None:
            return f'fold {self.index}: empty'
        return (f'fold {self.index}: train {len(self.train):,}w -> '
                f'test {len(self.test):,}w '
                f'[{self.test_start:%Y-%m-%d} .. {self.test_end:%Y-%m-%d}]')


def purged_walk_forward(
    window_index: pd.DatetimeIndex,
    *,
    n_folds: int = 6,
    embargo_minutes: int = 1440,
    min_train_windows: int = 500,
    scheme: str = 'calendar',
) -> list[WindowFold]:
    """Split distinct window opens into expanding folds with a purged gap.

    The timeline is cut into `n_folds + 1` blocks; fold *i* trains on blocks
    0..i and tests on block i+1, with any training window inside the embargo of
    the test start removed. The first block is never a test block, which is what
    makes the first fold's training set non-trivial.

    **`scheme` decides what "equal" means, and it matters more than it sounds.**

    `'count'` cuts into equal numbers of windows — the original behaviour, and
    what every ledger entry before 2026-09-03 used. With data density varying 8x
    across the span (30 to 244 windows/day, as `--complete-cases` intersects
    three sources whose coverage improved through 2026), equal counts give
    calendar spans of 21 to 78 days. So the folds are not the same experiment:
    one fitted model tested across 78 days of drift is averaged with one tested
    across 21. Worse, appending data re-cuts EVERY boundary, so a fold's skill
    moves between runs when only the splitting changed — which happens on every
    run once collection is continuous.

    `'calendar'` cuts into equal spans of time. Earlier boundaries then stay put
    as data arrives, so fold-level results are comparable run to run. Measured
    on 12,909 complete-case windows it is also simply a better estimator:

        equal-count      +0.00276 +/- 0.00088  t=+3.15  6/6
        equal-calendar   +0.00312 +/- 0.00075  t=+4.18  6/6

    The trade is unavoidable: with density varying you can balance time or
    count, never both. Equal-calendar put 491 windows in one fold against 5,671
    in another, so folds carry equal weight at unequal precision — inefficient
    but unbiased, and the price of a boundary that does not move.
    """
    index = pd.DatetimeIndex(sorted(pd.DatetimeIndex(window_index).unique()))
    if len(index) < (n_folds + 1) * 2:
        raise ValueError(
            f'{len(index)} windows cannot support {n_folds} folds — need at least '
            f'{(n_folds + 1) * 2}'
        )
    if scheme not in ('calendar', 'count'):
        raise ValueError(
            f"scheme={scheme!r}; expected 'calendar' or 'count'")
    if scheme == 'count':
        cuts = [index[e] if e < len(index) else index[-1] + pd.Timedelta(1)
                for e in np.linspace(0, len(index), n_folds + 2, dtype=int)]
    else:
        # Equal spans of TIME. The last edge is nudged past the final window so
        # the closing block is inclusive, exactly as the count form's slice is.
        cuts = list(pd.date_range(index.min(), index.max(), periods=n_folds + 2))
        cuts[-1] = cuts[-1] + pd.Timedelta(minutes=1)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    folds: list[WindowFold] = []
    for i in range(n_folds):
        test = index[(index >= cuts[i + 1]) & (index < cuts[i + 2])]
        if len(test) == 0:
            continue
        train_pool = index[index < cuts[i + 1]]
        train = train_pool[train_pool < test[0] - embargo]
        if len(train) < min_train_windows:
            logger.warning(
                'fold %d: %d training windows is under the %d minimum, skipped',
                i, len(train), min_train_windows,
            )
            continue
        folds.append(WindowFold(index=i, train=train, test=test,
                                embargo_minutes=embargo_minutes))
    if not folds:
        raise ValueError('no fold had enough training windows')
    return folds


def assert_no_leakage(fold: WindowFold) -> None:
    """Refuse a fold whose sets overlap or whose embargo is not honoured."""
    overlap = fold.train.intersection(fold.test)
    if len(overlap):
        raise LeakageError(
            f'fold {fold.index}: {len(overlap)} window opens are in both train and test'
        )
    if fold.train_end is None or fold.test_start is None:
        return
    if fold.train_end >= fold.test_start:
        raise LeakageError(
            f'fold {fold.index}: training ends {fold.train_end} at or after the '
            f'test start {fold.test_start}'
        )
    if fold.gap_minutes < fold.embargo_minutes:
        raise LeakageError(
            f'fold {fold.index}: gap of {fold.gap_minutes:.0f} minutes is under the '
            f'{fold.embargo_minutes}-minute embargo — a training row this close '
            f'computes its 1440-minute features from test-period bars'
        )


def rows_for(table: pd.DataFrame, window_opens: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask selecting every row belonging to these windows.

    Membership, not a timestamp comparison. A `>=`/`<` slice on `decision_time`
    would split a window across the boundary — the offset-3 row inside train and
    the offset-12 row inside test — which is the exact leak this module exists
    to prevent.
    """
    return table['window_open'].isin(window_opens)


def effective_observations(table: pd.DataFrame) -> int:
    """Distinct windows, not rows.

    Four offsets per window means a row count overstates the sample fourfold,
    and a standard error computed from it is half what it should be. Every
    reported error bar in this system divides by this.
    """
    if table.empty:
        return 0
    return int(table.drop_duplicates(['symbol', 'window_open']).shape[0])


def recency_weights(
    window_opens: pd.Series,
    half_life_days: Optional[float],
) -> Optional[np.ndarray]:
    """Exponential decay by age, or None when disabled.

    Off by default here, unlike the previous incarnation of this project, where
    a 50-day half-life meant five years of history bought one effective
    observation over one year. At 15-minute windows the sample is large enough
    that decay costs more than the non-stationarity it buys — but it is exposed
    so a run can disagree, and `scripts/evaluate.py` reports the sweep.
    """
    if not half_life_days:
        return None
    times = pd.DatetimeIndex(window_opens)
    age_days = (times.max() - times).total_seconds() / 86400.0
    return np.power(0.5, age_days / float(half_life_days))
