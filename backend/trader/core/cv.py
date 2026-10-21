"""Cross-validation at the window level, because the row is not the unit.

Several decision offsets share one settlement, so every split is on `window_open`,
and row frames are selected by membership, not sliced.

The embargo is a day, not fifteen minutes: a training row just after a test
block computes `log_rv_1440` from bars inside that block.

Folds are expanding: each fold trains on everything before its test block.
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
    min_test_windows: int = 200,
    scheme: str = 'calendar',
    fold_block_days: float = 21.0,
    train_days: Optional[float] = None,
) -> list[WindowFold]:
    """Split distinct window opens into expanding folds with a purged gap.

    Fold *i* trains on blocks before its test block, minus any window inside
    the embargo of the test start. The first block is never a test block.

    `scheme`:
    - `'count'`: blocks of equal window counts. Boundaries move as data arrives.
    - `'calendar'`: blocks of `fold_block_days`, anchored to the first window.
      Boundaries stay put as data arrives, so fold results compare across runs.
      Folds then carry equal time at unequal sample size.
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
        # Equal spans of time, anchored to the first window. The block length
        # is a parameter, so existing boundaries never move when data is appended.
        block = pd.Timedelta(days=float(fold_block_days))
        origin = index.min()
        span = index.max() - origin
        # Complete blocks only. The trailing stub is not a test block: its width
        # would change every run. Up to `fold_block_days` of the newest data
        # stays out of the test set until its block completes.
        n_blocks = max(int(span // block), 1)
        edges = [origin + k * block for k in range(n_blocks + 1)]
        # Every complete block, not only the newest ones. Folds accumulate as
        # history grows, so gates that count folds must be proportions.
        cuts = edges
        if len(cuts) < 3:
            # Fewer than two blocks: subdivide the span instead, and warn,
            # because these boundaries move as data arrives.
            logger.warning(
                'span %s is shorter than two %.0f-day blocks, so folds are '
                'subdivided proportionally and their boundaries WILL move as '
                'data arrives. Lower fold_block_days to anchor them.',
                span, float(fold_block_days))
            cuts = list(pd.date_range(index.min(), index.max(),
                                      periods=n_folds + 2))
            cuts[-1] = cuts[-1] + pd.Timedelta(minutes=1)
    embargo = pd.Timedelta(minutes=embargo_minutes)
    folds: list[WindowFold] = []
    # The count scheme yields exactly `n_folds` folds. The calendar scheme tests
    # every complete block after the first: `n_folds` does not cap it, because an
    # explicit 6 cannot be told from the default and capping halves coverage.
    # To test fewer calendar blocks, widen `fold_block_days`.
    available = max(len(cuts) - 2, 0)
    wanted = available if scheme == 'calendar' else min(int(n_folds), available)
    first = max(available - wanted, 0)
    for i in range(first, available):
        test = index[(index >= cuts[i + 1]) & (index < cuts[i + 2])]
        if len(test) == 0:
            continue
        # A block thin on data is not a fold, even when it is full on time.
        if len(test) < min_test_windows:
            logger.warning(
                'fold %d: %d test windows in [%s .. %s) is under the %d '
                'minimum, skipped — the block is full on time and thin on '
                'data', i, len(test), cuts[i + 1], cuts[i + 2],
                min_test_windows)
            continue
        train_pool = index[index < cuts[i + 1]]
        train = train_pool[train_pool < test[0] - embargo]
        # Rolling training window, when asked for. Cut from the embargo's far
        # edge, so the window holds `train_days` of usable history.
        # Expanding is the default because the weekly retrain deploys it;
        # change both together.
        if train_days is not None and len(train):
            floor = (test[0] - embargo) - pd.Timedelta(days=float(train_days))
            train = train[train >= floor]
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


def folds_for_config(window_index, config) -> list[WindowFold]:
    """`purged_walk_forward` with every geometry field read from the Config.

    The single place folds are built from a Config, so no caller can pass half
    of the geometry and fall back to this module's defaults for the rest.
    """
    return purged_walk_forward(
        window_index,
        n_folds=config.n_folds,
        embargo_minutes=config.embargo_minutes,
        scheme=getattr(config, 'fold_scheme', 'calendar'),
        fold_block_days=getattr(config, 'fold_block_days', 21.0),
        min_test_windows=getattr(config, 'min_test_windows', 200),
        min_train_windows=getattr(config, 'min_train_windows', 500),
        train_days=getattr(config, 'fold_train_days', None))


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

    Membership, not a time slice: a slice on `decision_time` would split one
    window's offsets across train and test.
    """
    return table['window_open'].isin(window_opens)


def effective_observations(table: pd.DataFrame) -> int:
    """Distinct windows, not rows.

    Offsets share a window, so a row count overstates the sample.
    """
    if table.empty:
        return 0
    return int(table.drop_duplicates(['symbol', 'window_open']).shape[0])


def recency_weights(
    window_opens: pd.Series,
    half_life_days: Optional[float],
) -> Optional[np.ndarray]:
    """Exponential decay by age, or None when disabled (the default)."""
    if not half_life_days:
        return None
    times = pd.DatetimeIndex(window_opens)
    age_days = (times.max() - times).total_seconds() / 86400.0
    return np.power(0.5, age_days / float(half_life_days))
