"""Cross-validation for overlapping financial labels.

Ordinary k-fold is invalid here for two reasons, and both need fixing or the
validation score is optimistic:

**Labels overlap.** A triple-barrier label opened at bar t resolves somewhere in
[t, t+h]. Two labels opened an hour apart share almost all of their outcome, so a
train row and a test row an hour apart are near-duplicates. *Purging* removes
train rows whose label window reaches into the test set; *embargo* removes train
rows immediately after it, where serial correlation still leaks.

**Overlapping labels are not independent observations.** With a 72-hour horizon
on hourly bars, 2,880 rows carry roughly 40 independent outcomes. Weighting each
row by its average uniqueness stops the model from treating 72 views of one event
as 72 events.

Two splitters are provided. `purged_walk_forward` gives the single chronological
path a live system actually experiences. `combinatorial_purged_splits` gives many
paths over the same history, so a Sharpe becomes a distribution rather than one
number — which is the difference between "this worked" and "this works".

A caveat on what CPCV does *not* buy you. The 11 paths reuse the same history, so
they are 11 correlated views of one sample, not 11 independent samples. The
spread across them measures sensitivity to how the data was cut — worth knowing,
and the thing a single walk-forward hides — but it does not add evidence. On a
120-day window at a 72-hour horizon the whole sample carries about 40 independent
observations, and an individual CPCV test block carries about 7. Widening the
sample means more instruments, not more folds: see `effective_sample_size`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

# López de Prado's default geometry: 12 groups, 2 held out per split.
# C(12,2) = 66 splits recombining into C(11,1) = 11 distinct backtest paths.
DEFAULT_N_GROUPS = 12
DEFAULT_TEST_GROUPS = 2


@dataclass(frozen=True)
class CVFold:
    """One train/test split with its leakage controls recorded.

    `purge_bars` and `embargo_bars` are carried rather than assumed so a result
    can state the controls it was produced under.
    """

    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    purge_bars: int = 0
    embargo_bars: int = 0
    test_groups: tuple[int, ...] = ()

    @property
    def train_end(self) -> Optional[pd.Timestamp]:
        return self.train_idx.max() if len(self.train_idx) else None

    @property
    def test_start(self) -> Optional[pd.Timestamp]:
        return self.test_idx.min() if len(self.test_idx) else None

    @property
    def test_end(self) -> Optional[pd.Timestamp]:
        return self.test_idx.max() if len(self.test_idx) else None

    def __len__(self) -> int:
        return len(self.test_idx)


def bars_from_hours(index: pd.DatetimeIndex, hours: float) -> int:
    """Convert a duration to a bar count using the index's own spacing."""
    if hours <= 0 or len(index) < 2:
        return max(0, int(hours))
    step = pd.Series(index).diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return max(0, int(hours))
    return int(math.ceil(pd.Timedelta(hours=float(hours)) / step))


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def _purge_train(
    index: pd.DatetimeIndex,
    test_positions: np.ndarray,
    *,
    purge_bars: int,
    embargo_bars: int,
) -> np.ndarray:
    """Positions usable for training given the test positions.

    Drops `purge_bars` before each contiguous test block — those rows have
    labels that resolve inside the test window — and `embargo_bars` after it.
    Works on position arrays so a test set can be several disjoint blocks, which
    is exactly what CPCV produces.
    """
    n = len(index)
    blocked = np.zeros(n, dtype=bool)
    blocked[test_positions] = True

    for start, stop in _contiguous_blocks(test_positions):
        blocked[max(0, start - purge_bars):start] = True
        blocked[stop + 1:min(n, stop + 1 + embargo_bars)] = True

    return np.flatnonzero(~blocked)


def _contiguous_blocks(positions: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, stop) runs in a sorted position array."""
    if positions.size == 0:
        return []
    sorted_positions = np.sort(positions)
    breaks = np.flatnonzero(np.diff(sorted_positions) > 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks, [sorted_positions.size - 1]])
    return [(int(sorted_positions[a]), int(sorted_positions[b])) for a, b in zip(starts, stops)]


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def purged_walk_forward(
    index: pd.DatetimeIndex,
    *,
    n_folds: int = 6,
    min_train_bars: int = 720,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    expanding: bool = True,
) -> list[CVFold]:
    """Chronological splits: train on the past, test on the next block.

    This is the honest simulation of a live system, and the only splitter whose
    result can be compared to paper trading. Its weakness is that it produces one
    path, so it cannot say how much of the result was the particular cut —
    `combinatorial_purged_splits` answers that.
    """
    index = pd.DatetimeIndex(index).sort_values()
    n = len(index)
    if n <= min_train_bars + n_folds:
        return []

    testable = n - min_train_bars
    fold_size = testable // n_folds
    if fold_size < 1:
        return []

    folds: list[CVFold] = []
    for i in range(n_folds):
        test_start = min_train_bars + i * fold_size
        test_stop = n - 1 if i == n_folds - 1 else min_train_bars + (i + 1) * fold_size - 1
        test_positions = np.arange(test_start, test_stop + 1)
        if test_positions.size == 0:
            continue

        train_positions = _purge_train(
            index, test_positions, purge_bars=purge_bars, embargo_bars=embargo_bars
        )
        # Walk-forward never trains on the future, whatever the purge leaves.
        train_positions = train_positions[train_positions < test_start]
        if expanding is False:
            train_positions = train_positions[train_positions >= test_start - min_train_bars]
        if train_positions.size < min_train_bars // 2:
            continue

        folds.append(CVFold(
            train_idx=index[train_positions],
            test_idx=index[test_positions],
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            test_groups=(i,),
        ))
    return folds


# ---------------------------------------------------------------------------
# Combinatorial purged cross-validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCVLayout:
    """The geometry of a CPCV run, so a result can report what produced it."""

    n_groups: int
    test_groups: int
    n_splits: int
    n_paths: int

    @classmethod
    def for_(cls, n_groups: int, test_groups: int) -> "CPCVLayout":
        n_splits = math.comb(n_groups, test_groups)
        # Each group is tested in C(N-1, k-1) splits, and that is also how many
        # complete out-of-sample paths the splits recombine into.
        n_paths = math.comb(n_groups - 1, test_groups - 1)
        return cls(n_groups, test_groups, n_splits, n_paths)

    def __str__(self) -> str:
        return (
            f"{self.n_groups} groups, {self.test_groups} held out -> "
            f"{self.n_splits} splits, {self.n_paths} paths"
        )


def group_positions(index: pd.DatetimeIndex, n_groups: int) -> list[np.ndarray]:
    """Split the timeline into contiguous, near-equal groups.

    Contiguous rather than interleaved: a group must be a period of time, or
    purging cannot remove the rows whose labels bleed into it.
    """
    n = len(index)
    if n_groups < 2 or n < n_groups:
        return []
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(n_groups)]


def combinatorial_purged_splits(
    index: pd.DatetimeIndex,
    *,
    n_groups: int = DEFAULT_N_GROUPS,
    test_groups: int = DEFAULT_TEST_GROUPS,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    min_train_bars: int = 240,
) -> list[CVFold]:
    """Every combination of `test_groups` held out, each purged and embargoed.

    With the defaults this is 66 splits over 12 groups. On its own that is just a
    lot of folds; `assemble_paths` is what turns them into complete backtest
    paths.
    """
    index = pd.DatetimeIndex(index).sort_values()
    groups = group_positions(index, n_groups)
    if not groups:
        return []

    folds: list[CVFold] = []
    for combo in combinations(range(n_groups), test_groups):
        test_positions = np.sort(np.concatenate([groups[g] for g in combo]))
        train_positions = _purge_train(
            index, test_positions, purge_bars=purge_bars, embargo_bars=embargo_bars
        )
        if train_positions.size < min_train_bars:
            continue
        folds.append(CVFold(
            train_idx=index[train_positions],
            test_idx=index[test_positions],
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            test_groups=combo,
        ))
    return folds


def assemble_paths(
    folds: Sequence[CVFold],
    *,
    n_groups: int = DEFAULT_N_GROUPS,
) -> list[list[tuple[int, CVFold]]]:
    """Recombine splits into complete out-of-sample paths.

    Each group is tested in several splits. Taking the j-th of those for every
    group gives one path that covers the whole timeline exactly once, out of
    sample throughout. Doing that for every j gives `CPCVLayout.n_paths` distinct
    histories the strategy could have lived through — and therefore a
    distribution of Sharpe instead of a point estimate.

    Returns one list of `(group, fold)` pairs per path.
    """
    by_group: dict[int, list[CVFold]] = {g: [] for g in range(n_groups)}
    for fold in folds:
        for group in fold.test_groups:
            by_group[group].append(fold)

    populated = {g: f for g, f in by_group.items() if f}
    if not populated:
        return []

    n_paths = min(len(f) for f in populated.values())
    paths: list[list[tuple[int, CVFold]]] = []
    for j in range(n_paths):
        paths.append([(g, folds_for_g[j]) for g, folds_for_g in sorted(populated.items())])
    return paths


def path_test_index(path: Sequence[tuple[int, CVFold]], index: pd.DatetimeIndex,
                    n_groups: int = DEFAULT_N_GROUPS) -> pd.DatetimeIndex:
    """The timestamps one path covers — the whole timeline, once each."""
    groups = group_positions(pd.DatetimeIndex(index).sort_values(), n_groups)
    if not groups:
        return pd.DatetimeIndex([])
    positions = np.sort(np.concatenate([groups[g] for g, _ in path]))
    return pd.DatetimeIndex(index).sort_values()[positions]


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------


def _label_spans(n: int, horizon_bars: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inclusive (first, last) bar positions whose returns each label consumes.

    A label opened at bar i is decided by the returns over (i, i+h] — bars i+1
    through i+h. The opening bar is excluded: its return is already history when
    the position is entered. Getting this wrong by one bar makes a 1-bar horizon
    look like it halves the sample, because consecutive labels appear to share
    their opening bar when in fact they share no returns at all.

    Also returns which labels are *resolvable*. A label opened within `h` bars of
    the end has no full window to resolve into, so it is not an observation and
    must not be counted as one — the labeller drops those rows, and counting them
    here would overstate the evidence at the recent edge of the sample.
    """
    horizon = max(1, int(horizon_bars))
    positions = np.arange(n)
    starts = np.minimum(positions + 1, max(n - 1, 0))
    stops = np.minimum(positions + horizon, max(n - 1, 0))
    resolvable = (positions + horizon) <= (n - 1)
    return starts, stops, resolvable


def label_concurrency(index: pd.DatetimeIndex, horizon_bars: int) -> np.ndarray:
    """How many label windows consume each bar's return.

    Concurrency at bar t is the number of labels whose outcome depends on the
    return realised at t, and it is the denominator of uniqueness.
    """
    n = len(index)
    if n == 0:
        return np.zeros(0)

    starts, stops, resolvable = _label_spans(n, horizon_bars)
    # Only resolvable labels occupy their window; the rest are not observations.
    starts, stops = starts[resolvable], stops[resolvable]
    # Difference array: +1 where a span starts, -1 just past where it ends.
    deltas = np.zeros(n + 1, dtype=float)
    np.add.at(deltas, starts, 1.0)
    np.add.at(deltas, stops + 1, -1.0)
    return np.cumsum(deltas)[:n]


def average_uniqueness(index: pd.DatetimeIndex, horizon_bars: int) -> np.ndarray:
    """Mean of 1/concurrency over each label's span, in [0, 1].

    A label sharing its window with 71 others scores about 1/72. Summed over the
    sample this recovers the effective number of independent observations, which
    is the honest denominator for any significance test. A 1-bar horizon scores
    1.0 throughout: consecutive labels then share no returns. Labels too close to
    the end of the sample to resolve score 0.
    """
    n = len(index)
    if n == 0:
        return np.zeros(0)

    concurrency = label_concurrency(index, horizon_bars)
    inverse = np.divide(
        1.0, concurrency, out=np.zeros_like(concurrency), where=concurrency > 0
    )
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])

    starts, stops, resolvable = _label_spans(n, horizon_bars)
    spans = np.maximum(stops - starts + 1, 1).astype(float)
    uniqueness = (cumulative[stops + 1] - cumulative[starts]) / spans
    # A label with no full forward window carries no information; zero weight
    # keeps it out of both the training weights and the effective sample count.
    return np.where(resolvable, uniqueness, 0.0)


def effective_sample_size(index: pd.DatetimeIndex, horizon_bars: int) -> float:
    """Independent-observation count implied by label overlap.

    The number to quote when reporting how much evidence a fold contains. For
    hourly bars and a 72-hour horizon it is roughly `len(index) / 72`, not
    `len(index)` — the distinction between forty observations and three thousand.
    Unresolvable labels at the end of the sample contribute nothing.
    """
    return float(average_uniqueness(index, horizon_bars).sum())


def recency_weights(index: pd.DatetimeIndex, half_life_days: float) -> np.ndarray:
    """Exponential decay toward the present, normalised to mean 1."""
    n = len(index)
    if n == 0:
        return np.zeros(0)
    if half_life_days <= 0:
        return np.ones(n)

    age_days = (index.max() - index).total_seconds() / 86_400.0
    weights = np.power(0.5, np.asarray(age_days) / float(half_life_days))
    mean = weights.mean()
    return weights / mean if mean > 0 else np.ones(n)


def sample_weights(
    index: pd.DatetimeIndex,
    *,
    horizon_bars: int,
    half_life_days: float = 0.0,
) -> np.ndarray:
    """Per-row training weights: uniqueness, optionally decayed by age.

    Normalised to mean 1 so a change in weighting scheme does not silently
    rescale the model's effective learning rate.
    """
    weights = average_uniqueness(index, horizon_bars)
    if half_life_days > 0:
        weights = weights * recency_weights(index, half_life_days)
    mean = weights.mean() if weights.size else 0.0
    return weights / mean if mean > 0 else np.ones(len(index))


# ---------------------------------------------------------------------------
# Per-fold preprocessing
# ---------------------------------------------------------------------------


@dataclass
class FoldPreprocessor:
    """Scaling and selection fitted on a fold's training rows only.

    Fitting a scaler on the whole sample leaks the test set's mean and variance
    into training. The effect is small per feature and compounds across 77 of
    them, and it is invisible in the fold score — which is precisely why it has
    to be structural rather than remembered.
    """

    scaler_factory: Optional[Callable[[], Any]] = None
    selector_factory: Optional[Callable[[], Any]] = None

    def __post_init__(self) -> None:
        self.scaler = self.scaler_factory() if self.scaler_factory else None
        self.selector = self.selector_factory() if self.selector_factory else None

    def fit_transform(self, x_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> pd.DataFrame:
        out = x_train
        if self.selector is not None:
            self.selector.fit(out, y_train)
            out = pd.DataFrame(self.selector.transform(out), index=out.index)
        if self.scaler is not None:
            self.scaler.fit(out)
            out = pd.DataFrame(self.scaler.transform(out), index=out.index, columns=out.columns)
        return out

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        out = x_data
        if self.selector is not None:
            out = pd.DataFrame(self.selector.transform(out), index=out.index)
        if self.scaler is not None:
            out = pd.DataFrame(self.scaler.transform(out), index=out.index, columns=out.columns)
        return out


def preprocess_fold(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    preprocessor: Optional[FoldPreprocessor] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FoldPreprocessor]:
    """Fit on train, apply to both. Returns the fitted preprocessor for reuse."""
    fold = preprocessor or FoldPreprocessor()
    return fold.fit_transform(x_train, y_train=y_train), fold.transform(x_test), fold


# ---------------------------------------------------------------------------
# Leakage assertion
# ---------------------------------------------------------------------------


def assert_no_leakage(fold: CVFold, *, horizon_bars: int) -> None:
    """Raise if any training label's window reaches into the test set.

    Cheap enough to call on every fold. A purge that is shorter than the label
    horizon is the single easiest way to produce a validation score that cannot
    be reproduced live, and it is silent.
    """
    if not len(fold.train_idx) or not len(fold.test_idx):
        return

    step = pd.Series(fold.test_idx).diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return
    horizon = step * max(1, int(horizon_bars))

    # Blocks derived from gaps in the test index itself. `np.arange` is
    # contiguous by construction, so `_contiguous_blocks` on it always returned a
    # single block spanning the whole set — meaning only the FIRST block's start
    # was ever checked. `combinatorial_purged_splits` is the splitter that
    # produces multi-block test sets, so that is exactly the case this guard
    # exists for: a fold purged around block 1 but not block 2 passed.
    gaps = pd.Series(fold.test_idx).diff()
    boundaries = [0, *np.flatnonzero((gaps > step * 1.5).to_numpy()[1:]) + 1,
                  len(fold.test_idx)]
    test_blocks = [
        (boundaries[i], boundaries[i + 1] - 1)
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    for start, stop in test_blocks:
        block_start = fold.test_idx[start]
        block_end = fold.test_idx[stop]
        # A training label opened at t resolves by t + horizon. It leaks if that
        # window overlaps this test block.
        offenders = fold.train_idx[
            (fold.train_idx < block_start)
            & (fold.train_idx + horizon >= block_start)
        ]
        if len(offenders):
            raise ValueError(
                f"{len(offenders)} training labels resolve inside the test block "
                f"{block_start}..{block_end}; purge_bars={fold.purge_bars} is "
                f"shorter than the {horizon_bars}-bar label horizon"
            )
