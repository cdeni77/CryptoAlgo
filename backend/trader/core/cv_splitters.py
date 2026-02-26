from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class CVFold:
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_bars: int
    embargo_bars: int


def _bars_from_days(index: pd.DatetimeIndex, days: Optional[float]) -> int:
    if not days or days <= 0:
        return 0
    if len(index) < 2:
        return int(days)
    step = index.to_series().diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return int(days)
    return int(math.ceil(pd.Timedelta(days=float(days)) / step))


def _resolve_bars(index: pd.DatetimeIndex, *, days: Optional[int] = None, bars: Optional[int] = None) -> int:
    if bars is not None:
        return max(0, int(bars))
    return _bars_from_days(index, days)


def create_walk_forward_splits(index: pd.DatetimeIndex, n_folds: int, min_train_days: int, purge_days: int) -> List[CVFold]:
    start, end = index.min(), index.max()
    total_days = (end - start).days
    min_test_days = 60
    if total_days < min_train_days + min_test_days:
        boundary = start + pd.Timedelta(days=int(total_days * 0.7))
        train_idx = index[index < boundary]
        test_idx = index[index >= boundary]
        return [CVFold(train_idx=train_idx, test_idx=test_idx, train_end=boundary, test_start=boundary, test_end=end, purge_bars=0, embargo_bars=0)]

    n_folds = min(n_folds, max(1, (total_days - min_train_days) // min_test_days))
    purge_delta = pd.Timedelta(days=max(0, purge_days))
    test_zone_start = start + pd.Timedelta(days=min_train_days + max(0, purge_days))
    if test_zone_start >= end:
        boundary = start + pd.Timedelta(days=int(total_days * 0.7))
        train_idx = index[index < boundary]
        test_idx = index[index >= boundary]
        return [CVFold(train_idx=train_idx, test_idx=test_idx, train_end=boundary, test_start=boundary, test_end=end, purge_bars=0, embargo_bars=0)]

    fold_days = max(1, (end - test_zone_start).days // n_folds)
    folds: List[CVFold] = []
    for i in range(n_folds):
        test_start = test_zone_start + pd.Timedelta(days=i * fold_days)
        test_end = test_start + pd.Timedelta(days=fold_days) if i < n_folds - 1 else end
        train_end = test_start - purge_delta
        train_idx = index[index < train_end]
        test_idx = index[(index >= test_start) & (index <= test_end)]
        folds.append(
            CVFold(
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_bars=0,
                embargo_bars=0,
            )
        )
    return folds


def create_purged_embargo_splits(
    index: pd.DatetimeIndex,
    n_folds: int,
    min_train_days: int,
    purge_days: Optional[int] = None,
    purge_bars: Optional[int] = None,
    embargo_days: Optional[int] = None,
    embargo_bars: Optional[int] = None,
    embargo_frac: Optional[float] = None,
) -> List[CVFold]:
    wf_folds = create_walk_forward_splits(index, n_folds=n_folds, min_train_days=min_train_days, purge_days=0)
    purge_n = _resolve_bars(index, days=purge_days, bars=purge_bars)
    embargo_n = _resolve_bars(index, days=embargo_days, bars=embargo_bars)
    folds: List[CVFold] = []

    for fold in wf_folds:
        test_idx = fold.test_idx
        if len(test_idx) == 0:
            continue
        if embargo_frac is not None and embargo_frac > 0:
            embargo_n = max(embargo_n, int(len(test_idx) * float(embargo_frac)))

        first_test = index.get_loc(test_idx[0])
        last_test = index.get_loc(test_idx[-1])
        left_cut = max(0, first_test - purge_n)
        right_cut = min(len(index), last_test + 1 + embargo_n)

        train_idx = index[:left_cut].append(index[right_cut:])
        # keep a minimum anchor window to avoid pathological tiny train sets
        min_anchor = index.min() + pd.Timedelta(days=min_train_days)
        train_idx = train_idx[train_idx < min_anchor].append(train_idx[train_idx >= min_anchor])

        folds.append(
            CVFold(
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=train_idx.max() if len(train_idx) else test_idx.min(),
                test_start=test_idx.min(),
                test_end=test_idx.max(),
                purge_bars=purge_n,
                embargo_bars=embargo_n,
            )
        )
    return folds
