import numpy as np
import pandas as pd

from core.cv_splitters import create_purged_embargo_splits
from core.preprocessing_cv import FoldPreprocessor, fit_transform_fold


def _index(n: int = 400) -> pd.DatetimeIndex:
    return pd.date_range('2024-01-01', periods=n, freq='h')


def test_purged_embargo_splits_remove_train_test_overlap() -> None:
    idx = _index()
    folds = create_purged_embargo_splits(
        idx,
        n_folds=4,
        min_train_days=5,
        purge_bars=6,
        embargo_bars=8,
    )

    for fold in folds:
        assert len(fold.test_idx.intersection(fold.train_idx)) == 0
        first_test = idx.get_loc(fold.test_idx[0])
        left_block = idx[max(0, first_test - 6):first_test]
        assert len(left_block.intersection(fold.train_idx)) == 0


def test_embargo_blocks_adjacent_samples_after_test_window() -> None:
    idx = _index()
    folds = create_purged_embargo_splits(
        idx,
        n_folds=3,
        min_train_days=5,
        purge_bars=2,
        embargo_bars=12,
    )

    for fold in folds:
        last_test = idx.get_loc(fold.test_idx[-1])
        embargo_slice = idx[last_test + 1:last_test + 1 + 12]
        assert len(embargo_slice.intersection(fold.train_idx)) == 0


def test_purged_embargo_splits_support_day_based_purge_and_embargo() -> None:
    idx = _index()
    folds = create_purged_embargo_splits(
        idx,
        n_folds=3,
        min_train_days=5,
        purge_days=2,
        embargo_days=1,
    )

    for fold in folds:
        assert fold.purge_bars == 48
        assert fold.embargo_bars == 24


class MeanRecorderScaler:
    def fit(self, x: pd.DataFrame) -> None:
        self.mean_ = x.mean()

    def transform(self, x: pd.DataFrame):
        return x - self.mean_


def test_fold_preprocessing_fits_only_train_data() -> None:
    train = pd.DataFrame({'a': [0.0, 0.0, 0.0], 'b': [1.0, 1.0, 1.0]})
    test = pd.DataFrame({'a': [100.0, 100.0], 'b': [101.0, 101.0]})

    x_train_t, x_test_t, pre = fit_transform_fold(
        train,
        test,
        preprocessor=FoldPreprocessor(scaler_factory=MeanRecorderScaler),
    )

    assert np.isclose(pre.scaler.mean_['a'], 0.0)
    assert np.isclose(pre.scaler.mean_['b'], 1.0)
    # If preprocessing fit globally, centered test mean would be close to 0.
    assert float(x_test_t['a'].mean()) > 50.0
    assert float(x_test_t['b'].mean()) > 50.0
    assert np.isclose(float(x_train_t['a'].mean()), 0.0)
