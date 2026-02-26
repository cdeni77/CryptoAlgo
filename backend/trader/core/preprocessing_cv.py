from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pandas as pd


@dataclass
class FoldPreprocessor:
    scaler_factory: Optional[Callable[[], object]] = None
    selector_factory: Optional[Callable[[], object]] = None

    def __post_init__(self) -> None:
        self.scaler = self.scaler_factory() if self.scaler_factory else None
        self.selector = self.selector_factory() if self.selector_factory else None

    def fit_transform(self, x_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> pd.DataFrame:
        x_out = x_train
        if self.selector is not None:
            self.selector.fit(x_out, y_train)
            x_out = pd.DataFrame(self.selector.transform(x_out), index=x_out.index)
        if self.scaler is not None:
            self.scaler.fit(x_out)
            x_out = pd.DataFrame(self.scaler.transform(x_out), index=x_out.index)
        return x_out

    def transform(self, x_data: pd.DataFrame) -> pd.DataFrame:
        x_out = x_data
        if self.selector is not None:
            x_out = pd.DataFrame(self.selector.transform(x_out), index=x_out.index)
        if self.scaler is not None:
            x_out = pd.DataFrame(self.scaler.transform(x_out), index=x_out.index)
        return x_out


def fit_transform_fold(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    preprocessor: Optional[FoldPreprocessor] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, FoldPreprocessor]:
    fold_preprocessor = preprocessor or FoldPreprocessor()
    x_train_t = fold_preprocessor.fit_transform(x_train, y_train=y_train)
    x_test_t = fold_preprocessor.transform(x_test)
    return x_train_t, x_test_t, fold_preprocessor
