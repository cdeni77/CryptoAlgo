"""`bar_mean` must survive a `mean_price` column that is not already float.

`to_numeric` leaves a nullable-integer or object column as a masked integer
array, and `.where` then tries to cast the float `(O+H+L+C)/4` fallback into
int32 and raises. That took down the whole window build, so every evaluate in
stack mode failed on a TypeError from deep inside pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.windows import bar_mean


def frame(mean_price):
    return pd.DataFrame({
        'open': [10.0, 20.0], 'high': [12.0, 22.0],
        'low': [8.0, 18.0], 'close': [10.0, 20.0],
        'mean_price': mean_price,
    })


def test_a_float_column_is_used_as_is():
    out = bar_mean(frame([11.0, 21.0]))
    assert list(out) == [11.0, 21.0]


@pytest.mark.parametrize('values', [
    pd.array([11, 21], dtype='Int64'),          # nullable integer
    pd.array([11, 21], dtype='int64'),          # plain integer
    pd.Series([11, 21], dtype='object'),        # object
])
def test_a_non_float_column_still_works(values):
    out = bar_mean(frame(values))
    assert list(out) == [11.0, 21.0]
    assert out.dtype == np.float64


def test_a_missing_value_falls_back_to_the_ohlc_mean():
    out = bar_mean(frame(pd.array([11, None], dtype='Int64')))
    assert out.iloc[0] == 11.0
    assert out.iloc[1] == 20.0, 'the (O+H+L+C)/4 of the second row'


def test_no_column_at_all_is_the_untouched_path():
    f = frame([1.0, 1.0]).drop(columns=['mean_price'])
    assert list(bar_mean(f)) == [10.0, 20.0]
