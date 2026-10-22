"""The strike and the settlement value must use the venue's own mean.

Both ends of the target are `bar_mean`. The venue settles on the simple
average of sixty seconds of its index, and the CF Benchmarks feed publishes
exactly that as `avg_60s_data`, stored as `mean_price`. Computing
`(O+H+L+C)/4` when the real number is in the next column throws away the
reason that feed exists, and does it at both ends of the label at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.windows import bar_mean


def _frame(mean=None):
    # Deliberately NOT a ramp: on a monotonic path the OHLC average equals
    # the true mean by symmetry, so a ramp would pass either way.
    d = {'open': [100.0], 'high': [160.0], 'low': [99.0], 'close': [101.0]}
    if mean is not None:
        d['mean_price'] = [mean]
    return pd.DataFrame(d)


def test_without_the_column_it_is_the_ohlc_proxy():
    assert bar_mean(_frame()).iloc[0] == pytest.approx((100 + 160 + 99 + 101) / 4)


def test_with_the_column_it_is_the_venue_number():
    got = bar_mean(_frame(mean=101.5)).iloc[0]
    assert got == pytest.approx(101.5)
    assert got != pytest.approx((100 + 160 + 99 + 101) / 4)


def test_a_missing_value_falls_back_row_by_row():
    """A partially covered span must not be dropped or silently zeroed."""
    f = pd.DataFrame({'open': [100.0, 100.0], 'high': [160.0, 160.0],
                      'low': [99.0, 99.0], 'close': [101.0, 101.0],
                      'mean_price': [101.5, np.nan]})
    got = bar_mean(f)
    assert got.iloc[0] == pytest.approx(101.5)
    assert got.iloc[1] == pytest.approx((100 + 160 + 99 + 101) / 4)


def test_the_index_writers_populate_the_column_this_reads():
    """Two names that must agree, in files that do not import each other."""
    from core.datastore import SCHEMAS
    from core.index_bars import rows_for_store
    assert 'mean_price' in SCHEMAS['minute_bars']
    rows = rows_for_store([{'symbol': 'BTC-USD', 'minute': 1792643040,
                            'open': 1.0, 'high': 2.0, 'low': 0.5, 'close': 1.5,
                            'mean_price': 1.25, 'mean_source': 'venue_avg_60s',
                            'n_prints': 60, 'complete': True}],
                          available_time=pd.Timestamp(1792643100, unit='s',
                                                      tz='UTC'))
    assert 'mean_price' in rows[0]
