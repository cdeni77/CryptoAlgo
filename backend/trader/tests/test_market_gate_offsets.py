"""`model_minus_market` scored every offset while the loop trades one.

The gate asks whether the model beats the PRICE — the thing this whole stack
exists not to fail. But it pooled all four decision offsets, and
`--entry-offsets 12` means only +12m can open a position. Measured on 5,622 live
rows:

    offset   model - mkt      t     days+
    +3m       -0.00259     -1.11     2/6
    +6m       -0.00644     -1.63     3/6
    +9m       +0.00068     +0.18     3/6
    +12m      +0.00550     +1.09     5/6      <- the only offset that trades
    pooled    -0.00072     -0.26     2/6      <- what the gate read

So the gate has now rejected two candidates for losing to the market at offsets
they never trade, while the offset they DO trade was positive and the best of
the four. That is the same defect as the entry-offsets bug — a measurement
describing a policy nobody runs — and it will keep rejecting good candidates
until it stops.

When `entry_offsets` is None every offset can enter, so pooling is correct and
nothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.metrics import market_rows_from_scored


def _frame():
    rows = []
    for offset, model_p in ((3, 0.20), (6, 0.20), (9, 0.20), (12, 0.80)):
        rows.append({
            'symbol': 'BTC-USD',
            'window_open': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
            'offset': offset,
            'market_probability': 0.50,
            'model_probability': model_p,
            'baseline_probability': 0.50,
            'outcome': 1.0,
            'quote_age_seconds': 2.0,
        })
    return pd.DataFrame(rows)


def test_only_the_offsets_that_can_trade_are_scored():
    rows = market_rows_from_scored(_frame(), entry_offsets=(12,))
    assert len(rows) == 1
    assert rows[0][2] == 12


def test_every_offset_is_scored_when_entries_are_unrestricted():
    """None means any offset may enter, so pooling is the honest measure."""
    assert len(market_rows_from_scored(_frame(), entry_offsets=None)) == 4


def test_several_entry_offsets_are_all_kept():
    rows = market_rows_from_scored(_frame(), entry_offsets=(9, 12))
    assert sorted(r[2] for r in rows) == [9, 12]


def test_the_quote_age_bar_still_applies():
    frame = _frame()
    frame.loc[frame['offset'] == 12, 'quote_age_seconds'] = 900.0
    assert market_rows_from_scored(frame, entry_offsets=(12,)) == []
