"""A row written with an unknown quality is INVISIBLE, so the write is refused.

`ResearchStore.read` keeps `QUALITY_LEVELS[index(min_quality):]` and the
default `min_quality` is 'valid', so that set is exactly ('valid',). An
unrecognised value is therefore not a weaker grade — it is a row no default
read will ever return.

Measured 2026-10-21: `scripts/backfill_trades.py` wrote `quality='backfill'`
and all 31,462 rows of `venue_trades` were unreadable while the files held the
data. It also broke that script's own resume check, so every run refetched the
whole history. Nothing raised.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from core.datastore import QUALITY_LEVELS, DataStoreError, ResearchStore

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _row(quality):
    return pd.DataFrame([{
        'venue': 'kalshi', 'symbol': 'BTC-USD',
        'event_time': NOW, 'available_time': NOW, 'quality': quality,
        'market_ticker': 'KXBTC15M-26OCT211200-00',
        'window_open': NOW, 'offset_minutes': 3, 'taker_book_side': 'bid',
        'n_prints': 4, 'contracts': 10.0,
        'min_price': 0.41, 'max_price': 0.44, 'vwap': 0.42,
    }])


def test_an_unknown_quality_is_refused_rather_than_written_invisibly(tmp_path):
    store = ResearchStore(root=tmp_path)
    with pytest.raises(DataStoreError) as excinfo:
        store.write('venue_trades', _row('backfill'))
    message = str(excinfo.value)
    assert 'backfill' in message
    assert 'invisible' in message.lower(), 'the error must say WHY it matters'


def test_every_known_quality_is_accepted(tmp_path):
    store = ResearchStore(root=tmp_path)
    for quality in QUALITY_LEVELS:
        assert store.write('venue_trades', _row(quality)) >= 0


def test_a_valid_row_is_readable_by_default(tmp_path):
    """The symptom that started this: written, but `read()` returned nothing."""
    store = ResearchStore(root=tmp_path)
    store.write('venue_trades', _row('valid'))
    assert len(store.read('venue_trades')) == 1


def test_a_demoted_row_is_still_hidden_by_default_but_readable_explicitly(tmp_path):
    """The guard must not break the grading it protects: 'suspicious' is a
    real grade and SHOULD be hidden from a default read."""
    store = ResearchStore(root=tmp_path)
    store.write('venue_trades', _row('suspicious'))
    assert len(store.read('venue_trades')) == 0
    assert len(store.read('venue_trades', min_quality=None)) == 1
