"""The tick sink must never touch the store's read-concat-rewrite path.

At a measured 862 frames a second a `(venue, symbol, month)` partition would
reach tens of gigabytes and be rewritten on every flush — which is exactly the
stall `run_live.py` arranges the whole live process to avoid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.datastore import ResearchStore
from core.spool import FrameSpool, compact, event_rows, prune
from data_collection.stream.base import BookEvent

HOUR = 1756200000.0          # 2025-08-26T09:20:00Z, mid-hour
NEXT_HOUR = HOUR + 3600.0
LATER = HOUR + 7200.0        # so both files above are closed


def rec(t=HOUR, ticker='K', price=0.30, size=10.0):
    return {'t': t, 'venue': 'kalshi', 'symbol': 'BTC-USD',
            'market_ticker': ticker, 'seq': 1, 'kind': 'delta', 'side': 'yes',
            'price': price, 'size': size, 'absolute': False}


def test_a_snapshot_becomes_one_row_per_level():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                      received=HOUR, seq=1, yes=[(0.3, 1.0), (0.31, 2.0)],
                      no=[(0.68, 3.0)])
    rows = list(event_rows(event, 'BTC-USD'))
    assert len(rows) == 3
    assert [r['side'] for r in rows] == ['yes', 'yes', 'no']
    assert all(r['absolute'] for r in rows)


def test_a_delta_records_that_its_size_is_a_signed_change():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='delta',
                      received=HOUR, seq=2, yes=[(0.3, -5.0)], no=[],
                      absolute=False)
    rows = list(event_rows(event, 'BTC-USD'))
    assert rows == [{'t': HOUR, 'venue': 'kalshi', 'symbol': 'BTC-USD',
                     'market_ticker': 'K', 'seq': 2, 'kind': 'delta',
                     'side': 'yes', 'price': 0.3, 'size': -5.0,
                     'absolute': False}]


def test_appending_never_reads_the_file_back(tmp_path):
    spool = FrameSpool(tmp_path, 'kalshi')
    spool.extend(rec(t=HOUR + i * 0.001) for i in range(500))
    spool.flush()
    written = list(tmp_path.rglob('*.jsonl'))
    assert len(written) == 1
    assert sum(1 for _ in written[0].open()) == 500


def test_frames_roll_into_separate_hourly_files(tmp_path):
    spool = FrameSpool(tmp_path, 'kalshi')
    spool.append(rec(t=HOUR))
    spool.append(rec(t=NEXT_HOUR))
    spool.flush()
    assert len(list(tmp_path.rglob('*.jsonl'))) == 2


def test_compaction_writes_immutable_parquet_the_store_can_read(tmp_path):
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec())
    spool.close()

    assert compact(spool_root, store_root, keep_days=0.0, now=LATER) == 1
    got = ResearchStore(store_root).read('venue_book_events', min_quality=None)
    assert len(got) == 1
    assert got.iloc[0]['kind'] == 'delta' and got.iloc[0]['size'] == 10.0
    assert not list(spool_root.rglob('*.jsonl')), 'compacted files are removed'


def test_the_hour_still_being_appended_to_is_never_compacted(tmp_path):
    """Compacting an open file archives a prefix and deletes the rest."""
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec(t=HOUR))
    spool.flush()
    assert compact(spool_root, store_root, keep_days=0.0, now=HOUR + 60) == 0
    assert list(spool_root.rglob('*.jsonl')), 'the open hour survives'


def test_compaction_never_rewrites_an_existing_partition(tmp_path):
    """The whole point: two hours are two files, not one file written twice."""
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec(t=HOUR))
    spool.append(rec(t=NEXT_HOUR))
    spool.close()
    compact(spool_root, store_root, keep_days=0.0, now=LATER)
    files = sorted((store_root / 'venue_book_events').rglob('*.parquet'))
    assert len(files) == 2, [f.name for f in files]


def test_pruning_bounds_the_raw_tier(tmp_path):
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append(rec(t=HOUR))
    spool.close()
    compact(spool_root, store_root, keep_days=0.0, now=LATER)
    assert list((store_root / 'venue_book_events').rglob('*.parquet'))
    # Everything on disk is older than a cutoff far in the future.
    import time as _t
    assert prune(store_root, keep_days=1.0, now=_t.time() + 86400 * 10) == 1
    assert not list((store_root / 'venue_book_events').rglob('*.parquet'))
