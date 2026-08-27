"""Turning collected JSONL into the two stored layers.

The properties that matter here are all about surviving how collection
actually ends: a 47-hour job gets killed, retried and re-run, so the converter
meets torn lines and duplicate windows as a matter of course rather than as an
exception. Losing a partition to either would mean re-collecting it, which is
the one cost this whole design is arranged to avoid.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research.collect import consolidate
from research.collect.fetchers import FIELDS


def _record(market_id='KXBTC15M-1', window='2026-02-01T00:00:00+00:00', n=2):
    """One collected window: n packed snapshots plus their raw ladders."""
    series = [[1769904000000 + i * 1000, 44 + i, 46 + i, 25, 99,
               25, 99, 30, 120, 3, 2, 154, 139] for i in range(n)]
    return {
        'venue': 'kalshi', 'symbol': 'BTC-USD', 'market_id': market_id,
        'window_open': window, 'fields': list(FIELDS), 'n': n,
        'series': series,
        'ladders': [{'timestamp': s[0], 'yes_bids': [{'price': s[1], 'size': 25}],
                     'yes_asks': [{'price': s[2], 'size': 99}]} for s in series],
    }


@pytest.fixture()
def collected(tmp_path, monkeypatch):
    """A collection tree the converter can read, redirected into tmp_path."""
    monkeypatch.setattr(consolidate, 'ARCHIVE_IN', tmp_path / 'archive')
    monkeypatch.setattr(consolidate, 'DERIVED_OUT', tmp_path / 'derived')
    monkeypatch.setattr(consolidate, 'LADDERS_OUT', tmp_path / 'ladders')

    def write(records, venue='kalshi', symbol='BTC-USD', month='2026-02'):
        path = (tmp_path / 'archive' / f'venue={venue}' / f'symbol={symbol}'
                / f'month={month}' / 'windows.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as handle:
            for rec in records:
                handle.write(json.dumps(rec) + '\n' if isinstance(rec, dict) else rec)
        return path
    return write


def _derived(tmp_path, venue='kalshi', symbol='BTC-USD', month='2026-02'):
    return pd.read_parquet(tmp_path / 'derived' / f'venue={venue}'
                           / f'symbol={symbol}' / f'month={month}' / 'data.parquet')


def test_every_snapshot_becomes_a_row(collected, tmp_path):
    collected([_record(n=3)])
    snaps, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                         [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert (snaps, windows) == (3, 1)


def test_the_packed_fields_are_restored_under_their_names(collected, tmp_path):
    collected([_record(n=1)])
    consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                        [next((tmp_path / 'archive').rglob('*.jsonl'))])
    row = _derived(tmp_path).iloc[0]
    assert row['best_bid'] == 44 and row['best_ask'] == 46
    assert row['bid_vol'] == 154 and row['ask_vol'] == 139


def test_a_snapshot_carries_its_offset_into_the_window(collected, tmp_path):
    """Features are defined at an offset, so the converter computes it once
    rather than making every reader rediscover the unit of `ts`."""
    collected([_record(n=2)])
    consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                        [next((tmp_path / 'archive').rglob('*.jsonl'))])
    frame = _derived(tmp_path)
    assert set(frame['offset_seconds']) == {0.0, 1.0}


def test_a_torn_final_line_does_not_lose_the_partition(collected, tmp_path):
    """A kill -9 mid-write leaves half a line. That must cost one window, not
    the month — the whole reason collection appends JSONL instead of writing
    Parquet as it goes."""
    collected([_record(market_id='good', n=2), '{"venue": "kalshi", "sym'])
    snaps, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                         [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert (snaps, windows) == (2, 1)


def test_a_window_collected_twice_is_kept_once(collected, tmp_path):
    """`error` rows are retried, so a window can legitimately be written more
    than once. Counting it twice would inflate every per-window statistic."""
    collected([_record(market_id='dup', n=2), _record(market_id='dup', n=2)])
    snaps, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                         [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert (snaps, windows) == (2, 1)


def test_distinct_windows_of_the_same_market_are_both_kept(collected, tmp_path):
    """Deduplication is on (market, window), not market alone."""
    collected([_record(market_id='m', window='2026-02-01T00:00:00+00:00'),
               _record(market_id='m', window='2026-02-01T00:15:00+00:00')])
    _, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                     [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert windows == 2


def test_the_raw_ladders_are_archived_alongside(collected, tmp_path):
    collected([_record(n=2)])
    consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                        [next((tmp_path / 'archive').rglob('*.jsonl'))])
    ladders = pd.read_parquet(tmp_path / 'ladders' / 'venue=kalshi'
                              / 'symbol=BTC-USD' / 'month=2026-02' / 'data.parquet')
    assert len(ladders) == 1
    assert json.loads(ladders.iloc[0]['ladders'])[0]['yes_bids'][0]['price'] == 44


def test_the_archive_can_be_skipped_when_only_features_are_wanted(collected, tmp_path):
    collected([_record(n=2)])
    consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                        [next((tmp_path / 'archive').rglob('*.jsonl'))],
                        keep_ladders=False)
    assert not (tmp_path / 'ladders').exists()
    assert _derived(tmp_path).shape[0] == 2


def test_an_empty_partition_produces_no_files_rather_than_an_empty_one(collected, tmp_path):
    collected(['\n'])
    snaps, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                         [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert (snaps, windows) == (0, 0)
    assert not (tmp_path / 'derived').exists()


def test_converting_twice_is_idempotent(collected, tmp_path):
    """A re-run after an interrupted conversion must not double the rows."""
    collected([_record(n=3)])
    path = next((tmp_path / 'archive').rglob('*.jsonl'))
    consolidate.convert('kalshi', 'BTC-USD', '2026-02', [path])
    consolidate.convert('kalshi', 'BTC-USD', '2026-02', [path])
    assert len(_derived(tmp_path)) == 3
