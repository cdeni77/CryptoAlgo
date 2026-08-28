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


def test_a_recollected_window_supersedes_its_truncated_first_copy(collected, tmp_path):
    """3,099 Polymarket windows were archived truncated at the 2,000-snapshot
    page cap before `fetch_pm` paginated. They were reset to pending and
    re-collected, so the archive holds BOTH copies — and keeping the first
    occurrence would keep the truncated one, making the re-collection
    pointless. The fuller record wins."""
    truncated = _record(market_id='pm-1', n=2)
    complete = _record(market_id='pm-1', n=5)
    collected([truncated, complete])
    snaps, windows = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                         [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert windows == 1
    assert snaps == 5, 'must keep the complete copy, not the truncated one'


def test_order_does_not_decide_which_copy_wins(collected, tmp_path):
    """The complete copy may be appended before or after the truncated one
    depending on when the retry ran."""
    collected([_record(market_id='pm-2', n=7), _record(market_id='pm-2', n=2)])
    snaps, _ = consolidate.convert('kalshi', 'BTC-USD', '2026-02',
                                   [next((tmp_path / 'archive').rglob('*.jsonl'))])
    assert snaps == 7


# --- memory: the converter froze the machine twice --------------------------
#
# `convert` built a dict of every RECORD in a partition before writing
# anything, and a record carries its raw ladders -- 97% of the bytes. Measured
# on the real archive: a single Polymarket record reaches 43 MB of JSON, and
# the largest partition is 2.0 GB gzipped, roughly 66 GB decompressed, more
# again as live Python objects. Against 31 GB of RAM that is not an OOM kill;
# it is a hard lockup with nothing flushed to the journal, which is exactly
# what the boot history showed both times.
#
# The fix is two passes: find WHERE the fullest copy of each window lives,
# keeping only the location, then re-read and stream those records out in
# batches. Peak memory then depends on the batch, not on the partition.

def test_selecting_copies_keeps_locations_not_records(collected, tmp_path):
    """The index must hold a place in a file, never the payload found there."""
    collected([_record(market_id='m', n=3)])
    path = next((tmp_path / 'archive').rglob('*.jsonl'))
    chosen = consolidate.select_copies([path])
    assert list(chosen) == [('m', '2026-02-01T00:00:00+00:00')]
    where = next(iter(chosen.values()))
    assert isinstance(where[1], int), 'second element should be a line number'
    assert 'series' not in repr(where) and 'ladders' not in repr(where)


def test_selecting_copies_points_at_the_fullest_copy(collected, tmp_path):
    """Same rule as before, decided without holding either record."""
    collected([_record(market_id='pm', n=2), _record(market_id='pm', n=9),
               _record(market_id='pm', n=4)])
    path = next((tmp_path / 'archive').rglob('*.jsonl'))
    chosen = consolidate.select_copies([path])
    assert next(iter(chosen.values()))[1] == 1, 'the n=9 copy is on line 1'


def test_a_torn_line_is_skipped_by_the_index_too(collected, tmp_path):
    collected(['{"venue": "kalshi", "sym\n', _record(market_id='ok', n=2)])
    path = next((tmp_path / 'archive').rglob('*.jsonl'))
    assert list(consolidate.select_copies([path])) == [
        ('ok', '2026-02-01T00:00:00+00:00')]


def test_peak_memory_does_not_scale_with_the_partition(collected, tmp_path):
    """The regression test for the crash, stated as the property that failed:
    doubling the partition must not double the memory. The old converter held
    every record, so peak tracked the partition exactly -- which is how a 2.0 GB
    gzipped month took down a 31 GB machine."""
    import tracemalloc

    def peak_for(count, sub):
        root = tmp_path / 'archive' / 'venue=kalshi' / 'symbol=BTC-USD' / f'month={sub}'
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'windows.jsonl'
        with open(path, 'w') as handle:
            for i in range(count):
                handle.write(json.dumps(_record(market_id=f'm{i}', n=300)) + '\n')
        tracemalloc.start()
        consolidate.convert('kalshi', 'BTC-USD', sub, [path],
                            keep_ladders=False, batch_rows=500)
        got = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return got

    small = peak_for(15, '2026-03')
    large = peak_for(60, '2026-04')          # 4x the partition
    assert large < small * 1.6, (
        f'peak went {small/1e6:.2f}MB -> {large/1e6:.2f}MB for 4x the data '
        f'-- memory is still scaling with the partition')


def test_batching_does_not_change_the_output(collected, tmp_path):
    """Row-group size is a memory knob, never a data one."""
    collected([_record(market_id=f'm{i}', n=50) for i in range(10)])
    path = next((tmp_path / 'archive').rglob('*.jsonl'))
    consolidate.convert('kalshi', 'BTC-USD', '2026-02', [path], batch_rows=7)
    small = _derived(tmp_path).sort_values(['market_id', 'ts']).reset_index(drop=True)
    consolidate.convert('kalshi', 'BTC-USD', '2026-02', [path], batch_rows=10_000)
    big = _derived(tmp_path).sort_values(['market_id', 'ts']).reset_index(drop=True)
    assert len(small) == len(big) == 500
    pd.testing.assert_frame_equal(small, big)
