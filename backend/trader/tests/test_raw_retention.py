"""Raw-tier retention, the free-disk guard, and the Coinbase trade archive."""
from __future__ import annotations

import json
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.datastore import ResearchStore
from core.spool import (
    RAW_TABLES, FrameSpool, SafeSpool, compact, compact_all, compact_table,
    free_space_needed, guard_disk, prune, retention_days, table_spool,
)

Usage = namedtuple('Usage', 'total used free')
DAY = 86400.0
NOW = datetime(2026, 10, 14, 12, 30, tzinfo=timezone.utc).timestamp()


def _hour(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y%m%dT%H')


def _parquet(store: Path, dataset: str, t: float, size: int = 1000) -> Path:
    hour = _hour(t)
    out = (store / dataset / 'venue=x' / f'date={hour[:4]}-{hour[4:6]}-{hour[6:8]}'
           / f'{hour}.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b'\0' * size)
    return out


class FakeDisk:
    """A volume whose `used` is exactly the bytes under the store."""

    def __init__(self, store: Path, total: int, other: int = 0):
        self.store, self.total, self.other = store, total, other
        self.calls = 0

    def __call__(self, _path):
        self.calls += 1
        used = self.other + sum(p.stat().st_size for p in self.store.rglob('*')
                                if p.is_file())
        return Usage(self.total, used, self.total - used)


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------

def test_retention_defaults_and_env(monkeypatch):
    monkeypatch.delenv('RAW_RETENTION_DAYS', raising=False)
    monkeypatch.delenv('INDEX_PROXY_RETENTION_DAYS', raising=False)
    monkeypatch.delenv('SPOT_TRADES_RETENTION_DAYS', raising=False)
    assert retention_days('venue_book_events') == 60.0
    # **A year, not the 60-day raw tier it used to inherit.** `spot_trades` is
    # a feature source in waiting; at 60 days it tops out near 5,760 windows,
    # the floor of a usable training window, so it could never have fed a
    # model however long it ran. This assertion used to read 60.0 and was
    # encoding that cap as if it were intended.
    assert retention_days('spot_trades') == 365.0
    assert retention_days('index_proxy_1s') == 365.0
    monkeypatch.setenv('RAW_RETENTION_DAYS', '90')
    monkeypatch.setenv('INDEX_PROXY_RETENTION_DAYS', '30')
    monkeypatch.setenv('SPOT_TRADES_RETENTION_DAYS', '45')
    assert retention_days('venue_book_events') == 90.0
    assert retention_days('index_proxy_1s') == 30.0
    # Its own knob: RAW_RETENTION_DAYS must no longer move it.
    assert retention_days('spot_trades') == 45.0


def test_a_garbage_retention_falls_back_rather_than_pruning_everything(monkeypatch):
    monkeypatch.setenv('RAW_RETENTION_DAYS', 'sixty')
    assert retention_days('venue_book_events') == 60.0


def test_prune_reads_the_hour_from_the_name_not_the_mtime(tmp_path):
    old = _parquet(tmp_path, 'venue_book_events', NOW - 61 * DAY)
    young = _parquet(tmp_path, 'venue_book_events', NOW - 59 * DAY)
    # Both files were just written, so an mtime rule would keep both.
    assert prune(tmp_path, keep_days=60, now=NOW) == 1
    assert not old.exists() and young.exists()


def test_each_table_is_pruned_to_its_own_retention(tmp_path, monkeypatch):
    monkeypatch.delenv('RAW_RETENTION_DAYS', raising=False)
    monkeypatch.delenv('INDEX_PROXY_RETENTION_DAYS', raising=False)
    monkeypatch.delenv('SPOT_TRADES_RETENTION_DAYS', raising=False)
    spool, store = tmp_path / 'spool', tmp_path / 'store'
    book = _parquet(store, 'venue_book_events', NOW - 100 * DAY)
    trades = _parquet(store, 'spot_trades', NOW - 100 * DAY)
    stale_trades = _parquet(store, 'spot_trades', NOW - 400 * DAY)
    index = _parquet(store, 'index_proxy_1s', NOW - 100 * DAY)
    compact_all(spool, store, now=NOW,
                disk_usage=lambda _p: Usage(10**12, 0, 10**12))
    assert not book.exists(), 'raw frames are a 60-day replay buffer'
    # 100 days old and KEPT: this used to assert the opposite, which is the
    # cap that made the dataset untrainable.
    assert trades.exists(), 'spot_trades is kept a year'
    assert not stale_trades.exists(), 'but a year is still a limit'
    assert index.exists(), 'the index proxy is kept a year'


# ---------------------------------------------------------------------------
# the disk guard
# ---------------------------------------------------------------------------

def test_the_guard_does_nothing_with_room(tmp_path):
    f = _parquet(tmp_path, 'venue_book_events', NOW - 10 * DAY)
    disk = FakeDisk(tmp_path, total=10**6)
    assert guard_disk(tmp_path, min_free_fraction=0.15, min_free_bytes=0,
                      now=NOW, disk_usage=disk) == 0
    assert f.exists()


def test_the_floor_is_the_larger_of_fraction_and_bytes(monkeypatch):
    monkeypatch.delenv('DISK_GUARD_MIN_FREE_FRACTION', raising=False)
    monkeypatch.delenv('DISK_GUARD_MIN_FREE_GB', raising=False)
    assert free_space_needed(1e12) == 0.15e12  # 15% of 1 TB > 20 GB
    assert free_space_needed(50e9) == 20e9  # 20 GB > 15% of 50 GB
    monkeypatch.setenv('DISK_GUARD_MIN_FREE_FRACTION', '0.5')
    assert free_space_needed(1e12) == 0.5e12


def test_the_guard_sheds_oldest_book_frames_first_until_above_the_floor(tmp_path, caplog):
    files = [_parquet(tmp_path, 'venue_book_events', NOW - d * DAY, size=1000)
             for d in (30, 20, 10, 5)]
    index = _parquet(tmp_path, 'index_proxy_1s', NOW - 300 * DAY, size=1000)
    trades = _parquet(tmp_path, 'spot_trades', NOW - 40 * DAY, size=1000)
    # 6,000 bytes used of 8,000; the floor wants 3,500 free.
    disk = FakeDisk(tmp_path, total=8000)
    with caplog.at_level('ERROR', logger='spool'):
        removed = guard_disk(tmp_path, min_free_fraction=0.0, min_free_bytes=3500,
                             now=NOW, disk_usage=disk)
    assert removed == 2
    assert not files[0].exists() and not files[1].exists()
    assert files[2].exists() and files[3].exists()
    # The most expendable table goes first, not the oldest file on disk.
    assert index.exists() and trades.exists()
    assert 'DISK GUARD' in caplog.text


def test_the_guard_moves_to_trades_then_the_index_only_when_it_must(tmp_path):
    book = _parquet(tmp_path, 'venue_book_events', NOW - 5 * DAY)
    trades = _parquet(tmp_path, 'spot_trades', NOW - 5 * DAY)
    index = _parquet(tmp_path, 'index_proxy_1s', NOW - 5 * DAY)
    disk = FakeDisk(tmp_path, total=3000)
    guard_disk(tmp_path, min_free_fraction=0.0, min_free_bytes=2000,
               now=NOW, disk_usage=disk)
    assert not book.exists() and not trades.exists() and index.exists()


def test_the_guard_never_deletes_the_newest_day(tmp_path, caplog):
    today = _parquet(tmp_path, 'venue_book_events', NOW - 3600)
    disk = FakeDisk(tmp_path, total=2000, other=1500)
    with caplog.at_level('ERROR', logger='spool'):
        assert guard_disk(tmp_path, min_free_fraction=0.0, min_free_bytes=1000,
                          now=NOW, disk_usage=disk) == 0
    assert today.exists()
    assert 'STILL' in caplog.text


def test_compaction_survives_a_guard_that_raises(tmp_path):
    def broken(_path):
        raise OSError('statvfs failed')
    assert compact_all(tmp_path / 'spool', tmp_path / 'store', now=NOW,
                       disk_usage=broken) == {d: 0 for d in RAW_TABLES}


# ---------------------------------------------------------------------------
# table layout
# ---------------------------------------------------------------------------

LATER = NOW + 7200.0


def test_book_compaction_never_reads_another_tables_spool(tmp_path):
    spool_root, store = tmp_path / 'spool', tmp_path / 'store'
    trades = table_spool(spool_root, 'spot_trades', 'coinbase')
    trades.append({'t': NOW, 'venue': 'coinbase', 'symbol': 'BTC-USD',
                   'product_id': 'BTC-USD', 'trade_id': '1', 'time': NOW,
                   'price': 1.0, 'size': 1.0, 'side': 'BUY', 'sequence_num': 3})
    trades.close()
    assert compact(spool_root, store, keep_days=0, now=LATER) == 0
    assert list(spool_root.rglob('*.jsonl')), 'left for its own compaction'
    assert compact_table(spool_root, store, 'spot_trades', keep_days=0,
                         now=LATER) == 1


def test_the_legacy_book_layout_still_compacts(tmp_path):
    spool_root, store = tmp_path / 'spool', tmp_path / 'store'
    spool = FrameSpool(spool_root, 'kalshi')
    spool.append({'t': NOW, 'venue': 'kalshi', 'symbol': 'BTC-USD',
                  'market_ticker': 'K', 'seq': 1, 'kind': 'delta', 'side': 'yes',
                  'price': 0.3, 'size': 1.0, 'absolute': False})
    spool.close()
    assert compact_all(spool_root, store, now=LATER,
                       disk_usage=lambda _p: Usage(10**12, 0, 10**12)
                       )['venue_book_events'] == 1


# ---------------------------------------------------------------------------
# SafeSpool
# ---------------------------------------------------------------------------

class _Exploding:
    def __init__(self):
        self.calls = 0

    def extend(self, records):
        self.calls += 1
        raise OSError(28, 'No space left on device')

    def flush(self):
        raise OSError(28, 'No space left on device')

    def close(self):
        pass


def test_a_failing_spool_drops_rows_instead_of_raising():
    clock = [0.0]
    bad = _Exploding()
    spool = SafeSpool(lambda: bad, retry_after=60.0, clock=lambda: clock[0])
    spool.append({'t': 1.0})
    spool.extend([{'t': 2.0}, {'t': 3.0}])
    spool.flush()
    assert spool.dropped == 3
    assert bad.calls == 1, 'no retry inside the cool-down'
    clock[0] = 61.0
    spool.append({'t': 4.0})
    assert bad.calls == 2 and spool.dropped == 4


def test_a_spool_that_cannot_even_open_does_not_raise(tmp_path):
    def factory():
        raise PermissionError('read-only volume')
    spool = SafeSpool(factory)
    spool.append({'t': NOW})
    spool.flush()
    spool.close()
    assert spool.dropped == 1


def test_safe_spool_writes_through_when_healthy(tmp_path):
    spool = SafeSpool(lambda: table_spool(tmp_path, 'spot_trades', 'coinbase'))
    spool.append({'t': NOW, 'x': 1})
    spool.flush()
    lines = list((tmp_path / 'spot_trades' / 'coinbase').glob('*.jsonl'))
    assert len(lines) == 1 and json.loads(lines[0].read_text()) == {'t': NOW, 'x': 1}


# ---------------------------------------------------------------------------
# the Coinbase trade archive
# ---------------------------------------------------------------------------

from scripts.record_spot_stream import SequenceTracker, handle, trade_record  # noqa: E402
from core.trade_bars import TradeTape  # noqa: E402

PRINT = '2026-10-14T12:30:05.123456789Z'


def _message(trade_id='77', kind='update', seq=5, price='62000.5'):
    return {'channel': 'market_trades', 'sequence_num': seq, 'events': [{
        'type': kind, 'trades': [{
            'trade_id': trade_id, 'product_id': 'BTC-USD', 'price': price,
            'size': '0.01', 'side': 'BUY', 'time': PRINT}]}]}


def test_every_field_the_archive_promises_is_recorded():
    record = trade_record(_message()['events'][0]['trades'][0],
                          received=NOW + 5.2, sequence_num=5)
    assert record['product_id'] == 'BTC-USD' and record['symbol'] == 'BTC-USD'
    assert record['trade_id'] == '77' and record['side'] == 'BUY'
    assert record['price'] == 62000.5 and record['size'] == 0.01
    assert record['time'] == pytest.approx(
        pd.Timestamp(PRINT).value / 1e9, abs=1e-6)
    assert record['t'] == NOW + 5.2 and record['sequence_num'] == 5


def test_a_trade_without_an_id_is_not_archived():
    assert trade_record({'product_id': 'BTC-USD', 'time': PRINT},
                        received=NOW, sequence_num=1) is None


class _List:
    def __init__(self):
        self.rows = []

    def append(self, row):
        self.rows.append(row)


def test_the_socket_handler_spools_and_still_folds_the_tape():
    tape, spool = TradeTape(), _List()
    handle(_message(), tape, SequenceTracker(), received=NOW + 5.2,
           last_received=None, spool=spool)
    assert len(spool.rows) == 1
    assert tape.products() == ['BTC-USD']


def test_a_broken_spool_never_costs_the_tape_a_trade():
    class Broken:
        def append(self, row):
            raise RuntimeError('boom')
    tape = TradeTape()
    handle(_message(), tape, SequenceTracker(), received=NOW + 5.2,
           last_received=None, spool=Broken())
    assert tape.products() == ['BTC-USD']


def test_a_redelivered_trade_is_stored_once(tmp_path):
    """A trade replayed after a reconnect, in a later hour's file, is stored once."""
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = table_spool(spool_root, 'spot_trades', 'coinbase')
    first = _message(seq=5)['events'][0]['trades'][0]
    spool.append(trade_record(first, received=NOW + 5.2, sequence_num=5))
    spool.append(trade_record(first, received=NOW + 5.9, sequence_num=6))
    other = dict(first, trade_id='78')
    spool.append(trade_record(other, received=NOW + 6.0, sequence_num=7))
    # The replay, an hour later, after a reconnect.
    spool.append(trade_record(first, received=NOW + 3600.0, sequence_num=1))
    spool.close()
    compact_table(spool_root, store_root, 'spot_trades', keep_days=0,
                  now=NOW + 3 * 3600)
    got = ResearchStore(store_root).read('spot_trades', min_quality=None)
    assert sorted(got['trade_id']) == ['77', '78']
    row = got[got['trade_id'] == '77'].iloc[0]
    assert row['product_id'] == 'BTC-USD' and row['side'] == 'BUY'
    assert row['available_time'] >= row['event_time']
    assert {'received_time', 'sequence_num', 'price', 'size'} <= set(got.columns)


def test_a_receipt_before_the_print_is_never_knowable_early(tmp_path):
    """A receipt time before the trade time is never used; the store refuses such a row."""
    spool_root, store_root = tmp_path / 'spool', tmp_path / 'store'
    spool = table_spool(spool_root, 'spot_trades', 'coinbase')
    trade = _message()['events'][0]['trades'][0]
    printed = pd.Timestamp(PRINT).value / 1e9
    spool.append(trade_record(trade, received=printed - 2.0, sequence_num=5))
    spool.close()
    compact_table(spool_root, store_root, 'spot_trades', keep_days=0,
                  now=NOW + 3 * 3600)
    got = ResearchStore(store_root).read('spot_trades', min_quality=None)
    assert got.iloc[0]['available_time'] == got.iloc[0]['event_time']
