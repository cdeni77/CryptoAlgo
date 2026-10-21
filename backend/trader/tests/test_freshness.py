"""A dataset that stops receiving rows must be loud.

`core.heartbeat` proves a component's loop completed a pass. It cannot tell a
loop that runs and writes nothing from one that is working, which is how
`venue_ladder` died for thirty-one days with every heartbeat green.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core import freshness

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _write(root, dataset, newest, *, column='event_time', rows=3):
    """One partition whose newest row is `newest`."""
    path = root / dataset / 'venue=kalshi' / 'symbol=BTC-USD'
    path.mkdir(parents=True, exist_ok=True)
    stamps = [newest - timedelta(minutes=i) for i in range(rows)][::-1]
    frame = pd.DataFrame({column: pd.to_datetime(stamps, utc=True),
                          'value': range(rows)})
    frame.to_parquet(path / 'data.parquet', index=False)
    return path / 'data.parquet'


def test_a_current_dataset_is_not_reported(tmp_path):
    _write(tmp_path, 'venue_ladder', NOW - timedelta(minutes=2))
    assert freshness.stale(str(tmp_path), datasets=['venue_ladder'], now=NOW) == []


def test_a_recorder_that_stopped_writing_is_reported(tmp_path):
    """The venue_ladder failure: the process lived, the rows stopped."""
    _write(tmp_path, 'venue_ladder', NOW - timedelta(days=31))
    late = freshness.stale(str(tmp_path), datasets=['venue_ladder'], now=NOW)
    assert len(late) == 1
    name, age, limit = late[0]
    assert name == 'venue_ladder'
    assert age == pytest.approx(31 * 86400, rel=1e-3)
    assert limit == 900.0


def test_a_dataset_with_no_files_is_reported_rather_than_passing(tmp_path):
    late = freshness.stale(str(tmp_path), datasets=['venue_ladder'], now=NOW)
    assert late and late[0][1] is None, 'absent must not read as fresh'


def test_the_newest_row_wins_across_partitions(tmp_path):
    """**Not the newest file by mtime.** The store partitions by venue, symbol
    and month, so the most recently TOUCHED file is not the one holding the
    most recent row -- reading only the newest few reported the LABEL as seven
    days stale when it was current."""
    _write(tmp_path, 'venue_settlements', NOW - timedelta(days=7))
    stale_file = tmp_path / 'venue_settlements' / 'venue=kalshi' / 'symbol=BTC-USD'
    other = tmp_path / 'venue_settlements' / 'venue=kalshi' / 'symbol=SOL-USD'
    other.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({'event_time': pd.to_datetime(
        [NOW - timedelta(minutes=5)], utc=True), 'value': [1]})
    frame.to_parquet(other / 'data.parquet', index=False)
    # Make the STALE partition the most recently touched.
    (stale_file / 'data.parquet').touch()
    assert freshness.stale(str(tmp_path), datasets=['venue_settlements'],
                           now=NOW) == []


def test_a_dataset_with_no_live_producer_is_not_a_fault(tmp_path, monkeypatch):
    """`None` means "nothing writes this continuously", and must not report.

    It used to be asserted against `venue_trades`, which was backfill-only —
    and that expectation is exactly why the tape went 22 days with an interior
    hole and nobody was told. `venue_trades` has a live producer now, so the
    mechanism is tested on its own rather than on a dataset that outgrew it.
    """
    monkeypatch.setitem(freshness.EXPECTED_MAX_AGE, 'venue_trades', None)
    _write(tmp_path, 'venue_trades', NOW - timedelta(days=30))
    assert freshness.stale(str(tmp_path), datasets=['venue_trades'], now=NOW) == []


def test_the_tape_is_now_expected_to_keep_arriving(tmp_path):
    """`scripts/record_trades.py` writes it every minute, so silence is a fault.

    The limit matches the other minute-grid recorders: fifteen minutes with
    nothing recorded is one whole market window gone. The recorder runs a
    couple of minutes behind the tape on purpose — it writes a minute only once
    that minute has fully elapsed and the poll covered all of it — so the limit
    has to be generous against that lag, not against the poll period.
    """
    assert freshness.EXPECTED_MAX_AGE['venue_trades'] == 900.0
    _write(tmp_path, 'venue_trades', NOW - timedelta(minutes=4))
    assert freshness.stale(str(tmp_path), datasets=['venue_trades'], now=NOW) == []

    _write(tmp_path, 'venue_trades', NOW - timedelta(days=7))
    late = freshness.stale(str(tmp_path), datasets=['venue_trades'], now=NOW)
    assert [name for name, _age, _limit in late] == ['venue_trades']


def test_the_probe_never_raises_on_a_broken_file(tmp_path):
    """A freshness probe that can take the caller down is worse than silent."""
    path = tmp_path / 'venue_ladder'
    path.mkdir(parents=True)
    (path / 'data.parquet').write_text('not parquet at all')
    assert freshness.newest_event_time(str(tmp_path), 'venue_ladder') is None
