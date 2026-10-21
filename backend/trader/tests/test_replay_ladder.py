"""Replaying the archive must rebuild the book, not something that looks like it.

`venue_ladder` is the one table this repository calls irrecoverable, so a
reconstruction of it is the worst place for a near-miss: nothing downstream can
tell a rebuilt ladder from a recorded one, and a backfill that is merely CLOSE
would become training data that nobody can audit afterwards.

Two things are therefore pinned here rather than argued:

* the regroup is the exact inverse of `core.spool.event_rows`, checked through
  the real spool-and-compact path rather than against hand-built rows, and
* a replayed row is labelled as reconstructed, lands BESIDE a live recording
  instead of superseding it, and re-running changes nothing.

The agreement number itself is not a unit test — it lives in
`research/validate/_validate_replay.py`, which folds the production archive
against the live rows the recorder wrote over the same minutes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.datastore import ResearchStore
from core.replay_book import (archive_events, events_from_rows, hour_files,
                              next_mark, replay_cache, row_tuple,
                              sampled_books, window_open)
from core.spool import FrameSpool, compact, event_rows
from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from scripts import replay_ladder

TICKER = 'KXBTC15M-26OCT180800-00'
OPEN = pd.Timestamp('2026-10-18T11:45:00Z')
T0 = OPEN.timestamp()
#: Well past every hour the tests spool into, so `compact` treats them as closed.
LATER = T0 + 86400.0


def snapshot(t, ticker=TICKER, seq=1, yes=((0.30, 100.0), (0.31, 50.0)),
             no=((0.68, 70.0),)):
    return BookEvent(venue='kalshi', market_ticker=ticker, kind='snapshot',
                     received=t, seq=seq, yes=sorted(yes), no=sorted(no))


def delta(t, ticker=TICKER, seq=2, price=0.30, change=25.0, side='yes'):
    level = [(price, change)]
    return BookEvent(venue='kalshi', market_ticker=ticker, kind='delta',
                     received=t, seq=seq, yes=level if side == 'yes' else [],
                     no=level if side == 'no' else [], absolute=False)


def archive(tmp_path: Path, events) -> Path:
    """Spool `events` and compact them, exactly as the live recorder does."""
    root = tmp_path / 'store'
    spool = FrameSpool(tmp_path / 'spool', 'kalshi')
    for event in events:
        spool.extend(event_rows(event, 'BTC-USD'))
    spool.close()
    compact(tmp_path / 'spool', root, keep_days=0, now=LATER)
    return root


# ---------------------------------------------------------------------------
# The regroup is the inverse of the split
# ---------------------------------------------------------------------------

def test_a_snapshot_survives_the_round_trip_through_level_rows():
    event = snapshot(T0 + 10)
    back = list(events_from_rows(
        row_tuple(r) for r in event_rows(event, 'BTC-USD')))
    assert back == [event]


def test_a_delta_keeps_its_signed_size_and_its_absolute_flag():
    event = delta(T0 + 20, change=-12.5)
    back = list(events_from_rows(
        row_tuple(r) for r in event_rows(event, 'BTC-USD')))
    assert back == [event]
    assert back[0].absolute is False


def test_two_snapshots_sharing_a_seq_stay_two_frames():
    """`seq` restarts at 1 on every resubscription, at least once per window.

    Regrouping on `(market_ticker, seq)` alone merges two ladders minutes apart
    into one impossible frame — and snapshots are exactly the frames a replay
    bootstraps from, so the loss falls entirely on the rows that matter.
    """
    first = snapshot(T0 + 10, seq=1)
    second = snapshot(T0 + 400, seq=1, yes=((0.40, 9.0),), no=((0.59, 8.0),))
    rows = [row_tuple(r) for e in (first, second)
            for r in event_rows(e, 'BTC-USD')]
    assert list(events_from_rows(rows)) == [first, second]


def test_the_archive_rebuilds_the_book_the_live_cache_held(tmp_path):
    """The whole claim, end to end: spool, compact, read back, fold, compare."""
    events = [snapshot(T0 + 10), delta(T0 + 20, seq=2),
              delta(T0 + 30, seq=3, price=0.31, change=-50.0),
              delta(T0 + 40, seq=4, price=0.68, change=5.0, side='no')]
    live = BookCache()
    for event in events:
        live.apply(event)

    root = archive(tmp_path, events)
    rebuilt = BookCache()
    for event in archive_events(hour_files(root)):
        rebuilt.apply(event)

    assert rebuilt.ladder(TICKER).yes == live.ladder(TICKER).yes
    assert rebuilt.ladder(TICKER).no == live.ladder(TICKER).no
    # The level the delta emptied is gone, not resting at float residue.
    assert 0.31 not in dict(rebuilt.ladder(TICKER).yes)


# ---------------------------------------------------------------------------
# Reading a window out of a ticker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('ticker, opens', [
    ('KXBTC15M-26OCT180800-00', '2026-10-18T11:45:00Z'),
    ('KXBTC15M-26SEP161915-15', '2026-09-16T23:00:00Z'),
    ('KXETH15M-26AUG262330-30', '2026-08-27T03:15:00Z'),
])
def test_the_ticker_names_the_close_and_the_window_opens_before_it(ticker, opens):
    assert window_open(ticker) == pd.Timestamp(opens)


@pytest.mark.parametrize('ticker', [
    'KXBTCD-26AUG2317-T86749.99',   # a threshold ladder, not an up/down market
    'KXBTC15M-26OCT180807-07',      # not on a quarter-hour boundary
    'KXBTC15M-26XXX180800',         # not a month
    '', 'nonsense',
])
def test_a_ticker_that_is_not_one_of_ours_is_skipped_rather_than_guessed_at(ticker):
    assert window_open(ticker) is None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_a_cache_on_the_wall_clock_is_refused_rather_than_mis_dated():
    """The failure this guard replaces produced a perfect ladder and a wrong date.

    A `BookCache()` left on `time.time` folds every frame correctly and reports
    an age of however long ago the archive is, so `ws_row` stamps
    `read_at - age` before the window opened and `available_time` silently
    collapses onto `event_time`.
    """
    stray, clock = BookCache(), replay_cache()[1]
    with pytest.raises(ValueError, match='replay_cache'):
        list(sampled_books([snapshot(T0 + 10)], stray, clock))


def test_marks_land_on_the_configured_phase():
    assert next_mark(T0, phase=25.0, period=60.0) == T0 + 25
    assert next_mark(T0 + 25, phase=25.0, period=60.0) == T0 + 85
    assert next_mark(T0 + 24.9, phase=25.0, period=60.0) == T0 + 25


def test_a_sample_is_only_emitted_once_a_later_frame_proves_the_fold_reached_it():
    """No row for the stretch after the last frame.

    A book dated by a clock with nothing behind it is an assertion, and at the
    end of an archive it is precisely the row somebody reads as "the market
    went quiet".
    """
    cache, clock = replay_cache()
    events = [snapshot(T0 + 10), delta(T0 + 20), delta(T0 + 50, seq=3)]
    marks = list(sampled_books(events, cache, clock, phase=25.0))
    assert marks == [T0 + 25]


def test_a_dark_stretch_invalidates_every_book_and_is_not_filled_in():
    """Silence at ~800 frames a second means the recorder was down.

    Folding across it would carry a pre-outage book forward under a
    post-outage timestamp, which is a wrong answer that looks entirely right.
    """
    cache, clock = replay_cache()
    events = [snapshot(T0 + 10), delta(T0 + 20), delta(T0 + 40, seq=3),
              delta(T0 + 700, seq=4), delta(T0 + 750, seq=5)]
    marks = list(sampled_books(events, cache, clock, phase=25.0,
                               max_event_gap=60.0))
    # Nothing between the two bursts, and the book is condemned until a
    # snapshot rebuilds it.
    assert marks == [T0 + 25, T0 + 745]
    assert cache.gapped(TICKER)


def test_the_replayed_row_is_dated_by_the_frame_not_by_a_wall_clock(tmp_path):
    cache, clock = replay_cache()
    sampler = replay_ladder.Sampler()
    events = [snapshot(T0 + 10), delta(T0 + 20), delta(T0 + 50, seq=3)]
    rows = []
    for mark in sampled_books(events, cache, clock, phase=25.0):
        rows.extend(sampler.rows(cache, mark))

    assert len(rows) == 1
    row = rows[0]
    assert row['event_time'] == OPEN
    # The last frame folded was at T0+20, and nothing in this row comes from a
    # clock that was running while it was built.
    assert row['available_time'] == pd.Timestamp(T0 + 20, unit='s', tz='UTC')
    assert row['book_age_ms'] == pytest.approx(5000.0)
    assert row['minute_into_window'] == pytest.approx(25 / 60.0, abs=1e-3)
    assert row['transport'] == 'ws_replay'
    assert json.loads(row['yes_levels']) == [[0.30, 125.0], [0.31, 50.0]]


def test_nothing_is_sampled_outside_a_market_s_own_fifteen_minutes():
    """The same filter the live recorder applies, and for a sharper reason here.

    A settled Kalshi market serves an empty book, and a snapshot with both
    sides empty leaves NO rows in the archive at all — so the replay would hold
    the last pre-settlement ladder forever and happily stamp it onto minute 40.
    """
    cache, clock = replay_cache()
    sampler = replay_ladder.Sampler()
    late = T0 + 25 * 60
    events = [snapshot(T0 + 10), delta(T0 + 20), delta(T0 + 40, seq=3),
              delta(late, seq=4), delta(late + 40, seq=5)]
    rows = []
    for mark in sampled_books(events, cache, clock, phase=25.0):
        rows.extend(sampler.rows(cache, mark))

    assert {r['minute_into_window'] <= 15 for r in rows} == {True}
    # And the settled market is dropped rather than left to go quietly stale.
    assert TICKER not in cache.tickers()


def test_a_condemned_book_is_written_and_written_as_condemned():
    cache, clock = replay_cache()
    sampler = replay_ladder.Sampler()
    events = [snapshot(T0 + 10, seq=1), delta(T0 + 15, seq=2),
              delta(T0 + 20, seq=9),            # seq 3..8 never arrived
              delta(T0 + 50, seq=10)]
    rows = []
    for mark in sampled_books(events, cache, clock, phase=25.0):
        rows.extend(sampler.rows(cache, mark))

    assert [r['quality'] for r in rows] == ['suspicious']


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _run(root, **kwargs):
    store = ResearchStore(root)
    options = dict(start=None, end=None, phase=25.0, window_minutes=15,
                   transport=replay_ladder.DEFAULT_TRANSPORT,
                   max_event_gap=60.0, write=True)
    options.update(kwargs)
    return store, replay_ladder.replay(store, **options)


def _events():
    # Spaced well under `max_event_gap`, so the stream reads as a live one
    # rather than as a recorder that kept falling over.
    return [snapshot(T0 + 10)] + [
        delta(T0 + t, seq=i + 2) for i, t in enumerate(range(20, 210, 30))]


def test_a_dry_run_writes_nothing(tmp_path):
    root = archive(tmp_path, _events())
    store, stats = _run(root, write=False)
    assert stats['rows'] > 0
    assert not list((root / 'venue_ladder').rglob('*.parquet')) \
        if (root / 'venue_ladder').exists() else True
    assert store.read('venue_ladder').empty


def test_the_rows_land_in_the_store_and_read_back(tmp_path):
    root = archive(tmp_path, _events())
    store, stats = _run(root)
    assert stats['written'] == stats['rows'] > 0
    got = store.read('venue_ladder')
    assert len(got) == stats['rows']
    assert set(got['transport']) == {'ws_replay'}
    assert set(got['symbol']) == {'BTC-USD'}


def test_running_twice_changes_nothing(tmp_path):
    root = archive(tmp_path, _events())
    store, first = _run(root)
    before = store.read('venue_ladder')
    _, second = _run(root)
    after = store.read('venue_ladder')
    assert second['rows'] == first['rows']
    pd.testing.assert_frame_equal(before, after)


def test_a_replayed_row_never_supersedes_a_recorded_one(tmp_path):
    """`transport` is part of the event key, so the two are independent.

    Without that the later `available_time` silently wins and a reconstruction
    quietly replaces the recording it is supposed to be checked against — the
    exact failure already recorded for `venue_depth`, where 58 overlapping
    pairs were on disk and invisible to every read.
    """
    root = archive(tmp_path, _events())
    store = ResearchStore(root)
    store.write('venue_ladder', pd.DataFrame([{
        'venue': 'kalshi', 'symbol': 'BTC-USD', 'event_time': OPEN,
        'available_time': OPEN + pd.Timedelta(seconds=30), 'quality': 'valid',
        'market_ticker': TICKER, 'window_open': OPEN,
        'minute_into_window': 0.5, 'yes_levels': '[[0.9, 1.0]]',
        'no_levels': '[]', 'yes_total': 1.0, 'no_total': 0.0,
        'transport': 'ws', 'book_age_ms': 1.0}]))
    _run(root)

    got = store.read('venue_ladder')
    same_minute = got[got['event_time'] == OPEN]
    assert set(same_minute['transport']) == {'ws', 'ws_replay'}
    recorded = same_minute[same_minute['transport'] == 'ws'].iloc[0]
    assert recorded['yes_levels'] == '[[0.9, 1.0]]'


def test_the_recorder_s_own_partition_file_is_never_rewritten(tmp_path):
    """The live recorder is writing this partition every minute.

    `ResearchStore.write` reads the whole `(venue, symbol, month)` partition,
    merges and rewrites it. Two of those over one file is a lost write, on the
    table whose entire justification is that a minute lost is gone — so the
    replay writes its own immutable file beside the recorder's instead.
    """
    root = archive(tmp_path, _events())
    store = ResearchStore(root)
    store.write('venue_ladder', pd.DataFrame([{
        'venue': 'kalshi', 'symbol': 'BTC-USD', 'event_time': OPEN,
        'available_time': OPEN, 'quality': 'valid', 'market_ticker': TICKER,
        'window_open': OPEN, 'minute_into_window': 0.5,
        'yes_levels': '[]', 'no_levels': '[]', 'yes_total': 0.0,
        'no_total': 0.0, 'transport': 'ws', 'book_age_ms': 1.0}]))
    live = (root / 'venue_ladder' / 'venue=kalshi' / 'symbol=BTC-USD'
            / 'month=2026-10' / 'data.parquet')
    stamp, payload = live.stat().st_mtime_ns, live.read_bytes()

    _run(root)

    assert live.read_bytes() == payload
    assert live.stat().st_mtime_ns == stamp
    assert (live.parent / 'replay-2026-10-18.parquet').exists()


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_the_default_is_a_dry_run(tmp_path):
    root = archive(tmp_path, _events())
    assert replay_ladder.main(['--research-store', str(root)]) == 0
    assert ResearchStore(root).read('venue_ladder').empty


def test_it_refuses_to_label_a_reconstruction_as_a_recording(tmp_path):
    root = archive(tmp_path, _events())
    for label in ('ws', 'rest'):
        assert replay_ladder.main(
            ['--research-store', str(root), '--transport', label, '--write']) == 2
    assert ResearchStore(root).read('venue_ladder').empty


def test_dry_run_and_write_together_are_refused_rather_than_ranked(tmp_path):
    root = archive(tmp_path, _events())
    assert replay_ladder.main(
        ['--research-store', str(root), '--dry-run', '--write']) == 2
    assert ResearchStore(root).read('venue_ladder').empty


def test_a_second_run_resumes_past_the_days_already_written(tmp_path, caplog):
    root = archive(tmp_path, _events())
    assert replay_ladder.main(['--research-store', str(root), '--write']) == 0
    before = ResearchStore(root).read('venue_ladder')

    with caplog.at_level('INFO'):
        assert replay_ladder.main(['--research-store', str(root), '--write']) == 0
    assert 'already replayed' in caplog.text
    pd.testing.assert_frame_equal(before, ResearchStore(root).read('venue_ladder'))


# ---------------------------------------------------------------------------
# resume: a silent no-op that logs "done" is the failure this module repairs
# ---------------------------------------------------------------------------

def test_an_explicit_start_is_never_overridden():
    """`--start 2026-10-02 --end 2026-10-02T23:59` became the inverted window
    `[2026-10-19, 2026-10-02)` -- zero hour files, zero rows, exit success."""
    from scripts.replay_ladder import resume_point
    import pandas as pd

    done = {'2026-09-30', '2026-10-01', '2026-10-03', '2026-10-18'}
    asked = pd.Timestamp('2026-10-02', tz='UTC')
    start, _holes = resume_point(done, asked)
    assert start == asked, 'an explicit start is an instruction, not a hint'


def test_resume_stops_at_a_hole_rather_than_jumping_past_it():
    """`max(done) + 1 day` assumes the finished days are contiguous. A
    parallel run left 2026-10-02 unwritten while 2026-10-18 succeeded, so the
    resume point jumped to 10-19 and the hole could never be filled."""
    from scripts.replay_ladder import resume_point
    import pandas as pd

    done = {'2026-09-30', '2026-10-01', '2026-10-03', '2026-10-18'}
    start, holes = resume_point(done, None)
    assert start == pd.Timestamp('2026-10-02', tz='UTC'), \
        'resume must land ON the hole, not past it'
    assert [d.date().isoformat() for d in holes] == ['2026-10-03', '2026-10-18']


def test_a_contiguous_run_resumes_after_the_last_day():
    from scripts.replay_ladder import resume_point
    import pandas as pd

    done = {'2026-09-30', '2026-10-01', '2026-10-02'}
    start, holes = resume_point(done, None)
    assert start == pd.Timestamp('2026-10-03', tz='UTC')
    assert holes == []


def test_force_leaves_the_start_alone():
    from scripts.replay_ladder import resume_point

    done = {'2026-09-30', '2026-10-01'}
    start, holes = resume_point(done, None, force=True)
    assert start is None and holes == []


def test_nothing_done_means_nothing_to_resume_from():
    from scripts.replay_ladder import resume_point

    assert resume_point(set(), None) == (None, [])
