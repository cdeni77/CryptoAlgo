"""Tests for rebuilding the strike-ladder implied sigma from history.

The inversion is reused from `scripts/record_implied_vol.implied_sigma`, so live
and backfill share one definition. What is new is the cross-section: history is
one tick series per strike, aligned here onto a common instant.
"""

from __future__ import annotations

import datetime as dt

from research.collect.implied_vol_backfill import (
    LADDER_SERIES, cross_section, pack_series, sample_instants,
)


def _packed(by_strike: dict) -> dict:
    """Strikes to packed tick series, through the real code path."""
    return {k: pack_series(v) for k, v in by_strike.items()}

UTC = dt.timezone.utc


def _snap(ts_ms, bid, ask):
    return {'timestamp': ts_ms,
            'yes_bids': [{'price': bid, 'size': 10}] if bid is not None else [],
            'yes_asks': [{'price': ask, 'size': 10}] if ask is not None else []}


def test_the_cross_section_is_the_last_quote_at_or_before_the_instant():
    """The cross-section takes the last quote at or before the instant, never a later one.
    """
    series = {
        70000.0: [_snap(1_000, 30, 31), _snap(5_000, 40, 41)],
        71000.0: [_snap(2_000, 20, 21), _snap(9_000, 25, 26)],
    }
    rungs = cross_section(_packed(series), at_ms=6_000)
    assert dict(rungs) == {70000.0: 0.405, 71000.0: 0.205}


def test_a_strike_with_no_tick_yet_is_omitted_rather_than_guessed():
    series = {70000.0: [_snap(9_000, 30, 31)]}
    assert cross_section(_packed(series), at_ms=1_000) == []


def test_a_one_sided_quote_is_omitted_because_it_has_no_mid():
    """A one-sided quote has no mid, so its strike is omitted."""
    series = {70000.0: [_snap(1_000, 30, None)],
              71000.0: [_snap(1_000, 20, 21)]}
    assert [s for s, _ in cross_section(_packed(series), at_ms=2_000)] == [71000.0]


def test_the_cross_section_is_sorted_by_strike():
    series = {72000.0: [_snap(1, 10, 11)], 70000.0: [_snap(1, 30, 31)],
              71000.0: [_snap(1, 20, 21)]}
    assert [s for s, _ in cross_section(_packed(series), at_ms=5)] == [70000.0, 71000.0, 72000.0]


def test_probabilities_come_back_as_fractions_not_cents():
    series = {70000.0: [_snap(1, 38, 39)]}
    _, p = cross_section(_packed(series), at_ms=5)[0]
    assert 0.0 < p < 1.0 and abs(p - 0.385) < 1e-9


def test_sample_instants_walk_the_quarter_hour_grid_inside_the_ladder():
    """Samples sit on the quarter-hour grid where the windows decide."""
    o = dt.datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    got = sample_instants(o, c, minutes=15)
    assert got == [o, o + dt.timedelta(minutes=15), o + dt.timedelta(minutes=30),
                   o + dt.timedelta(minutes=45)]


def test_sample_instants_excludes_the_close_itself():
    """The close itself is excluded: no time remains, so sigma is undefined."""
    o = dt.datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 20, 30, tzinfo=UTC)
    assert c not in sample_instants(o, c, minutes=15)


def test_all_three_assets_have_a_ladder_series():
    """All three assets have a ladder series."""
    assert set(LADDER_SERIES.values()) == {'BTC-USD', 'ETH-USD', 'SOL-USD'}


def test_the_inversion_itself_is_reused_not_reimplemented():
    from research.collect import implied_vol_backfill as m
    from scripts.record_implied_vol import implied_sigma
    assert m.implied_sigma is implied_sigma


# --- fetching a ladder's cross-section concurrently -------------------------
#
# Strikes are fetched concurrently because the run is bound by round-trip
# latency, not the rate limit. Concurrency must not change what is collected.

class _FakeApi:
    """Answers a book for every strike except those in `empty_for`, and records requests.
    """

    def __init__(self, empty_for=()):
        self.empty_for = set(empty_for)
        self.asked = []
        self._lock = __import__('threading').Lock()

    def get(self, path, params, **kw):
        with self._lock:
            self.asked.append(params.get('ticker'))
        if params.get('ticker') in self.empty_for:
            return {'snapshots': []}, True
        return {'snapshots': [_snap(1_000, 30, 31)]}, True


def _strikes():
    return [(70000.0, 'T70000'), (71000.0, 'T71000'), (72000.0, 'T72000')]


def test_fetching_a_cross_section_concurrently_matches_the_serial_result():
    """One worker and eight workers return the same cross-section."""
    from research.collect.implied_vol_backfill import fetch_rungs
    one = fetch_rungs(_FakeApi(), _strikes(), 0, 1, workers=1)
    many = fetch_rungs(_FakeApi(), _strikes(), 0, 1, workers=8)
    assert set(one) == set(many) == {70000.0, 71000.0, 72000.0}
    assert one == many


def test_a_strike_whose_book_is_empty_is_omitted_not_stored_blank():
    """A strike with an empty book is omitted, not stored as zero."""
    from research.collect.implied_vol_backfill import fetch_rungs
    got = fetch_rungs(_FakeApi(empty_for={'T71000'}), _strikes(), 0, 1, workers=4)
    assert set(got) == {70000.0, 72000.0}


def test_every_strike_is_asked_for_exactly_once():
    """Each strike is requested exactly once."""
    from research.collect.implied_vol_backfill import fetch_rungs
    api = _FakeApi()
    fetch_rungs(api, _strikes(), 0, 1, workers=4)
    assert sorted(api.asked) == ['T70000', 'T71000', 'T72000']


# --- sampling density ------------------------------------------------------
#
# Each strike's tick series is already fetched for the whole window, so extra
# sample instants cost no requests. The trader decides at several offsets, so
# the feature needs sigma at several minutes-to-close.

def test_a_finer_sample_step_asks_more_questions_of_the_same_fetched_books():
    """A finer step samples more instants from the same fetched books."""
    o = dt.datetime(2026, 7, 29, 20, 40, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    coarse = sample_instants(o, c, minutes=15)
    fine = sample_instants(o, c, minutes=5)
    assert len(coarse) == 2
    assert [(c - t).total_seconds() / 60 for t in fine] == [20.0, 15.0, 10.0, 5.0]


def test_the_sample_step_is_configurable_and_defaults_below_the_window():
    """`SAMPLE_MINUTES` stays below `TAIL_MINUTES`, or the grid collapses to one instant.
    """
    from research.collect import implied_vol_backfill as m
    assert m.SAMPLE_MINUTES < m.TAIL_MINUTES


def test_the_finer_grid_still_excludes_the_close():
    """The finer grid still excludes the close."""
    o = dt.datetime(2026, 7, 29, 20, 40, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    assert c not in sample_instants(o, c, minutes=1)


# --- the fetch window must start before the first instant -------------------
#
# A book is a step function, so the state at T is the last tick at or before T.
# If the fetch starts at T, nothing precedes T and the first instant is empty.
# Rung count grows with time to close until the ladder opens, so samples sit in
# the 20-50 minute band before the close.

def test_the_fetch_window_starts_strictly_before_the_first_sampled_instant():
    """The fetch starts strictly before the first sampled instant."""
    from research.collect.implied_vol_backfill import windows_for
    opened = dt.datetime(2026, 4, 10, 10, 0, tzinfo=UTC)
    closes = dt.datetime(2026, 4, 10, 11, 0, tzinfo=UTC)
    fetch_from, instants = windows_for(opened, closes)
    assert instants, 'no instants to sample'
    assert fetch_from < instants[0]


def test_the_sampled_instants_sit_in_the_band_where_rungs_exist():
    """Sampled instants fall between 5 and 50 minutes before the close."""
    from research.collect.implied_vol_backfill import windows_for
    opened = dt.datetime(2026, 4, 10, 10, 0, tzinfo=UTC)
    closes = dt.datetime(2026, 4, 10, 11, 0, tzinfo=UTC)
    _, instants = windows_for(opened, closes)
    mins = [(closes - t).total_seconds() / 60 for t in instants]
    assert max(mins) <= 50, f'samples an unopened ladder: {max(mins)}'
    assert min(mins) >= 5, f'samples too close to expiry: {min(mins)}'
    assert len(instants) >= 8, f'only {len(instants)} instants, density wasted'


def test_a_ladder_that_opened_late_is_clamped_not_sampled_before_it_existed():
    """A ladder that opened late is not sampled before it opened."""
    from research.collect.implied_vol_backfill import windows_for
    closes = dt.datetime(2026, 4, 10, 11, 0, tzinfo=UTC)
    opened = closes - dt.timedelta(minutes=25)
    fetch_from, instants = windows_for(opened, closes)
    assert fetch_from >= opened
    assert all(t >= opened for t in instants)


def test_the_close_itself_is_never_sampled():
    from research.collect.implied_vol_backfill import windows_for
    closes = dt.datetime(2026, 4, 10, 11, 0, tzinfo=UTC)
    _, instants = windows_for(closes - dt.timedelta(hours=1), closes)
    assert closes not in instants


# --- the stride selects from all ladders, not from the unfinished ones -------
#
# Applying the stride after removing finished ladders re-samples the remainder
# and changes the scope on every restart.

def test_the_stride_selects_the_same_ladders_regardless_of_what_is_done():
    from research.collect.implied_vol_backfill import select_ladders
    every = [f'L{i:03d}' for i in range(10)]
    fresh = select_ladders(every, done=set(), stride=2)
    partly = select_ladders(every, done={'L000', 'L002'}, stride=2)
    assert fresh == ['L000', 'L002', 'L004', 'L006', 'L008']
    # The same target set, minus what is finished. Never a new selection.
    assert partly == ['L004', 'L006', 'L008']


def test_resuming_never_adds_a_ladder_the_stride_had_skipped():
    """Resuming never adds a ladder the stride skipped."""
    from research.collect.implied_vol_backfill import select_ladders
    every = [f'L{i:03d}' for i in range(100)]
    target = set(select_ladders(every, done=set(), stride=3))
    done = set(list(target)[:10])
    assert set(select_ladders(every, done=done, stride=3)) <= target


def test_a_stride_of_one_is_every_ladder():
    from research.collect.implied_vol_backfill import select_ladders
    every = [f'L{i:03d}' for i in range(5)]
    assert select_ladders(every, done=set(), stride=1) == every


def test_max_ladders_caps_after_the_stride_not_before():
    """`max_ladders` applies after the stride, so the stride is not collapsed to the
    earliest ladders.
    """
    from research.collect.implied_vol_backfill import select_ladders
    every = [f'L{i:03d}' for i in range(20)]
    got = select_ladders(every, done=set(), stride=4, max_ladders=3)
    assert got == ['L000', 'L004', 'L008']


def test_ladders_are_ordered_by_close_time_not_ticker_text():
    """Ladders are ordered by close time, not ticker text (which sorts by month name).
    """
    import datetime as dt
    from research.collect.implied_vol_backfill import select_ladders
    utc = dt.timezone.utc
    every = {'KXBTCD-26AUG0117': (dt.datetime(2026, 8, 1, tzinfo=utc),
                                  dt.datetime(2026, 8, 1, 17, tzinfo=utc)),
             'KXBTCD-26APR0117': (dt.datetime(2026, 4, 1, tzinfo=utc),
                                  dt.datetime(2026, 4, 1, 17, tzinfo=utc)),
             'KXBTCD-26JUL0117': (dt.datetime(2026, 7, 1, tzinfo=utc),
                                  dt.datetime(2026, 7, 1, 17, tzinfo=utc))}
    assert select_ladders(every, done=set(), stride=1) == [
        'KXBTCD-26APR0117', 'KXBTCD-26JUL0117', 'KXBTCD-26AUG0117']
    recent = select_ladders(every, done=set(), stride=1,
                            since=dt.datetime(2026, 7, 1, tzinfo=utc))
    assert recent == ['KXBTCD-26JUL0117', 'KXBTCD-26AUG0117']
