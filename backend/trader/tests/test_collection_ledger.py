"""Tests for the collection ledger, which records the state of every window.

The ledger separates pending, empty and error, retryable from terminal, and
survives a kill mid-window. "Never asked" is a state, not an inference from absence.
"""

from __future__ import annotations

import datetime as dt

import pytest

from research.collect.ledger import Ledger, MAX_ATTEMPTS


def _w(day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(2026, 1, day, hour, minute, tzinfo=dt.timezone.utc)


@pytest.fixture()
def ledger(tmp_path):
    return Ledger(str(tmp_path / 'ledger.db'))


def _items(n=3, venue='kalshi'):
    return [(venue, 'BTC-USD', _w(8, h), f'TICK-{h}') for h in range(n)]


# -- seeding -----------------------------------------------------------------

def test_seeding_creates_pending_rows(ledger):
    ledger.seed(_items(3))
    assert ledger.counts()['pending'] == 3


def test_a_seeded_row_carries_the_venues_own_market_id(ledger):
    ledger.seed([('kalshi', 'BTC-USD', _w(8), 'KXBTC15M-26JAN080015-15')])
    item = ledger.claim(1)[0]
    assert item.market_id == 'KXBTC15M-26JAN080015-15'


def test_reseeding_is_idempotent_and_never_resurrects_finished_work(ledger):
    """Re-seeding never resets finished work."""
    ledger.seed(_items(3))
    done = ledger.claim(3)
    for item in done:
        ledger.record(item, 'ok', snapshots=100)
    ledger.seed(_items(3))  # same items again
    assert ledger.counts()['ok'] == 3
    assert ledger.counts().get('pending', 0) == 0


def test_reseeding_adds_genuinely_new_windows(ledger):
    ledger.seed(_items(2))
    ledger.seed(_items(4))
    assert ledger.counts()['pending'] == 4


# -- pending vs empty vs error ------------------------------------------------

def test_empty_is_a_result_and_is_not_retried(ledger):
    """`empty` is an answer from the venue, so it is not retried."""
    ledger.seed(_items(1))
    ledger.record(ledger.claim(1)[0], 'empty')
    assert ledger.counts()['empty'] == 1
    assert ledger.claim(10) == []


def test_error_is_retried_because_the_question_was_never_answered(ledger):
    ledger.seed(_items(1))
    ledger.record(ledger.claim(1)[0], 'error', error='429')
    again = ledger.claim(10)
    assert len(again) == 1, 'a failed request must come back around'


def test_ok_is_terminal(ledger):
    ledger.seed(_items(1))
    ledger.record(ledger.claim(1)[0], 'ok', snapshots=1234)
    assert ledger.claim(10) == []


def test_error_stops_being_retried_once_attempts_are_exhausted(ledger):
    """After `MAX_ATTEMPTS` errors a window stops being retried, so it cannot block the
    queue.
    """
    ledger.seed(_items(1))
    for _ in range(MAX_ATTEMPTS):
        batch = ledger.claim(10)
        assert batch, 'should still be retryable'
        ledger.record(batch[0], 'error', error='boom')
    assert ledger.claim(10) == [], 'must give up after MAX_ATTEMPTS'


def test_attempts_accumulate_across_retries(ledger):
    ledger.seed(_items(1))
    for _ in range(3):
        ledger.record(ledger.claim(1)[0], 'error', error='x')
    row = ledger.rows()[0]
    assert row['attempts'] == 3


def test_the_last_error_is_kept_for_diagnosis(ledger):
    ledger.seed(_items(1))
    ledger.record(ledger.claim(1)[0], 'error', error='500 upstream exploded')
    assert 'upstream exploded' in ledger.rows()[0]['last_error']


# -- resume ------------------------------------------------------------------

def test_resume_after_a_hard_kill_loses_at_most_the_in_flight_window(tmp_path):
    """Resume is a query: a row claimed but never recorded comes back as pending."""
    path = str(tmp_path / 'l.db')
    first = Ledger(path)
    first.seed(_items(5))
    batch = first.claim(2)
    first.record(batch[0], 'ok', snapshots=10)
    del first  # simulate the process dying

    second = Ledger(path)
    assert second.counts()['ok'] == 1
    assert len(second.claim(10)) == 4


def test_claiming_does_not_mark_work_done(ledger):
    """A claim that is never recorded is not lost."""
    ledger.seed(_items(3))
    ledger.claim(3)
    assert Ledger(ledger.path).counts()['pending'] == 3


# -- coverage ----------------------------------------------------------------

def test_coverage_separates_empty_from_error_from_never_asked(ledger):
    """Coverage counts ok, empty and error separately."""
    ledger.seed(_items(3))
    batch = ledger.claim(3)
    ledger.record(batch[0], 'ok', snapshots=5)
    ledger.record(batch[1], 'empty')
    ledger.record(batch[2], 'error', error='timeout')

    cov = ledger.coverage()
    row = [r for r in cov if r['month'] == '2026-01'][0]
    assert row['ok'] == 1 and row['empty'] == 1 and row['error'] == 1


def test_coverage_reports_yield_over_answered_windows_only(ledger):
    """Yield is over answered windows only, so an in-progress run does not look like a
    failing one.
    """
    ledger.seed(_items(4))
    batch = ledger.claim(2)
    ledger.record(batch[0], 'ok', snapshots=5)
    ledger.record(batch[1], 'empty')
    row = [r for r in ledger.coverage() if r['month'] == '2026-01'][0]
    assert row['yield_pct'] == pytest.approx(50.0)


# -- scheduling --------------------------------------------------------------

def test_work_is_claimed_interleaved_by_month_not_venue_by_venue(tmp_path):
    """Work is claimed by month across venues, so each finished month is usable on both
    venues.
    """
    ledger = Ledger(str(tmp_path / 'l.db'))
    ledger.seed([('kalshi', 'BTC-USD', _w(8), 'k1'),
                 ('polymarket', 'BTC-USD', _w(8), 'p1')])
    venues = {item.venue for item in ledger.claim(2)}
    assert venues == {'kalshi', 'polymarket'}


def test_earlier_months_are_claimed_before_later_ones(ledger):
    ledger.seed([('kalshi', 'BTC-USD', dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc), 'm3'),
                 ('kalshi', 'BTC-USD', dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), 'm1')])
    assert ledger.claim(1)[0].market_id == 'm1'


def test_claim_can_be_limited_to_one_month_for_phased_runs(ledger):
    ledger.seed([('kalshi', 'BTC-USD', dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc), 'a'),
                 ('kalshi', 'BTC-USD', dt.datetime(2026, 2, 5, tzinfo=dt.timezone.utc), 'b')])
    got = ledger.claim(10, month='2026-01')
    assert [i.market_id for i in got] == ['a']


def test_claim_can_start_from_a_date_so_a_trial_can_target_a_known_overlap(ledger):
    """`since` lets a trial start at a known date, such as the overlap with live-recorded
    ladders.
    """
    ledger.seed([('kalshi', 'BTC-USD', _w(8), 'early'),
                 ('kalshi', 'BTC-USD', _w(25), 'late')])
    got = ledger.claim(10, since=_w(20))
    assert [i.market_id for i in got] == ['late']


def test_since_and_month_compose(ledger):
    ledger.seed([('kalshi', 'BTC-USD', _w(25), 'jan25'),
                 ('kalshi', 'BTC-USD', dt.datetime(2026, 2, 25, tzinfo=dt.timezone.utc), 'feb25')])
    got = ledger.claim(10, month='2026-01', since=_w(20))
    assert [i.market_id for i in got] == ['jan25']
