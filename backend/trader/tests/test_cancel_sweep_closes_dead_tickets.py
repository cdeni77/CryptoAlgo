"""A cancel the venue answers with 404 must close the ticket.

`cancel_stale_resting` runs between the book read and `decided_at`, so a
ticket it can never cancel costs the FORECAST a round trip on every later
cycle. Measured on one container life: 585 cancel attempts, all 404, against
65 distinct orders, 3-4s per sweep against a ~3s decision budget.
"""

from __future__ import annotations

import asyncio
import types

import pandas as pd
import pytest

from data_collection.kalshi_client import KalshiError
from scripts import live


class Ticket:
    def __init__(self, tid=1, order_id='abc', age_seconds=600):
        self.id = tid
        self.venue_order_id = order_id
        self.created_at = (pd.Timestamp.now(tz='UTC')
                           - pd.Timedelta(seconds=age_seconds))


class Writer:
    def __init__(self, tickets):
        self._tickets = tickets
        self.resolved = []

    def open_tickets(self, limit=50):
        return list(self._tickets)

    def resolve_ticket(self, tid, **fields):
        self.resolved.append((tid, fields))


class Kalshi:
    def __init__(self, error):
        self.error = error
        self.attempts = []

    async def cancel_order(self, order_id):
        self.attempts.append(order_id)
        if self.error:
            raise self.error
        return {}


CONFIG = types.SimpleNamespace(maker_entry=True, maker_cancel_seconds=60)


def run(writer, kalshi):
    return asyncio.run(
        live.cancel_stale_resting(writer, kalshi, CONFIG,
                                  now=pd.Timestamp.now(tz='UTC')))


def test_a_404_closes_the_ticket_so_it_is_never_swept_again():
    writer = Writer([Ticket()])
    kalshi = Kalshi(KalshiError('DELETE ... -> 404: not_found', status=404))
    run(writer, kalshi)

    assert len(writer.resolved) == 1, 'the dead ticket must be closed'
    tid, fields = writer.resolved[0]
    assert tid == 1
    assert fields['status'] not in live.PgWriter.OPEN_TICKET_STATUSES, \
        'the ticket must leave the open set, or it is swept again'
    assert 'no such order' in fields['note']


def test_a_transient_failure_leaves_the_ticket_open_for_a_retry():
    """A 500 is not evidence the order is gone."""
    writer = Writer([Ticket()])
    kalshi = Kalshi(KalshiError('DELETE ... -> 503: busy', status=503))
    run(writer, kalshi)
    assert writer.resolved == [], 'a transient error must not close the ticket'


def test_a_successful_cancel_still_resolves_as_before():
    writer = Writer([Ticket()])
    kalshi = Kalshi(None)
    assert run(writer, kalshi) == 1
    assert writer.resolved[0][1]['status'] == 'cancelled'
    assert 'unfilled' in writer.resolved[0][1]['note']


def test_a_ticket_younger_than_the_cutoff_is_left_alone():
    writer = Writer([Ticket(age_seconds=5)])
    kalshi = Kalshi(None)
    run(writer, kalshi)
    assert kalshi.attempts == [] and writer.resolved == []


def test_every_dead_ticket_is_closed_in_one_sweep():
    """The leak was 65 tickets re-swept forever; one pass must clear them."""
    writer = Writer([Ticket(tid=i, order_id=f'o{i}') for i in range(65)])
    kalshi = Kalshi(KalshiError('-> 404: not_found', status=404))
    run(writer, kalshi)
    assert len(kalshi.attempts) == 65
    assert len(writer.resolved) == 65
