"""The Polymarket socket recorder, and the failures the Kalshi one paid for.

**A silent socket cannot wake itself.** `record_stream.consume` first checked
its refresh deadline inside the loop body, so when the subscribed markets
settled at a window boundary the venue stopped sending and the loop waited
forever for a frame that was never coming. Nothing raised, `supervise` saw a
coroutine legitimately awaiting, and the recorder sat dead behind a container
reporting healthy. So silence is a CONDITION here, enforced on a timeout — at
~300 frames a second, fifteen seconds of nothing means the market settled or the
connection died, and both are repaired by resubscribing.

**The window rolls every fifteen minutes** and the token ids change with it. A
recorder that subscribes once holds a book for a market that has settled, which
looks exactly like a quiet market.

**What it publishes is what `cross_venue_row` reads**, in the same shape as
`record_pm_ladder.CACHE`: best bid and ask in CENTS with an `at` stamp, so the
staleness guard added today keeps working and the two sources are swappable.
"""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from scripts import record_pm_stream as rec


def test_silence_is_a_condition_not_a_wait():
    """The exact deadlock: a socket that goes quiet must return a reason, not
    block until a frame that is never coming."""
    async def _run():
        async def never():
            await asyncio.sleep(3600)

        reason = await rec.consume(_Stream(never), until=None,
                                   silence_seconds=0.2)
        return reason

    reason = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    assert 'silen' in reason.lower(), reason


def test_the_refresh_deadline_fires_even_while_frames_pour_in():
    """The other half: a busy socket must still hit its window rollover."""
    async def _run():
        async def flood():
            return {'event_type': 'price_change', 'price_changes': []}
        return await rec.consume(_Stream(flood), until=0.2, silence_seconds=5.0)

    reason = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    assert 'refresh' in reason.lower() or 'window' in reason.lower(), reason


def test_publish_matches_the_shape_cross_venue_row_reads():
    """Same keys and units as `record_pm_ladder.CACHE`: cents, plus a stamp for
    the staleness guard."""
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}],
                'asks': [{'price': '0.46', 'size': '150'}]})
    rec.publish(book, {'T': 'BTC-USD'})
    entry = rec.CACHE['BTC-USD']
    assert entry['best_bid'] == pytest.approx(44.0), 'cents, not dollars'
    assert entry['best_ask'] == pytest.approx(46.0)
    assert isinstance(entry['at'], pd.Timestamp)


def test_a_one_sided_book_is_not_published_as_a_touch():
    """A lone bid says the probability is at LEAST something, which is not a
    probability — the same rule `_two_sided_mid` applies."""
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}], 'asks': []})
    rec.publish(book, {'T': 'BTC-USD'})
    assert 'BTC-USD' not in rec.CACHE or rec.CACHE['BTC-USD'].get('best_ask') is None


class _Stream:
    """A stand-in socket whose next frame comes from `factory`."""

    def __init__(self, factory):
        self._factory = factory

    def events(self):
        factory = self._factory

        async def _gen():
            while True:
                out = await factory()
                if out is not None:
                    yield out
        return _gen()


def test_the_streamed_window_is_the_one_kalshi_settles():
    """The alignment that "looks entirely right" when wrong.

    Kalshi's ticker is keyed on the SETTLE (KXBTC15M-26SEP032245-45 settles at
    22:45), Polymarket's slug on the window OPEN. They name the same fifteen
    minutes only if the slug's stamp is the settle minus the window length. Read
    the slug as a close instead and every window shifts fifteen minutes, the
    request still returns a healthy book, and agreement with our label drops to
    a coin flip — which is how this was found the first time.
    """
    import datetime as dt
    from scripts.record_pm_ladder import slug_for, window_of

    # A decision inside the 22:30-22:45 window, as Kalshi's -2245 ticker names.
    now = dt.datetime(2026, 9, 3, 22, 37, tzinfo=dt.timezone.utc)
    opened = window_of(slug_for('btc', now))
    assert opened == pd.Timestamp('2026-09-03 22:30', tz='UTC'), (
        'the slug must name the window OPEN that Kalshi settles at 22:45')
    assert opened + pd.Timedelta(minutes=15) == pd.Timestamp(
        '2026-09-03 22:45', tz='UTC')


def test_a_decision_at_the_boundary_takes_the_window_it_is_inside():
    """Floor, never ceil. At exactly 22:45 the new window has opened, and
    naming the NEXT one returns a real book for a market fifteen minutes
    ahead."""
    import datetime as dt
    from scripts.record_pm_ladder import slug_for, window_of

    at_open = dt.datetime(2026, 9, 3, 22, 45, tzinfo=dt.timezone.utc)
    assert window_of(slug_for('btc', at_open)) == pd.Timestamp(
        '2026-09-03 22:45', tz='UTC')
    one_before = dt.datetime(2026, 9, 3, 22, 44, 59, tzinfo=dt.timezone.utc)
    assert window_of(slug_for('btc', one_before)) == pd.Timestamp(
        '2026-09-03 22:30', tz='UTC')
