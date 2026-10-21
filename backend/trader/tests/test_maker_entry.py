"""Maker entry: rest at the bid instead of lifting the ask.

Measured 2026-10-18 on 54,100 live-recorded rows, the SAME model with no
retraining -- only the execution price changed:

    taker, ask + fee          7,259 trades    -$77.25   -0.108pp/ct  t=-0.17
    MAKER, bid + fee=0       13,400 trades  +$2445.76   +1.608pp/ct  t=+2.91
    MAKER + no 35-65c band   12,588 trades  +$3781.70   +2.346pp/ct  t=+4.74

Every one of those assumes a resting order FILLS, and none ever has on this
account -- all 1,556 recorded fills are `is_taker: true`. So the tests that
matter most here are the ones asserting the path stays OFF unless asked, and
that an order which rests cannot outlive the forecast behind it.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import DEFAULT_CONFIG
from data_collection.kalshi_client import KalshiClient, NotLiveError, Quote


def _quote(**kw):
    base = dict(ticker='KXBTC15M-26OCT180015-15', yes_bid=0.65, yes_ask=0.66,
                no_bid=0.34, no_ask=0.35, last_price=0.65, volume=0,
                open_interest=0, close_time=None, status='active',
                yes_bid_size=441.0, yes_ask_size=120.0)
    base.update(kw)
    return Quote(**base)


def test_a_maker_buys_the_bid_and_a_taker_lifts_the_ask():
    q = _quote()
    assert q.ask_for('up') == pytest.approx(0.66)
    assert q.bid_for('up') == pytest.approx(0.65)
    assert q.ask_for('down') == pytest.approx(0.35)
    assert q.bid_for('down') == pytest.approx(0.34)


def test_resting_is_cheaper_than_crossing_by_the_whole_spread():
    """The economics in one line: a 1c spread is 1c per contract, and on a
    measured 1.000c median spread that is worth as much as the fee itself."""
    q = _quote()
    assert q.ask_for('up') - q.bid_for('up') == pytest.approx(0.01)
    assert q.ask_for('down') - q.bid_for('down') == pytest.approx(0.01)


def test_the_queue_ahead_is_not_the_size_we_could_take():
    """`size_for` caps a taker's order; `rest_size_for` is the queue AHEAD of a
    resting one. Confusing them would read a deep book as a likely fill."""
    q = _quote()
    assert q.size_for('up') == pytest.approx(120.0)        # the ask we'd cross
    assert q.rest_size_for('up') == pytest.approx(441.0)   # the bid we'd join
    assert q.size_for('down') == pytest.approx(441.0)
    assert q.rest_size_for('down') == pytest.approx(120.0)


def test_maker_entry_is_off_by_default():
    """The default is load-bearing: no resting order has ever filled here."""
    assert DEFAULT_CONFIG.maker_entry is False
    # `maker_time_in_force` carries a real value even when maker entry is off:
    # it is the venue's required enum, not a switch. Nothing rests unless
    # `maker_entry` is set, which is the flag that matters here.


def test_the_resting_time_in_force_is_the_value_the_venue_accepts():
    """**Required, and spelled with one 'l'.** Omitting it was measured on
    2026-10-18: all 48 maker orders came back `400 missing_parameters —
    'CreateOrderV2Request.TimeInForce' failed on the 'required' tag`. There is
    no venue default to fall back to."""
    assert DEFAULT_CONFIG.maker_time_in_force == 'good_till_canceled'
    assert 'cancelled' not in DEFAULT_CONFIG.maker_time_in_force, \
        "the venue spells it 'canceled'; 'cancelled' is not a valid value"


def test_a_client_that_cannot_place_also_cannot_cancel():
    """Same guard as `place_order`: a script built to observe must not mutate."""
    client = KalshiClient(live=False)
    with pytest.raises(NotLiveError):
        import asyncio
        asyncio.run(client.cancel_order('abc123'))


def test_a_resting_order_must_not_outlive_its_forecast():
    """The leash is the order's own AGE, not an offset into the window.

    Keyed on the window, one flag meant two different leashes: an order placed
    at +3m lived six minutes and one placed at +12m lived none. What expires is
    the FORECAST -- `sigma_remaining` is computed for the decision instant --
    so the bound has to be the order's age.
    """
    assert DEFAULT_CONFIG.maker_cancel_seconds == 60.0
    # Shorter than a single window, or it cannot bound anything.
    assert 0 < DEFAULT_CONFIG.maker_cancel_seconds < 15 * 60


@pytest.mark.asyncio
async def test_the_sweep_is_inert_unless_maker_entry_is_on():
    """Nothing about the taker path changes by adding this."""
    from scripts.live import cancel_stale_resting

    class Boom:
        async def cancel_order(self, *a, **k):
            raise AssertionError('the sweep must not touch a taker account')

    class NoTickets:
        def open_tickets(self, limit=50):
            raise AssertionError('the sweep must not even look')

    assert await cancel_stale_resting(NoTickets(), Boom(), DEFAULT_CONFIG) == 0
    off = replace(DEFAULT_CONFIG, maker_entry=True, maker_cancel_seconds=0)
    assert await cancel_stale_resting(NoTickets(), Boom(), off) == 0


# --------------------------------------------------------------------------
# pricing the BACKTEST the way the account will actually enter
# --------------------------------------------------------------------------

def test_the_backtest_entry_moves_from_the_ask_to_the_bid():
    """`--assume-maker` alone only zeroes the FEE and leaves the backtest
    lifting the ask, understating the maker case by a whole spread. On a 1c
    book both sides improve by exactly a cent."""
    import pandas as pd
    from core.decide import as_maker_prices
    taker = pd.DataFrame({'ask_up': [0.66], 'ask_down': [0.35]})
    maker = as_maker_prices(taker)
    assert maker.ask_up.iloc[0] == pytest.approx(0.65)
    assert maker.ask_down.iloc[0] == pytest.approx(0.34)
    # and the original is untouched — the transform returns a copy
    assert taker.ask_up.iloc[0] == pytest.approx(0.66)


def test_the_derived_bid_is_exact_when_both_asks_share_a_snapshot():
    """yes_bid = 1 - no_ask holds by construction on one snapshot, which is why
    the backtest can derive what live reads straight off the quote."""
    import pandas as pd
    from core.decide import as_maker_prices
    for yes_bid, yes_ask in ((0.30, 0.31), (0.05, 0.07), (0.90, 0.91)):
        no_ask = 1.0 - yes_bid
        out = as_maker_prices(pd.DataFrame({'ask_up': [yes_ask],
                                            'ask_down': [no_ask]}))
        assert out.ask_up.iloc[0] == pytest.approx(yes_bid)
        assert out.ask_down.iloc[0] == pytest.approx(1.0 - yes_ask)


def test_a_frame_without_quotes_is_left_alone():
    """A backtest with no book prices against the baseline; there is no bid to
    move to, and inventing one would turn a counterfactual into a fill."""
    import pandas as pd
    from core.decide import as_maker_prices
    f = pd.DataFrame({'model_probability': [0.6]})
    assert as_maker_prices(f).equals(f)
    assert as_maker_prices(pd.DataFrame()).empty


def test_resting_at_the_bid_implies_paying_the_maker_fee():
    """Pricing the entry as a maker while charging the taker schedule would
    understate the net edge by the whole fee. Implied, not validated, so the
    two can never be set into disagreement."""
    cfg = replace(DEFAULT_CONFIG, maker_entry=True)
    assert cfg.assume_maker is True
    assert DEFAULT_CONFIG.assume_maker is False       # default untouched


def test_the_maker_fee_the_venue_actually_charges_is_zero():
    """`maker_fee_multiplier` is 0.0 and the venue's own order ticket agrees:
    a 12-contract limit at 63c against a 65/66 book quotes "$7.56 ($0 fee)"."""
    from core.costs import fee_per_contract
    cfg = replace(DEFAULT_CONFIG, maker_entry=True)
    assert float(fee_per_contract(0.63, cfg)) == pytest.approx(0.0)
    taker = float(fee_per_contract(0.63, DEFAULT_CONFIG))
    assert taker > 0.0, 'a taker is charged; only the maker is free'


# --------------------------------------------------------------------------
# the field that actually decides whether an order rests
# --------------------------------------------------------------------------

def _sent_body(**kw):
    """Capture the order body without touching the network."""
    import asyncio
    client = KalshiClient(live=True, key_id='x', private_key_pem=None)
    captured = {}

    async def fake_request(method, path, *, body=None, **_):
        captured['method'], captured['path'], captured['body'] = method, path, body
        return {'order': {'order_id': 'abc', 'status': 'resting'}}

    client._request = fake_request                       # noqa: SLF001
    args = dict(ticker='KXBTC15M-26OCT180015-15', side='up', contracts=1,
                limit_price=0.65)
    args.update(kw)
    asyncio.run(client.place_order(**args))
    return captured['body']


def test_a_resting_order_sets_self_trade_prevention_to_maker():
    """**This field, not `time_in_force`, decides whether the order crosses.**
    `taker_at_cross` crosses the spread; `maker` rests. Pricing the entry at the
    bid while sending `taker_at_cross` would put out a crossing order at a price
    nothing is offering — it would mostly fail to fill and read as "resting does
    not work", when what was wrong was the flag."""
    body = _sent_body(rest=True, time_in_force='good_till_canceled')
    assert body['self_trade_prevention_type'] == 'maker'
    assert body['time_in_force'] == 'good_till_canceled', \
        'the field is REQUIRED; omitting it is a 400 missing_parameters'
    # The combination that actually rests: GTC + maker. Either alone is not it.
    assert body['side'] == 'bid'


def test_a_taker_order_is_unchanged():
    """The default path must be byte-identical to what has placed 1,556 fills."""
    body = _sent_body()
    assert body['self_trade_prevention_type'] == 'taker_at_cross'
    assert body['time_in_force'] == 'immediate_or_cancel'


def test_the_resting_limit_still_rounds_so_it_can_fill():
    """A bid fills against asks at or below it, so the limit rounds UP onto the
    venue's own tick. Resting must not quietly change that."""
    body = _sent_body(rest=True, time_in_force='good_till_canceled')
    assert float(body['price']) >= 0.65 - 1e-9
    assert body['side'] == 'bid'


# --------------------------------------------------------------------------
# the directional tilt gate
# --------------------------------------------------------------------------

def test_a_constant_bullish_lean_fails_the_tilt_gate():
    """**A systematic lean is not a forecast.** Measured 2026-10-17, the
    deployed model ran +0.883pp more bullish than the price at t=+25.9 while
    every outcome-based statistic on the same sample was noise. Against a 1.5pp
    edge gate that makes the real hurdle ~0.6pp one way and ~2.4pp the other:
    live took 83 UP trades and ZERO DOWN over four days."""
    import numpy as np
    import pandas as pd
    from core.metrics import market_gate_values, MARKET_COLUMNS

    n = 400
    rng = np.random.default_rng(3)
    market = rng.uniform(0.2, 0.8, n)
    t0 = pd.Timestamp('2026-10-01', tz='UTC')
    rows = []
    for i in range(n):
        w = t0 + pd.Timedelta(minutes=15 * i)
        mkt = float(market[i])
        rows.append(('BTC-USD', w, 3,                       # symbol, window, offset
                     mkt,                                   # market
                     mkt,                                   # baseline
                     float(np.clip(mkt + 0.0088, 1e-6, 1 - 1e-6)),   # model: +0.88pp
                     int(rng.random() < mkt),               # outcome
                     w, f'f{i // 50}', True))               # decision_time, fold, labelled
    got = market_gate_values(rows)
    assert got['directional_tilt_pp'] == pytest.approx(0.88, abs=0.05)
    assert got['directional_tilt_abs'] > 0.5, 'this is the case the gate exists for'


def test_an_unmeasured_tilt_is_nan_not_zero():
    """A gate that reads 0.0 when nothing was measured passes by claiming a
    model with no observed lean has no lean."""
    import math
    from core.metrics import market_gate_values
    got = market_gate_values([])
    assert math.isnan(got['directional_tilt_abs'])
    assert math.isnan(got['directional_tilt_pp'])


# --------------------------------------------------------------------------
# a resting order is not a dead one
# --------------------------------------------------------------------------

def test_a_resting_ticket_is_still_the_loops_problem():
    """**The bug that produced unrecorded positions.** `filled_from_order`
    reads the POST reply; a `good_till_canceled` order necessarily comes back
    with zero fills because it is sitting ON the book. Booking that as 'killed'
    left the venue holding 1 ETH @ 0.79 and 2 SOL @ 0.80 on 2026-10-18, both
    `is_taker: false`, while our own positions table had neither — and
    `exposure`, `max_positions_per_window` and `max_window_exposure_fraction`
    are all computed from OUR book, so limits did not bind on them.

    A resting ticket therefore has to stay in the open set: the reconcile books
    any fill, and `cancel_stale_resting` retires it if none comes.
    """
    from core.pg_writer import PgWriter
    assert 'resting' in PgWriter.OPEN_TICKET_STATUSES
    assert 'new' in PgWriter.OPEN_TICKET_STATUSES
    # 'killed' must NOT be: a dead order needs no sweep and re-cancelling one
    # would be noise against the venue.
    assert 'killed' not in PgWriter.OPEN_TICKET_STATUSES
    assert 'filled' not in PgWriter.OPEN_TICKET_STATUSES


def test_a_resting_result_does_not_trigger_the_kill_resend():
    """`execute_entry` resends once when an immediate_or_cancel takes nothing.
    A resting order has taken nothing YET, and resending would double the
    intended size on one fifteen-minute move."""
    import inspect

    from livetrade import execution
    src = inspect.getsource(execution.execute_entry)
    assert "status == 'killed'" in src, 'the resend keys on killed'
    assert "status == 'resting'" not in src, \
        'a resting order must not be resent — it is still live at the venue'


def test_the_shipped_artifacts_alpha_is_gated_not_just_the_walk_forwards():
    """**The surprise of 2026-10-18.** `residual_scale` aggregates the folds;
    `choose_candidate` ships a refit on all data that fits its own alpha. The
    gate passed at 0.940 and the artifact installed at 0.015 — a model that
    departed from the price by 0.027pp, could barely clear its own 0.5pp edge
    gate, and sized every stake to one contract. A gate that does not measure
    the thing that trades is not a gate."""
    from core.metrics import GATE_NOTES, GATE_SPEC
    spec = {g.name: g for g in GATE_SPEC}
    assert 'deployed_residual_scale' in spec, \
        'the installed artifact must be gated, not only the walk-forward'
    assert spec['deployed_residual_scale'].threshold == \
        spec['residual_scale'].threshold, 'same quantity, same bar'
    assert spec['deployed_residual_scale'].direction == 'min'
    assert GATE_NOTES.get('deployed_residual_scale')


def test_an_alpha_of_point_zero_one_five_would_now_be_refused():
    """The specific value that shipped, against the specific gate."""
    from core.metrics import GATE_SPEC
    g = {x.name: x for x in GATE_SPEC}['deployed_residual_scale']
    assert 0.015 < g.threshold, 'the artifact that traded must fail this'
    assert 0.940 >= g.threshold, 'the walk-forward value would have passed it'


# --------------------------------------------------------------------------
# a resting order must round AWAY from the spread
# --------------------------------------------------------------------------

def test_a_resting_bid_rounds_down_and_a_crossing_bid_rounds_up():
    """**Why 25 of the first 35 maker-mode fills were takers.** The rounding
    was taker-shaped and correct for what it was written for: a bid fills
    against asks at or below it, so rounding UP buys reach, and under
    fill_or_kill rounding the wrong way is a guaranteed kill.

    For a resting order it is the bug. On a measured 1.000c spread one tick of
    upward rounding puts the bid into the ask, so the order crosses, pays the
    fee and gives up the spread — the whole maker case, lost to a rounding
    direction.
    """
    taker = _sent_body(limit_price=0.655, time_in_force='immediate_or_cancel')
    maker = _sent_body(limit_price=0.655, rest=True,
                       time_in_force='good_till_canceled')
    assert taker['side'] == maker['side'] == 'bid'
    assert float(taker['price']) == pytest.approx(0.66), 'a taker reaches for the ask'
    assert float(maker['price']) == pytest.approx(0.65), 'a maker stays on the bid'
    assert float(maker['price']) < float(taker['price'])


def test_a_resting_ask_rounds_up_and_a_crossing_ask_rounds_down():
    """The mirror. Buying DOWN places a YES ask: to rest it must stay ABOVE the
    bid, so it rounds up, where a crossing ask rounds down to reach the bid."""
    import asyncio

    def body(rest):
        client = KalshiClient(live=True, key_id='x')
        out = {}

        async def fake(method, path, *, body=None, **_):
            out.update(body or {})
            return {'order': {'order_id': 'abc'}}

        client._request = fake                                # noqa: SLF001
        asyncio.run(client.place_order(
            ticker='KXBTC15M-26OCT180015-15', side='down', contracts=1,
            limit_price=0.345, rest=rest,
            time_in_force='good_till_canceled' if rest else 'immediate_or_cancel'))
        return out

    taker, maker = body(False), body(True)
    assert taker['side'] == maker['side'] == 'ask'
    assert float(maker['price']) > float(taker['price']), \
        'a resting ask must sit further from the bid than a crossing one'


def test_the_rounding_never_costs_more_than_one_tick():
    """Resting away from the spread is a real price concession; it must be
    bounded at a single tick or it is a different trade."""
    for limit in (0.655, 0.123, 0.876, 0.505):
        t = float(_sent_body(limit_price=limit,
                             time_in_force='immediate_or_cancel')['price'])
        m = float(_sent_body(limit_price=limit, rest=True,
                             time_in_force='good_till_canceled')['price'])
        assert abs(t - m) <= 0.01 + 1e-9, f'{limit}: {t} vs {m} is over a cent'


# --------------------------------------------------------------------------
# resting orders that FILL must become positions
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_filled_resting_order_is_booked_as_a_position():
    """**The second half of the resting fix.** Marking a maker order `resting`
    instead of `killed` stopped it being written off, but nothing moved it to
    `filled` — so the fill was lost. Measured 2026-10-18: the venue held 35
    contracts from 30 fills while our positions table showed 6, and the cancel
    sweep reported `404 not_found` every minute because the order had already
    filled.
    """
    from dataclasses import replace

    from scripts.live import book_resting_fills

    class Ticket:
        id, symbol, side = 7, 'BTC-USD', 'up'
        window_open = settle_time = '2026-10-18T20:00:00+00:00'
        offset_minutes, contracts, prediction_id = 3, 2, None
        venue_order_id, status = 'ord-1', 'resting'

    booked = {}

    class Writer:
        def open_tickets(self, limit=50):
            return [Ticket()]

        def open_position(self, **f):
            booked.update(f)
            return 1

        def adjust_account(self, **f):
            booked['adjust'] = f

        def resolve_ticket(self, tid, **f):
            booked['resolved'] = f

    class Venue:
        async def fills(self, limit=200):
            return [{'order_id': 'ord-1', 'count_fp': '2.00',
                     'yes_price_dollars': '0.6500'}]

    cfg = replace(DEFAULT_CONFIG, maker_entry=True)
    n = await book_resting_fills(Writer(), Venue(), cfg)
    assert n == 1, 'the filled resting order must be booked'
    assert booked['contracts'] == 2
    assert booked['price'] == pytest.approx(0.65)
    assert booked['side'] == 'up'
    assert booked['resolved']['status'] == 'filled'
    # a maker pays no fee, so the outlay is price x contracts exactly
    assert booked['fee'] == pytest.approx(0.0)
    assert booked['outlay'] == pytest.approx(1.30)


@pytest.mark.asyncio
async def test_a_down_fill_books_the_NO_price_not_the_yes_price():
    """V2 quotes everything from the YES side. Reading `yes_price` on a NO fill
    books a 30c purchase as 70c — the same class of error as inverting the
    side outright."""
    from dataclasses import replace

    from scripts.live import book_resting_fills

    class Ticket:
        id, symbol, side = 8, 'ETH-USD', 'down'
        window_open = settle_time = '2026-10-18T20:00:00+00:00'
        offset_minutes, contracts, prediction_id = 6, 1, None
        venue_order_id, status = 'ord-2', 'resting'

    got = {}

    class Writer:
        def open_tickets(self, limit=50): return [Ticket()]
        def open_position(self, **f): got.update(f); return 1
        def adjust_account(self, **f): pass
        def resolve_ticket(self, tid, **f): pass

    class Venue:
        async def fills(self, limit=200):
            return [{'order_id': 'ord-2', 'count_fp': '1.00',
                     'yes_price_dollars': '0.3000'}]

    await book_resting_fills(Writer(), Venue(), replace(DEFAULT_CONFIG, maker_entry=True))
    assert got['price'] == pytest.approx(0.70), \
        'buying DOWN at a 0.30 yes price costs 0.70 for the NO contract'


@pytest.mark.asyncio
async def test_nothing_is_booked_when_maker_entry_is_off():
    from scripts.live import book_resting_fills

    class Boom:
        def open_tickets(self, limit=50):
            raise AssertionError('must not even look on the taker path')

    assert await book_resting_fills(Boom(), object(), DEFAULT_CONFIG) == 0
