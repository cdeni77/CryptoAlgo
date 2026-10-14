"""An entry cycle decides first and keeps its books afterwards.

Every second between reading the book and the order reaching the venue is a
second the market can move away; the old cycle spent ~5s there on a REST bar
fetch, four authenticated reconcile calls and one order at a time. These drive
the real `run_cycle` against a recording venue and assert on WHEN each call
happened, not on the source text.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import time

import numpy as np
import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.pg_writer import OrderTicket, PgWriter
from data_collection.kalshi_client import KalshiError, Quote
from livetrade import artifact as live_artifact
from livetrade import features as live_features
from livetrade import state as live_state
from scripts import live

TICKERS = {'BTC-USD': 'KXBTC15M-T1', 'ETH-USD': 'KXETH15M-T1', 'SOL-USD': 'KXSOL15M-T1'}


def make_quote(symbol, *, bid=0.59, ask=0.60, source='stream'):
    return Quote(ticker=TICKERS[symbol], yes_bid=bid, yes_ask=ask, no_bid=1 - ask,
                 no_ask=1 - bid, last_price=None, volume=0, open_interest=0,
                 close_time=None, status='active', yes_bid_size=500.0,
                 yes_ask_size=500.0, exchange_index=2, book_source=source,
                 book_time=time.time())


class Venue:
    """Records every call, in order, with a scripted reply per order key."""

    def __init__(self, *, replies=None, delay=0.0, fresh_ask=0.60):
        self.events: list[tuple] = []
        self.replies = replies or {}
        self.delay = delay
        self.fresh_ask = fresh_ask
        self.sent: list[dict] = []
        self.started: list[float] = []

    async def place_order(self, **body):
        self.events.append(('order', body['ticker'], body.get('client_order_id')))
        self.sent.append(body)
        self.started.append(time.perf_counter())
        await asyncio.sleep(self.delay)
        reply = self.replies.get(body.get('client_order_id'))
        if reply is None:
            reply = {'order_id': 'o-' + str(len(self.sent)), 'status': 'executed',
                     'fill_count': f"{body['contracts']}.00"}
        if isinstance(reply, BaseException):
            raise reply
        return dict(reply)

    async def reconcile(self, exchange_index=None):
        self.events.append(('reconcile',))
        return {'balance': float('nan'), 'positions': [], 'fills': [],
                'settlements': []}

    async def orderbook(self, ticker):
        self.events.append(('diagnose', ticker))
        return {'yes_bid': 0.59, 'yes_ask': 0.60}

    async def orders(self, **kw):
        return []

    async def book_quote(self, meta):
        self.events.append(('book', meta.ticker))
        symbol = next(s for s, t in TICKERS.items() if t == meta.ticker)
        return make_quote(symbol, bid=self.fresh_ask - 0.01, ask=self.fresh_ask,
                          source='rest')


class Model:
    scoring, groups, init_score_source, version = object(), (), 'baseline', 'm1'

    def predict(self, table):
        return np.array([0.80] * len(table))

    def score(self, table):
        return pd.DataFrame({'model_probability': self.predict(table)},
                            index=table.index)


def run(tmp_path, monkeypatch, venue, *, offset=12, reconciled=True,
        record_touch=False, **config_over):
    writer = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    writer.ensure_account(1000.0, mode='live')
    now = pd.Timestamp.now('UTC').floor('min')
    stamps = pd.date_range(now - pd.Timedelta(minutes=30), now, freq='1min')
    bars = {s: pd.DataFrame({'event_time': stamps, 'open': 100.0, 'high': 100.0,
                             'low': 100.0, 'close': 100.0, 'volume': 1.0,
                             'quote_volume': 100.0, 'trade_count': 10.0})
            for s in DEFAULT_CONFIG.symbols}

    async def fake_bars(config, minutes=None, **kw):
        venue.events.append(('bars', bool(kw.get('prefer_stream'))))
        return bars

    async def fake_quotes(kalshi, symbols, settle_time):
        venue.events.append(('quotes',))
        return {s: (make_quote(s), TICKERS[s]) for s in symbols}

    def fake_score(bars_, scoring, config, *, window_open, offset, groups=None,
                   deferred=()):
        n = len(DEFAULT_CONFIG.symbols)
        return pd.DataFrame({
            'symbol': list(DEFAULT_CONFIG.symbols), 'window_open': [window_open] * n,
            'settle_time': [window_open + pd.Timedelta(minutes=15)] * n,
            'offset': [offset] * n, 'strike': [100.0] * n, 'last_price': [100.4] * n,
            'displacement': [0.004] * n, 'sigma_remaining': [0.005] * n,
            'z_score': [0.8] * n, 'baseline_probability': [0.60] * n,
            'strike_source': [None] * n})

    monkeypatch.setattr(live, 'fetch_bars', fake_bars)
    monkeypatch.setattr(live, 'fetch_quotes', fake_quotes)
    monkeypatch.setattr(live, 'score_live', fake_score)
    # Looked up where they live: `prewarm_fit_lookup`/`latest_fit` in
    # livetrade.features, the reconcile clock in livetrade.state.
    monkeypatch.setattr(live_features, '_read_recent_fits', lambda: pd.DataFrame())
    monkeypatch.setenv('RECORD_TOUCH', '1' if record_touch else '0')
    monkeypatch.setattr(live_state, '_LAST_RECONCILED',
                        time.monotonic() if reconciled else None)
    config = dataclasses.replace(
        DEFAULT_CONFIG, min_usable_bars=1, min_remaining_seconds=0,
        starting_bankroll=1000.0, entry_offsets=(12,), **config_over)
    args = argparse.Namespace(offset=offset, reconcile=True, mode='live',
                              place_orders=True, dry_run=False)
    decisions = asyncio.run(live.run_cycle(args, config, writer, Model(), venue))
    return writer, decisions


def first(events, kind):
    return next(i for i, e in enumerate(events) if e[0] == kind)


def last(events, kind):
    return max(i for i, e in enumerate(events) if e[0] == kind)


class TestOrdering:
    def test_an_entry_cycle_orders_before_it_reconciles(self, tmp_path, monkeypatch):
        venue = Venue()
        writer, decisions = run(tmp_path, monkeypatch, venue)
        events = venue.events
        assert events[0] == ('bars', True), 'the entry cycle asks for stream bars'
        assert first(events, 'quotes') < first(events, 'order')
        assert last(events, 'order') < first(events, 'reconcile'), events
        assert len(writer.open_positions()) == 3
        assert sum(d.traded for d in decisions) == 3

    def test_a_non_entry_cycle_reconciles_before_the_book(self, tmp_path, monkeypatch):
        venue = Venue()
        run(tmp_path, monkeypatch, venue, offset=9)
        assert venue.events[0] == ('bars', False)
        assert first(venue.events, 'reconcile') < first(venue.events, 'quotes')
        assert not [e for e in venue.events if e[0] == 'order']

    def test_an_unreconciled_venue_is_reconciled_first(self, tmp_path, monkeypatch):
        venue = Venue()
        run(tmp_path, monkeypatch, venue, reconciled=False)
        assert first(venue.events, 'reconcile') < first(venue.events, 'quotes')
        assert first(venue.events, 'order') > first(venue.events, 'quotes')

    def test_a_successful_reconcile_records_its_time(self, tmp_path, monkeypatch):
        """The entry-cycle staleness guard reads this clock. If bookkeeping
        stopped recording it, every entry cycle would reconcile first again."""
        before = time.monotonic()
        run(tmp_path, monkeypatch, Venue(), offset=9, reconciled=False)
        assert live_state._LAST_RECONCILED is not None
        assert live_state._LAST_RECONCILED >= before

    def test_predictions_are_still_written_for_every_symbol(self, tmp_path, monkeypatch):
        writer, _ = run(tmp_path, monkeypatch, Venue())
        with writer._session() as session:            # noqa: SLF001
            from core.pg_writer import Prediction
            assert session.query(Prediction).count() == 3


class TestConcurrentOrders:
    def test_every_order_is_in_flight_at_once(self, tmp_path, monkeypatch):
        venue = Venue(delay=0.3)
        started = time.perf_counter()
        run(tmp_path, monkeypatch, venue)
        assert len(venue.started) == 3
        assert max(venue.started) - min(venue.started) < 0.15, 'orders went out in series'
        assert time.perf_counter() - started < 3.0

    def test_a_kill_is_diagnosed_after_every_order_is_out(self, tmp_path, monkeypatch):
        class KillFirst(Venue):
            async def place_order(self, **body):
                if body['ticker'] == TICKERS['BTC-USD']:
                    self.replies[body['client_order_id']] = {
                        'order_id': 'k', 'status': 'canceled', 'fill_count': '0.00'}
                return await super().place_order(**body)

        venue = KillFirst(delay=0.1)
        writer, _ = run(tmp_path, monkeypatch, venue, kill_retries=0)
        assert first(venue.events, 'diagnose') > last(venue.events, 'order')
        assert len(writer.open_positions()) == 2


class TestOneResendAfterAKill:
    class KillOnce(Venue):
        """BTC's first attempt is killed; anything after takes the scripted reply."""

        async def place_order(self, **body):
            key = body['client_order_id']
            if body['ticker'] == TICKERS['BTC-USD'] and not key.endswith('-r1'):
                self.replies.setdefault(key, {'order_id': 'k', 'status': 'canceled',
                                              'fill_count': '0.00'})
            return await super().place_order(**body)

    def test_a_kill_is_repriced_and_resent_once(self, tmp_path, monkeypatch):
        venue = self.KillOnce(fresh_ask=0.61)
        writer, decisions = run(tmp_path, monkeypatch, venue)
        keys = [b['client_order_id'] for b in venue.sent if b['ticker'] == TICKERS['BTC-USD']]
        assert len(keys) == 2 and keys[1] == keys[0] + '-r1'
        assert ('book', TICKERS['BTC-USD']) in venue.events, 'the touch was re-read'
        resent = [b for b in venue.sent if b['client_order_id'].endswith('-r1')][0]
        first_try = [b for b in venue.sent if b['client_order_id'] == keys[0]][0]
        assert resent['limit_price'] > first_try['limit_price'], 'not re-priced'
        assert len(writer.open_positions()) == 3
        with writer._session() as session:            # noqa: SLF001
            ticket = session.query(OrderTicket).filter_by(symbol='BTC-USD').one()
            assert ticket.status == 'filled'

    def test_a_partial_fill_is_never_topped_up(self, tmp_path, monkeypatch):
        class Partial(Venue):
            async def place_order(self, **body):
                if body['ticker'] == TICKERS['BTC-USD']:
                    self.replies[body['client_order_id']] = {
                        'order_id': 'p', 'status': 'canceled', 'fill_count': '3.00'}
                return await super().place_order(**body)

        venue = Partial()
        writer, _ = run(tmp_path, monkeypatch, venue)
        assert not [b for b in venue.sent if b['client_order_id'].endswith('-r1')]
        btc = [p for p in writer.open_positions() if p.symbol == 'BTC-USD']
        assert btc and btc[0].contracts == 3

    def test_no_resend_when_the_fresh_price_no_longer_clears(self, tmp_path, monkeypatch):
        venue = self.KillOnce(fresh_ask=0.79)
        writer, _ = run(tmp_path, monkeypatch, venue)
        assert not [b for b in venue.sent if b['client_order_id'].endswith('-r1')]
        assert len(writer.open_positions()) == 2

    def test_no_resend_outside_the_window(self, tmp_path, monkeypatch):
        venue = self.KillOnce()
        run(tmp_path, monkeypatch, venue, kill_retry_window_seconds=-1.0)
        assert not [b for b in venue.sent if b['client_order_id'].endswith('-r1')]

    def test_the_resend_respects_the_window_exposure_of_the_others(
            self, tmp_path, monkeypatch):
        # Room for exactly three $25 positions: the resend may take only what
        # the other two in-flight orders left.
        venue = self.KillOnce(fresh_ask=0.61)
        writer, _ = run(tmp_path, monkeypatch, venue,
                        max_window_exposure_fraction=0.06)
        staked = sum(float(p.outlay) for p in writer.open_positions())
        assert staked <= 0.06 * 1000.0 + 1e-6


# ---------------------------------------------------------------------------
# A POST nobody heard back from
# ---------------------------------------------------------------------------

from tests.test_act_on import FakeKalshi, args, decision  # noqa: E402


class LookupKalshi(FakeKalshi):
    def __init__(self, *, raises, record=None, lookup_raises=None):
        super().__init__(raises=raises)
        self.record, self.lookup_raises, self.lookups = record, lookup_raises, []

    async def order_by_client_id(self, client_order_id, *, ticker=None):
        self.lookups.append(client_order_id)
        if self.lookup_raises is not None:
            raise self.lookup_raises
        return self.record


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    w.ensure_account(100.0, mode='live')
    return w


def _ticket(writer):
    with writer._session() as session:                # noqa: SLF001
        return session.query(OrderTicket).one()


class TestAmbiguousPost:
    def test_a_timeout_books_what_the_venue_says_it_filled(self, writer):
        kalshi = LookupKalshi(raises=asyncio.TimeoutError(), record={
            'order_id': 'x', 'status': 'executed', 'fill_count_fp': '5.00',
            'taker_fill_cost_dollars': '3.0000', 'outcome_side': 'yes'})
        assert asyncio.run(live.act_on(args(), writer, kalshi, decision(), None)) is True
        assert len(kalshi.orders) == 1, 'looked up, never re-sent'
        assert kalshi.lookups == [kalshi.orders[0]['client_order_id']]
        position = writer.open_positions()[0]
        assert position.contracts == 5 and float(position.price) == pytest.approx(0.60)
        assert _ticket(writer).status == 'filled'

    def test_a_5xx_with_no_order_at_the_venue_books_nothing(self, writer):
        kalshi = LookupKalshi(raises=KalshiError('-> 503', status=503), record=None)
        assert asyncio.run(live.act_on(args(), writer, kalshi, decision(), None)) is False
        assert len(kalshi.lookups) == 2, 'asked twice before concluding'
        assert writer.open_positions() == []
        assert _ticket(writer).status == 'skipped'

    def test_a_failed_lookup_leaves_the_ticket_unknown(self, writer):
        kalshi = LookupKalshi(raises=asyncio.TimeoutError(),
                              lookup_raises=KalshiError('-> 500', status=500))
        assert asyncio.run(live.act_on(args(), writer, kalshi, decision(), None)) is False
        assert _ticket(writer).status == 'unknown', 'unknown still blocks the window'

    def test_a_4xx_is_a_refusal_and_not_looked_up(self, writer):
        kalshi = LookupKalshi(raises=KalshiError('-> 409', status=409))
        assert asyncio.run(live.act_on(args(), writer, kalshi, decision(), None)) is False
        assert kalshi.lookups == []
        assert _ticket(writer).status == 'skipped'


class TestClientLookup:
    def test_order_by_client_id_lists_the_market_and_matches(self):
        from data_collection.kalshi_client import KalshiClient

        client = KalshiClient(live=False)
        seen = []

        async def fake_request(method, path, *, params=None, body=None, base=None):
            seen.append((method, path, dict(params or {})))
            return {'orders': [{'order_id': 'a', 'client_order_id': 'other'},
                               {'order_id': 'b', 'client_order_id': 'BTC-USD-1'}]}

        client._request = fake_request
        found = asyncio.run(client.order_by_client_id('BTC-USD-1', ticker='T'))
        assert found['order_id'] == 'b'
        assert seen[0][:2] == ('GET', '/portfolio/orders')
        assert seen[0][2]['ticker'] == 'T'
        assert asyncio.run(client.order_by_client_id('nope', ticker='T')) is None


# ---------------------------------------------------------------------------
# Maker entry, opt-in
# ---------------------------------------------------------------------------

class TestMakerEntry:
    def test_it_is_off_by_default(self):
        assert DEFAULT_CONFIG.entry_style == 'taker'

    def test_the_posted_price_and_maker_fee_price_the_decision(self):
        config = dataclasses.replace(DEFAULT_CONFIG, entry_style='maker')
        quote = make_quote('BTC-USD', bid=0.57, ask=0.60)
        row = {'symbol': 'BTC-USD', 'window_open': pd.Timestamp('2026-10-14 12:00', tz='UTC'),
               'settle_time': pd.Timestamp('2026-10-14 12:15', tz='UTC'), 'offset': 12,
               'model_probability': 0.80, 'baseline_probability': 0.60,
               'ask_up': 0.60, 'ask_down': 0.43, 'market_ticker': TICKERS['BTC-USD']}
        d, maker = live.entry_decision(row, config, quote=quote, bankroll=100.0,
                                       exposure=live.WindowExposure(),
                                       require_quote=True, halted=False)
        assert maker and d.traded
        assert d.price == pytest.approx(0.58), 'one tick inside the 0.57 bid'
        assert d.effective_cost == pytest.approx(0.58 + config.maker_fee_rate)

    def test_a_bid_one_tick_under_the_ask_posts_at_the_bid(self):
        up, down = live.maker_prices(make_quote('BTC-USD', bid=0.59, ask=0.60))
        assert up == pytest.approx(0.59)
        assert down == pytest.approx(0.40)

    def test_the_maker_order_rests_then_cancels_and_books_the_maker_fill(self, writer):
        calls = []

        class Maker:
            async def place_order(self, **body):
                calls.append(('place', body))
                return {'order_id': 'm1', 'status': 'resting'}

            async def cancel(self, order_id, *, ticker=None):
                calls.append(('cancel', order_id))
                return {}

            async def order(self, order_id):
                calls.append(('order', order_id))
                return {'order_id': 'm1', 'status': 'canceled', 'fill_count_fp': '5.00',
                        'maker_fill_cost_dollars': '2.9000', 'outcome_side': 'yes'}

        config = dataclasses.replace(DEFAULT_CONFIG, entry_style='maker',
                                     maker_rest_seconds=0.0, min_remaining_seconds=0)
        d = dataclasses.replace(decision(price=0.58),
                                settle_time=pd.Timestamp.now(tz='UTC') + pd.Timedelta(minutes=3))
        result = asyncio.run(live.place_and_book(args(), writer, Maker(), d, None,
                                                 config=config, maker=True))
        assert result.booked and result.filled == 5
        body = calls[0][1]
        assert body['post_only'] is True
        assert body['time_in_force'] == 'good_till_canceled'
        assert body['expiration_time'] > time.time()
        assert [c[0] for c in calls] == ['place', 'cancel', 'order']
        position = writer.open_positions()[0]
        assert float(position.fee) == pytest.approx(5 * config.maker_fee_rate)
        assert float(position.price) == pytest.approx(0.58)

    def test_the_wire_body_for_a_post_only_order(self):
        from data_collection.kalshi_client import KalshiClient

        client = KalshiClient(live=True)
        sent = []

        async def fake_request(method, path, *, params=None, body=None, base=None):
            sent.append(body)
            return {'order': {'order_id': 'z'}}

        client._request = fake_request
        asyncio.run(client.place_order(ticker='T', side='up', contracts=2,
                                       limit_price=0.0513, post_only=True,
                                       time_in_force='good_till_canceled',
                                       expiration_time=123))
        asyncio.run(client.place_order(ticker='T', side='down', contracts=2,
                                       limit_price=0.58, post_only=True,
                                       time_in_force='good_till_canceled'))
        asyncio.run(client.place_order(ticker='T', side='up', contracts=2,
                                       limit_price=0.58))
        # A maker bid rounds DOWN onto the 0.1c tail tick; a maker YES ask UP.
        assert sent[0]['price'] == '0.0510' and sent[0]['post_only'] is True
        assert sent[0]['expiration_time'] == 123
        assert sent[1]['side'] == 'ask' and sent[1]['price'] == '0.4200'
        assert 'post_only' not in sent[2], 'a taker order is unchanged'


# ---------------------------------------------------------------------------
# The loop survives a bad cycle, and reloads a new artifact without a restart
# ---------------------------------------------------------------------------

class TestTheLoop:
    def _main(self, tmp_path, monkeypatch, cycle):
        writer = PgWriter(database_url=f'sqlite:///{tmp_path}/loop.db')

        class M:
            deployable, version, scoring = True, 'm', type('S', (), {'summary': lambda s: ''})()

            def verify(self, config):
                pass

            def summary(self):
                return ''

        # `main` loads through scripts.live; a hot reload goes through
        # livetrade.artifact._load_artifact, so both are patched.
        for module in (live, live_artifact):
            monkeypatch.setattr(module, 'load_live', lambda config=None: M())
            monkeypatch.setattr(module, 'config_for_artifact', lambda c, m, mode: c)
        monkeypatch.setattr(live, 'PgWriter', lambda: writer)
        monkeypatch.setattr(live, 'seconds_until_next_decision', lambda *a, **k: 0.0)
        monkeypatch.setattr(live, 'run_cycle', cycle)
        return asyncio.run(live.main(['--loop', '--no-require-gates']))

    def test_an_exception_costs_one_cycle_not_the_loop(self, tmp_path, monkeypatch, caplog):
        calls = []

        async def cycle(*a):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('boom')
            raise KeyboardInterrupt

        with caplog.at_level('ERROR'):
            assert self._main(tmp_path, monkeypatch, cycle) == 0
        assert len(calls) == 2, 'the loop did not continue past the failure'
        assert 'cycle failed' in caplog.text and 'boom' in caplog.text

    def test_cancellation_still_stops_it(self, tmp_path, monkeypatch):
        async def cycle(*a):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            self._main(tmp_path, monkeypatch, cycle)


class TestHotReload:
    def test_it_reloads_only_when_the_file_changes(self, tmp_path, monkeypatch):
        import os
        path = tmp_path / 'forecast.joblib'
        path.write_bytes(b'one')
        signature = live.artifact_signature(path)
        loads = []
        monkeypatch.setattr(live_artifact, '_load_artifact',
                            lambda a, c: loads.append(1) or ('new', 'cfg'))
        out = asyncio.run(live.reload_if_changed(None, None, path, 'old', 'old_cfg',
                                                 signature))
        assert out == ('old', 'old_cfg', signature) and loads == []

        staged = tmp_path / 'staged.joblib'
        staged.write_bytes(b'two!')
        os.replace(staged, path)
        model, config, sig = asyncio.run(live.reload_if_changed(
            None, None, path, 'old', 'old_cfg', signature))
        assert (model, config) == ('new', 'cfg') and sig != signature

    def test_a_failed_load_keeps_the_old_model(self, tmp_path, monkeypatch, caplog):
        import os
        path = tmp_path / 'forecast.joblib'
        path.write_bytes(b'one')
        signature = live.artifact_signature(path)
        staged = tmp_path / 'staged.joblib'
        staged.write_bytes(b'broken')
        os.replace(staged, path)

        def refuse(a, c):
            raise SystemExit('the newest attempt was blocked')

        monkeypatch.setattr(live_artifact, '_load_artifact', refuse)
        with caplog.at_level('ERROR'):
            model, config, sig = asyncio.run(live.reload_if_changed(
                None, None, path, 'old', 'old_cfg', signature))
        assert (model, config) == ('old', 'old_cfg')
        assert sig == live.artifact_signature(path), 'not retried every window'
        assert 'could not load the new artifact' in caplog.text


def test_config_offsets_are_the_running_configs(monkeypatch):
    monkeypatch.setattr(live_state, '_RUNNING_CONFIG', dataclasses.replace(
        DEFAULT_CONFIG, decision_offsets=(4, 8, 12), entry_offsets=None))
    assert live.config_offsets() == (4, 8, 12)
    live.reset_gap_history()
    window = pd.Timestamp('2026-10-14 12:00', tz='UTC')
    live.gap_change('BTC-USD', 0.01, window_open=window, offset=4)
    assert live.gap_change('BTC-USD', 0.03, window_open=window,
                           offset=8) == pytest.approx(0.02)
    live.reset_gap_history()
