"""The venue's balance is adopted AFTER our own settlements are credited.

This is not a style preference. Measured on the first live night, on real money,
with the two operations the other way round:

    09:15:38  ours $147.03, venue $168.03 (+21.00)   <- venue credited a payout
    09:16:42  ours $189.03, venue $168.03 (-21.00)   <- we credited the same one
    09:19:56  ours $160.67, venue $161.07 ( +0.41)   <- reconciled back

The venue credits a settlement the moment it settles, so reading its balance
first and settling second books every payout twice. The bankroll self-heals on
the next cycle — no money moved — but Kelly sized off an inflated bankroll for a
full cycle, and every settlement produced a large spurious drift warning. The
drift log is the only thing standing between an unrecorded fill and silence, and
it cannot do that job while it cries wolf on every win.

The test drives the real `run_cycle`, because the thing under test is an ordering
*inside* it. Asserting a hand-written sequence of the two calls would only test
the sequence I wrote in the test.

The bars are deliberately stale, so `stale_symbols` sets `offset = None` and the
cycle returns right after the settle/adopt block — no model, no quotes needed.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.pg_writer import PgWriter
from scripts import live as live_mod

WINDOW = datetime(2026, 8, 23, 0, 30, tzinfo=timezone.utc)
SETTLE = WINDOW + timedelta(minutes=15)
TICKER = 'KXBTC15M-26AUG230045'
PAYOUT = 10.0          # 10 contracts of a winning YES at settlement
CONTRACTS = 10


def _stale_bars() -> dict[str, pd.DataFrame]:
    """The shape `fetch_bars` returns, old enough that `stale_symbols` flags
    every symbol so the cycle stops right after the settle/adopt block."""
    newest = pd.Timestamp('2026-08-23 01:00:00', tz='UTC')  # past SETTLE, still a day old
    stamps = pd.date_range(newest - pd.Timedelta(minutes=60), newest, freq='1min')
    frame = pd.DataFrame({
        'event_time': stamps,
        'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
        'volume': 1.0, 'quote_volume': 100.0, 'trade_count': 10.0,
    })
    return {symbol: frame.copy() for symbol in DEFAULT_CONFIG.symbols}


class VenueStub:
    """Reports a balance that ALREADY includes the payout, which is what the
    real venue does the instant a market settles."""

    def __init__(self, balance: float):
        self.balance = balance
        self.reads = 0

    async def reconcile(self, *, exchange_index=None):
        self.reads += 1
        return {
            'balance': self.balance,
            'settlements': [{'ticker': TICKER, 'settled_up': True,
                             'revenue': PAYOUT}],
            'positions': [],
        }


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    w.ensure_account(100.0, mode='live')
    return w


def _open_a_winning_position(writer: PgWriter) -> None:
    writer.write_prediction(
        symbol='BTC-USD', window_open=WINDOW, settle_time=SETTLE,
        offset_minutes=3, decision_time=WINDOW + timedelta(minutes=3),
        strike=100.0, last_price=100.0, displacement=0.0, sigma_remaining=0.001,
        z_score=0.0, baseline_probability=0.5, model_probability=0.6,
        market_probability=0.6, price_source='quote', reason='traded',
        traded=True, side='up', price=0.6, effective_cost=0.62, edge=0.01,
        contracts=CONTRACTS, model_version=None)
    writer.write_ticket(
        symbol='BTC-USD', window_open=WINDOW, settle_time=SETTLE,
        offset_minutes=3, market_ticker=TICKER, side='up', contracts=CONTRACTS,
        limit_price=0.6, max_price=0.61, expected_cost=6.2,
        model_probability=0.6, edge=0.01)
    writer.open_position(
        symbol='BTC-USD', window_open=WINDOW, settle_time=SETTLE,
        offset_minutes=3, side='up', contracts=CONTRACTS, price=0.6,
        outlay=6.0, fee=0.2, model_probability=0.6, baseline_probability=0.5,
        edge=0.01)


def _args(**over) -> argparse.Namespace:
    base = dict(offset=None, reconcile=True, mode='live', place_orders=False,
                dry_run=True)
    base.update(over)
    return argparse.Namespace(**base)


def _run(writer, venue, monkeypatch):
    async def fake_fetch(config, minutes=None):
        return _stale_bars()

    monkeypatch.setattr(live_mod, 'fetch_bars', fake_fetch)
    return asyncio.run(live_mod.run_cycle(
        _args(), DEFAULT_CONFIG, writer, None, venue))


def test_a_settled_payout_is_credited_once_not_twice(writer, monkeypatch):
    """The whole finding, in one assertion.

    Bankroll 100, a position whose payout is $10. The venue already reports 104
    (100 - 6 outlay - 0.2 fee + 10 payout = 103.80, rounded to what it holds).
    If the balance is adopted before we settle, our figure ends up 103.80 + 10.
    """
    writer.update_account(bankroll=93.8)        # outlay and fee already out
    _open_a_winning_position(writer)
    venue = VenueStub(balance=103.8)            # payout already credited

    _run(writer, venue, monkeypatch)

    assert venue.reads == 1
    bankroll = float(writer.account().bankroll)
    assert bankroll == pytest.approx(103.8), (
        f'expected the venue figure 103.80, got {bankroll:.2f}. '
        f'{bankroll:.2f} == 113.80 means the payout was booked twice: the '
        f'balance was adopted before settle_due credited it.'
    )


def test_the_position_is_still_marked_settled(writer, monkeypatch):
    """Adopting the balance later must not skip our own bookkeeping — the
    position's outcome and realised PnL are ours to record, not the venue's."""
    writer.update_account(bankroll=93.8)
    _open_a_winning_position(writer)

    _run(writer, VenueStub(balance=103.8), monkeypatch)

    assert writer.open_positions() == [], 'the position was never settled'


def test_a_real_disagreement_still_reaches_the_log(writer, monkeypatch, caplog):
    """The reorder must not silence the alarm it exists to sharpen. A venue
    balance that disagrees AFTER our credits are in is a genuine drift."""
    writer.update_account(bankroll=93.8)
    _open_a_winning_position(writer)

    with caplog.at_level('WARNING'):
        # 90.0 is nowhere near 103.80: an unrecorded fill would look like this.
        _run(writer, VenueStub(balance=90.0), monkeypatch)

    assert any('balance drift' in r.message for r in caplog.records), (
        'a real disagreement produced no warning'
    )
    assert float(writer.account().bankroll) == pytest.approx(90.0), (
        'the venue is the account of record even when it disagrees'
    )


def test_no_venue_means_no_balance_write(writer, monkeypatch):
    """Paper mode, and `--no-reconcile`. Our arithmetic stands alone, and the
    settlement credit must survive."""
    writer.update_account(bankroll=93.8)
    _open_a_winning_position(writer)

    async def fake_fetch(config, minutes=None):
        return _stale_bars()

    monkeypatch.setattr(live_mod, 'fetch_bars', fake_fetch)
    asyncio.run(live_mod.run_cycle(
        _args(reconcile=False), DEFAULT_CONFIG, writer, None, None))

    assert float(writer.account().bankroll) > 93.8, (
        'the payout was not credited with no venue to read'
    )
