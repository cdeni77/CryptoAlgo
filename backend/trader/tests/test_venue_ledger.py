"""Tests for parsing, storing and totalling the venue's ledger.

Each test rules out a plausible but wrong P&L. Fixtures use unsuffixed `fee_cost`, as on
the wire.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from core import venue_ledger
from core.pg_writer import PgWriter
from data_collection.kalshi_client import (
    HISTORICAL_BASE_URL, KalshiClient, KalshiError, _money, parse_fill,
    parse_settlement, parse_trade,
)


def _writer() -> PgWriter:
    return PgWriter('sqlite:///:memory:')


# ---------------------------------------------------------------- parsing

def test_a_zero_revenue_is_a_measurement_and_not_a_missing_field():
    """A zero revenue is a measurement (a losing position), not a missing field.

    Reusing the quote parser, where zero means absent, would turn every loser into a
    null.
    """
    assert _money({'revenue': 0}, 'revenue') == 0.0
    assert _money({'revenue_dollars': '0.0000'}, 'revenue') == 0.0
    assert _money({}, 'revenue') is None
    assert _money({'revenue': 'nonsense'}, 'revenue') is None


def test_dollars_win_over_cents_so_the_ledger_is_not_a_hundred_times_wrong():
    """`revenue_dollars: "21.0000"` and `revenue: 2100` are the same number; dollars win."""
    assert _money({'revenue_dollars': '21.0000', 'revenue': 2100}, 'revenue') == 21.0
    # Cents only, the legacy encoding.
    assert _money({'revenue': 2100}, 'revenue') == 21.0


def test_a_no_fill_is_priced_from_the_no_side():
    """A NO fill is priced from `no_price_dollars`; the YES price would invert the cost basis."""
    fill = parse_fill({
        'trade_id': 't1', 'ticker': 'KXBTC15M-A', 'side': 'no', 'action': 'buy',
        'count_fp': '5.00', 'yes_price_dollars': '0.7000',
        'no_price_dollars': '0.3000', 'created_time': '2026-08-25T10:00:00Z',
    })
    assert fill.side == 'down'
    assert fill.price == pytest.approx(0.30)
    assert fill.contracts == pytest.approx(5.0)

    yes = parse_fill({'trade_id': 't2', 'ticker': 'A', 'side': 'yes',
                      'count_fp': '2.00', 'yes_price_dollars': '0.7000',
                      'no_price_dollars': '0.3000'})
    assert yes.side == 'up'
    assert yes.price == pytest.approx(0.70)


def test_an_unfamiliar_side_is_carried_through_rather_than_guessed():
    fill = parse_fill({'trade_id': 't', 'ticker': 'A', 'side': 'sideways'})
    assert fill.side == 'sideways'
    assert fill.price is None


def test_settlement_pnl_is_revenue_minus_cost_minus_fee():
    """5 NO contracts at 30c, won: $5.00 back on $1.50 of cost and a 7c fee."""
    settled = parse_settlement({
        'ticker': 'KXBTC15M-A', 'market_result': 'no', 'no_count_fp': '5.00',
        'no_total_cost_dollars': '1.5000', 'revenue_dollars': '5.0000',
        'fee_cost': 0.0700, 'settled_time': '2026-08-25T10:15:00Z',
    })
    assert settled.cost == pytest.approx(1.50)
    assert settled.pnl == pytest.approx(3.43)


def test_a_loser_settles_at_zero_revenue_and_a_negative_pnl():
    settled = parse_settlement({
        'ticker': 'A', 'market_result': 'yes', 'no_count_fp': '5.00',
        'no_total_cost_dollars': '1.5000', 'revenue_dollars': '0.0000',
        'fee_cost': 0.0700,
    })
    assert settled.revenue == 0.0
    assert settled.pnl == pytest.approx(-1.57)


def test_a_missing_field_leaves_the_pnl_unknown_rather_than_break_even():
    """A missing field leaves P&L as None, not 0.0."""
    settled = parse_settlement({'ticker': 'A', 'no_count_fp': '5.00',
                                'no_total_cost_dollars': '1.5000'})
    assert settled.revenue is None
    assert settled.pnl is None


def test_a_winning_favourite_that_nets_negative_is_still_a_win():
    """The win rate reads `market_result`, never the sign of the P&L.

    A favourite bought at 97c can win and still net negative after fees.
    """
    assert venue_ledger.won(market_result='yes', yes_contracts=100.0,
                            no_contracts=0.0) is True
    assert venue_ledger.won(market_result='no', yes_contracts=100.0,
                            no_contracts=0.0) is False
    # No result named, or neither side held: no answer, not a loss.
    assert venue_ledger.won(market_result=None, yes_contracts=5.0,
                            no_contracts=0.0) is None
    assert venue_ledger.won(market_result='yes', yes_contracts=0.0,
                            no_contracts=0.0) is None
    # Holding both sides of one market has no single answer.
    assert venue_ledger.won(market_result='yes', yes_contracts=1.0,
                            no_contracts=1.0) is None


# The venue's documented `/historical/trades` payload, as one literal, so the
# parser sees the real field set.
TAPE_PRINT = {
    'trade_id': 'x', 'ticker': 'A', 'count_fp': '10.00',
    'yes_price_dollars': '0.5600', 'no_price_dollars': '0.4400',
    'taker_outcome_side': 'yes', 'taker_book_side': 'bid',
    'created_time': '2023-11-07T05:31:56Z', 'is_block_trade': True,
    'taker_side': 'yes',
}


def test_the_tape_is_anonymous_and_carries_no_side_of_ours():
    """`/historical/trades` is the public tape, with no account attribution.

    A `Trade` has no position and no P&L. `taker_side` names the aggressor of a print
    from any account, so it is not a side of ours.
    """
    trade = parse_trade(TAPE_PRINT)
    assert trade.contracts == pytest.approx(10.0)
    assert trade.yes_price == pytest.approx(0.56)
    assert trade.no_price == pytest.approx(0.44)
    assert trade.is_block_trade is True
    assert not hasattr(trade, 'pnl')
    assert not hasattr(trade, 'contracts_held')


def test_the_taker_side_is_translated_and_the_book_side_is_not():
    """'yes'/'no' become 'up'/'down'; 'bid'/'ask' stay in the venue's terms, since a
    book side is not a direction.
    """
    trade = parse_trade(TAPE_PRINT)
    assert trade.taker_side == 'up'
    assert trade.taker_book_side == 'bid'

    no_side = parse_trade({**TAPE_PRINT, 'taker_outcome_side': 'no',
                           'taker_side': 'no', 'taker_book_side': 'ask'})
    assert no_side.taker_side == 'down'
    assert no_side.taker_book_side == 'ask'


def test_the_renamed_taker_field_wins_over_the_alias():
    """`taker_outcome_side` takes priority over the older `taker_side` alias."""
    trade = parse_trade({**TAPE_PRINT, 'taker_outcome_side': 'no',
                         'taker_side': 'yes'})
    assert trade.taker_side == 'down'
    # The alias alone still works.
    alias_only = parse_trade({'trade_id': 'y', 'ticker': 'A', 'taker_side': 'no'})
    assert alias_only.taker_side == 'down'
    # An unrecognised value is carried through, not guessed.
    assert parse_trade({'trade_id': 'z', 'ticker': 'A',
                        'taker_side': 'sideways'}).taker_side == 'sideways'
    assert parse_trade({'trade_id': 'w', 'ticker': 'A'}).taker_side is None


def test_a_print_whose_two_prices_do_not_sum_to_a_dollar_is_read_as_served():
    """Neither price is derived as `1 - the other`; both are read as served."""
    trade = parse_trade({**TAPE_PRINT, 'yes_price_dollars': '0.5600',
                         'no_price_dollars': '0.5600'})
    assert trade.yes_price == pytest.approx(0.56)
    assert trade.no_price == pytest.approx(0.56)


# ------------------------------------------------------- reading both tiers

class _Recorder:
    """A client whose `_request` is a scripted, recording stub."""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = responses
        self.calls: list[tuple[str, dict, str | None]] = []

    def attach(self, client: KalshiClient) -> KalshiClient:
        async def _request(method, path, *, params=None, body=None, base=None):
            self.calls.append((path, dict(params or {}), base))
            pages = self.responses.get(path)
            if pages is None:
                raise KalshiError(f'GET {path} -> 404')
            index = sum(1 for c in self.calls if c[0] == path) - 1
            return pages[min(index, len(pages) - 1)]

        client._request = _request  # noqa: SLF001 - that is the seam under test
        return client


def test_both_tiers_are_read_and_the_overlap_is_deduplicated():
    """Both ledger tiers are read and their overlap is deduplicated.

    The live routes cover only a recent window and older records are on
    `/historical/...`. The same fill can come back from both.
    """
    client = KalshiClient(key_id='k', private_key_pem='')
    recorder = _Recorder({
        '/portfolio/fills': [{'fills': [
            {'trade_id': 'a', 'ticker': 'K', 'side': 'yes', 'count_fp': '1.00',
             'yes_price_dollars': '0.50', 'created_time': '2026-08-25T10:00:00Z'},
            {'trade_id': 'b', 'ticker': 'K', 'side': 'no', 'count_fp': '2.00',
             'no_price_dollars': '0.30', 'created_time': '2026-08-24T10:00:00Z'},
        ]}],
        '/portfolio/fills/historical': [{'fills': [
            # 'b' again (the overlap), and one only the historical tier has.
            {'trade_id': 'b', 'ticker': 'K', 'side': 'no', 'count_fp': '2.00',
             'no_price_dollars': '0.30', 'created_time': '2026-08-24T10:00:00Z'},
            {'trade_id': 'c', 'ticker': 'K', 'side': 'yes', 'count_fp': '4.00',
             'yes_price_dollars': '0.20', 'created_time': '2026-01-01T10:00:00Z'},
        ]}],
    })
    recorder.attach(client)

    fills = asyncio.run(client.all_fills())
    assert [f.trade_id for f in fills] == ['a', 'b', 'c'], 'newest first, deduplicated'

    paths = [c[0] for c in recorder.calls]
    assert '/portfolio/fills' in paths and '/portfolio/fills/historical' in paths

    # The historical tier is on a different host; a base of None means the trading host.
    by_path = {c[0]: c[2] for c in recorder.calls}
    assert by_path['/portfolio/fills'] is None
    assert by_path['/portfolio/fills/historical'] == HISTORICAL_BASE_URL


def test_a_missing_historical_tier_does_not_lose_the_live_one():
    client = KalshiClient(key_id='k', private_key_pem='')
    _Recorder({
        '/portfolio/settlements': [{'settlements': [
            {'ticker': 'K1', 'market_result': 'yes', 'yes_count_fp': '1.00',
             'yes_total_cost_dollars': '0.50', 'revenue_dollars': '1.00',
             'fee_cost': 0.02,
             'settled_time': '2026-08-25T10:15:00Z'},
        ]}],
        # /portfolio/settlements/historical absent -> the stub raises KalshiError
    }).attach(client)

    settled = asyncio.run(client.all_settlements())
    assert [s.ticker for s in settled] == ['K1']
    assert settled[0].pnl == pytest.approx(0.48)


def test_pagination_follows_the_cursor_and_stops_when_it_repeats():
    client = KalshiClient(key_id='k', private_key_pem='')
    recorder = _Recorder({
        '/portfolio/fills': [
            {'fills': [{'trade_id': '1', 'ticker': 'K', 'side': 'yes'}],
             'cursor': 'page2'},
            {'fills': [{'trade_id': '2', 'ticker': 'K', 'side': 'yes'}],
             'cursor': 'page2'},  # the venue repeats itself
        ],
        '/portfolio/fills/historical': [{'fills': []}],
    })
    recorder.attach(client)

    fills = asyncio.run(client.all_fills())
    assert {f.trade_id for f in fills} == {'1', '2'}
    live_calls = [c for c in recorder.calls if c[0] == '/portfolio/fills']
    assert len(live_calls) == 2, 'stopped on the repeated cursor rather than looping'
    assert live_calls[1][1]['cursor'] == 'page2'


def test_the_tape_reads_the_historical_route_on_the_historical_host():
    client = KalshiClient(key_id='k', private_key_pem='')
    recorder = _Recorder({
        '/historical/trades': [{'trades': [
            {'trade_id': 'x', 'ticker': 'K', 'count_fp': '3.00',
             'yes_price_dollars': '0.40', 'no_price_dollars': '0.60'},
        ]}],
        '/markets/trades': [{'trades': []}],
    })
    recorder.attach(client)

    trades = asyncio.run(client.market_trades(ticker='K', historical=True))
    assert len(trades) == 1
    assert ('/historical/trades', {'limit': 100, 'ticker': 'K'},
            HISTORICAL_BASE_URL) in recorder.calls

    asyncio.run(client.market_trades(ticker='K'))
    assert any(c[0] == '/markets/trades' and c[2] is None for c in recorder.calls)


def test_an_unreadable_cutoff_is_not_fatal():
    """An unreadable cutoff returns {}, so both tiers are queried."""
    client = KalshiClient(key_id='k', private_key_pem='')
    _Recorder({}).attach(client)
    assert asyncio.run(client.historical_cutoff()) == {}


# ----------------------------------------------------------------- storing

def test_a_resynced_fill_does_not_double_the_cost_basis():
    """Fills are idempotent on the venue's `trade_id`, since both writers store the same
    fill repeatedly.
    """
    writer = _writer()
    fill = parse_fill({'trade_id': 't1', 'ticker': 'K', 'side': 'no',
                       'count_fp': '5.00', 'no_price_dollars': '0.30',
                       'created_time': '2026-08-25T10:00:00Z'})
    row = venue_ledger.fill_row(fill)
    writer.upsert_venue_fills([row])
    writer.upsert_venue_fills([row])
    writer.upsert_venue_fills([row])

    stored = writer.venue_fills()
    assert len(stored) == 1
    assert stored[0].price == pytest.approx(0.30)
    assert stored[0].contracts == pytest.approx(5.0)


def test_a_resync_heals_a_row_rather_than_preserving_a_bad_parse():
    """On conflict the row is overwritten, so a later, better parse replaces a bad one."""
    writer = _writer()
    writer.upsert_venue_settlements([venue_ledger.settlement_row(
        parse_settlement({'ticker': 'K1', 'yes_count_fp': '1.00'}))])
    assert writer.venue_settlements()[0].pnl is None

    writer.upsert_venue_settlements([venue_ledger.settlement_row(
        parse_settlement({'ticker': 'K1', 'market_result': 'yes',
                          'yes_count_fp': '1.00',
                          'yes_total_cost_dollars': '0.50',
                          'revenue_dollars': '1.00',
                          'fee_cost': 0.02}))])
    rows = writer.venue_settlements()
    assert len(rows) == 1
    assert rows[0].pnl == pytest.approx(0.48)


def test_a_fill_without_a_trade_id_is_skipped_rather_than_keyed_on_nothing():
    writer = _writer()
    written = writer.upsert_venue_fills([{'trade_id': '', 'ticker': 'K',
                                          'side': 'up', 'contracts': 1.0}])
    assert written == 0
    assert writer.venue_fills() == []


def test_the_settlement_total_is_over_everything_by_default():
    """The settlement total covers the whole ledger by default; a truncated total would
    look the same.
    """
    writer = _writer()
    base = datetime(2026, 8, 20, tzinfo=timezone.utc)
    for i in range(30):
        writer.upsert_venue_settlements([venue_ledger.settlement_row(
            parse_settlement({
                'ticker': f'K{i}', 'market_result': 'yes', 'yes_count_fp': '1.00',
                'yes_total_cost_dollars': '0.50', 'revenue_dollars': '1.00',
                'fee_cost': 0.00,
                'settled_time': (base + timedelta(minutes=15 * i)).isoformat(),
            }))])
    assert len(writer.venue_settlements()) == 30
    assert len(writer.venue_settlements(limit=5)) == 5
    assert venue_ledger.summarise(
        writer.venue_settlements()).realized_pnl == pytest.approx(15.0)


def test_the_balance_sample_records_both_figures_and_their_difference():
    writer = _writer()
    now = datetime.now(timezone.utc)
    writer.write_venue_balance(timestamp=now, balance=103.43, exchange_index=2,
                               our_bankroll=100.0)
    row = writer.latest_venue_balance()
    assert row.balance == pytest.approx(103.43)
    assert row.our_bankroll == pytest.approx(100.0)
    assert row.drift == pytest.approx(3.43)
    assert row.exchange_index == 2

    writer.prune_venue_balances(now + timedelta(minutes=1))
    assert writer.latest_venue_balance() is None


def test_the_minute_price_cycle_also_prunes_old_balance_samples():
    """The minute price cycle also prunes old `venue_balance` rows."""
    from livetrade.ledger import BALANCE_RETENTION_DAYS, record_minute_prices

    writer = _writer()
    now = datetime.now(timezone.utc)
    stale = now - timedelta(days=BALANCE_RETENTION_DAYS + 1)
    fresh = now - timedelta(hours=1)
    for stamp in (stale, fresh):
        writer.write_venue_balance(timestamp=stamp, balance=100.0,
                                   exchange_index=2, our_bankroll=100.0)

    record_minute_prices(writer, {})

    kept = writer.venue_balances_since(now - timedelta(days=365))
    assert len(kept) == 1, 'the stale sample survived the cycle'
    assert abs((kept[0].timestamp.replace(tzinfo=timezone.utc) - fresh)
               .total_seconds()) < 1


def test_a_settlement_joins_to_our_position_through_the_ticket():
    """A settlement joins to our position through `order_tickets`.

    A null `our_pnl` means the venue settled a market we have no record of buying.
    """
    writer = _writer()
    window = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)
    settle = window + timedelta(minutes=15)
    writer.ensure_account(100.0, mode='live')
    writer.write_ticket(symbol='BTC-USD', window_open=window, settle_time=settle,
                        offset_minutes=3, market_ticker='KXBTC15M-A', side='down',
                        contracts=5, limit_price=0.30, max_price=0.32,
                        expected_cost=1.52, model_probability=0.75, edge=0.03)
    position_id = writer.open_position(
        symbol='BTC-USD', window_open=window, settle_time=settle,
        offset_minutes=3, side='down', contracts=5, price=0.30, outlay=1.52,
        fee=0.02, model_probability=0.75, baseline_probability=0.70, edge=0.03)
    writer.settle_position(position_id, settled_up=False)

    found = writer.position_for_ticker('KXBTC15M-A')
    assert found is not None and found.id == position_id
    assert writer.position_for_ticker('KXBTC15M-NOTOURS') is None

    row = venue_ledger.settlement_row(
        parse_settlement({'ticker': 'KXBTC15M-A', 'market_result': 'no',
                          'no_count_fp': '5.00',
                          'no_total_cost_dollars': '1.5000',
                          'revenue_dollars': '5.0000',
                          'fee_cost': 0.0200,
                          'settled_time': settle.isoformat()}),
        position=found)
    assert row['position_id'] == position_id
    assert row['our_pnl'] == pytest.approx(found.pnl)
    assert row['pnl'] == pytest.approx(3.48)


# --------------------------------------------------------------- totalling

def test_an_incomplete_settlement_is_counted_and_not_added_as_zero():
    """An incomplete settlement is counted, not added as zero, since the total is short."""
    rows = [
        {'pnl': 1.0, 'fee_cost': 0.02, 'revenue': 2.0, 'yes_cost': 1.0,
         'no_cost': None, 'yes_contracts': 2.0, 'no_contracts': 0.0,
         'market_result': 'yes', 'settled_time': datetime(2026, 8, 25, tzinfo=timezone.utc)},
        {'pnl': None, 'fee_cost': None, 'revenue': None, 'yes_cost': None,
         'no_cost': None, 'yes_contracts': 3.0, 'no_contracts': 0.0,
         'market_result': None, 'settled_time': None},
    ]
    summary = venue_ledger.summarise(rows)
    assert summary.settlements == 2
    assert summary.realized_pnl == pytest.approx(1.0)
    assert summary.incomplete == 1
    assert summary.wins == 1
    assert summary.undecided == 1
    assert summary.contracts == pytest.approx(5.0)


def test_a_ledger_of_nothing_but_gaps_totals_to_none_not_to_zero():
    summary = venue_ledger.summarise([
        {'pnl': None, 'fee_cost': None, 'revenue': None, 'yes_cost': None,
         'no_cost': None, 'yes_contracts': 1.0, 'no_contracts': 0.0,
         'market_result': None, 'settled_time': None},
    ])
    assert summary.realized_pnl is None
    assert summary.fees is None
    assert summary.win_rate is None


def test_the_curve_skips_a_gap_rather_than_drawing_a_flat_step():
    t0 = datetime(2026, 8, 25, 10, 15, tzinfo=timezone.utc)
    rows = [
        {'ticker': 'A', 'pnl': 3.0, 'settled_time': t0},
        {'ticker': 'B', 'pnl': None, 'settled_time': t0 + timedelta(minutes=15)},
        {'ticker': 'C', 'pnl': -1.0, 'settled_time': t0 + timedelta(minutes=30)},
        {'ticker': 'D', 'pnl': 2.0, 'settled_time': None},
    ]
    points = venue_ledger.cumulative_curve(rows, starting_balance=100.0)
    assert [p['ticker'] for p in points] == ['A', 'C']
    assert [p['cumulative_pnl'] for p in points] == pytest.approx([3.0, 2.0])
    assert [p['equity'] for p in points] == pytest.approx([103.0, 102.0])


def test_the_curve_is_pnl_by_default_rather_than_a_back_projected_equity():
    """Without a starting balance the curve is P&L, with a null equity.

    Back-projecting the balance would assume no deposits.
    """
    points = venue_ledger.cumulative_curve(
        [{'ticker': 'A', 'pnl': 3.0,
          'settled_time': datetime(2026, 8, 25, tzinfo=timezone.utc)}])
    assert points[0]['equity'] is None
    assert points[0]['cumulative_pnl'] == pytest.approx(3.0)


def test_the_cash_flow_check_recovers_the_starting_balance():
    """The cash-flow check recovers the starting balance.

    $100 in, one 5-contract NO fill at 30c ($1.50 out), settled for $5.00 with a 7c fee:
    the balance is $103.43 and the implied start is exactly $100.
    """
    check = venue_ledger.balance_check(
        venue_balance=103.43,
        settlements=[{'revenue': 5.0, 'fee_cost': 0.07}],
        fills=[{'price': 0.30, 'contracts': 5.0}],
    )
    assert check['net_flow'] == pytest.approx(3.43)
    assert check['implied_starting_balance'] == pytest.approx(100.0)
    assert check['fills_without_price'] == 0


def test_a_double_counted_fee_shows_up_in_the_implied_start():
    """A double-counted fee moves the implied starting balance by the full fee bill."""
    settlements = [{'revenue': 1.0, 'fee_cost': 0.02} for _ in range(100)]
    fills = [{'price': 0.50, 'contracts': 1.0} for _ in range(100)]
    honest = venue_ledger.balance_check(
        venue_balance=100.0 + 100 * (1.0 - 0.50 - 0.02),
        settlements=settlements, fills=fills)
    assert honest['implied_starting_balance'] == pytest.approx(100.0)

    # The same ledger against a balance where the fee never left: the implied start
    # moves by the full fee bill.
    doubled = venue_ledger.balance_check(
        venue_balance=100.0 + 100 * (1.0 - 0.50),
        settlements=settlements, fills=fills)
    assert doubled['implied_starting_balance'] == pytest.approx(102.0)


def test_an_unpriced_fill_is_reported_rather_than_treated_as_free():
    check = venue_ledger.balance_check(
        venue_balance=100.0, settlements=[],
        fills=[{'price': None, 'contracts': 5.0}])
    assert check['fills_without_price'] == 1
    assert check['spent'] == pytest.approx(0.0)
