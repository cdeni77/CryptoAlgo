"""The maker-fill simulator, on synthetic books with a known answer.

The stream input reaches the simulator as raw `venue_book_events` rows (a
snapshot, then signed deltas) built by `core/spool.event_rows`, the same path
the recorder archives through. The snapshot input is the packed Predexon touch
that `research/collect` writes.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.config import DEFAULT_CONFIG
from core.spool import event_rows
from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from research.analysis import maker_fill as mf

OPEN = pd.Timestamp('2026-10-01T00:00Z')
T0 = OPEN.timestamp()
DEC = T0 + 3 * 60 + 3                       # offset 3, 3s delay
# The window closes 20:15 ET on 2026-09-30, so the ticker names that close.
TICKER = 'KXBTC15M-26SEP302015-15'
SYMBOL = 'BTC-USD'
POLICY = mf.Policy(offsets=(3,), rest_seconds=20.0, horizons=(10, 30, 60),
                   contracts=10, maker_fee_rate=0.0025)

_seq = iter(range(1, 10_000))


def snap(t, yes, no, ticker=TICKER):
    return BookEvent(venue='kalshi', market_ticker=ticker, kind='snapshot',
                     received=t, seq=next(_seq), yes=list(yes.items()),
                     no=list(no.items()))


def delta(t, side, price, change, ticker=TICKER):
    levels = [(price, change)]
    return BookEvent(venue='kalshi', market_ticker=ticker, kind='delta',
                     received=t, seq=next(_seq), yes=levels if side == 'yes' else [],
                     no=levels if side == 'no' else [], absolute=False)


def rows(events, symbol=SYMBOL):
    records = [r for e in events for r in event_rows(e, symbol)]
    frame = pd.DataFrame(records)
    frame['event_time'] = pd.to_datetime(frame.pop('t'), unit='s', utc=True)
    return frame


# YES bid 0.80 (100 resting), YES ask 1 - 0.18 = 0.82, mid 0.81: bias buys UP.
UP_BOOK = ({0.70: 20.0, 0.78: 40.0, 0.80: 100.0}, {0.17: 30.0, 0.18: 50.0})


def run_stream(events, policy=POLICY, settlements=None):
    settlements = settlements or mf.Settlements({}, {})
    skips = Counter()
    orders = mf.simulate_stream(rows(events), settlements, policy, skips=skips)
    return orders, skips


def only(orders):
    assert len(orders) == 1, orders
    return orders[0]


# ---- replay ------------------------------------------------------------------

def test_replay_reproduces_the_book_bookcache_builds():
    events = [snap(T0 + 1, *UP_BOOK),
              delta(T0 + 2, 'yes', 0.80, -30.0),
              delta(T0 + 2, 'no', 0.19, 12.0),       # same instant, new seq
              delta(T0 + 3, 'yes', 0.78, -40.0),     # level removed
              delta(T0 + 4, 'no', 0.17, 5.5)]
    direct = BookCache(max_age_seconds=math.inf)
    for event in events:
        direct.apply(event)

    replayed = BookCache(max_age_seconds=math.inf)
    (ticker, symbol, updates), = list(mf.stream_updates(rows(events)))
    assert (ticker, symbol, len(updates)) == (TICKER, SYMBOL, len(events))
    for _, event in updates:
        replayed.apply(event)
    a, b = direct.ladder(TICKER), replayed.ladder(TICKER)
    assert a.yes == pytest.approx(b.yes) and a.no == pytest.approx(b.no)
    assert dict(b.yes) == pytest.approx({0.70: 20.0, 0.80: 70.0})


def test_window_comes_from_the_ticker_when_no_settlement_row():
    order = only(run_stream([snap(DEC - 10, *UP_BOOK)])[0])
    assert order.window_open == OPEN
    assert order.decided == pytest.approx(DEC)


# ---- the fill rules -------------------------------------------------------------

def test_post_joins_the_best_bid_on_the_favourite():
    order = only(run_stream([snap(DEC - 10, *UP_BOOK)])[0])
    assert (order.side, order.price, order.ahead) == ('UP', 0.80, 100.0)
    assert (order.bid, order.ask) == pytest.approx((0.80, 0.82))
    assert not order.fills


def test_improve_posts_one_tick_above_the_bid():
    policy = mf.Policy(offsets=(3,), improve=True)
    order = only(run_stream([snap(DEC - 10, *UP_BOOK)], policy)[0])
    assert order.price == pytest.approx(0.81) and order.improved and order.ahead == 0.0


def test_improve_never_crosses():
    assert mf.post_price(0.80, 0.81, True) == (0.80, False)
    assert mf.post_price(0.950, 0.953, True) == (pytest.approx(0.951), True)


def test_a_trade_through_fills_in_both_modes():
    events = [snap(DEC - 10, *UP_BOOK),
              delta(DEC + 5, 'yes', 0.80, -100.0),
              delta(DEC + 5, 'no', 0.20, 10.0)]       # YES ask 0.80
    order = only(run_stream(events)[0])
    assert order.fills == {'conservative': pytest.approx(DEC + 5),
                           'queue': pytest.approx(DEC + 5)}


def test_a_touch_without_trade_through_does_not_fill():
    events = [snap(DEC - 10, *UP_BOOK),
              delta(DEC + 2, 'no', 0.19, 10.0),       # ask 0.81: one tick away
              delta(DEC + 4, 'yes', 0.80, -30.0)]     # 30 of the 100 ahead
    order = only(run_stream(events)[0])
    assert order.fills == {}


def test_queue_mode_fills_only_after_the_queue_ahead_is_gone():
    events = [snap(DEC - 10, *UP_BOOK),
              delta(DEC + 2, 'yes', 0.80, -60.0),
              delta(DEC + 3, 'yes', 0.80, 20.0),      # joins behind us
              delta(DEC + 4, 'yes', 0.80, -40.0),     # 100 gone: at the front
              delta(DEC + 6, 'yes', 0.80, -5.0)]      # one more: filled
    order = only(run_stream(events)[0])
    assert 'conservative' not in order.fills
    assert order.fills['queue'] == pytest.approx(DEC + 6)


def test_no_frame_after_the_cancel_can_fill():
    late = [snap(DEC - 10, *UP_BOOK), delta(DEC + 20.5, 'no', 0.20, 10.0)]
    assert only(run_stream(late)[0]).fills == {}
    at_cancel = [snap(DEC - 10, *UP_BOOK), delta(DEC + 20, 'no', 0.20, 10.0)]
    assert set(only(run_stream(at_cancel)[0]).fills) == {'conservative', 'queue'}


def test_down_side_buys_no_and_is_filled_by_a_yes_bid():
    yes, no = {0.18: 50.0}, {0.78: 40.0, 0.80: 100.0}  # YES mid 0.19
    events = [snap(DEC - 10, yes, no), delta(DEC + 5, 'yes', 0.20, 10.0)]
    order = only(run_stream(events)[0])
    assert (order.side, order.price) == ('DOWN', 0.80)
    assert order.ask == pytest.approx(0.82)
    assert 'conservative' in order.fills


def test_all_mode_posts_both_sides():
    orders, _ = run_stream([snap(DEC - 10, *UP_BOOK)], mf.Policy(offsets=(3,), select='all'))
    assert sorted(o.side for o in orders) == ['DOWN', 'UP']
    down = next(o for o in orders if o.side == 'DOWN')
    assert (down.bid, down.ask) == pytest.approx((0.18, 0.20))


def test_bias_mode_skips_a_mid_outside_the_band():
    orders, skips = run_stream([snap(DEC - 10, {0.50: 10.0}, {0.48: 10.0})])
    assert orders == [] and skips['no side selected'] == 1


def test_model_mode_buys_the_side_with_positive_edge():
    policy = mf.Policy(offsets=(3,), select='model')
    model = {(SYMBOL, OPEN, 3): 0.10}               # P(down) 0.90 against NO at 0.18
    orders = mf.simulate_stream(rows([snap(DEC - 10, *UP_BOOK)]), mf.Settlements({}, {}),
                                policy, model=model)
    assert only(orders).side == 'DOWN'


def test_a_stale_book_is_not_used():
    orders, skips = run_stream([snap(DEC - 120, *UP_BOOK)])
    assert orders == [] and skips['no current book'] == 1


# ---- marks and P&L ---------------------------------------------------------------

def _filled_then(after):
    return [snap(DEC - 10, *UP_BOOK),
            delta(DEC + 5, 'yes', 0.80, -100.0),
            delta(DEC + 5, 'no', 0.20, 10.0),        # fill; bid 0.78 ask 0.80
            *after]


def test_adverse_selection_is_positive_when_the_mid_moves_against_us():
    order = only(run_stream(_filled_then([delta(DEC + 12, 'yes', 0.78, -40.0)]))[0])
    assert order.mid_fill['conservative'] == pytest.approx(0.79)
    assert order.marks[('conservative', 10)] == pytest.approx(0.75)
    frame = mf.order_rows([order], mf.Settlements({}, {}), POLICY)
    assert (frame['adverse_10s'] > 0).all()


def test_adverse_selection_is_negative_when_the_mid_moves_for_us():
    order = only(run_stream(_filled_then([delta(DEC + 12, 'yes', 0.79, 10.0)]))[0])
    frame = mf.order_rows([order], mf.Settlements({}, {}), POLICY)
    assert frame['adverse_10s'].iloc[0] == pytest.approx(0.79 - 0.795)


def test_pnl_includes_the_maker_fee_and_the_taker_comparison():
    order = only(run_stream(_filled_then([]))[0])
    frame = mf.order_rows([order], mf.Settlements({TICKER: True}, {TICKER: OPEN}), POLICY)
    cons = frame[frame['mode'] == 'conservative'].iloc[0]
    assert cons['payout'] == 1.0
    assert cons['maker_fee'] == pytest.approx(0.0025)
    assert cons['pnl'] == pytest.approx(1.0 - 0.80 - 0.0025)
    taker_fee = math.ceil(round(0.07 * 10 * 0.82 * 0.18 * 10_000, 6)) / 10_000 / 10
    assert cons['taker_fee'] == pytest.approx(taker_fee)
    assert cons['taker_pnl'] == pytest.approx(1.0 - 0.82 - taker_fee)
    assert cons['adverse_settle'] == pytest.approx(0.79 - 1.0)


def test_a_losing_side_and_an_unfilled_order():
    order = only(run_stream([snap(DEC - 10, *UP_BOOK)])[0])
    frame = mf.order_rows([order], mf.Settlements({TICKER: False}, {}), POLICY)
    assert (frame['payout'] == 0.0).all() and frame['pnl'].isna().all()
    assert frame['taker_pnl'].iloc[0] == pytest.approx(-0.82 - frame['taker_fee'].iloc[0])


def test_maker_fee_defaults_to_config():
    assert mf.Policy().maker_fee_rate == DEFAULT_CONFIG.maker_fee_rate


# ---- point in time -------------------------------------------------------------------

def test_a_frame_after_the_decision_cannot_move_the_post():
    events = [snap(DEC - 10, *UP_BOOK), delta(DEC + 0.001, 'yes', 0.81, 10.0)]
    order = only(run_stream(events)[0])
    assert (order.price, order.ahead) == (0.80, 100.0)


def test_a_frame_at_the_decision_instant_does_set_the_post():
    events = [snap(DEC - 10, *UP_BOOK), delta(DEC, 'yes', 0.81, 10.0)]
    assert only(run_stream(events)[0]).price == pytest.approx(0.81)


# ---- input 2: snapshots ----------------------------------------------------------------

def snapshot_rows(series, ticker=TICKER, symbol=SYMBOL, opened=OPEN):
    return pd.DataFrame([{'symbol': symbol, 'market_id': ticker, 'window_open': opened,
                          'ts': int(round(t * 1000)), 'best_bid': b, 'best_ask': a}
                         for t, b, a in series])


def test_snapshots_fill_only_on_trade_through_and_label_the_input():
    series = [(DEC - 10, 80, 82),
              (DEC + 0.5, 81, 82),      # after the decision: cannot set the post
              (DEC + 3, 81, 81),        # touch, no trade-through
              (DEC + 5, 79, 80)]        # trade-through
    skips = Counter()
    order = only(mf.simulate_snapshots(snapshot_rows(series), POLICY, skips=skips))
    assert (order.input, order.price, order.ask) == ('snapshots', 0.80, 0.82)
    assert order.fills == {'conservative': pytest.approx(DEC + 5)}
    frame = mf.order_rows([order], mf.Settlements({TICKER: True}, {}), POLICY)
    assert list(frame['mode']) == ['conservative']
    assert frame['pnl'].iloc[0] == pytest.approx(1.0 - 0.80 - 0.0025)
    text = mf.report(frame, POLICY, input='snapshots', skips=skips, n_boot=50)
    assert 'input=snapshots  fill=conservative' in text and 'fill=queue' not in text


def test_snapshots_after_the_cancel_cannot_fill():
    series = [(DEC - 10, 80, 82), (DEC + 21, 79, 80)]
    assert only(mf.simulate_snapshots(snapshot_rows(series), POLICY)).fills == {}


def test_snapshot_down_side_converts_the_yes_touch():
    series = [(DEC - 10, 18, 20), (DEC + 5, 20, 21)]   # YES bid 0.20: NO ask 0.80
    order = only(mf.simulate_snapshots(snapshot_rows(series), POLICY))
    assert order.side == 'DOWN' and order.price == pytest.approx(0.80)
    assert 'conservative' in order.fills


def test_load_snapshots_reads_derived_and_archive(tmp_path):
    from research.collect.consolidate import DERIVED_SCHEMA
    from research.collect.fetchers import FIELDS

    series = [(DEC - 10, 80, 82), (DEC + 5, 79, 80)]
    derived = snapshot_rows(series)
    full = pd.DataFrame({f: [float('nan')] * len(derived) for f in FIELDS[1:]})
    full['ts'] = derived['ts']
    full['best_bid'], full['best_ask'] = derived['best_bid'], derived['best_ask']
    full['venue'], full['symbol'], full['market_id'] = 'kalshi', SYMBOL, TICKER
    full['window_open'] = OPEN
    full['event_time'] = pd.to_datetime(full['ts'], unit='ms', utc=True)
    full['offset_seconds'] = 0.0
    path = tmp_path / 'derived/venue=kalshi/symbol=BTC-USD/month=2026-10/data.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(full, schema=DERIVED_SCHEMA,
                                        preserve_index=False), path)

    eth = 'KXETH15M-26SEP302015-15'
    archive = tmp_path / 'archive/venue=kalshi/symbol=ETH-USD/month=2026-10'
    archive.mkdir(parents=True)
    packed = [[int(round(t * 1000)), b, a] + [0] * (len(FIELDS) - 3) for t, b, a in series]
    with gzip.open(archive / 'windows.jsonl.gz', 'at') as handle:
        handle.write(json.dumps({'venue': 'kalshi', 'symbol': 'ETH-USD',
                                 'market_id': eth, 'window_open': OPEN.isoformat(),
                                 'fields': list(FIELDS), 'series': packed,
                                 'ladders': []}) + '\n')

    loaded = mf.load_snapshots(tmp_path, ['BTC-USD', 'ETH-USD'], OPEN,
                               OPEN + pd.Timedelta(minutes=15))
    assert sorted(loaded['market_id'].unique()) == sorted([eth, TICKER])
    orders = mf.simulate_snapshots(loaded, POLICY)
    assert sorted(o.symbol for o in orders) == ['BTC-USD', 'ETH-USD']
    assert all('conservative' in o.fills for o in orders)
    outside = mf.load_snapshots(tmp_path, ['BTC-USD'], OPEN + pd.Timedelta(minutes=15),
                                OPEN + pd.Timedelta(minutes=30))
    assert outside.empty


# ---- statistics -----------------------------------------------------------------------

def test_cluster_bootstrap_resamples_whole_windows():
    values = [1.0, 1.0, -1.0, -1.0, 0.5, 0.5]
    clusters = ['a', 'a', 'b', 'b', 'c', 'c']
    mean, lo, hi = mf.cluster_bootstrap(values, clusters, n_boot=2000, seed=1)
    assert mean == pytest.approx(1.0 / 6.0)
    assert lo < mean < hi and lo >= -1.0 and hi <= 1.0
    assert mf.cluster_bootstrap(values, clusters, n_boot=2000, seed=1) == (mean, lo, hi)
    one = mf.cluster_bootstrap([1.0, 2.0], ['a', 'a'], n_boot=100)
    assert one[0] == 1.5 and math.isnan(one[1])


def test_summary_reports_fill_rate_and_both_modes():
    filled = only(run_stream(_filled_then([]))[0])
    unfilled = only(run_stream([snap(DEC - 10, *UP_BOOK)])[0])
    frame = mf.order_rows([filled, unfilled], mf.Settlements({TICKER: True}, {}), POLICY)
    table = mf.summarise(frame[frame['mode'] == 'queue'], n_boot=100)
    assert (table['orders'].iloc[0], table['fills'].iloc[0]) == (2, 1)
    assert table['fill_rate'].iloc[0] == 0.5
    assert table['maker_c'].iloc[0] == pytest.approx((1.0 - 0.80 - 0.0025) * 100)
    assert table['taker_n'].iloc[0] == 2
    filled_taker = frame[(frame['mode'] == 'queue') & frame['filled']]['taker_pnl']
    assert table['taker_filled_c'].iloc[0] == pytest.approx(filled_taker.iloc[0] * 100)
    text = mf.report(frame, POLICY, input='stream', skips=Counter(), n_boot=50)
    assert 'input=stream  fill=conservative' in text and 'input=stream  fill=queue' in text


# ---- the CLI, end to end -----------------------------------------------------------------

def test_cli_reads_the_store_and_reports_both_inputs(tmp_path, capsys):
    from core.datastore import ResearchStore
    from core.spool import FrameSpool, compact

    events = _filled_then([delta(DEC + 30, 'yes', 0.78, -40.0)])
    spool = FrameSpool(tmp_path / 'spool', 'kalshi')
    for event in events:
        spool.extend(event_rows(event, SYMBOL))
    spool.close()
    store_root = tmp_path / 'store'
    expected = sum(len(list(event_rows(e, SYMBOL))) for e in events)
    assert compact(tmp_path / 'spool', store_root, keep_days=0, now=T0 + 7200) == expected
    store = ResearchStore(store_root)
    store.write('venue_settlements', pd.DataFrame([{
        'venue': 'kalshi', 'symbol': SYMBOL, 'event_time': OPEN,
        'available_time': OPEN + pd.Timedelta(minutes=20), 'quality': 'valid',
        'market_ticker': TICKER, 'window_open': OPEN,
        'close_time': OPEN + pd.Timedelta(minutes=15), 'settlement_time': None,
        'result': 'yes', 'settled_up': True, 'volume': None,
        'open_interest': None, 'last_price': None}]))

    out = tmp_path / 'rows.csv'
    common = ['--store', str(store_root), '--symbols', SYMBOL,
              '--start', OPEN.isoformat(),
              '--end', (OPEN + pd.Timedelta(hours=1)).isoformat(),
              '--offsets', '3', '--contracts', '10', '--boot', '200']
    assert mf.main(common + ['--out', str(out)]) == 0
    text = capsys.readouterr().out
    assert 'input=stream  fill=conservative  all orders' in text
    assert 'input=stream  fill=queue  by offset' in text
    written = pd.read_csv(out)
    assert set(written['mode']) == {'conservative', 'queue'}
    assert written['pnl'].tolist() == pytest.approx([1.0 - 0.80 - 0.0025] * 2)

    root = tmp_path / 'collection'
    path = root / 'derived/venue=kalshi/symbol=BTC-USD/month=2026-10/data.parquet'
    path.parent.mkdir(parents=True)
    snapshot_rows([(DEC - 10, 80, 82), (DEC + 5, 79, 80)]).drop(
        columns='symbol').to_parquet(path, index=False)
    assert mf.main(common + ['--input', 'snapshots', '--collect-data', str(root)]) == 0
    text = capsys.readouterr().out
    assert 'input=snapshots  fill=conservative  all orders' in text
    assert 'fill=queue' not in text


def test_the_quadratic_maker_fee_follows_the_venue_schedule():
    """round_up(M * 0.0175 * C * P * (1 - P)), per order, to a hundredth of a cent."""
    from core.config import DEFAULT_CONFIG
    from research.analysis.maker_fill import maker_fee

    # 10 contracts at 50c, M=1: 0.0175 * 10 * 0.25 = 0.04375 -> 0.0438 per order.
    assert maker_fee(0.50, 10, DEFAULT_CONFIG, multiplier=1.0) == pytest.approx(0.00438)
    assert maker_fee(0.50, 10, DEFAULT_CONFIG, multiplier=0.0) == 0.0
    # The default series multiplier is zero, so the flat config value is ignored.
    assert maker_fee(0.90, 10, DEFAULT_CONFIG, multiplier=0.0) == 0.0
