"""The live engine: score the current window, price it against the book, act.

One script for the whole operational loop, because the previous incarnation of
this project had four (`signals`, `paper_engine`, `live_orchestrator`, and a
promotion cadence inside the orchestrator) and they disagreed about what had
already happened.

Each cycle, in this order:

1. **Fetch bars.** The last day of one-minute bars for the three symbols,
   straight from Coinbase. A day is what the longest feature lookback needs.
2. **Record minute prices**, so the dashboard can draw the path against the
   strike. This is the only reason they are stored in the serving database.
3. **Settle** any position whose window has closed, from the bar that opens on
   its settlement minute. Settlement first, always: a position matures at the
   instant the next window opens, and deciding before settling would stake the
   same dollars twice.
4. **Score** the current window at the nearest configured offset, through the
   same `core.dataset` path the backtest uses.
5. **Price** it. Live, against the venue's own ask; without a venue, against the
   calibrated baseline, and the row says which.
6. **Decide** with `core.decide.decide` — the same function the backtest calls —
   and place, or record, or abstain.

**Modes, and what each will actually do.**

    --mode paper                 score, price, record. Places nothing.
    --mode live --dry-run        talk to Kalshi, read the real book, size the
                                 order, write the ticket, place nothing.
    --mode live --place-orders    place them.

`--place-orders` is a separate flag from `--mode live` on purpose. The failure
worth designing against is a script that was meant to observe and instead
traded, and one flag guarding that is one typo away from being wrong.

**The gates still apply.** `--require-gates` (the default) refuses to trade an
artifact whose promotion was blocked. Overriding it needs `--force` and a written
reason, which is recorded on every prediction the run writes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG, find_fee_config
from core.dataset import score_live
from core.decide import Decision, Reason, WindowExposure, decide, rejection_histogram
from core.pg_writer import PgWriter
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live
from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.kalshi_client import KalshiClient, KalshiError, Quote

logger = logging.getLogger('live')

# How much history each cycle fetches. The longest feature lookback is 1,440
# minutes, and the seasonal factor is a fitted lookup rather than a rolling
# window, so a day plus a margin is sufficient and a week is waste.
FETCH_MINUTES = 1_500

# Kalshi series for the 15-minute up/down markets, per Coinbase spot symbol.
# Resolved to an actual market by close time — see
# `KalshiClient.resolve_window_market` — so a series rename fails loudly here
# rather than silently trading the wrong contract.
#
# The `15M` suffix is load-bearing. `KXBTCD` was tried first and every window
# abstained: it is the *hourly* series, and its tickers carry an explicit strike
# (`KXBTCD-26AUG2317-T86749.99`), making it a threshold ladder rather than an
# up/down market. `KXBTC15M-26AUG230030` is series + date + HHMM with no strike
# suffix, which is the tell — the strike is the price at the window's open, and
# that is exactly what `core/windows.py` builds a target from.
SERIES_BY_SYMBOL = {
    'BTC-USD': os.getenv('KALSHI_SERIES_BTC', 'KXBTC15M'),
    'ETH-USD': os.getenv('KALSHI_SERIES_ETH', 'KXETH15M'),
    'SOL-USD': os.getenv('KALSHI_SERIES_SOL', 'KXSOL15M'),
}

# Minute prices older than this are dropped from the serving store each cycle.
PRICE_RETENTION_HOURS = 48


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    parser.add_argument('--dry-run', action='store_true', default=None,
                        help='Read the real book, size the order, place nothing. '
                             'Implied unless --place-orders is given.')
    parser.add_argument('--place-orders', action='store_true',
                        help='Actually place orders. Requires --mode live, and is '
                             'a separate flag from it deliberately.')
    parser.add_argument('--loop', action='store_true',
                        help='Run every cycle-seconds until interrupted.')
    parser.add_argument('--cycle-seconds', type=int, default=60)
    parser.add_argument('--bankroll', type=float, default=None,
                        help='Starting bankroll, used only when creating the account.')
    parser.add_argument('--offset', type=int, default=None,
                        help='Force a decision offset instead of using whichever '
                             'configured offset the clock has just passed.')
    parser.add_argument('--require-gates', dest='require_gates',
                        action='store_true', default=True)
    parser.add_argument('--no-require-gates', dest='require_gates', action='store_false')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--reason', type=str, default=None,
                        help='Required with --force. Recorded on every row written.')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Artifact path (default {MODELS_ROOT}/{LIVE_MODEL})')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def current_window(now: datetime, config: Config) -> tuple[pd.Timestamp, int]:
    """The window being decided, and how many minutes into it we are."""
    stamp = pd.Timestamp(now).tz_convert('UTC').floor(f'{config.window_minutes}min')
    elapsed = int((pd.Timestamp(now).tz_convert('UTC') - stamp).total_seconds() // 60)
    return stamp, elapsed


def choose_offset(elapsed: int, config: Config) -> Optional[int]:
    """The latest configured offset the clock has reached.

    Not the nearest: an offset in the future has not happened, and scoring at one
    would read a bar that does not exist. Returns None before the first offset,
    which is an abstention rather than an error.
    """
    reached = [o for o in sorted(config.decision_offsets) if o <= elapsed]
    return reached[-1] if reached else None


async def fetch_bars(config: Config, minutes: int = FETCH_MINUTES) -> dict[str, pd.DataFrame]:
    """One-minute bars for the universe, straight from the venue."""
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(minutes=minutes)
    client = CoinbaseRESTClient(
        api_key=os.getenv('COINBASE_API_KEY'),
        api_secret=os.getenv('COINBASE_API_SECRET'),
    )
    out: dict[str, pd.DataFrame] = {}
    try:
        for symbol in config.symbols:
            bars = await client.get_candles_range(symbol, '1m', start, end)
            if not bars:
                logger.error('%s: the venue returned no one-minute bars', symbol)
                continue
            frame = pd.DataFrame([{
                'event_time': pd.Timestamp(b.event_time, tz='UTC'),
                'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
                'volume': b.volume, 'quote_volume': getattr(b, 'quote_volume', np.nan),
                'trade_count': getattr(b, 'trade_count', np.nan),
            } for b in bars]).sort_values('event_time', ignore_index=True)
            out[symbol] = frame
            logger.info('%s: %d bars to %s', symbol, len(frame),
                        frame['event_time'].iloc[-1])
    finally:
        close = getattr(client, 'close', None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
    return out


def record_minute_prices(writer: PgWriter, bars: dict[str, pd.DataFrame],
                         *, hours: int = 6) -> int:
    """Store the last few hours of bars so the dashboard can draw the path."""
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
    rows = []
    for symbol, frame in bars.items():
        recent = frame.loc[frame['event_time'] >= cutoff]
        for bar in recent.itertuples():
            rows.append({
                'symbol': symbol, 'minute': bar.event_time.to_pydatetime(),
                'open': float(bar.open), 'high': float(bar.high),
                'low': float(bar.low), 'close': float(bar.close),
            })
    written = writer.write_minute_prices(rows) if rows else 0
    writer.prune_minute_prices(
        (pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=PRICE_RETENTION_HOURS))
        .to_pydatetime())
    return written


def settle_due(writer: PgWriter, bars: dict[str, pd.DataFrame]) -> list[tuple[int, float]]:
    """Settle every matured position from the bar opening on its settle minute.

    `open` of the settle minute, not `close` of the minute before: the strike was
    read the same way, so the window's return is open-to-open. Anchoring one end
    on a last trade and the other on a first trade is how this project once
    manufactured 98% of an apparent edge.
    """
    now = datetime.now(timezone.utc)
    settled: list[tuple[int, float]] = []
    for position in writer.positions_due(now):
        frame = bars.get(position.symbol)
        if frame is None:
            continue
        settle_minute = pd.Timestamp(position.settle_time).tz_convert('UTC')
        row = frame.loc[frame['event_time'] == settle_minute]
        if row.empty:
            logger.warning('%s window %s: no bar at the settle minute yet, waiting',
                           position.symbol, position.window_open)
            continue
        settle_price = float(row['open'].iloc[0])
        settled_up = settle_price > float(position.price_reference) \
            if hasattr(position, 'price_reference') else None
        # The strike is on the prediction row, not the position, so read it back.
        strike = _strike_for(writer, position)
        if strike is None:
            logger.error('%s window %s: no strike recorded, cannot settle',
                         position.symbol, position.window_open)
            continue
        settled_up = settle_price > strike
        pnl = writer.settle_position(position.id, settled_up=settled_up)
        if pnl is not None:
            settled.append((position.id, pnl))
            logger.info('settled %s %s: %s at %.4f vs strike %.4f -> %+.2f',
                        position.symbol, position.window_open,
                        'up' if settled_up else 'down', settle_price, strike, pnl)
    return settled


def _strike_for(writer: PgWriter, position) -> Optional[float]:
    from core.pg_writer import Prediction
    with writer._session() as session:  # noqa: SLF001 - same package, one query
        row = (session.query(Prediction)
               .filter(Prediction.symbol == position.symbol,
                       Prediction.window_open == position.window_open)
               .order_by(Prediction.offset_minutes)
               .first())
        return float(row.strike) if row is not None else None


async def fetch_quotes(
    kalshi: Optional[KalshiClient],
    symbols: list[str],
    settle_time: pd.Timestamp,
) -> dict[str, tuple[Quote, str]]:
    """Resolve each symbol's market for this window and read its book."""
    if kalshi is None:
        return {}
    quotes: dict[str, tuple[Quote, str]] = {}
    for symbol in symbols:
        series = SERIES_BY_SYMBOL.get(symbol)
        if not series:
            logger.warning('%s has no Kalshi series configured, no quote', symbol)
            continue
        try:
            market = await kalshi.resolve_window_market(
                series, settle_time.to_pydatetime())
            if market is None:
                continue
            ticker = str(market.get('ticker', ''))
            quote = await kalshi.quote(ticker)
            if not quote.tradeable():
                logger.info('%s %s is not tradeable (%s), abstaining',
                            symbol, ticker, quote.status)
                continue
            quotes[symbol] = (quote, ticker)
            logger.info('%s %s: %.2f / %.2f (spread %.0fc, vol %d)',
                        symbol, ticker, quote.yes_bid, quote.yes_ask,
                        (quote.spread or 0) * 100, quote.volume)
        except KalshiError as exc:
            logger.error('%s: could not read the book (%s)', symbol, exc)
    return quotes


async def run_cycle(args, config: Config, writer: PgWriter, model,
                    kalshi: Optional[KalshiClient]) -> list[Decision]:
    now = datetime.now(timezone.utc)
    window_open, elapsed = current_window(now, config)
    offset = args.offset if args.offset is not None else choose_offset(elapsed, config)

    bars = await fetch_bars(config)
    if not bars:
        logger.error('no bars, nothing to do this cycle')
        return []
    record_minute_prices(writer, bars)
    settle_due(writer, bars)

    if offset is None:
        logger.info('%d minutes into the window; first decision offset is +%dm',
                    elapsed, min(config.decision_offsets))
        return []

    scored = score_live(bars, model.scoring, config,
                        window_open=window_open, offset=offset,
                        groups=model.groups or None)
    scored['model_probability'] = model.predict(scored)

    settle_time = window_open + pd.Timedelta(minutes=config.window_minutes)
    quotes = await fetch_quotes(kalshi, list(scored['symbol'].unique()), settle_time)
    scored['ask_up'] = [
        quotes[s][0].ask_for('up') if s in quotes else np.nan for s in scored['symbol']]
    scored['ask_down'] = [
        quotes[s][0].ask_for('down') if s in quotes else np.nan for s in scored['symbol']]
    scored['market_ticker'] = [
        quotes[s][1] if s in quotes else None for s in scored['symbol']]

    account = writer.ensure_account(config.starting_bankroll, mode=args.mode)
    exposure = WindowExposure()
    decisions: list[Decision] = []

    for _, row in scored.sort_values('symbol').iterrows():
        decision = decide(row, config, bankroll=account.bankroll, exposure=exposure)
        decisions.append(decision)
        writer.write_prediction(
            symbol=decision.symbol, window_open=window_open, settle_time=settle_time,
            offset_minutes=offset, decision_time=window_open + pd.Timedelta(minutes=offset),
            strike=float(row['strike']), last_price=float(row['last_price']),
            displacement=float(row['displacement']),
            sigma_remaining=_finite(row.get('sigma_remaining')),
            z_score=_finite(row.get('z_score')),
            baseline_probability=float(row['baseline_probability']),
            model_probability=float(row['model_probability']),
            market_probability=_finite(row.get('ask_up')),
            price_source=decision.price_source,
            reason=decision.reason.value, traded=decision.traded,
            side=decision.side.value if decision.side else None,
            price=_finite(decision.price), effective_cost=_finite(decision.effective_cost),
            edge=_finite(decision.edge), contracts=decision.contracts or None,
            model_version=getattr(model, 'version', None),
        )
        logger.info(decision.describe())
        if not decision.traded:
            continue
        exposure = exposure.with_(decision)
        await act_on(args, writer, kalshi, decision, row)

    writer.write_equity_point(
        timestamp=now, equity=account.bankroll,
        bankroll=account.bankroll,
        staked=sum(p.outlay for p in writer.open_positions()),
        open_positions=len(writer.open_positions()),
        realized_pnl=account.realized_pnl,
    )
    return decisions


async def act_on(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                 decision: Decision, row) -> None:
    """Record the position, and place the order when asked to twice."""
    ticket_id = writer.write_ticket(
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        market_ticker=decision.market_ticker, side=decision.side.value,
        contracts=decision.contracts, limit_price=decision.price,
        # The worst price still worth paying: where the edge reaches zero. A
        # ticket without this is an instruction to pay anything.
        max_price=float(min(0.99, decision.price + max(0.0, decision.edge))),
        expected_cost=decision.stake, model_probability=decision.model_probability,
        edge=decision.edge,
    )

    placed_price = decision.price
    filled = decision.contracts
    if args.place_orders and kalshi is not None and decision.market_ticker:
        try:
            order = await kalshi.place_order(
                ticker=decision.market_ticker, side=decision.side.value,
                contracts=decision.contracts,
                limit_price=float(min(0.99, decision.price + max(0.0, decision.edge))),
                client_order_id=f'{decision.symbol}-{decision.window_open:%Y%m%d%H%M}',
            )
            filled = int(order.get('count', decision.contracts))
            writer.resolve_ticket(ticket_id, status='placed',
                                  filled_contracts=filled,
                                  filled_price=placed_price,
                                  note=str(order.get('order_id', ''))[:200])
            logger.info('placed order %s', order.get('order_id'))
        except (KalshiError, ValueError) as exc:
            writer.resolve_ticket(ticket_id, status='skipped', note=str(exc)[:400])
            logger.error('order refused, no position recorded: %s', exc)
            return
    elif args.mode == 'live':
        logger.info('dry run: ticket %d written, no order placed', ticket_id)

    writer.open_position(
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        side=decision.side.value, contracts=filled, price=placed_price,
        outlay=decision.stake, fee=decision.fee,
        model_probability=decision.model_probability,
        baseline_probability=decision.baseline_probability, edge=decision.edge,
    )
    account = writer.account()
    if account is not None:
        writer.update_account(bankroll=account.bankroll - decision.stake,
                              fees_paid=account.fees_paid + decision.fee)


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-14s %(message)s',
        datefmt='%H:%M:%S', stream=sys.stdout)

    if args.place_orders and args.mode != 'live':
        raise SystemExit('--place-orders requires --mode live')
    if args.force and not args.reason:
        raise SystemExit('--force needs --reason, and it is recorded on every row')

    config = DEFAULT_CONFIG.with_fee_assumptions(find_fee_config())
    if args.bankroll is not None:
        config = config.with_overrides(starting_bankroll=args.bankroll)

    model = load_live() if args.model is None else __import__(
        'core.model', fromlist=['ForecastModel']).ForecastModel.load(args.model)
    if model is None:
        raise SystemExit(
            f'no artifact at {MODELS_ROOT / LIVE_MODEL}. Run '
            f'`python -m scripts.promote` first — promotion is the only path to '
            f'a live model, deliberately.')
    if not model.deployable:
        raise SystemExit(
            'this artifact carries no scoring bundle, so it cannot score a window '
            'it has never seen. Re-run `python -m scripts.promote` with the '
            'current code — artifacts from before the bundle existed can be '
            'evaluated but not deployed.')
    if args.require_gates and not args.force:
        _refuse_if_blocked()

    print('=' * 78)
    print(f'Quarter — {args.mode} mode'
          + ('  [PLACING ORDERS]' if args.place_orders else '  [dry run]'))
    print('=' * 78)
    print(model.summary())
    print(model.scoring.summary())
    print(f'bankroll          ${config.starting_bankroll:.2f}, '
          f'{config.kelly_fraction:.2f} Kelly, gate {config.min_edge_pp:.2f}pp, '
          f'cap ${config.max_stake_dollars}')
    print()

    writer = PgWriter()
    kalshi: Optional[KalshiClient] = None
    if args.mode == 'live':
        kalshi = KalshiClient(live=bool(args.place_orders))
        if not kalshi.configured:
            raise SystemExit(
                'live mode needs Kalshi credentials: KALSHI_KEY_ID and either '
                'KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH.')
        await kalshi.__aenter__()
        logger.info('Kalshi balance $%.2f', await kalshi.balance())

    try:
        while True:
            decisions = await run_cycle(args, config, writer, model, kalshi)
            if decisions:
                counts = rejection_histogram(decisions)
                logger.info('cycle: %s', counts[counts > 0].to_dict())
            if not args.loop:
                return 0
            await asyncio.sleep(args.cycle_seconds)
    except KeyboardInterrupt:
        logger.info('stopped')
        return 0
    finally:
        if kalshi is not None:
            await kalshi.close()


def _refuse_if_blocked() -> None:
    from core.promotion import history

    frame = history()
    if frame.empty:
        raise SystemExit('no promotion attempt recorded; refusing to trade')
    latest = frame.iloc[0]
    if not bool(latest.get('installed')):
        raise SystemExit(
            f"the newest attempt {latest.get('version')} was blocked on "
            f"{latest.get('failed_gates')}. Trading it needs --no-require-gates "
            f"and --force with a written reason.")


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
