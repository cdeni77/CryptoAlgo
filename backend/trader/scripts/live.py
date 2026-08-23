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

**Live, the venue is the account of record.** In paper mode the bankroll is
arithmetic — start at the configured figure, subtract each outlay, add each
payout — and settlement comes from our own bars. Live, both of those are
*estimates of someone else's ledger*, and where they disagree the venue is right
and we are wrong:

* **Balance** comes from `/portfolio/balance` each cycle. Our running figure is
  kept alongside and the gap is logged, because a widening gap is the first sign
  of an unrecorded fill or a partial.
* **Settlement** comes from the venue where it can. We approximate sixty seconds
  of CF Benchmarks BRTI with a one-minute OHLC mean of Coinbase bars, which is a
  close proxy and not the same number — so a position settled from our bars can
  disagree with what was actually paid. `--reconcile` prefers the venue's
  settlements and falls back to bars only for what it has not resolved yet.
* **Fills** are read back rather than assumed. An order placed is not an order
  filled, and a `fill_or_kill` that killed leaves a ticket and no position.
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
from core.dataset import DatasetError
from core.pg_writer import AccountModeMismatch, PgWriter, TraderAlreadyRunning
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live
from core.windows import bar_mean
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
    # Mutually exclusive, at the parser. `--dry-run` used to be declared and
    # never read — `args.dry_run` appeared nowhere in this file — so
    # `--mode live --dry-run --place-orders` parsed cleanly and placed real
    # orders. A flag documented as a safety guard has to be either honoured or
    # a usage error; silently ignored is the one option that gets money lost.
    orders = parser.add_mutually_exclusive_group()
    orders.add_argument('--dry-run', action='store_true', default=False,
                        help='Read the real book, size the order, place nothing '
                             'and book nothing. Implied unless --place-orders.')
    orders.add_argument('--place-orders', action='store_true',
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
    parser.add_argument('--reconcile', dest='reconcile', action='store_true',
                        default=True,
                        help='Live only. Take balance, fills and settlements from '
                             'the venue rather than from our own arithmetic. On by '
                             'default because the venue is the account of record.')
    parser.add_argument('--no-reconcile', dest='reconcile', action='store_false')
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


def venue_settled_up(row: dict, side: str) -> Optional[bool]:
    """Did the *up* side win, according to the venue's settlement row?

    The venue is the account of record, so this is preferred over any bar-derived
    answer. Kalshi has published more than one shape for this, so read an
    explicit result where there is one and fall back to inferring from revenue
    and the side we held. Returns None when neither is legible — the caller then
    falls back to bars and says so, rather than guessing.
    """
    for key in ('market_result', 'result', 'settlement_result'):
        value = row.get(key)
        if isinstance(value, str) and value.strip().lower() in ('yes', 'no'):
            return value.strip().lower() == 'yes'
    revenue = row.get('revenue_dollars', row.get('revenue'))
    try:
        revenue = float(revenue)
    except (TypeError, ValueError):
        return None
    # Revenue is what the position paid out. A paid-out `up` holding means yes
    # won; a paid-out `down` holding means it lost.
    won = revenue > 0.0
    if side == 'up':
        return won
    if side == 'down':
        return not won
    return None


def settle_due(writer: PgWriter, bars: dict[str, pd.DataFrame],
               *, venue_settlements: Optional[dict[str, dict]] = None,
               ) -> list[tuple[int, float]]:
    """Settle every matured position, on the venue's rule.

    **This must agree with `core/windows.py:build_windows` exactly**, because the
    model is trained against that label and the money is booked against this one.
    It did not. This function read the *open* of the bar starting at `settle_time`
    and compared with a strict `>`, while the target is the `(O+H+L+C)/4` mean of
    the minute *ending* at `settle_time` compared with `>=`. Three deviations at
    once — wrong minute, wrong estimator, wrong comparison — and measured on real
    bars they disagreed on 3.4-8.2% of windows. Its docstring justified the
    `open` because "the strike was read the same way", which stopped being true
    when the averaged target landed and left the comment arguing for the bug.

    The venue's own settlement wins wherever it is available. Ours is an OHLC
    mean of Coinbase standing in for sixty seconds of CF Benchmarks BRTI, which
    is a close proxy and not the same number; every disagreement is logged
    because a persistent one is the basis risk becoming measurable.
    """
    now = datetime.now(timezone.utc)
    venue_settlements = venue_settlements or {}
    settled: list[tuple[int, float]] = []
    for position in writer.positions_due(now):
        strike = _strike_for(writer, position)
        if strike is None:
            logger.error('%s window %s: no strike recorded, cannot settle',
                         position.symbol, position.window_open)
            continue

        # The venue first, where it knows.
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        from_venue = None
        if ticker and ticker in venue_settlements:
            from_venue = venue_settled_up(venue_settlements[ticker], position.side)

        # Ours: the mean over the minute ENDING at settle_time, and `>=`.
        settle_minute = (pd.Timestamp(position.settle_time).tz_convert('UTC')
                         - pd.Timedelta(minutes=1))
        frame = bars.get(position.symbol)
        from_bars, settle_price = None, float('nan')
        if frame is not None:
            row = frame.loc[frame['event_time'] == settle_minute]
            if not row.empty:
                settle_price = float(bar_mean(row).iloc[0])
                if np.isfinite(settle_price):
                    from_bars = settle_price >= strike

        if from_venue is not None and from_bars is not None and from_venue != from_bars:
            logger.warning(
                '%s window %s: the venue settled %s and our bars say %s '
                '(mean %.4f vs strike %.4f). Taking the venue. A persistent '
                'disagreement here is the Coinbase-vs-BRTI basis, measured.',
                position.symbol, position.window_open,
                'up' if from_venue else 'down', 'up' if from_bars else 'down',
                settle_price, strike)

        settled_up = from_venue if from_venue is not None else from_bars
        if settled_up is None:
            logger.warning(
                '%s window %s: neither the venue nor a bar at %s can settle this '
                'yet, waiting', position.symbol, position.window_open,
                settle_minute)
            continue

        pnl = writer.settle_position(position.id, settled_up=bool(settled_up))
        if pnl is not None:
            settled.append((position.id, pnl))
            logger.info('settled %s %s: %s (%s) vs strike %.4f -> %+.2f',
                        position.symbol, position.window_open,
                        'up' if settled_up else 'down',
                        'venue' if from_venue is not None else f'mean {settle_price:.4f}',
                        strike, pnl)
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


async def reconcile_with_venue(writer: PgWriter, kalshi: KalshiClient) -> dict[str, dict]:
    """Make the venue's ledger the one we report.

    Three comparisons, each of which has a specific failure it catches:

    * **balance** — our running figure against theirs. A gap that grows is an
      unrecorded fill, a partial, or a fee we mispriced. Logged rather than
      silently overwritten on the first cycle, then written, because the venue is
      right either way and a silent overwrite hides how wrong we were.
    * **settlements** — what a position actually paid. Ours are settled from an
      OHLC mean of Coinbase standing in for CF Benchmarks BRTI, which will
      sometimes disagree.
    * **open positions** — a position we think is open and the venue does not is
      an order that never filled.
    """
    state = await kalshi.reconcile()
    venue_balance = float(state['balance'])
    account = writer.account()
    if account is not None:
        ours = float(account.bankroll)
        drift = venue_balance - ours
        if abs(drift) > 0.01:
            logger.warning(
                'balance drift: ours $%.2f, venue $%.2f (%+.2f). The venue is the '
                'account of record — writing theirs. A drift that grows means a '
                'fill we did not record, a partial, or a mispriced fee.',
                ours, venue_balance, drift)
        writer.update_account(bankroll=venue_balance)

    # Settle from the venue where it knows, keyed on the market ticker we stored.
    # This dict used to be built, logged, and dropped on the floor — `revenue` was
    # assigned and never read (ruff F841), `resolved` was used only as a
    # membership set for the warning below, and `settle_position` was called from
    # exactly one place: `settle_due`, off our own bars. So the documented
    # "settlement from /portfolio/settlements where it knows" did not exist. It is
    # returned now, and `run_cycle` hands it to `settle_due`.
    resolved: dict[str, dict] = {}
    for row in state.get('settlements', []):
        ticker = str(row.get('ticker', ''))
        if not ticker:
            continue
        resolved[ticker] = row
    if resolved:
        logger.info('venue reports %d settlement(s) to reconcile', len(resolved))

    venue_open = {str(p.get('ticker', '')) for p in state.get('positions', [])
                  if int(p.get('position') or 0) != 0}
    for position in writer.open_positions():
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        if ticker and ticker not in venue_open and ticker not in resolved:
            logger.warning(
                '%s window %s: we hold %d contracts the venue does not report. '
                'Most likely the order never filled — a fill_or_kill that killed '
                'leaves a ticket and no position.',
                position.symbol, position.window_open, position.contracts)

    # The reverse direction, which was never checked: a position the venue holds
    # and we do not. That is what an order POST that timed out after the venue
    # accepted it leaves behind, and it is the one discrepancy that costs money
    # silently.
    ours = set()
    for position in writer.open_positions():
        ticket = _ticket_for(writer, position)
        ticker = getattr(ticket, 'market_ticker', None) if ticket else None
        if ticker:
            ours.add(ticker)
    for ticker in sorted(venue_open - ours):
        logger.error(
            'the venue reports an open position in %s that we have no record of. '
            'An order was filled and not booked — most likely a POST that timed '
            'out after being accepted. Reconcile by hand before trading again.',
            ticker)
    return resolved


def _ticket_for(writer: PgWriter, position):
    from core.pg_writer import OrderTicket

    with writer._session() as session:  # noqa: SLF001 - same package, one query
        return (session.query(OrderTicket)
                .filter(OrderTicket.symbol == position.symbol,
                        OrderTicket.window_open == position.window_open)
                .one_or_none())


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

    # The venue first, where there is one: it knows what actually settled and
    # what the balance actually is. Bars only fill in what it has not resolved.
    venue_settlements: dict[str, dict] = {}
    if kalshi is not None and args.reconcile:
        try:
            venue_settlements = await reconcile_with_venue(writer, kalshi)
        except KalshiError as exc:
            logger.error('reconciliation failed (%s); falling back to our own '
                         'bookkeeping for this cycle', exc)
    settle_due(writer, bars, venue_settlements=venue_settlements)

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

    # The venue publishes the number it will settle against, as `floor_strike`,
    # the moment the window opens. Prefer it over the one built from bars: ours is
    # a one-minute OHLC mean standing in for sixty seconds of CF Benchmarks BRTI,
    # and the difference is a basis we can simply not take when the real figure is
    # on the wire. The bar-derived strike stays for the backtest, which has no
    # market to ask.
    for index, row in scored.iterrows():
        quote = quotes.get(row['symbol'], (None, None))[0]
        if quote is None or quote.floor_strike is None:
            continue
        venue_strike = float(quote.floor_strike)
        ours = float(row['strike'])
        drift_bps = abs(venue_strike / ours - 1) * 10_000
        if drift_bps > 25:
            logger.warning(
                '%s: our strike %.2f differs from the venue\'s %.2f by %.1fbp. '
                'Ours is an OHLC mean of Coinbase bars; theirs is BRTI. Using '
                'theirs, but a gap this wide suggests a stale bar feed.',
                row['symbol'], ours, venue_strike, drift_bps)
        scored.loc[index, 'strike_source'] = 'venue'
        scored.loc[index, 'strike'] = venue_strike
        scored.loc[index, 'displacement'] = float(row['last_price']) / venue_strike - 1.0
        if np.isfinite(row.get('sigma_remaining', np.nan)) and row['sigma_remaining'] > 0:
            scored.loc[index, 'z_score'] = (
                scored.loc[index, 'displacement'] / row['sigma_remaining'])

    if 'strike_source' not in scored.columns:
        scored['strike_source'] = 'bars'
    scored['strike_source'] = scored['strike_source'].fillna('bars')

    # The displacement moved, so the barrier probability has to be recomputed
    # from it rather than carried over from the bar-derived strike.
    if (scored['strike_source'] == 'venue').any():
        from core.baseline import attach_baseline
        scored = attach_baseline(scored.drop(
            columns=['baseline_probability', 'baseline_probability_logit'],
            errors='ignore'), model.scoring.baseline)
        scored['model_probability'] = model.predict(scored)

    # Depth at the touch caps the stake. Measured, unlike
    # `Config.max_stake_dollars`, which is a standing guess — so when the book
    # tells us, believe the book.
    for index, row in scored.iterrows():
        quote = quotes.get(row['symbol'], (None, None))[0]
        if quote is None:
            continue
        for side in ('up', 'down'):
            depth = quote.depth_dollars(side)
            if depth is not None:
                scored.loc[index, f'depth_{side}'] = depth

    account = writer.account()
    if account is None:
        raise RuntimeError('no account row; main() must call ensure_account first')

    # Seed exposure from what is already committed for THIS window, so
    # `ALREADY_ENTERED` / `POSITION_LIMIT` / `WINDOW_EXPOSURE` survive a new
    # cycle, a new offset and a process restart.
    entered, staked, n_entered = writer.entries_for_window(
        window_open.to_pydatetime() if hasattr(window_open, 'to_pydatetime') else window_open)
    exposure = WindowExposure(stake=staked, positions=n_entered, symbols_entered=entered)
    if entered:
        logger.info('window %s already holds %s ($%.2f); those symbols will refuse',
                    window_open, sorted(entered), staked)
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
        # Only count exposure we actually took on. `act_on` returns False when
        # the order was refused, killed, or never sent.
        if await act_on(args, writer, kalshi, decision, row):
            exposure = exposure.with_(decision)

    # Re-read: `account` was loaded before the decisions and is now stale by
    # every stake debited this cycle.
    account = writer.account()
    open_now = writer.open_positions()
    writer.write_equity_point(
        timestamp=now,
        equity=(account.bankroll + sum(p.outlay for p in open_now)) if account else 0.0,
        bankroll=account.bankroll if account else 0.0,
        staked=sum(p.outlay for p in open_now),
        open_positions=len(open_now),
        realized_pnl=account.realized_pnl if account else 0.0,
    )
    return decisions


def order_limit_price(decision: Decision) -> float:
    """The worst price still worth paying for this decision.

    Was `price + edge` — the break-even price. Under `fill_or_kill` that lets a
    thin book walk the whole order to a zero-EV fill and still call it a trade;
    measured, it sent 0.7832 against a 0.60 ask, 18c of tolerance on a 1c
    measured spread. Pay away a bounded fraction of the edge instead, capped in
    cents, so a fill always keeps most of what the forecast claimed.
    """
    config = DEFAULT_CONFIG
    edge = decision.edge if np.isfinite(decision.edge) else 0.0
    allowance = min(max(0.0, edge) * config.slippage_share_of_edge,
                    config.max_slippage_cents / 100.0)
    return float(min(0.99, decision.price + allowance))


def filled_from_order(order: dict, requested: int) -> tuple[int, float]:
    """Contracts actually filled, and at what price, from the venue's reply.

    This used to be `int(order.get('count', decision.contracts))`. `count` is the
    size *requested*, and `status`, `remaining_count` and `taker_fill_count` were
    never read — so a killed `fill_or_kill`, a partial fill, and an HTTP 200 with
    an empty body all recorded a full fill and debited the bankroll for contracts
    nobody held. The documented claim that "a fill_or_kill that killed leaves a
    ticket and no position" was false.

    Returns `(0, nan)` for anything that is not a confirmed fill. Assuming a fill
    is the one error here that cannot be reconciled later: the position is
    invented, `settle_due` settles it, and the PnL is fiction.
    """
    status = str(order.get('status', '') or '').strip().lower()
    if status in ('canceled', 'cancelled', 'killed', 'rejected', 'expired'):
        return 0, float('nan')

    filled = None
    for key in ('taker_fill_count', 'filled_count', 'fill_count'):
        if order.get(key) is not None:
            try:
                filled = int(order[key])
                break
            except (TypeError, ValueError):
                return 0, float('nan')
    if filled is None and order.get('remaining_count') is not None:
        try:
            filled = requested - int(order['remaining_count'])
        except (TypeError, ValueError):
            return 0, float('nan')
    if filled is None:
        # No fill field at all. Only 'executed'/'filled' justifies believing the
        # whole order traded; anything else (including an empty body) is unknown,
        # and unknown must not become a position.
        if status in ('executed', 'filled'):
            filled = requested
        else:
            return 0, float('nan')

    filled = max(0, min(int(filled), requested))
    price = float('nan')
    for key in ('average_fill_price_dollars', 'avg_price_dollars',
                'average_fill_price', 'yes_price', 'no_price'):
        raw = order.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        # Integer-cent fields on the same keys as dollar ones; the venue serves
        # both shapes. Above 1.0 it can only be cents.
        price = value / 100.0 if value > 1.0 else value
        break
    return filled, price


async def act_on(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                 decision: Decision, row) -> bool:
    """Record the ticket, place the order when asked to twice, book the fill.

    Returns whether a position was booked, so the caller only counts exposure it
    actually took on.

    **A position is written only when money actually moved.** Previously control
    fell through to `open_position` in every branch: an unresolved market (no
    ticker) wrote a position and debited the bankroll having sent no order, and so
    did `--mode live --dry-run`. Both produced holdings the venue had never heard
    of, which `settle_due` then settled into invented PnL — the exact failure the
    `price_source` column exists to make visible.
    """
    placing = bool(args.place_orders) and kalshi is not None and not args.dry_run

    if placing and not decision.market_ticker:
        logger.error(
            '%s window %s: no market resolved, so nothing can be bought. '
            'Abstaining rather than booking a position against our own baseline.',
            decision.symbol, decision.window_open)
        return False

    ticket_id = writer.write_ticket(
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        market_ticker=decision.market_ticker, side=decision.side.value,
        contracts=decision.contracts, limit_price=decision.price,
        # The worst price still worth paying. Capped at a fraction of the edge:
        # paying the whole edge away leaves a zero-EV fill, and under
        # fill_or_kill that is what walking the book to break-even buys.
        max_price=order_limit_price(decision),
        expected_cost=decision.stake, model_probability=decision.model_probability,
        edge=decision.edge,
    )

    filled = decision.contracts
    placed_price = decision.price
    if placing:
        try:
            order = await kalshi.place_order(
                ticker=decision.market_ticker, side=decision.side.value,
                contracts=decision.contracts,
                limit_price=order_limit_price(decision),
                client_order_id=f'{decision.symbol}-{decision.window_open:%Y%m%d%H%M}',
            )
        except (KalshiError, ValueError) as exc:
            writer.resolve_ticket(ticket_id, status='skipped', note=str(exc)[:400])
            logger.error('order refused, no position recorded: %s', exc)
            return False
        except (asyncio.TimeoutError, OSError) as exc:
            # The request may well have reached the venue. Do NOT book a position
            # and do NOT retry: `client_order_id` is deterministic per
            # (symbol, window), so the next cycle's attempt is the venue's problem
            # to deduplicate, and reconciliation will surface a fill we never saw.
            writer.resolve_ticket(ticket_id, status='unknown', note=str(exc)[:400])
            logger.error(
                'the order request to %s failed in flight (%s). It may have been '
                'accepted. Not booking a position and not retrying; the next '
                'reconcile will report a venue position we do not hold.',
                decision.market_ticker, exc)
            return False

        filled, fill_price = filled_from_order(order, decision.contracts)
        if filled <= 0:
            writer.resolve_ticket(
                ticket_id, status='killed',
                note=f"status={order.get('status')!r} order_id={str(order.get('order_id',''))[:60]}")
            logger.warning(
                '%s window %s: the order did not fill (status %r). No position.',
                decision.symbol, decision.window_open, order.get('status'))
            return False
        if np.isfinite(fill_price):
            placed_price = fill_price
        writer.resolve_ticket(
            ticket_id, status='filled', filled_contracts=filled,
            filled_price=placed_price,
            note=str(order.get('order_id', ''))[:200])
        if filled < decision.contracts:
            logger.warning('%s window %s: partial fill %d of %d',
                           decision.symbol, decision.window_open,
                           filled, decision.contracts)
        logger.info('filled %d @ %.4f (order %s)', filled, placed_price,
                    order.get('order_id'))
    elif args.mode == 'live':
        # A real book was read and priced; nothing was bought. Recording a
        # position here would put a holding on the books that does not exist.
        logger.info('dry run: ticket %d written, no order placed, no position',
                    ticket_id)
        return False

    outlay = decision.stake * (filled / decision.contracts) if decision.contracts else 0.0
    fee = decision.fee * (filled / decision.contracts) if decision.contracts else 0.0
    writer.open_position(
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        side=decision.side.value, contracts=filled, price=placed_price,
        outlay=outlay, fee=fee,
        model_probability=decision.model_probability,
        baseline_probability=decision.baseline_probability, edge=decision.edge,
    )
    # Relative, in one statement: a read-then-write across two transactions loses
    # one of two overlapping debits, and nothing enforces a single writer.
    writer.adjust_account(bankroll_delta=-outlay, fees_delta=fee)
    return True


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
    try:
        account = writer.ensure_account(config.starting_bankroll, mode=args.mode)
    except AccountModeMismatch as exc:
        raise SystemExit(str(exc))
    logger.info('account #%s mode=%s bankroll $%.2f realized $%+.2f',
                account.id, account.mode, account.bankroll, account.realized_pnl)

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
        with writer.exclusive_trader_lock():
            while True:
                try:
                    decisions = await run_cycle(args, config, writer, model, kalshi)
                except DatasetError as exc:
                    # One unscoreable cycle is not a reason to exit. This used to
                    # be fatal: `score_live` raised on every cycle and the loop
                    # caught only KeyboardInterrupt, so the process died and
                    # `restart: unless-stopped` crash-looped it forever.
                    logger.error('cycle skipped, nothing scored: %s', exc)
                    decisions = []
                if decisions:
                    counts = rejection_histogram(decisions)
                    logger.info('cycle: %s', counts[counts > 0].to_dict())
                if not args.loop:
                    return 0
                await asyncio.sleep(args.cycle_seconds)
    except TraderAlreadyRunning as exc:
        raise SystemExit(str(exc))
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
    # `installed` and `passed` are different questions. `--force` installs an
    # artifact whose gates FAILED and records installed=True, so testing
    # `installed` alone let a gate-failing model trade silently — and
    # `POST /jobs/scripts.promote {"args": ["--force", "--reason", "x"]}` reaches
    # that from a single HTTP request. Test the gates.
    if not bool(latest.get('installed')):
        raise SystemExit(
            f"the newest attempt {latest.get('version')} was blocked on "
            f"{latest.get('failed_gates')}. Trading it needs --no-require-gates "
            f"and --force with a written reason.")
    if 'passed' in frame.columns and not bool(latest.get('passed')):
        raise SystemExit(
            f"the newest attempt {latest.get('version')} was force-installed "
            f"with failing gates ({latest.get('failed_gates')}). It is on disk, "
            f"but --require-gates means what it says. Trading it needs "
            f"--no-require-gates and --force with a written reason.")


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
