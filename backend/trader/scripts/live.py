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
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG, SERIES_BY_SYMBOL, find_fee_config
from core.costs import trade_fee
from core.dataset import score_live
from core.decide import (
    Decision, Reason, Side, WindowExposure, decide, rejection_histogram,
)
from core.dataset import DatasetError
from core.pg_writer import AccountModeMismatch, PgWriter, TraderAlreadyRunning
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live
from core import venue_ledger
from core.windows import bar_mean
from data_collection.coinbase_connector import CoinbaseRESTClient
from data_collection.kalshi_client import (
    KalshiClient, KalshiError, Quote, parse_fill, parse_settlement,
)

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
# SERIES_BY_SYMBOL now lives in core/config.py — this used to be the only
# place the KALSHI_SERIES_* env vars reached; see core/config.py for why
# that was a trap once other scripts read a hardcoded copy of the same map.

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
    parser.add_argument('--entry-offsets', type=int, nargs='+', default=[12],
                        metavar='M',
                        help='Which decision offsets may OPEN a position. Every '
                             'configured offset is still scored and recorded; this '
                             'restricts only entries. Default 12: measured over 70 '
                             'days, taking the earliest offset that cleared the '
                             'gate returned 0.040c per contract (t=0.10) against '
                             '3.304c at +12m alone (t=5.98), and in production 90%% '
                             'of entries landed at +3m. Pass "9 12" for the '
                             'conservative widening, or every offset to restore the '
                             'old behaviour.')
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
    parser.add_argument('--clear-halt', action='store_true',
                        help='Clear a circuit-breaker halt and exit, placing no '
                             'orders. Requires --reason, because a breaker cleared '
                             'without one recorded is a breaker nobody learns from.')
    parser.add_argument('--max-daily-loss-fraction', type=float, default=None,
                        metavar='F',
                        help='Override the daily-loss breaker, as a fraction of '
                             'the STARTING bankroll (default 0.15). A flag rather '
                             'than an edit to the default, so what the loop is '
                             'actually running with is visible in the deploy '
                             'config and in `ps`, and reverting is deleting a '
                             'line. 1.0 or more effectively disables it, leaving '
                             'only the ruin floor.')
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


# The rolling window, kept between cycles. See `fetch_bars`.
_BAR_CACHE: dict[str, pd.DataFrame] = {}

# How far back to re-ask on an incremental fetch. The newest cached bar is the
# minute still forming, so it is refetched to get its final values; the extra
# minutes cover a bar the venue revised or filled in late.
BAR_REFETCH_MINUTES = 3


def _merge_bars(cached: pd.DataFrame, fresh: pd.DataFrame,
                *, floor: pd.Timestamp) -> pd.DataFrame:
    """Cached bars plus fresh ones, the fresh copy winning on any overlap.

    `keep='last'` matters: the overlap is deliberately the minutes that were
    still forming when they were last read, and the newer read is the finished
    one. Keeping the cached copy would freeze a partial candle into the window
    forever, and every `log_rv_*` feature is built off these closes.
    """
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.sort_values('event_time', ignore_index=True)
    combined = combined.drop_duplicates(subset='event_time', keep='last')
    return combined[combined['event_time'] >= floor].reset_index(drop=True)


async def fetch_bars(config: Config, minutes: int = FETCH_MINUTES) -> dict[str, pd.DataFrame]:
    """One-minute bars for the universe, straight from the venue.

    **Only the tail is fetched.** `log_rv_1440` and `beta_1440` need a day of
    history, so the window is 1,500 minutes — but the cycle runs every 60
    seconds, and re-downloading twenty-five hours to learn one new minute was
    measured at 3.2s, the largest single cost in the cycle. `get_candles_range`
    pages at 300 candles, so a full window is ~6 sequential round trips per
    symbol, which is why fetching the three symbols concurrently barely helped.

    The window is kept between cycles and only the newest few minutes are
    re-asked. A symbol with no usable cache — first cycle, or a gap wider than
    the window after an outage — falls back to the full fetch, so this is an
    optimisation and never a source of missing history.
    """
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(minutes=minutes)
    floor = pd.Timestamp(start, tz='UTC')
    client = CoinbaseRESTClient(
        api_key=os.getenv('COINBASE_API_KEY'),
        api_secret=os.getenv('COINBASE_API_SECRET'),
    )
    out: dict[str, pd.DataFrame] = {}
    try:
        # **Concurrently, not one symbol after another.** Measured at 3.17s for
        # three symbols — three independent round trips to the same host, run in
        # series for no reason. This is the largest single cost in the cycle, and
        # while reordering already moved it off the book-to-order path, it still
        # sets how stale the bars are at decision time and how long the whole
        # cycle takes.
        #
        # `return_exceptions` so one symbol's failure costs only that symbol.
        # Losing all three because SOL timed out would turn a partial outage into
        # a total one, and the loop already handles a missing symbol.
        spans: dict[str, datetime] = {}
        for symbol in config.symbols:
            cached = _BAR_CACHE.get(symbol)
            spans[symbol] = start
            if cached is None or cached.empty:
                continue
            last = pd.Timestamp(cached['event_time'].iloc[-1])
            gap_start = (last.tz_convert(None).to_pydatetime()
                         - timedelta(minutes=BAR_REFETCH_MINUTES))
            # Only incremental if the cache still reaches back far enough to
            # cover the window on its own; otherwise the tail would leave a hole
            # in the middle, which no feature would notice and every `rv` would
            # be computed across.
            if gap_start > start and pd.Timestamp(cached['event_time'].iloc[0]) <= floor:
                spans[symbol] = gap_start

        results = await asyncio.gather(
            *(client.get_candles_range(symbol, '1m', spans[symbol], end)
              for symbol in config.symbols),
            return_exceptions=True)
        for symbol, bars in zip(config.symbols, results):
            if isinstance(bars, BaseException):
                logger.error('%s: the venue refused one-minute bars (%s)',
                             symbol, str(bars)[:120])
                continue
            if not bars:
                logger.error('%s: the venue returned no one-minute bars', symbol)
                continue
            frame = pd.DataFrame([{
                'event_time': pd.Timestamp(b.event_time, tz='UTC'),
                'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
                'volume': b.volume, 'quote_volume': getattr(b, 'quote_volume', np.nan),
                'trade_count': getattr(b, 'trade_count', np.nan),
            } for b in bars]).sort_values('event_time', ignore_index=True)
            cached = _BAR_CACHE.get(symbol)
            if cached is not None and spans[symbol] > start:
                frame = _merge_bars(cached, frame, floor=floor)
            else:
                frame = frame[frame['event_time'] >= floor].reset_index(drop=True)
            _BAR_CACHE[symbol] = frame
            out[symbol] = frame
            logger.debug('%s: %d bars to %s (%s)', symbol, len(frame),
                         frame['event_time'].iloc[-1],
                         'incremental' if spans[symbol] > start else 'full')
    finally:
        close = getattr(client, 'close', None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
    return out


def check_circuit_breakers(writer: PgWriter, config: Config,
                           *, now: datetime) -> Optional[str]:
    """Halt the account on a bad day or a bad streak. Returns the reason, or None.

    `Account.halted` and `halted_reason` existed as columns, were rendered on the
    dashboard as a safety chip, and **were never written by anything** — the only
    code that set them was `core/book.py`, the backtest's in-memory account,
    which `scripts/live.py` does not import. So the indicator was structurally
    incapable of turning on, and the `halted` promotion gate read the simulated
    account rather than the real one.

    The ruin floor in `decide` was the only live limit, and it fires after half
    the account is gone. At 96 windows a day across three symbols that is a long
    way to bleed: the nominal worst case was $768/day against a $100 bankroll.

    A halt is sticky. Clearing it is a manual decision, because a breaker that
    resets itself at midnight is a speed bump.
    """
    account = writer.account()
    if account is None:
        return None
    if account.halted:
        return str(account.halted_reason or 'halted')

    def halt(reason: str) -> str:
        writer.update_account(halted=True, halted_reason=reason[:400])
        logger.error('HALTED: %s. No further entries until this is cleared by '
                     'hand.', reason)
        return reason

    if not np.isfinite(float(account.bankroll)):
        return halt('the bankroll is not a finite number, so nothing about the '
                    'account can be trusted')

    since = pd.Timestamp(now).tz_convert('UTC').normalize().to_pydatetime()
    today = writer.settled_positions_since(since)
    if today:
        realised = sum(float(p.pnl or 0.0) for p in today)
        limit = -abs(config.max_daily_loss_fraction) * config.starting_bankroll
        if realised <= limit:
            return halt(f'today is {realised:+.2f} against a limit of {limit:.2f} '
                        f'({config.max_daily_loss_fraction:.0%} of the starting '
                        f'bankroll) over {len(today)} settlements')

    # Peak-to-current drawdown on realised equity.
    #
    # **The daily rule cannot see this shape, and it is the shape that happened.**
    # Equity ran $100 -> $166.86 by 13:00 UTC on the second day and gave back
    # $63.92 over the next ten hours — all inside one UTC day, so that day's
    # realised was +$3.81 against a -$15.00 limit. The daily rule saw a good day
    # while the account sat 38.3% below its high. `max_drawdown <= 0.35` was
    # already a promotion gate on the *backtest*; a threshold worth enforcing on a
    # simulation is worth enforcing on the money.
    #
    # Realised basis on both sides: `realised_high_water()` is the running maximum
    # of the same series `account.realized_pnl` currently holds, derived from the
    # settlements rather than stored, so there is no second source of truth to
    # drift.
    peak_realised = writer.realised_high_water()
    start = float(config.starting_bankroll)
    peak_equity = start + max(peak_realised, 0.0)
    current_equity = start + float(account.realized_pnl or 0.0)
    if peak_equity > 0:
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= config.max_drawdown_fraction:
            # **Only a halt once it is also real capital loss.** The first
            # version halted on the drawdown alone, and immediately stopped an
            # account sitting at $107.94 on a $100 start — up 8%, having never
            # cost anything, because it had once been up 67%. That is a fund's
            # rule, for protecting banked gains. This account exists to find out
            # whether an edge is real, and what it has to protect is the stake.
            #
            # The stake is already guarded twice: `ruin_floor_fraction` refuses
            # every trade below 50% of the starting bankroll, and the daily rule
            # bounds one day's loss at 15% of it. So above water this is a signal
            # rather than a stop, and it says so in the log.
            if current_equity < start:
                return halt(
                    f'drawdown {drawdown:.1%} from a peak of ${peak_equity:.2f} '
                    f'to ${current_equity:.2f}, at or over the '
                    f'{config.max_drawdown_fraction:.0%} limit, and below the '
                    f'${start:.2f} starting bankroll')
            logger.warning(
                'drawdown %.1f%% from a peak of $%.2f to $%.2f — over the %.0f%% '
                'limit but still above the $%.2f starting bankroll, so this is a '
                'signal and not a halt. The stake is guarded by the ruin floor '
                '($%.2f) and the daily loss limit.',
                100 * drawdown, peak_equity, current_equity,
                100 * config.max_drawdown_fraction, start,
                start * config.ruin_floor_fraction)

    recent = writer.settled_positions_since(
        (pd.Timestamp(now).tz_convert('UTC') - pd.Timedelta(days=7)).to_pydatetime())
    streak = 0
    for position in sorted(recent, key=lambda p: p.settled_at or since, reverse=True):
        if float(position.pnl or 0.0) < 0:
            streak += 1
        else:
            break
    if streak >= config.max_consecutive_losses:
        return halt(f'{streak} consecutive losing settlements, at or over the '
                    f'limit of {config.max_consecutive_losses}')
    return None


def stale_symbols(bars: dict[str, pd.DataFrame], config: Config,
                  *, now: datetime) -> dict[str, str]:
    """Symbols whose feed is too old, or absent, with the reason.

    There was no staleness guard anywhere. The last `event_time` was logged
    (`fetch_bars`) and never compared to the wall clock or to the decision
    minute, so a delayed or partial fetch was scored as though it were current.
    Two distinct hazards:

    * **An old feed.** The forward-fill inside `build_windows` is right for a
      minute in which nothing traded and cannot tell that from a minute not yet
      fetched, so a stale feed produces a fabricated `last_price` and the barrier
      treats it as a measurement.
    * **A missing symbol.** `fetch_bars` logged and continued, which silently
      redefines the *other* symbols' `cross_asset` features — measured,
      `beta_1440` moved 7.7x with no error and no NaN. The universe is part of
      the feature definition, so a short universe is a different model.
    """
    reasons: dict[str, str] = {}
    cutoff = pd.Timestamp(now).tz_convert('UTC') - pd.Timedelta(
        seconds=int(config.max_bar_age_seconds))
    for symbol in config.symbols:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            reasons[symbol] = 'no bars returned'
            continue
        newest = pd.Timestamp(frame['event_time'].iloc[-1])
        if newest < cutoff:
            age = (pd.Timestamp(now).tz_convert('UTC') - newest).total_seconds()
            reasons[symbol] = (f'newest bar {newest} is {age:.0f}s old, over the '
                               f'{config.max_bar_age_seconds}s limit')
    return reasons


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


def as_utc(value) -> pd.Timestamp:
    """A UTC timestamp, whether the source remembered the timezone or not.

    The ORM columns are `DateTime(timezone=True)`, so Postgres hands back
    tz-aware values and SQLite hands back naive ones — it has no timezone type.
    `pd.Timestamp(x).tz_convert('UTC')` raises on the naive case, so any code
    path that reads a timestamp back out of the store and assumes awareness works
    against Postgres and raises against SQLite. Everything stored is UTC, so
    localise when the tz is missing rather than guessing.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tz is None else stamp.tz_convert('UTC')


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


def resolve_window(strike: float, settle_time, symbol: str,
                   bars: dict[str, pd.DataFrame]) -> tuple[Optional[bool], float]:
    """Did this window settle up, per the trained rule? Plus the value used.

    One implementation, because `settle_due` and `settle_predictions` must agree —
    the whole reason the live settlement drifted from the training label in the
    first place was a second copy of this arithmetic.

    The rule is `core/windows.py`'s: the `(O+H+L+C)/4` mean of the minute *ending*
    at `settle_time`, compared with `>=` because the venue's `strike_type` is
    `greater_or_equal`. Returns `(None, nan)` when the bar is not there yet.
    """
    frame = bars.get(symbol)
    if frame is None:
        return None, float('nan')
    minute = as_utc(settle_time) - pd.Timedelta(minutes=1)
    row = frame.loc[frame['event_time'] == minute]
    if row.empty:
        return None, float('nan')
    value = float(bar_mean(row).iloc[0])
    if not np.isfinite(value):
        return None, float('nan')
    return value >= strike, value


def settle_predictions(writer: PgWriter, bars: dict[str, pd.DataFrame],
                       *, now: Optional[datetime] = None) -> int:
    """Fill in the realised outcome on every settled window, traded or not.

    Without this the market benchmark cannot exist. `market_probability` is
    recorded on each decision — the venue's own mid — and scoring it needs the
    answer beside it. `settle_due` only touches windows we *hold*, which is the
    ~6% we chose to trade: exactly the selected sample that cannot answer "is the
    market's probability better than ours".
    """
    # Injectable so a test can place a window in the past without waiting for the
    # wall clock to agree.
    now = now or datetime.now(timezone.utc)
    filled = 0
    for symbol, window_open, settle_time, strike in writer.windows_awaiting_outcome(now):
        settled_up, _ = resolve_window(strike, settle_time, symbol, bars)
        if settled_up is None:
            continue
        filled += writer.set_window_outcome(symbol, window_open, settled_up=settled_up)
    if filled:
        logger.info('recorded the outcome on %d prediction row(s)', filled)
    return filled


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
        settle_minute = as_utc(position.settle_time) - pd.Timedelta(minutes=1)
        from_bars, settle_price = resolve_window(
            strike, position.settle_time, position.symbol, bars)

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


def config_for_artifact(config, model, *, mode: str):
    """Adopt the artifact's init-score source, refusing the one mode that cannot
    supply it.

    The artifact records which forecaster its correction was fitted on top of.
    Live has no legitimate choice here — scoring a market-fitted residual on the
    baseline logit is exactly the silent failure `ForecastModel.verify` exists to
    catch — so the artifact wins and the mismatch becomes impossible rather than
    merely detected.

    The refusal is the other half. A market-init model needs the book to score:
    `fetch_quotes` returns {} when no Kalshi client was opened, which is every
    paper run. Adopting silently would swap a loud crash for a NaN prediction
    every cycle, which reads as a model that has decided to abstain.
    """
    # The economic policy the artifact was EVALUATED under. `scripts/live.py`
    # has no flag for any of these, so without this the loop traded whatever
    # Config defaulted to — 0.25 Kelly against the promoted 0.10, a 1.50pp gate
    # against 3.00pp. `verify` does not catch it because these change what the
    # strategy DOES, not what the model SAYS, and that is exactly why it went
    # unnoticed: the model was scored correctly and then acted on wrongly.
    #
    # CLAUDE.md measures the cost of getting Kelly wrong: 0.25 -> 0.10 moved
    # realised edge per contract +0.99pp -> +3.32pp and drawdown 58% -> 21%,
    # because a smaller fraction also floors marginal trades under one contract
    # and refuses them. It is an edge filter, not only a size.
    provenance = dict(getattr(model, 'config_provenance', None) or {})
    economics = {}
    for field in ('kelly_fraction', 'min_edge_pp', 'max_stake_dollars',
                  'max_stake_fraction', 'half_spread_cents', 'compound'):
        value = provenance.get(field)
        # None means the promoting run left it at ITS default, which is the
        # default here too. Adopting a None would erase a real setting.
        if value is not None and value != getattr(config, field, None):
            economics[field] = value
    if economics:
        logger.info('adopting the economics this artifact was measured under: %s',
                    ', '.join(f'{k}={v}' for k, v in sorted(economics.items())))
        config = replace(config, **economics)

    source = getattr(model, 'init_score_source', 'baseline')
    if source == getattr(config, 'init_score_source', 'baseline'):
        return config
    if source == 'market' and mode != 'live':
        raise SystemExit(
            f'this artifact was fitted on the market logit, and {mode} mode '
            f'opens no venue client, so there is no book to read a price from. '
            f'Every window would score NaN and look like an abstention. Run it '
            f'with --mode live (add --dry-run to place nothing).')
    return replace(config, init_score_source=source)


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
            logger.debug('%s %s: %.2f / %.2f (spread %.0fc, vol %d)',
                        symbol, ticker, quote.yes_bid, quote.yes_ask,
                        (quote.spread or 0) * 100, quote.volume)
        except KalshiError as exc:
            logger.error('%s: could not read the book (%s)', symbol, exc)
    return quotes


class VenueState(NamedTuple):
    """What the venue says, read before we settle anything ourselves.

    The balance travels with the settlements rather than being written on the
    spot, because the order of those two operations turned out to matter — see
    `adopt_venue_balance`.
    """

    settlements: dict[str, dict]
    balance: float


def adopt_venue_balance(writer: PgWriter, venue_balance: float, *,
                        exchange_index: Optional[int] = None) -> None:
    """Make the venue's balance the one we report. **Call this after settling.**

    Our running figure against theirs: a gap that grows is an unrecorded fill, a
    partial, or a fee we mispriced. Logged rather than silently overwritten,
    because the venue is right either way and a silent overwrite hides how wrong
    we were.

    **The ordering is the whole point, and getting it backwards cost the alarm its
    meaning.** This used to run at the top of the cycle, before `settle_due`. But
    the venue credits a settlement the moment it settles, so by the time we read
    the balance the payout is already in it — and then `settle_due` credited the
    same payout again. Measured over the first live night, on real money:

        09:15:38  ours $147.03, venue $168.03 (+21.00)   <- venue credited
        09:16:42  ours $189.03, venue $168.03 (-21.00)   <- we credited it again
        09:19:56  ours $160.67, venue $161.07 ( +0.41)   <- reconciled back

    The bankroll self-healed on the next cycle, so no money moved. Two things did
    go wrong. Kelly sized off a bankroll inflated by the payout for up to a full
    cycle; and the drift log filled with benign +/-$21 pairs, which is exactly the
    noise a genuine unrecorded fill would hide in. An alarm that cries wolf every
    time a position settles is not an alarm.

    Settling first makes the drift mean what it says: a disagreement between our
    bookkeeping and the venue's, with the ordering artifact removed.
    """
    if not np.isfinite(venue_balance):
        # The venue is the account of record only when it actually answered.
        # Writing an unreadable balance over a correct one is worse than keeping
        # ours and saying so.
        logger.error('the venue did not return a readable balance; keeping our own '
                     'figure this cycle rather than overwriting it')
        return
    account = writer.account()
    if account is None:
        return
    ours = float(account.bankroll)
    drift = venue_balance - ours
    if abs(drift) > 0.01:
        logger.warning(
            'balance drift: ours $%.2f, venue $%.2f (%+.2f). The venue is the '
            'account of record — writing theirs. A drift that grows means a '
            'fill we did not record, a partial, or a mispriced fee.',
            ours, venue_balance, drift)
    # Sampled *before* the overwrite, so the row keeps both figures as they stood
    # at the same instant. Recording it afterwards would store our bankroll as
    # the venue's and report a drift of zero forever — a self-fulfilling alarm.
    # A single log line cannot show a trend, and the trend is the diagnosis:
    # a drift that stays put is a starting-balance mismatch, one that grows is an
    # unrecorded fill.
    try:
        writer.write_venue_balance(
            timestamp=datetime.now(timezone.utc), balance=venue_balance,
            exchange_index=exchange_index, our_bankroll=ours)
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not sample the venue balance this cycle')
    writer.update_account(bankroll=venue_balance)


# The last shard actually observed on a book. See `run_cycle`: the balance query
# needs it, and the book is now read after reconciliation, so it cannot come from
# this cycle's quotes.
_LAST_EXCHANGE_INDEX: Optional[int] = None


def remember_exchange_index(quotes: dict) -> Optional[int]:
    """Learn the shard from the book that was just read.

    **A change here is loud, not silent.** Balances are local to a shard, so a
    stale one reads the wrong balance — and that is exactly the failure that once
    had every order refused `insufficient_balance` while the funds sat elsewhere.
    A move is rare (the venue re-categorising a series), costs at most the one
    cycle that already reconciled, and is corrected before the next.
    """
    global _LAST_EXCHANGE_INDEX
    seen = venue_exchange_index(quotes)
    if seen is None:
        return _LAST_EXCHANGE_INDEX
    if _LAST_EXCHANGE_INDEX is not None and seen != _LAST_EXCHANGE_INDEX:
        logger.warning(
            'the venue moved these markets from exchange shard %s to %s. This '
            'cycle reconciled against the old one; balances are per-shard, so '
            'treat this cycle\'s balance as suspect.',
            _LAST_EXCHANGE_INDEX, seen)
    _LAST_EXCHANGE_INDEX = seen
    return seen


def venue_exchange_index(quotes: dict) -> Optional[int]:
    """Which exchange shard the traded markets live on, per the venue.

    Kalshi shards its exchange by category and **balances are local to a shard**,
    so the only balance an order can draw on is the one on the shard holding its
    market. On 2026-08-25 the KX*15M series moved to shard 2 while this account's
    funds sat on shard 0, and every order was refused `insufficient_balance`
    against an apparently healthy $107.96 total.

    Read off the quotes rather than configured, for the same reason markets are
    resolved by asking rather than by building a ticker: a constant is a guess
    that keeps working until the venue moves, and then it is silently wrong.
    Returns None when no book was read, which means the whole-account total.
    """
    # **`fetch_quotes` stores `(Quote, ticker)`, not a bare Quote**, and reading
    # `.exchange_index` straight off the tuple silently yielded nothing: this
    # returned None on every live cycle, so the balance query always fell back to
    # the whole-account total instead of the shard the crypto markets are on.
    # The unit test passed because it built `{symbol: Quote}` — a shape the live
    # path never produces. Accept both, and let the test assert the real one.
    candidates = [v[0] if isinstance(v, tuple) else v for v in quotes.values()]
    seen = {int(q.exchange_index) for q in candidates
            if getattr(q, 'exchange_index', None) is not None}
    if not seen:
        return None
    if len(seen) > 1:
        # Never observed. Narrowing to one shard would understate the balance
        # available to the others, so decline to narrow at all.
        logger.warning('markets span exchange shards %s; using the account total',
                       sorted(seen))
        return None
    return seen.pop()


def persist_venue_ledger(writer: PgWriter, *, fills: list[dict],
                         settlements: list[dict]) -> tuple[int, int]:
    """Store the venue's own fills and settlements. Idempotent on the venue's keys.

    These arrive on every reconcile and were read, compared, and dropped. That is
    why the dashboard's P&L was our arithmetic rather than the venue's: nothing
    kept the rows that would have made the venue's version computable. The account
    curve is drawn from what this writes.

    Free, in API terms — `reconcile` already fetched both, so this adds no request
    to the cycle. It covers only what the live tier returned, which for a loop
    running every minute is everything that has happened since it started;
    `scripts.sync_venue` does the deep paginated backfill across the historical
    tier.

    Exceptions are caught and logged rather than raised. A cycle that cannot write
    telemetry must still trade and settle — the store is a record, not the
    account — and the previous behaviour of an unhandled write killing the loop
    was the wrong trade-off in a process that holds positions.
    """
    written_fills = written_settlements = 0
    try:
        # Two queries, then only the rows that are actually new. The venue returns
        # the same ~200 fills and ~200 settlements every cycle, and upserting all
        # of them would be four hundred read-then-write round trips a minute to
        # change nothing. A settlement already stored with a null `pnl` is *not*
        # in the skip set, so an incomplete parse is retried and heals.
        known_fills, known_settlements = writer.venue_ledger_keys()
    except Exception:  # noqa: BLE001 - telemetry must not stop the loop
        logger.exception('could not read the stored venue ledger; writing all rows')
        known_fills, known_settlements = set(), set()

    try:
        rows = []
        for raw in fills:
            parsed = parse_fill(raw)
            if parsed.trade_id and parsed.trade_id not in known_fills:
                rows.append(venue_ledger.fill_row(parsed))
        written_fills = writer.upsert_venue_fills(rows)
    except Exception:  # noqa: BLE001
        logger.exception('could not store the venue fills this cycle')
    try:
        rows = []
        for raw in settlements:
            parsed = parse_settlement(raw)
            if not parsed.ticker or parsed.ticker in known_settlements:
                continue
            rows.append(venue_ledger.settlement_row(
                parsed, position=writer.position_for_ticker(parsed.ticker)))
        written_settlements = writer.upsert_venue_settlements(rows)
    except Exception:  # noqa: BLE001
        logger.exception('could not store the venue settlements this cycle')
    if written_fills or written_settlements:
        logger.debug('venue ledger: %d fill(s), %d settlement(s) stored',
                     written_fills, written_settlements)
    return written_fills, written_settlements


async def reconcile_with_venue(writer: PgWriter, kalshi: KalshiClient, *,
                               exchange_index: Optional[int] = None) -> VenueState:
    """Read the venue's ledger, and keep it.

    Three comparisons, each of which has a specific failure it catches:

    * **balance** — carried out in `VenueState` and written by
      `adopt_venue_balance` *after* settlement, not here.
    * **settlements** — what a position actually paid. Ours are settled from an
      OHLC mean of Coinbase standing in for CF Benchmarks BRTI, which will
      sometimes disagree.
    * **open positions** — a position we think is open and the venue does not is
      an order that never filled.

    It also **stores** the fills and settlements now, via `persist_venue_ledger`.
    It used to compare them and drop them, which is why the account page could
    only ever show our own arithmetic: the venue's numbers passed through this
    function every cycle and nothing kept them.
    """
    state = await kalshi.reconcile(exchange_index=exchange_index)
    venue_balance = float(state['balance'])
    persist_venue_ledger(writer, fills=list(state.get('fills') or []),
                         settlements=list(state.get('settlements') or []))

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
        logger.debug('venue reports %d settlement(s) to reconcile', len(resolved))

    # `KalshiClient.position_size` and not `p['position']`: the field is
    # `position_fp`, a fixed-point string, and reading the name from the older
    # documentation made `venue_open` the empty set on every cycle. That broke
    # this check in both directions at once — the forward one warned that every
    # real position "the venue does not report", and the reverse one, for a
    # position the venue holds and we do not, could never fire at all. The
    # reverse is the case the audit called the one that costs money silently.
    venue_open = {str(p.get('ticker', '')) for p in state.get('positions', [])
                  if KalshiClient.position_size(p) != 0}
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
    return VenueState(settlements=resolved, balance=venue_balance)


def _ticket_for(writer: PgWriter, position):
    from core.pg_writer import OrderTicket

    with writer._session() as session:  # noqa: SLF001 - same package, one query
        return (session.query(OrderTicket)
                .filter(OrderTicket.symbol == position.symbol,
                        OrderTicket.window_open == position.window_open)
                .one_or_none())


def _cumulative(levels, *, best: float, within: float, invert: bool) -> float:
    """Resting size at prices at least as good as `best +/- within`.

    This, not the touch, is what decides a `fill_or_kill`. Our limit sits a cent
    or two past the touch, so the order fills only if the cumulative size up to
    that limit covers it. Measured on SOL: the touch held 6 contracts against a
    7-contract order, while cumulative within 2c was ~19 — the touch alone says
    "no fill" where the ladder says "fills".
    """
    total = 0.0
    for entry in levels or []:
        try:
            price, size = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        effective = (1.0 - price) if invert else price
        if (effective <= best + within) if invert else (effective >= best - within):
            total += size
    return total


def ladder_from_cache(cache, ticker: str):
    """(yes_levels, no_levels) from the stream's in-process book, or empty.

    **This is why the full feature set costs no latency.** `run_live` supervises
    `stream` and `trade` as asyncio tasks in ONE process, so `record_stream.CACHE`
    already holds every subscribed ladder folded from the socket — no fetch, and
    fresher than one: a frame arrives ~34ms after the venue stamps it against
    ~73ms for a REST round trip, at 100% top-of-book agreement and zero ladder
    drift against REST snapshots.

    Refused in three cases, all of which mean the book in hand may not be the
    book on the venue:

      * no cache — `scripts.live` also runs standalone, without the stream task;
      * stale — ten seconds of silence at 400+ frames a second is a sick
        transport, not a quiet market, and pricing against it is worse than
        having no depth;
      * gapped — `seq` is global per SUBSCRIPTION, so a miss condemns every book
        on the connection, this one included.

    Empty levels give NaN depth features, which `_warn_unscoreable_features`
    reports. That is the honest answer, not a silent substitution.
    """
    if cache is None or not ticker:
        return [], []
    try:
        if cache.gapped(ticker):
            return [], []
        ladder = cache.ladder(ticker)
    except Exception:                                         # noqa: BLE001
        return [], []
    if ladder is None or getattr(ladder, 'stale', True):
        return [], []
    return list(ladder.yes or []), list(ladder.no or [])


def _stream_cache():
    """The stream task's cache, when this process is running one."""
    try:
        from scripts.record_stream import CACHE
        return CACHE
    except Exception:                                         # noqa: BLE001
        return None


def _venue_caches() -> tuple[dict, dict]:
    """The recorders' in-process caches, empty when they are not running.

    Imported lazily and defensively: `scripts.live` also runs standalone, where
    neither recorder exists and every feature they feed is honestly unknown.
    """
    try:
        from scripts.record_pm_ladder import CACHE as pm
    except Exception:                                         # noqa: BLE001
        pm = {}
    try:
        from scripts.record_implied_vol import CACHE as iv
    except Exception:                                         # noqa: BLE001
        iv = {}
    return pm, iv


# The gap at each decision offset of the window in progress, per symbol.
#
# The backtest computes `venue_gap_change_5` as `shift(1)` over rows ordered by
# OFFSET and grouped by (symbol, window_open) — the previous decision offset in
# the SAME window. Live has to reproduce that exactly or it is fitting one
# feature and scoring another.
_GAP_HISTORY: dict = {}


def reset_gap_history() -> None:
    """Drop everything remembered. For tests, and for a clean restart."""
    _GAP_HISTORY.clear()


def gap_change(symbol: str, gap, *, window_open, offset: int) -> float:
    """The change in the cross-venue gap since the previous decision offset.

    NaN at the first offset of a window, by construction and not by accident:
    consecutive windows CHAIN — a window's strike is the previous window's
    settlement value — so differencing across a boundary produces a number that
    looks entirely correct and answers a different question. The backtest
    refuses it with `groupby(['symbol', 'window_open'])`, and so does this.

    A missing gap is a hole, not an observation: it neither differences nor
    displaces the last real reading, so the following offset still measures
    against something true.
    """
    key = (str(symbol), pd.Timestamp(window_open))
    # Evict by WINDOW, never by arrival. Three symbols are scored in every
    # cycle of the same window, so clearing whenever an unseen key turns up
    # meant BTC's reading was destroyed by ETH's and ETH's by SOL's — every
    # symbol then reported NaN on every cycle, forever.
    if key not in _GAP_HISTORY:
        for stale in [k for k in _GAP_HISTORY if k[1] != key[1]]:
            del _GAP_HISTORY[stale]
        _GAP_HISTORY[key] = {}
    previous = _GAP_HISTORY[key]
    try:
        value = float(gap)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(value):
        return float('nan')
    earlier = [(o, g) for o, g in previous.items() if o < int(offset)]
    previous[int(offset)] = value
    if not earlier:
        return float('nan')
    return float(value - max(earlier, key=lambda item: item[0])[1])


def cross_venue_row(pm: Optional[dict], quote) -> dict:
    """`cross_venue` from the in-process Polymarket cache.

    Two independent books on the same fifteen minutes, settling on different
    oracles — CF Benchmarks BRTI against Chainlink's BTC-USD TWAP-60s — that
    nonetheless agree on 99.52% of shared windows. So a price gap is information
    or liquidity, not a different question.

    `best_bid`/`best_ask` are CENTS on both sides, matching `core.book_features`.
    """
    from core.book_features import CROSS_VENUE

    out = {name: float('nan') for name in CROSS_VENUE}
    k_mid = _two_sided_mid(getattr(quote, 'yes_bid', np.nan) * 100.0,
                           getattr(quote, 'yes_ask', np.nan) * 100.0)
    p_mid = _two_sided_mid(*(
        (pm.get('best_bid'), pm.get('best_ask')) if pm else (np.nan, np.nan)))
    # Absence is not agreement: a zeroed gap would read as two venues concurring.
    out['pm_available'] = 0.0 if np.isnan(p_mid) else 1.0
    out['venue_prob_gap'] = k_mid - p_mid
    k_spread = _spread_cents(getattr(quote, 'yes_bid', np.nan) * 100.0,
                             getattr(quote, 'yes_ask', np.nan) * 100.0)
    p_spread = _spread_cents(*(
        (pm.get('best_bid'), pm.get('best_ask')) if pm else (np.nan, np.nan)))
    if k_spread > 0 and p_spread > 0:
        out['venue_spread_ratio'] = float(np.log(k_spread / p_spread))
    # `venue_gap_change_5` needs the gap five minutes ago, which the backtest
    # takes from the previous row of the same window. Live carries it per
    # symbol below; absent a prior observation it is honestly unknown.
    return out


def _two_sided_mid(bid, ask) -> float:
    """The mid as a probability, or NaN. A lone bid says the probability is at
    LEAST something, which is not a probability."""
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(bid) or np.isnan(ask):
        return float('nan')
    return (bid + ask) / 2.0 / 100.0


def _spread_cents(bid, ask) -> float:
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(bid) or np.isnan(ask):
        return float('nan')
    return ask - bid


def implied_vol_row(fit: Optional[dict], *, sigma_per_min, now) -> dict:
    """`implied_vol` from the in-process ladder-fit cache.

    The baseline scales a BACKWARD-looking realised vol. The strike ladder
    inverts to a FORWARD-looking one, and where they disagree the baseline's
    `sigma_remaining` is wrong in a knowable direction — the only quantity the
    barrier framing says needs forecasting at all.

    Staleness is a FEATURE, not a filter, up to `MAX_FIT_AGE_MINUTES`: coverage
    is ~15% of the timeline with a five-hour mean gap, so a sigma forward-filled
    from three hours ago is a different claim from a fresh one and the model has
    to be able to tell them apart. Beyond the cap it describes a different
    session and carrying it would be a fabrication.
    """
    from core.book_features import IMPLIED_VOL, MAX_FIT_AGE_MINUTES

    out = {name: float('nan') for name in IMPLIED_VOL}
    if not fit:
        return out
    age = (pd.Timestamp(now) - pd.Timestamp(fit['at'])).total_seconds() / 60.0
    out['iv_staleness_minutes'] = float(age)
    if age > MAX_FIT_AGE_MINUTES or age < 0:
        return out
    implied = float(fit.get('implied_sigma_per_min', np.nan))
    out['implied_sigma_per_min'] = implied
    out['iv_r2'] = float(fit.get('r2', np.nan))
    out['iv_n_strikes'] = float(fit.get('n_strikes', np.nan))
    try:
        realised = float(sigma_per_min)
    except (TypeError, ValueError):
        return out
    if implied > 0 and realised > 0:
        out['iv_minus_realised'] = float(np.log(implied / realised))
    return out


def _attach_book_features(scored: pd.DataFrame, quotes: dict, cache=None) -> None:
    """Book features onto the scoring rows, in place, from the touch.

    One row per symbol: the quote is the same for every offset in the cycle,
    and it is the book the order will actually be priced against.
    """
    from core.book_features import (
        CROSS_VENUE, IMPLIED_VOL, MARKET_PRICE, MARKET_STATE)

    pm_cache, iv_cache = _venue_caches()
    now = pd.Timestamp.now(tz='UTC')
    columns = (tuple(MARKET_STATE) + tuple(MARKET_PRICE)
               + tuple(CROSS_VENUE) + tuple(IMPLIED_VOL))
    for column in columns:
        if column not in scored.columns:
            scored[column] = np.nan
    if not len(scored):
        return
    base = pd.to_numeric(scored.get('baseline_probability'), errors='coerce')
    for i, symbol in enumerate(scored['symbol']):
        entry = quotes.get(symbol)
        if entry is None:
            continue
        try:
            yes_levels, no_levels = ladder_from_cache(cache, entry[1])
            row = book_feature_row(
                entry[0], yes_levels, no_levels,
                baseline_probability=float(base.iloc[i]) if len(base) > i else np.nan)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('%s: book features unavailable (%s)', symbol, str(exc)[:80])
            continue
        # The other two book groups. Both were declared in FEATURE_GROUPS and
        # fitted into the artifact, and neither was ever attached here — nine of
        # forty-nine features scored NaN every cycle. Dropping them instead is
        # not available: refitted without them, log_loss_skill goes +0.00307 ->
        # -0.00023 and folds positive 6/6 -> 3/6.
        row.update(cross_venue_row(pm_cache.get(symbol), entry[0]))
        row.update(implied_vol_row(
            iv_cache.get(symbol), now=now,
            sigma_per_min=scored['sigma_per_min'].iloc[i]
            if 'sigma_per_min' in scored.columns else np.nan))
        row['venue_gap_change_5'] = gap_change(
            symbol, row.get('venue_prob_gap'),
            window_open=scored['window_open'].iloc[i],
            offset=int(scored['offset'].iloc[i])
            if 'offset' in scored.columns
            else int(scored['offset_minutes'].iloc[i]))
        for column in columns:
            if column in row:
                scored.iloc[i, scored.columns.get_loc(column)] = row[column]


def _warn_unscoreable_features(scored: pd.DataFrame, model) -> None:
    """Say loudly when the model wants a feature this row cannot supply.

    A NaN feature does not raise; the booster substitutes the direction it
    learned in training. So the only signal that a live model is not the model
    that was measured is this line.
    """
    wanted = list(getattr(model, 'features', ()) or ())
    if not wanted or not len(scored):
        return
    missing = [c for c in wanted if c not in scored.columns]
    empty = [c for c in wanted
             if c in scored.columns and not pd.to_numeric(
                 scored[c], errors='coerce').notna().any()]
    if missing or empty:
        logger.warning(
            'scoring with features the row cannot supply — missing %s, all-NaN %s. '
            'LightGBM will substitute its learned default, so this is a different '
            'model from the one whose gates were measured.',
            missing or 'none', empty or 'none')


def _resting_total(levels) -> float:
    """Every resting contract on one side, at any price.

    `depth_bid_total` / `depth_ask_total` were the one thing `_record_touch`
    never derived, so `depth_ratio` could not be computed live at all.
    """
    total, seen = 0.0, False
    for entry in levels or []:
        try:
            total += float(entry[1])
            seen = True
        except (TypeError, ValueError, IndexError):
            continue
    return total if seen else float('nan')


def book_feature_row(quote, yes_levels, no_levels, *,
                     baseline_probability: float) -> dict:
    """The model's book features, from the book the fill will price against.

    **Live computed none of these.** `_record_touch` derives the depth every
    cycle and writes it to the store; the scoring row never saw it. That does
    not raise — LightGBM scores a NaN feature with the default direction it
    learned — so the loop would run silently as a DIFFERENT model from the one
    whose gates were measured.

    Built through `core.book_features.market_state_features`, the same function
    the backtest uses, off the same quote the order is priced against, so the
    two cannot drift. Prices go in as CENTS because that is what the snapshot
    shape uses; a one-sided book yields no probability, exactly as in the
    backtest.
    """
    import numpy as _np
    import pandas as _pd
    from core.book_features import market_state_features

    bid = quote.yes_bid if getattr(quote, 'yes_bid', None) is not None else _np.nan
    ask = quote.yes_ask if getattr(quote, 'yes_ask', None) is not None else _np.nan
    best_bid = float(bid) * 100.0 if _np.isfinite(_np.float64(bid)) else _np.nan
    best_ask = float(ask) * 100.0 if _np.isfinite(_np.float64(ask)) else _np.nan

    frame = _pd.DataFrame([{
        'best_bid': best_bid, 'best_ask': best_ask,
        'baseline_probability': baseline_probability,
        'bid_at_touch': float(getattr(quote, 'yes_bid_size', 0.0) or 0.0),
        'ask_at_touch': float(getattr(quote, 'yes_ask_size', 0.0) or 0.0),
        'bid_1c': _cumulative(yes_levels, best=float(bid), within=0.01,
                              invert=False) if _np.isfinite(_np.float64(bid)) else _np.nan,
        'ask_1c': _cumulative(no_levels, best=float(ask), within=0.01,
                              invert=True) if _np.isfinite(_np.float64(ask)) else _np.nan,
        'bid_5c': _cumulative(yes_levels, best=float(bid), within=0.05,
                              invert=False) if _np.isfinite(_np.float64(bid)) else _np.nan,
        'ask_5c': _cumulative(no_levels, best=float(ask), within=0.05,
                              invert=True) if _np.isfinite(_np.float64(ask)) else _np.nan,
        'bid_vol': _resting_total(yes_levels),
        'ask_vol': _resting_total(no_levels),
    }])
    return market_state_features(frame).iloc[0].to_dict()


async def _record_touch(scored: pd.DataFrame, quotes: dict, window_open, offset: int,
                        config: Config, kalshi) -> None:
    """Write the venue's book — top of book plus cumulative depth — to the store.

    Best effort: a failure here must never stop a trading cycle, because this is
    measurement and the cycle is money.

    The ladder comes from `GET /markets/{ticker}/orderbook`, which works while a
    market is open and returns empty once it settles. That asymmetry is the whole
    reason this runs live: depth is not in any historical endpoint, so a book not
    written down now is gone.
    """
    try:
        from core.datastore import ResearchStore

        rows = []
        for symbol in scored['symbol']:
            quote = quotes.get(symbol, (None, None))[0]
            if quote is None or quote.yes_bid is None or quote.yes_ask is None:
                continue
            ticker = quotes[symbol][1]
            yes_levels, no_levels = [], []
            if kalshi is not None and ticker:
                try:
                    book = await kalshi._request(  # noqa: SLF001
                        'GET', f'/markets/{ticker}/orderbook')
                    ladder = book.get('orderbook_fp') or book.get('orderbook') or {}
                    yes_levels = ladder.get('yes_dollars') or ladder.get('yes') or []
                    no_levels = ladder.get('no_dollars') or ladder.get('no') or []
                except Exception:                  # noqa: BLE001 - top of book still lands
                    pass
            event_time = pd.Timestamp(window_open) + pd.Timedelta(minutes=offset)
            rows.append({
                'venue': 'kalshi', 'symbol': symbol, 'event_time': event_time,
                'available_time': pd.Timestamp.now(tz='UTC'), 'quality': 'valid',
                'market_ticker': quotes[symbol][1], 'window_open': window_open,
                'offset_minutes': offset,
                'yes_bid': float(quote.yes_bid), 'yes_ask': float(quote.yes_ask),
                'yes_bid_size': float(quote.yes_bid_size or 0.0),
                'yes_ask_size': float(quote.yes_ask_size or 0.0),
                'depth_bid_1c': _cumulative(yes_levels, best=float(quote.yes_bid),
                                            within=0.01, invert=False),
                'depth_bid_5c': _cumulative(yes_levels, best=float(quote.yes_bid),
                                            within=0.05, invert=False),
                'depth_ask_1c': _cumulative(no_levels, best=float(quote.yes_ask),
                                            within=0.01, invert=True),
                'depth_ask_5c': _cumulative(no_levels, best=float(quote.yes_ask),
                                            within=0.05, invert=True),
                # The whole resting ladder, not just the levels near the
                # touch. `depth_ratio` and `book_convexity` read these, and
                # `book_feature_row` already derives them for the DECISION —
                # recording anything less would trade a feature the next
                # retrain cannot see.
                'depth_bid_total': _resting_total(yes_levels),
                'depth_ask_total': _resting_total(no_levels),
                'levels_bid': float(len(yes_levels)),
                'levels_ask': float(len(no_levels)),
                'seq': float('nan'), 'gaps': 0.0,
                # **Name the observer.** `source` is in
                # `EVENT_KEY_EXTRA['venue_depth']` so that independent
                # observations of one minute survive a read instead of
                # collapsing to whichever was published last. This writer left
                # it unset, so every row it has ever written sits under a NULL
                # observer — invisible to `_validate_depth`, which compares by
                # source, and indistinguishable from a row whose source the
                # store failed to record.
                #
                # It is deliberately NOT 'live': that belongs to the ladder
                # recorder's REST poll, and this is a different sample at a
                # different instant — the quote the trading loop actually
                # decided against. Colliding them would discard one.
                'source': 'live_touch',
            })
        if rows:
            ResearchStore().write('venue_depth', pd.DataFrame(rows))
    except Exception as exc:                       # noqa: BLE001 - never break a cycle
        logger.warning('could not record top of book (%s)', str(exc)[:120])


def prepare_init_score(scored: pd.DataFrame, model) -> pd.DataFrame:
    """Attach whatever init score this artifact was fitted on.

    The baseline logit is already on the table — `score_live` attaches it from the
    fold's own baseline. A market-initialised artifact needs the recorded quote's
    implied probability instead, taken from the de-spread mid rather than the ask:
    the init score is the market's estimate of the probability, not what we would
    pay for it.

    `Quote.mid` is None on a one-sided book, so those rows carry NaN and
    `ForecastModel.predict` returns NaN for them. That is the correct outcome — a
    correction to a price we could not read is not a forecast — and `decide()`
    abstains on it rather than falling back to the baseline under the wrong
    provenance.
    """
    source = getattr(model, 'init_score_source', 'baseline')
    if source != 'market':
        return scored
    from core.model import attach_market_logit

    out = scored.copy()
    out['market_probability'] = out['market_mid']
    return attach_market_logit(out)


# Wake just AFTER the offset, not before it.
#
# The first version of this woke 12 seconds early, reasoning that the cycle needs
# that long to fetch bars, reconcile and score before it reads the book. It made
# the lag four times worse — 2.64 minutes against the 0.62 it was meant to fix.
#
# `choose_offset(elapsed)` returns the largest offset at or below `elapsed`. Waking
# at 5.8 minutes therefore selects offset **3**, not 6, and scores the old offset
# nearly three minutes late. The simulation that vetted the early wake checked
# where the loop landed and never checked what `choose_offset` would do with it.
#
# One second past the boundary picks the intended offset, and the residual lag is
# then just the in-cycle latency before the book is read. Driving that below ~15s
# means reordering the cycle to fetch quotes first, which is a separate change.
DECISION_LAG_SECONDS = 1.0
# How far past an offset instant is still worth acting on immediately rather than
# waiting for the next one. Wider than a cycle takes (~10s), narrower than the gap
# between offsets (180s).
MISSED_TARGET_GRACE_SECONDS = 60.0
# How close to an offset instant a cycle must be to make a decision at all.
#
# **This is a latency budget, not a convenience window, and 75 was catastrophic.**
# Measured 2026-08-25 on the quote backfill: holding the signal at the offset it
# was built for and moving only the quote, one minute of staleness costs
# 0.025 nats at +3m and 0.074 nats at +12m, against a total model edge over the
# market of 0.002-0.005. Break-even lag is ~3s on the strict reading and ~10s on
# the generous one (price discovery near settlement is back-loaded, so linear
# interpolation over two minutes overstates the first seconds).
#
# At 75 the tolerance was WIDER than the ordinary cadence (60s), so after the
# scheduler fired on target the next routine cycle was still inside it and
# decided the same offset a full minute late. Observed on window 07:00 offset
# +3m: at +5s the model wanted ETH up at 0.81; at +76s it wanted ETH down at
# 0.07, reading the market's own 12-point move as a *larger* edge (7.43pp against
# 3.04pp). The on-time orders did not fill and the stale one did, which is the
# same mechanism as the ~30% no-fill rate seen from the other side.
#
# 15s admits the on-target cycle (observed +5s to +6s, book read first) with
# margin for a slow fetch, and refuses anything a cadence behind.
DECISION_TOLERANCE_SECONDS = 15.0


# Refusals that mean something is wrong rather than that the gates are working.
# A latched breaker, a breached bankroll floor, an unreadable book, or the model
# disagreeing with the market by an implausible margin are all worth a line; the
# ordinary "the edge was not there" outcomes are not.
LOUD_REFUSALS = frozenset({
    Reason.HALTED, Reason.BANKROLL_FLOOR, Reason.DISAGREEMENT_IMPLAUSIBLE,
    Reason.NO_QUOTE, Reason.PROBABILITY_INVALID, Reason.NOT_FINITE,
})


def decision_log_level(decision: Decision) -> int:
    """How loudly to report one decision. Every one is stored either way.

    `offset_not_traded` alone fires three symbols x two non-entry offsets x four
    windows an hour, forever, by design. Together with the other routine refusals
    it was the largest single source of noise in the live log, and none of it
    carries information a human needs — the cycle summary counts them.

    Anything that spends money is always INFO, and so is any refusal that
    indicates a fault rather than a gate doing its job.
    """
    if decision.traded:
        return logging.INFO
    if decision.reason in LOUD_REFUSALS:
        return logging.INFO
    return logging.DEBUG


def heartbeat_due(window_open, last_window) -> bool:
    """One line per window, so a quiet log still proves the loop is turning.

    Removing the per-cycle noise left the loop emitting nothing at all in steady
    state, which cannot be distinguished from a hang. The container healthcheck
    only proves the process is alive and can reach the database — not that it is
    still waking on the offsets and deciding. A restart emits one immediately,
    because that is exactly when an operator is watching.
    """
    return last_window is None or window_open != last_window


def heartbeat_summary(*, window_open, cycles: int, decisions: int, traded: int,
                      bankroll: float, lag_seconds: float) -> str:
    """The five things worth checking, on one line."""
    lag = f'{lag_seconds:.1f}s' if lag_seconds == lag_seconds else 'n/a'
    return (f'window {window_open:%H:%M} closed: {cycles} cycles, '
            f'{decisions} decisions, {traded} traded, bankroll ${bankroll:.2f}, '
            f'decision lag {lag}')


def decision_offset(elapsed: float, config: Config,
                    forced: Optional[int] = None) -> Optional[int]:
    """The offset this cycle may decide at, or None to settle and reconcile only.

    **Decide AT an offset, not on every cycle that happens to be past one.**
    `choose_offset` returns the largest offset at or below `elapsed`, so at 4, 5
    and 6 minutes into a window it returns 3, 3, 6 — and every cycle in between
    once produced a full decision and an order attempt. Measured: fourteen order
    attempts in three minutes where there should have been two. The intermediate
    cycles exist to settle and reconcile on a steady cadence; they were never
    meant to trade.

    The gate is `DECISION_TOLERANCE_SECONDS`, which is a measured latency budget
    — see the constant. A cycle further from its offset than that has watched the
    market move away from the signal it is holding, and the "edge" it computes is
    that displacement rather than a forecast. It abstains.

    An explicit `--offset` is a deliberate override — a backfill, a manual
    decision, or a test — and is honoured whatever the clock says. Only the offset
    the scheduler chose is subject to the tolerance.
    """
    if forced is not None:
        return forced
    offset = choose_offset(elapsed, config)
    if offset is None:
        return None
    lag_seconds = abs(elapsed - offset) * 60.0
    if lag_seconds > DECISION_TOLERANCE_SECONDS:
        logger.debug('%.2fm into the window, offset +%dm is %.0fs away (budget '
                     '%.0fs); settling and reconciling only',
                     elapsed, offset, lag_seconds, DECISION_TOLERANCE_SECONDS)
        return None
    return offset


def mark_decided(fired_targets: set, decisions: list) -> None:
    """Record every (window, offset) a cycle actually decided.

    **Two different things start a cycle, and only one of them was recording.**
    The `--cycle-seconds` cadence lands wherever it lands; if that is inside
    `DECISION_TOLERANCE_SECONDS` of an offset, the cycle decides and orders.
    `seconds_until_next_decision` then fired the same offset again a moment
    later, because its `already_fired` set only held targets the planner itself
    had scheduled. Measured on the live account: seventeen of forty-two order
    attempts were the venue refusing `order_already_exists` on a duplicate
    `client_order_id`, always five seconds after the first attempt.

    A refused decision counts as fired. The offset was evaluated, and re-running
    it produces the identical deterministic `client_order_id` — so a retry
    cannot reach the book, only the 409.
    """
    for decision in decisions:
        window = decision.window_open
        if not isinstance(window, pd.Timestamp):
            window = pd.Timestamp(window)
        fired_targets.add((window, int(decision.offset)))


def seconds_until_next_decision(config: Config, args, *,
                                now: Optional[datetime] = None,
                                already_fired: Optional[set] = None) -> float:
    """Sleep until just before the next decision offset, not a fixed interval.

    **The free-running timer was a measurement bias.** The loop slept
    `--cycle-seconds` from wherever it happened to be, so a decision nominally at
    +3m was taken whenever a cycle next landed in [3m, 4m). Measured over the
    first two live days: +3.62m on average, up to +4.16m.

    The features are built for the nominal offset while the price is read late, so
    the market gets up to a minute of information the model does not have — and
    one minute is worth ~0.027 nats, measured by shifting the offset, against a
    total model edge of +0.002. The bias was comparable to the whole effect and
    ran against us.

    Waking at `window_open + offset + 1s` collapses it to the in-cycle latency.
    Falls back to `--cycle-seconds` when no offset is ahead in this window, so
    settlement and reconciliation still run on their old cadence.
    """
    now = now or datetime.now(timezone.utc)
    window_open, _ = current_window(now, config)
    horizon = float(getattr(args, 'cycle_seconds', 60) or 60)
    if already_fired is not None:
        # Keep only this window and the next; anything older cannot fire again.
        keep = {window_open, window_open + timedelta(minutes=config.window_minutes)}
        already_fired.intersection_update(
            {k for k in already_fired if k[0] in keep})
    candidates = []
    for window in (window_open, window_open + timedelta(minutes=config.window_minutes)):
        for offset in config.decision_offsets:
            target = window + timedelta(minutes=offset) + timedelta(seconds=DECISION_LAG_SECONDS)
            delay = (target - now).total_seconds()
            # **A target just behind us must fire NOW, not be skipped.**
            #
            # The sleep is computed after the cycle finishes, so a cycle that runs
            # even slightly long ends up past the target it was aiming at. Dropping
            # it then jumps to the NEXT offset, capped at the ordinary cadence — so
            # a two-second overshoot at 02:03:01 became a decision at 02:04:10,
            # labelled +3m and taken at +4.17m. Every offset was missed the same
            # way, which is worse than the free-running timer this replaced.
            # A target just behind us fires NOW rather than being skipped — but
            # exactly ONCE. Without the `already_fired` guard this returns 0.5s,
            # the cycle runs, the target is still inside the grace window, and it
            # fires again: measured, fourteen order attempts in three minutes
            # where there should have been one or two. `already_entered` does not
            # save you, because a refused order books no position and is
            # legitimately retryable.
            key = (window, offset)
            if -MISSED_TARGET_GRACE_SECONDS < delay <= 1.0:
                if already_fired is None or key not in already_fired:
                    if already_fired is not None:
                        already_fired.add(key)
                    return 0.5
                continue
            if delay > 1.0:
                candidates.append(delay)
    if not candidates:
        return horizon
    # Never sleep past the ordinary cadence: settlement and venue reconciliation
    # are on the same cycle and should not wait for the next decision offset.
    return float(min(min(candidates), horizon))


async def run_cycle(args, config: Config, writer: PgWriter, model,
                    kalshi: Optional[KalshiClient]) -> list[Decision]:
    now = datetime.now(timezone.utc)
    window_open, elapsed = current_window(now, config)
    offset = decision_offset(elapsed, config, forced=args.offset)
    settle_time = window_open + pd.Timedelta(minutes=config.window_minutes)

    # **The book is read LAST of everything that can precede it**, because its age
    # is the only age paid at the touch. A quote that goes stale during the cycle
    # can only cost a fill, never an overpay — the order is `immediate_or_cancel`
    # at a limit derived from this price — so not filling is the direction to be
    # wrong in, and shortening the interval is how fills are bought back.
    # **Where the cycle spends its time between reading the book and sending the
    # order**, because that interval is what decides whether a fill happens.
    # Measured from the recorded tickets before this existed: median 4.97s, max
    # 5.56s from book-read to order-sent — against a book that moves ~1-1.5c in
    # that time and order allowances of 0.49c to 3c. Kills occurred even at the
    # full 3c cap, so this is a latency problem before it is a limit-width one.
    # `_record_touch` was the obvious suspect and is worth only ~300ms of it;
    # this says which of the rest actually costs.
    phase: dict[str, float] = {}
    _mark = time.perf_counter()

    def _phase(name: str) -> None:
        nonlocal _mark
        now_ = time.perf_counter()
        phase[name] = now_ - _mark
        _mark = now_

    # **Bars first, then the book.** Both are needed before a decision, but only
    # the BOOK's age is paid at the touch: the quote is what the order is priced
    # against, and every second between reading it and sending the order is a
    # second the market can move away. Bars are one-minute candles — three
    # seconds of age on them is nothing.
    #
    # Measured with the quote read first: quotes 0.39s, bars 3.17s, reconcile
    # 0.93s, score 0.14s, touch 0.30s — 4.55s from book to order, of which the
    # Coinbase call was 70%. Reading the book after the bars removes that 3.17s
    # from the staleness with no change to anything else: `settle_time` comes
    # from the clock, not from the bars, so nothing here depends on the order.
    bars = await fetch_bars(config)
    _phase('bars')

    if not bars:
        logger.error('no bars, nothing to do this cycle')
        return []
    record_minute_prices(writer, bars)

    halted = check_circuit_breakers(writer, config, now=now)
    if halted:
        # Note the offset is deliberately NOT cleared. That returns before
        # `score_live`, which stopped the recording as well as the trading — and
        # the measurement is the reason this loop is running. `decide()` refuses
        # every row with `Reason.HALTED` instead, so quotes, scores and outcomes
        # keep accruing for `market_benchmark` while no money moves.
        logger.error('account halted (%s); scoring and recording continue, '
                     'no new entries', halted)

    stale = stale_symbols(bars, config, now=now)
    if stale:
        # Settlement and reconciliation below still run: they read the past, which
        # a short feed does not invalidate, and a matured position should not be
        # left hanging because this cycle cannot decide.
        for symbol, why in sorted(stale.items()):
            logger.error('%s: %s', symbol, why)
        logger.error('the universe is %d of %d symbols this cycle, so no decision '
                     'is made — a short universe redefines every cross-asset '
                     'feature rather than simply omitting one row',
                     len(config.symbols) - len(stale), len(config.symbols))
        offset = None

    _phase('local')

    # Read the venue first, where there is one: it knows what actually settled,
    # and bars only fill in what it has not resolved. Its *balance* is adopted
    # after we settle, not here — the venue credits a payout the moment it
    # settles, so adopting first and settling second counted every payout twice
    # for a cycle. `adopt_venue_balance` has the measured log.
    venue_settlements: dict[str, dict] = {}
    venue_balance = float('nan')
    # **The shard is remembered between cycles, because the book is now read
    # after this.** Kalshi shards its exchange by category and balances are local
    # to a shard, so the balance query has to name the right one — and the only
    # place it was ever read from was the quotes, which is what forced the book
    # to be fetched first and cost ~1s of staleness on every order.
    #
    # It is a property of the series, not of a particular quote, so it is stable
    # between cycles; `remember_exchange_index` re-checks it against every book
    # actually read and complains if it ever moves. Until one has been seen, this
    # is None and `balance()` falls back to its own default — the same thing that
    # happens today on a cycle where no symbol quoted.
    shard = _LAST_EXCHANGE_INDEX
    if kalshi is not None and args.reconcile:
        try:
            venue = await reconcile_with_venue(writer, kalshi, exchange_index=shard)
            venue_settlements, venue_balance = venue.settlements, venue.balance
            _phase('reconcile')
        except KalshiError as exc:
            logger.error('reconciliation failed (%s); falling back to our own '
                         'bookkeeping for this cycle', exc)
    settle_due(writer, bars, venue_settlements=venue_settlements)
    _phase('settle')

    # **The book, last.** Everything above — bars, local bookkeeping, the venue
    # reconciliation, settlement — happens before the quote is read, so none of
    # it is staleness the order pays. Measured going in: 4.55s from book to
    # order, of which this reordering removes all but scoring.
    quotes: dict = {}
    quote_time = pd.Timestamp.now(tz='UTC')
    if offset is not None and kalshi is not None:
        try:
            quotes = await fetch_quotes(kalshi, list(config.symbols), settle_time)
            quote_time = pd.Timestamp.now(tz='UTC')
        except Exception as exc:              # noqa: BLE001 - the cycle still settles
            logger.error('could not read the book (%s); no decision this cycle', exc)
            quotes, offset = {}, None
    _phase('quotes')
    remember_exchange_index(quotes)
    # Independent of whether anything was held: this is what turns the recorded
    # market quotes into a scoreable sample.
    settle_predictions(writer, bars)
    # Now that our own credits are in, the drift is a real disagreement.
    if kalshi is not None and args.reconcile:
        adopt_venue_balance(writer, venue_balance, exchange_index=shard)

    if offset is None:
        # This used to read "N minutes into the window; first decision offset is
        # +3m", which at 8 minutes in was both false and not the reason. There are
        # three ways to arrive here and they are worth telling apart.
        reached = [o for o in sorted(config.decision_offsets) if o <= elapsed]
        if not reached:
            logger.debug('%.1fm into the window; first offset is +%dm',
                         elapsed, min(config.decision_offsets))
        else:
            logger.debug('%.1fm into the window; offset +%dm is %.0fs behind, over '
                         'the %.0fs budget — settling and reconciling only',
                         elapsed, reached[-1], (elapsed - reached[-1]) * 60.0,
                         DECISION_TOLERANCE_SECONDS)
        return []

    # Everything `_attach_book_features` fills after this returns. Declaring it
    # keeps the warning meaningful: a book feature that is empty because the
    # book has not been read yet is not the same as one nothing will ever fill,
    # and nine spurious warnings a cycle made the real case invisible.
    from core.book_features import (
        CROSS_VENUE as _CV, IMPLIED_VOL as _IV,
        MARKET_PRICE as _MP, MARKET_STATE as _MS)
    deferred = tuple(_MS) + tuple(_MP) + tuple(_CV) + tuple(_IV)
    scored = score_live(bars, model.scoring, config, deferred=deferred,
                        window_open=window_open, offset=offset,
                        groups=model.groups or None)

    # **When the book was actually read**, which is not `window_open + offset`.
    #
    # Measured over the first two live days: a decision nominally at +3m read its
    # quote at +3.62m on average and up to +4.16m. The features are built for the
    # nominal offset, so the market's price carries up to a minute of information
    # the model does not have — and one minute is worth ~0.027 nats, measured, as
    # against a total model edge of +0.002. The bias is comparable to the whole
    # effect, and it runs against us.
    #
    # It is not a trading bug: a fresh quote is what a fill would actually pay.
    # It is a *measurement* bug, and the fix is for the row to say what happened
    # rather than what was intended, so `market_benchmark` can filter or correct.
    # `quote_time` is stamped where the book is actually read, at the top of the
    # cycle.
    scored['ask_up'] = [
        quotes[s][0].ask_for('up') if s in quotes else np.nan for s in scored['symbol']]
    scored['ask_down'] = [
        quotes[s][0].ask_for('down') if s in quotes else np.nan for s in scored['symbol']]
    scored['market_ticker'] = [
        quotes[s][1] if s in quotes else None for s in scored['symbol']]
    # `Quote.mid` is None on a one-sided book, which is the honest answer: the
    # midpoint of a single quote is not a midpoint.
    scored['exchange_index'] = [
        quotes[s][0].exchange_index if s in quotes else 0 for s in scored['symbol']]
    scored['market_mid'] = [
        quotes[s][0].mid if s in quotes else np.nan for s in scored['symbol']]

    # Scoring happens *after* the book is read, not before. A baseline-initialised
    # model does not care — that is why this used to run above the quote fetch —
    # but a market-initialised one is a correction to the price and cannot be
    # evaluated without it. Leaving the old order in place would have left a
    # runtime failure waiting behind a config flag, which is the shape of most of
    # what the audit found. Behaviour is unchanged for the default source: the
    # only thing between the two positions was `settle_time`.
    # The book features, from the touch we just read. Without these the model
    # scores them as NaN — LightGBM uses the default direction it learned, so the
    # loop would run silently as a DIFFERENT model from the one whose gates were
    # measured, which is the `price_source` failure one level deeper.
    #
    # The ladder comes from the stream's in-process cache, so the depth features
    # cost no fetch and are fresher than REST would be (~34ms against ~73ms).
    # An earlier version attached touch-only on the grounds that a ladder fetch
    # would cost ~300ms against a 4.97s book-to-order budget — but that figure
    # predates the phase instrumentation, which measures 0.10-0.11s, and the
    # fetch is not needed at all.
    _attach_book_features(scored, quotes, _stream_cache())
    scored = prepare_init_score(scored, model)
    _phase('score')
    _warn_unscoreable_features(scored, model)
    scored['model_probability'] = model.predict(scored)

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

    # Everything above happened AFTER the book was read and BEFORE any order can
    # go out, so it is all quote staleness paid at the touch. Logged whether or
    # not this cycle trades, because a cycle that abstains still measures the
    # latency the next one will pay.
    if phase and offset is not None:
        # Only what ran AFTER the book was read. `phase` is insertion-ordered, so
        # this stays correct if the cycle is reordered again — summing "everything
        # except quotes" would have silently kept counting the bar fetch once the
        # bars moved ahead of it, and reported no improvement from the change that
        # produced all of it.
        names = list(phase)
        after = names[names.index('quotes') + 1:] if 'quotes' in phase else names
        since_quote = sum(phase[k] for k in after)
        logger.info('cycle latency: %s | %.2fs between the book and the order',
                    ' '.join(f'{k} {v:.2f}s' for k, v in phase.items()),
                    since_quote)

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

    # Live prices from the book or not at all. Without this a row with no quote
    # falls back to the backtest's counterfactual price — our own baseline — and
    # can still return TRADED, which is how an unresolved market booked a
    # position for an order that was never sent.
    require_quote = args.mode == 'live'
    for _, row in scored.sort_values('symbol').iterrows():
        # Re-read the clock. `now` was taken at the top of the cycle, and between
        # then and here sit a Coinbase fetch, four authenticated reconcile calls,
        # LightGBM inference and one quote call per symbol at up to 15s each.
        remaining = (settle_time - pd.Timestamp.now(tz='UTC')).total_seconds()
        if remaining < config.min_remaining_seconds:
            logger.warning(
                '%s: %.0fs left of the window, under the %ds floor — the sigma on '
                'this row is for %s and the clock has moved past it',
                row['symbol'], remaining, config.min_remaining_seconds, settle_time)
            break
        decision = decide(row, config, bankroll=account.bankroll,
                          exposure=exposure, require_quote=require_quote,
                          halted=bool(halted))
        decisions.append(decision)
        writer.write_prediction(
            symbol=decision.symbol, window_open=window_open, settle_time=settle_time,
            offset_minutes=offset, decision_time=quote_time,
            strike=float(row['strike']), last_price=float(row['last_price']),
            displacement=float(row['displacement']),
            sigma_remaining=_finite(row.get('sigma_remaining')),
            z_score=_finite(row.get('z_score')),
            baseline_probability=float(row['baseline_probability']),
            model_probability=float(row['model_probability']),
            # The venue's belief (the mid) and what a trade would cost (both
            # asks). This was `ask_up` recorded under the name
            # `market_probability` — a price rather than a probability, one side
            # of the book only, and read by nothing. The mid is the right input
            # to "does the model beat the market"; the ask carries half the
            # spread, so scoring against it would flatter us by exactly what we
            # pay to cross. Written on EVERY row, refused ones included, because
            # the sample that can answer that question has to be unselected.
            market_probability=_finite(row.get('market_mid')),
            market_ask_up=_finite(row.get('ask_up')),
            market_ask_down=_finite(row.get('ask_down')),
            price_source=decision.price_source,
            reason=decision.reason.value, traded=decision.traded,
            side=decision.side.value if decision.side else None,
            price=_finite(decision.price), effective_cost=_finite(decision.effective_cost),
            edge=_finite(decision.edge), contracts=decision.contracts or None,
            model_version=getattr(model, 'version', None),
        )
        logger.log(decision_log_level(decision), decision.describe())
        if not decision.traded:
            continue
        # Only count exposure we actually took on. `act_on` returns False when
        # the order was refused, killed, or never sent.
        if await act_on(args, writer, kalshi, decision, row, config=config,
                        quote_time=quote_time):
            exposure = exposure.with_(decision)

    # **Record the top of book we already have.** `Quote` parses `yes_bid_size`
    # and `yes_ask_size` from the same REST response the price comes from, and
    # `decide()` already uses them to cap the stake — and then they were dropped.
    #
    # They are the missing half of the economic question and they cost nothing:
    # no historical endpoint carries size (candlesticks give top-of-book price and
    # nothing behind it; the settled orderbook returns empty), so a size not
    # written down here is gone. Observed at the touch: BTC ~6,900 contracts, ETH
    # ~109, SOL ~6.9 — against orders of ~7, which makes SOL the case where this
    # actually decides whether a fill was possible.
    #
    # Kept behind a flag from the bisect that cleared it: this was the last change
    # deployed before every order began returning `market_not_found`, so it was
    # the first suspect. Disabling it changed nothing — the cause was the market's
    # `exchange_index` moving to 2 while the order body defaulted to 0.
    #
    # **After the orders, not before them.** It costs three REST orderbook calls
    # and a synchronous Parquet write — measured at 0.29s — and every millisecond
    # of that was staleness the order paid at the touch, for a row nothing in the
    # decision reads. Moving it changes only WHEN the archive is written, never
    # what it contains: `quotes` is the same object the decision priced against.
    if os.getenv('RECORD_TOUCH', '1') == '1' and offset is not None and quotes:
        await _record_touch(scored, quotes, window_open, offset, config, kalshi)
        _phase('touch')

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

    **Pay down to the gate, not to a fraction.** `min_edge_pp` is the threshold
    that admitted this trade, so any fill leaving at least that much is one the
    system has already said yes to — while no fill at all earns zero. Everything
    above the gate is therefore spendable.

    The rule this replaced was `min(edge * 0.25, 1c)`, which allowed the live
    BTC decision of 2026-08-26 just 0.92c of give against a book that moves ~2c
    in the seconds between reading it and the order landing. Measured over the
    first live nights: 26 of 42 attempts did not fill, and they missed on price
    rather than on size.

    Crossing further costs little: the order is `immediate_or_cancel` with
    `taker_at_cross`, so it fills against resting size from the touch outward
    and stops. The limit bounds the worst fill; it does not set the price. It
    only binds where the touch was too thin to fill at the old limit — which is
    precisely the case that used to return nothing at all.

    `max_slippage_cents` stays as a rail against a book so thin the order walks
    it absurdly far. That is not hypothetical: the original `price + edge` limit
    sent 0.7832 against a 0.60 ask, 18c of tolerance on a 1c measured spread.
    """
    config = DEFAULT_CONFIG
    edge = decision.edge if np.isfinite(decision.edge) else 0.0
    spendable = max(0.0, edge - config.min_edge_pp / 100.0)
    allowance = min(spendable, config.max_slippage_cents / 100.0)
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
    dead = status in ('canceled', 'cancelled', 'killed', 'rejected', 'expired')

    # `int(...)`, not `int(str)`: V2 returns these as fixed-point decimal strings
    # like "10.00", and `int("10.00")` raises ValueError — which this function
    # caught and turned into "nothing filled". A real, paid-for fill would have
    # been recorded as no fill and the position dropped on the floor, which is the
    # most expensive way to be wrong here. Float, not int, because the venue fills
    # fractionally: one live order came back as 0.43 + 0.57 on the same ticket.
    def count(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # **`fill_count_fp` first, because it is the only count a V2 order carries.**
    # Read off the live account: an order has `fill_count_fp`, `initial_count_fp`
    # and `remaining_count_fp`, and none of the three names below. So this loop
    # never matched, and every fill was decided by `status` alone — all-or-nothing,
    # which was harmless only while the client sent `fill_or_kill`. Under
    # `immediate_or_cancel` a partial fill is `status='canceled'` with a non-zero
    # `fill_count_fp`, and the status fallback books it as nothing.
    filled = None
    for key in ('fill_count_fp', 'taker_fill_count', 'filled_count', 'fill_count'):
        if order.get(key) is not None:
            filled = count(order[key])
            if filled is None:
                return 0, float('nan')
            break

    # **Never infer a fill from `remaining` on a dead order.** The live kill
    # reported `remaining_count_fp='0.00'` against `initial_count_fp='5.00'` and
    # nothing filled: remaining hits zero because the order left the book, not
    # because it traded, so `requested - remaining` would invent five contracts
    # out of a total miss. Remaining only means anything while the order is live.
    if filled is None and not dead:
        for key in ('remaining_count_fp', 'remaining_count'):
            if order.get(key) is not None:
                remaining = count(order[key])
                if remaining is None:
                    return 0, float('nan')
                filled = float(requested) - remaining
                break

    if filled is None:
        # No count at all. Only 'executed'/'filled' justifies believing the whole
        # order traded; anything else (including an empty body) is unknown, and
        # unknown must not become a position.
        if status in ('executed', 'filled'):
            filled = float(requested)
        else:
            return 0, float('nan')

    filled = max(0, min(int(round(filled)), requested))
    if filled <= 0:
        return 0, float('nan')

    # **The price actually paid, not the one we hoped for.** No price key the old
    # list looked for exists on a V2 order either, so this returned nan every time
    # and the caller fell back to the decision price. `taker_fill_cost_dollars` is
    # the money that actually left the account, so cost / count is the true
    # average fill — and it already accounts for filling better than the limit.
    price = float('nan')
    for key in ('taker_fill_cost_dollars', 'fill_cost_dollars'):
        cost = count(order.get(key))
        if cost is not None and cost > 0:
            paid = cost / float(filled)
            # That cost is denominated in the side actually BOUGHT, while this
            # function's contract is YES-denominated (the caller inverts for
            # DOWN). A NO buy at 7c has `taker_fill_cost_dollars=0.07`; returning
            # it unchanged would have the caller invert it and book 93c.
            outcome = str(order.get('outcome_side', '') or '').strip().lower()
            price = (1.0 - paid) if outcome == 'no' else paid
            break

    if not np.isfinite(price):
        for key in ('average_fill_price_dollars', 'avg_price_dollars',
                    'average_fill_price', 'yes_price_dollars', 'yes_price',
                    'no_price'):
            raw = order.get(key)
            if raw is None:
                continue
            value = count(raw)
            if value is None:
                continue
            # Integer-cent fields on the same keys as dollar ones; the venue serves
            # both shapes. Above 1.0 it can only be cents.
            price = value / 100.0 if value > 1.0 else value
            if key == 'no_price':
                price = 1.0 - price
            break
    return filled, price


async def act_on(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                 decision: Decision, row, *,
                 config: Config = DEFAULT_CONFIG,
                 quote_time: Optional[pd.Timestamp] = None) -> bool:
    """Record the ticket, place the order when asked to twice, book the fill.

    Returns whether a position was booked, so the caller only counts exposure it
    actually took on.

    **A position is written only when money actually moved.** Previously control
    fell through to `open_position` in every branch: an unresolved market (no
    ticker) wrote a position and debited the bankroll having sent no order, and so
    did `--mode live --dry-run`. Both produced holdings the venue had never heard
    of, which `settle_due` then settled into invented PnL — the exact failure the
    `price_source` column exists to make visible.

    **`quote_time` is when the book was read, not when this function runs.**
    `run_cycle` reads it once at the top of the cycle, then does a Coinbase
    fetch, four authenticated reconcile calls, inference and six 15-second quote
    calls before any order is sent — none of which revalidates that the book is
    still the one being traded on. `max_quote_age_seconds` was declared for
    exactly this and never checked anywhere. `None` (the default, and what every
    call site had before this) makes no staleness claim and is not refused —
    only a caller that supplies a timestamp is asking for the check.
    """
    placing = bool(args.place_orders) and kalshi is not None and not args.dry_run

    if placing and quote_time is not None:
        age = (pd.Timestamp.now(tz='UTC') - pd.Timestamp(quote_time)).total_seconds()
        if age > config.max_quote_age_seconds:
            writer.write_ticket(
                symbol=decision.symbol, window_open=decision.window_open,
                settle_time=decision.settle_time, offset_minutes=decision.offset,
                market_ticker=decision.market_ticker, side=decision.side.value,
                contracts=decision.contracts, limit_price=decision.price,
                max_price=order_limit_price(decision), expected_cost=decision.stake,
                model_probability=decision.model_probability, edge=decision.edge,
                status='skipped',
                note=(f'stale quote: {age:.0f}s old, over the '
                     f'{config.max_quote_age_seconds}s limit'))
            logger.warning(
                '%s window %s: quote is %.0fs old, over the %ds limit. '
                'Refusing rather than trading a book that may have moved.',
                decision.symbol, decision.window_open, age,
                config.max_quote_age_seconds)
            return False

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
    # Set only when the venue reported a real fill price, which is what licenses
    # recomputing the outlay from it rather than from the decision's stake.
    venue_fill_price: Optional[float] = None
    # The market's own exchange, not the order body's default of 0. See
    # `Quote.exchange_index` — a mismatch returns `404 market_not_found`, which
    # names the market rather than the mismatch and took a bisect to find.
    exchange_index = int(getattr(row, 'get', lambda *_: 0)('exchange_index', 0) or 0)
    if placing:
        try:
            order = await kalshi.place_order(
                ticker=decision.market_ticker, side=decision.side.value,
                exchange_index=exchange_index,
                contracts=decision.contracts,
                limit_price=order_limit_price(decision),
                # **The offset belongs in this key.** Keyed on (symbol, window)
                # alone it enforced one order *attempt* per window, while the
                # policy is one *position* per window — and those differ exactly
                # when an order does not fill. A fill_or_kill that kills still
                # consumes the id at the venue, so the first thin-volume kill
                # locked every later offset out of that window with
                # `409 order_already_exists`.
                #
                # Measured over the first live night: 57 of 69 unfilled attempts
                # were our own duplicate key, not the venue refusing us, and they
                # carried a *higher* claimed edge (8.37pp) than the ones that
                # filled (5.77pp). Only 9 were genuine
                # `fill_or_kill_insufficient_resting_volume`, at 3.94pp — the
                # lowest of the three. So the market was not selecting against
                # us; we were blocking ourselves out of the better half of our
                # own signal.
                #
                # Double-entry is not what this key was protecting. That is
                # `entries_for_window`, which counts a ticket in any status but
                # `skipped` — so a crash between sending an order and booking the
                # position still blocks the window, and `skipped` (nothing
                # bought) correctly reopens it. One attempt per offset is the
                # documented rule: walk the offsets in order, take the first that
                # clears every gate.
                client_order_id=(
                    f'{decision.symbol}-{decision.window_open:%Y%m%d%H%M}'
                    f'-{decision.offset:02d}'),
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
        # V2 quotes everything from the YES side, so a DOWN fill comes back as the
        # price we SOLD yes at. What we paid for the NO contract is `1 - that`.
        # Recording 0.69 where 0.31 was paid would not corrupt PnL — that is
        # computed from `outlay` — but it would put a wrong price on the position
        # and make `realised_edge_pp` and every displayed number wrong for half the
        # trades.
        if np.isfinite(fill_price) and decision.side is not Side.UP:
            fill_price = 1.0 - fill_price
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
            venue_fill_price = float(fill_price)
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

    # **Book the fill, not the intention.**
    #
    # These were pro-rated from `decision.stake` and `decision.fee`, both computed
    # at the *decision* price, while `price` on the position is taken from the
    # fill. A `fill_or_kill` fills at or better than its limit, so the cash
    # actually leaving the account was always <= the booked outlay and the
    # bankroll read systematically low — surfacing later as unexplained "balance
    # drift" against the venue, which the operator is told to read as an
    # unrecorded fill.
    #
    # Observed 2026-08-25: `placing down 1 @ 0.08` then `filled 1 @ 0.0530`. The
    # position recorded price 0.0530 and an outlay derived from 0.08.
    #
    # Recomputed ONLY when the venue actually reported a fill price. Without a
    # book, `decide()` folds the half-spread into `stake` (`crossing`), and
    # rebuilding the outlay from price alone would silently drop it and make paper
    # trading cheaper than it is. `outlay` is fee-inclusive, matching
    # `decide()`'s `stake`.
    ratio = (filled / decision.contracts) if decision.contracts else 0.0
    if filled and venue_fill_price is not None:
        fee = float(trade_fee(filled, venue_fill_price, config))
        outlay = float(filled) * float(venue_fill_price) + fee
    else:
        outlay = decision.stake * ratio
        fee = decision.fee * ratio
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


def config_from_args(args) -> Config:
    """The run's configuration, with everything that changes an answer explicit.

    `entry_offsets` is applied here rather than defaulted in `Config` because
    `scripts.evaluate` must keep measuring every offset — a narrowed library
    default would make the sweep unable to price the very cells that justify the
    narrowing.
    """
    config = DEFAULT_CONFIG.with_fee_assumptions(find_fee_config())
    overrides: dict = {}
    if args.bankroll is not None:
        overrides['starting_bankroll'] = args.bankroll
    entry = getattr(args, 'entry_offsets', None)
    if entry:
        overrides['entry_offsets'] = tuple(int(o) for o in entry)
    daily = getattr(args, 'max_daily_loss_fraction', None)
    if daily is not None:
        # Deliberately not clamped. A wider bound than the default is the whole
        # point of the flag, and a silent clamp would mean the loop ran with a
        # limit nobody asked for — worse than an obviously large number.
        overrides['max_daily_loss_fraction'] = float(daily)
    return config.with_overrides(**overrides) if overrides else config


async def main(argv: Optional[Sequence[str]] = None, *, gate=None) -> int:
    """The live loop. `argv` is explicit so `scripts.run_live` can compose it.

    `gate` is a `TradingGate` when this runs alongside the recorders in one
    process. The loop holds it across each cycle, so a recorder never starts
    work while a decision is in flight — the decision is the only
    latency-sensitive thing here, and a Parquet write on the event loop would
    land directly on it.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-14s %(message)s',
        datefmt='%H:%M:%S', stream=sys.stdout)

    if args.place_orders and args.mode != 'live':
        raise SystemExit('--place-orders requires --mode live')
    if args.force and not args.reason:
        raise SystemExit('--force needs --reason, and it is recorded on every row')

    if args.clear_halt:
        # Handled before the model is loaded, so a halt can be cleared on a box
        # with no artifact installed — otherwise recovering from a breaker would
        # require promoting a model first, which is absurd.
        #
        # The breakers are deliberately sticky: one that resets itself at midnight
        # is a speed bump rather than a breaker. But sticky with no way to clear it
        # is a trap, and that is what this was until now.
        if not args.reason:
            raise SystemExit(
                '--clear-halt needs --reason. The breaker fired for a cause, and '
                'clearing it without recording why you believe that cause is '
                'resolved is how the next one gets ignored too.')
        writer = PgWriter()
        account = writer.account()
        if account is None:
            raise SystemExit('no account row to clear')
        if not account.halted:
            print(f'account #{account.id} is not halted; nothing to clear '
                  f'(bankroll ${account.bankroll:.2f})')
            return 0
        previous = account.halted_reason
        writer.update_account(halted=False, halted_reason=None)
        logger.warning('halt CLEARED on account #%s. It had halted because: %s. '
                       'Reason given for clearing: %s',
                       account.id, previous, args.reason)
        print(f'cleared. it had halted because: {previous}')
        print(f'bankroll ${account.bankroll:.2f}, '
              f'realized ${account.realized_pnl:+.2f}')
        return 0

    config = config_from_args(args)

    # Load WITHOUT the config first. `verify(None)` still checks the things that
    # are true of the artifact alone — that the booster's columns match the
    # recorded feature list, and that the init source is one this can score —
    # but cannot compare against a configuration that does not yet know which
    # forecaster the artifact corrects. `config_for_artifact` settles that, and
    # the full check runs below against the config live will actually use.
    model = (load_live(config=None) if args.model is None
             else __import__('core.model', fromlist=['ForecastModel'])
             .ForecastModel.load(args.model, None))
    if model is not None:
        config = config_for_artifact(config, model, mode=args.mode)
        model.verify(config)
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
            # (window_open, offset) pairs already acted on, so a target that was
            # missed and fired late is not fired repeatedly for the whole grace
            # window. Bounded: pruned to the current and next window.
            fired_targets: set = set()
            # Per-window counters behind the heartbeat. Cheap, and the only proof
            # a quiet log gives that the loop is still turning.
            hb_window = None
            hb_cycles = hb_decisions = hb_traded = 0
            hb_lag = float('nan')
            while True:
                now = datetime.now(timezone.utc)
                window_open, _ = current_window(now, config)
                if heartbeat_due(window_open, hb_window):
                    if hb_window is not None:
                        logger.info(heartbeat_summary(
                            window_open=hb_window, cycles=hb_cycles,
                            decisions=hb_decisions, traded=hb_traded,
                            bankroll=float(writer.account().bankroll),
                            lag_seconds=hb_lag))
                    hb_window = window_open
                    hb_cycles = hb_decisions = hb_traded = 0
                    hb_lag = float('nan')
                hb_cycles += 1
                try:
                    if gate is not None:
                        async with gate.deciding():
                            decisions = await run_cycle(
                                args, config, writer, model, kalshi)
                    else:
                        decisions = await run_cycle(
                            args, config, writer, model, kalshi)
                except DatasetError as exc:
                    # One unscoreable cycle is not a reason to exit. This used to
                    # be fatal: `score_live` raised on every cycle and the loop
                    # caught only KeyboardInterrupt, so the process died and
                    # `restart: unless-stopped` crash-looped it forever.
                    logger.error('cycle skipped, nothing scored: %s', exc)
                    decisions = []
                if decisions:
                    counts = rejection_histogram(decisions)
                    noteworthy = any(d.traded or d.reason in LOUD_REFUSALS
                                     for d in decisions)
                    logger.log(logging.INFO if noteworthy else logging.DEBUG,
                               'cycle: %s', counts[counts > 0].to_dict())
                    hb_decisions += len(decisions)
                    hb_traded += sum(1 for d in decisions if d.traded)
                    # How late this cycle read the book, against the offset it
                    # was deciding. The last one in the window is representative.
                    hb_lag = (datetime.now(timezone.utc) - (
                        decisions[0].window_open.to_pydatetime()
                        + timedelta(minutes=int(decisions[0].offset)))
                    ).total_seconds()
                if not args.loop:
                    return 0
                # Before sleeping, record what this cycle actually decided —
                # otherwise the planner refires an offset the cadence already
                # traded, and the venue refuses the duplicate order_id.
                mark_decided(fired_targets, decisions)
                await asyncio.sleep(seconds_until_next_decision(
                    config, args, already_fired=fired_targets))
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
