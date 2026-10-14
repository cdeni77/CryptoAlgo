"""The live engine: score the current window, price it against the book, act.

Each cycle, in this order:

1. Fetch bars. On an entry cycle the newest minutes come from the Coinbase
   trade stream where it verifies them, else from REST.
2. Bookkeeping: reconcile with the venue, settle, record minute prices. On an
   entry cycle this runs after the orders.
3. Score the current window through the `core.dataset` path the backtest uses.
4. Price it against the stream book or one REST `/orderbook` read. Without a
   venue, price against the calibrated baseline; the row says which.
5. Decide with `core.decide.decide` and send every order at once. Resend once
   after a kill.

Modes:

    --mode paper                 score, price, record. Places nothing.
    --mode live --dry-run        read the real book, size the order, write the
                                 ticket, place nothing.
    --mode live --place-orders   place orders.

`--place-orders` is separate from `--mode live`, so one typo cannot place
orders. `--require-gates` (the default) refuses an artifact whose promotion
was blocked; overriding it needs `--force` and a `--reason`, recorded on every
row.

Live, the venue is the account of record. Balance comes from the venue each
cycle. Settlement comes from the venue where known, else from Coinbase bars as
a proxy for BRTI. Fills are read back, never assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core import heartbeat
from core.config import (                                          # noqa: F401
    Config, DEFAULT_CONFIG, SERIES_BY_SYMBOL, find_fee_config,
)
from core.dataset import score_live
from core.decide import (
    Decision, Reason, WindowExposure, decide, rejection_histogram,
)
from core.dataset import DatasetError
from core.pg_writer import AccountModeMismatch, PgWriter, TraderAlreadyRunning
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live
from data_collection.kalshi_client import KalshiClient

# The engine's parts live in `livetrade` and are re-exported here, so
# `from scripts.live import X` works. Module state is not re-exported, because
# a copy would go stale.
from livetrade import state
from livetrade.artifact import (                                    # noqa: F401
    _refuse_if_blocked, artifact_signature, artifact_version,
    config_for_artifact, reload_if_changed, stamp_version,
)
from livetrade.bars import (                                        # noqa: F401
    BAR_REFETCH_MINUTES, BAR_SETTLED_SECONDS, FETCH_MINUTES, fetch_bars,
    stream_bars,
)
from livetrade.execution import (                                   # noqa: F401
    OrderResult, act_on, client_order_id_for, diagnose_and_note, diagnose_kill,
    execute_entry, filled_from_order, kill_diagnosis,
    lookup_after_ambiguity, order_limit_price, place_and_book,
    sent_yes_price,
)
from livetrade.features import (                                    # noqa: F401
    EXPECTED_NAN, _attach_book_features, _warn_unscoreable_features,
    book_feature_row, config_offsets, cross_venue_row, gap_change,
    PEER_USED, implied_vol_row, latest_fit, peer_max_age_seconds,
    peer_refusal, prepare_init_score, prewarm_fit_lookup, reset_fit_lookup,
    reset_gap_history, take_peer_counts, warn_if_peer_cap_tight,
)
from livetrade.ledger import (                                      # noqa: F401
    PRICE_RETENTION_HOURS, VenueState, _write_equity_point, adopt_venue_balance,
    as_utc, bookkeeping, persist_venue_ledger, reconcile_with_venue,
    record_minute_prices, remember_exchange_index, resolve_window,
    settle_due, settle_predictions, venue_exchange_index, venue_settled_up,
)
from livetrade.quotes import (                                      # noqa: F401
    _record_touch, _stream_cache, depth_dollars, fetch_quotes, fresh_quote,
    ladder_from_cache, quote_age_seconds, quote_ladder, resolve_markets,
    stream_quote,
)
from livetrade.risk import (                                        # noqa: F401
    MIN_USABLE_BARS, check_circuit_breakers, short_symbols, stale_symbols,
)
from livetrade.util import _db

logger = logging.getLogger('live')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper')
    # Mutually exclusive at the parser: a safety flag is honoured or is a
    # usage error, never ignored.
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
    # Size off the running balance instead of the starting bankroll. Edge
    # measures are per contract, so they do not depend on this. An explicit
    # value beats the artifact's provenance (see `config_for_artifact`).
    parser.add_argument('--compound', action='store_true', default=None,
                        help='size off the running balance (default: off)')
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
    # Diagnostic levers. A flag keeps a forced run visible in the deploy config.
    # An explicit value beats the artifact's provenance: `config_for_artifact`
    # skips any field in `cli_overrides`.
    parser.add_argument('--require-peer-book', dest='require_peer_book',
                        action='store_true', default=None,
                        help='refuse a decision whose cross-venue peer book is '
                             'missing. `--complete-cases` required the peer on '
                             'every TRAINING row, so without this live scores '
                             'rows the artifact never saw. Settable nowhere '
                             'until now, despite config.py naming it "what '
                             'makes live agree".')
    parser.add_argument('--no-require-peer-book', dest='require_peer_book',
                        action='store_false')
    parser.add_argument('--require-model-version', default=None, metavar='VERSION',
                        help='place orders only while this artifact version is '
                             'loaded (as recorded in predictions.model_version); '
                             'rows from any other model are refused '
                             'model_not_pinned but still scored and recorded')
    parser.add_argument('--min-edge-pp', type=float, default=None, metavar='PP',
                        help='Override the edge gate, in probability points '
                             '(promoted policy: 3.0). A NEGATIVE value admits '
                             'every decision and is a fill test, not a strategy: '
                             'it trades known-negative expectancy on purpose so '
                             'the order path can be observed. Kelly still floors '
                             'marginal trades under one contract, so this does '
                             'not guarantee an order.')
    parser.add_argument('--max-stake-dollars', type=float, default=None, metavar='D',
                        help='Hard ceiling on one position, in dollars. It stands '
                             'in for market DEPTH, so it is a claim about the book '
                             'rather than about risk appetite — the measured BTC '
                             '15-minute ladder held thousands of contracts within '
                             'a few cents and 441 at the touch (~$48). Needed as a '
                             'flag because `config_for_artifact` otherwise adopts '
                             "the promoting run's value, so changing the default "
                             'alone would not reach a running loop.')
    parser.add_argument('--kelly-fraction', type=float, default=None, metavar='F',
                        help='Override the Kelly fraction (promoted policy: 0.10). '
                             'Present because it is not only a size: `decide()` '
                             'floors the stake to whole contracts, so a small '
                             'fraction pushes marginal trades under one contract '
                             'and refuses them as `below_min_contracts`. Raising '
                             'it is therefore how a forced run gets a small-edge '
                             'decision to actually place an order. Every other '
                             'script already takes this flag via `_common.py`; '
                             'live was the only one that could not say it.')
    parser.add_argument('--max-positions-per-window', type=int, default=None,
                        metavar='N',
                        help='How many of the three symbols may open a position '
                             'in one window (default 3). 1 restricts a forced run '
                             'to a single order per window — the first symbol '
                             'alphabetically — without narrowing `symbols`, which '
                             'would also narrow the bar fetch and NaN out every '
                             'cross_asset feature.')
    parser.add_argument('--model', type=str, default=None,
                        help=f'Artifact path (default {MODELS_ROOT}/{LIVE_MODEL})')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def current_window(now: datetime, config: Config) -> tuple[pd.Timestamp, int]:
    """Return the window being decided and the whole minutes elapsed in it."""
    stamp = pd.Timestamp(now).tz_convert('UTC').floor(f'{config.window_minutes}min')
    elapsed = int((pd.Timestamp(now).tz_convert('UTC') - stamp).total_seconds() // 60)
    return stamp, elapsed


def choose_offset(elapsed: int, config: Config) -> Optional[int]:
    """Return the latest configured offset the clock has reached, or None.

    Never a future offset, because its bar does not exist yet.
    """
    reached = [o for o in sorted(config.decision_offsets) if o <= elapsed]
    return reached[-1] if reached else None


# Wake just after the offset, never before it: `choose_offset` picks the largest
# offset at or below `elapsed`, so an early wake selects the previous offset.
DECISION_LAG_SECONDS = 1.0


# How far past an offset instant a missed target still fires at once. Wider
# than one cycle, narrower than the gap between offsets.
MISSED_TARGET_GRACE_SECONDS = 60.0


# How close to an offset instant a cycle must be to decide. This is a latency
# budget: a late decision reads the market's own move as edge. It must stay
# below the 60s cadence, so the next routine cycle cannot decide the same
# offset late.
DECISION_TOLERANCE_SECONDS = 15.0


# Refusals that indicate a fault, not a gate doing its job. These log at INFO.
LOUD_REFUSALS = frozenset({
    Reason.HALTED, Reason.BANKROLL_FLOOR, Reason.DISAGREEMENT_IMPLAUSIBLE,
    Reason.NO_QUOTE, Reason.PROBABILITY_INVALID, Reason.NOT_FINITE,
    Reason.MODEL_NOT_PINNED,
})


def decision_log_level(decision: Decision) -> int:
    """Return the log level for one decision: INFO if traded or a fault, else DEBUG."""
    if decision.traded:
        return logging.INFO
    if decision.reason in LOUD_REFUSALS:
        return logging.INFO
    return logging.DEBUG


def heartbeat_due(window_open, last_window) -> bool:
    """Return whether a new window started, so one heartbeat line is due.

    The line proves the loop still turns when the log is otherwise quiet.
    """
    return last_window is None or window_open != last_window


def heartbeat_summary(*, window_open, cycles: int, decisions: int, traded: int,
                      bankroll: float, lag_seconds: float,
                      peer: Optional[dict] = None) -> str:
    """Return the heartbeat line for one window.

    `peer` is `take_peer_counts` for the window.
    """
    lag = f'{lag_seconds:.1f}s' if lag_seconds == lag_seconds else 'n/a'
    line = (f'window {window_open:%H:%M} closed: {cycles} cycles, '
            f'{decisions} decisions, {traded} traded, bankroll ${bankroll:.2f}, '
            f'decision lag {lag}')
    if peer:
        used = int(peer.get(PEER_USED, 0))
        refused = {k: int(v) for k, v in peer.items() if k != PEER_USED and v}
        line += f', peer used {used}/{used + sum(refused.values())}'
        if refused:
            line += ' refused ' + ' '.join(
                f'{k}={v}' for k, v in sorted(refused.items()))
    return line


def decision_offset(elapsed: float, config: Config,
                    forced: Optional[int] = None) -> Optional[int]:
    """Return the offset this cycle may decide at, or None to settle only.

    A cycle decides only within `DECISION_TOLERANCE_SECONDS` of an offset.
    Cycles between offsets only settle and reconcile. An explicit `--offset`
    is always honoured.
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
    """Add every (window, offset) a cycle decided to `fired_targets`.

    Both the cadence and the planner can start a decision, so the planner must
    skip a target the cadence already decided. A refused decision counts as
    fired: a retry reuses the same `client_order_id` and gets a 409.
    """
    for decision in decisions:
        window = decision.window_open
        if not isinstance(window, pd.Timestamp):
            window = pd.Timestamp(window)
        fired_targets.add((window, int(decision.offset)))


def seconds_until_next_decision(config: Config, args, *,
                                now: Optional[datetime] = None,
                                already_fired: Optional[set] = None) -> float:
    """Return the seconds to sleep until the next decision offset.

    Targets `window_open + offset + DECISION_LAG_SECONDS`, because features are
    built for the nominal offset and a late book read gives the market extra
    information. Never sleeps past `--cycle-seconds`, so settlement keeps its
    cadence.
    """
    now = now or datetime.now(timezone.utc)
    window_open, _ = current_window(now, config)
    horizon = float(getattr(args, 'cycle_seconds', 60) or 60)
    if already_fired is not None:
        # Keep only this window and the next; older targets cannot fire again.
        keep = {window_open, window_open + timedelta(minutes=config.window_minutes)}
        already_fired.intersection_update(
            {k for k in already_fired if k[0] in keep})
    candidates = []
    for window in (window_open, window_open + timedelta(minutes=config.window_minutes)):
        for offset in config.decision_offsets:
            target = window + timedelta(minutes=offset) + timedelta(seconds=DECISION_LAG_SECONDS)
            delay = (target - now).total_seconds()
            # A target just behind us fires now, exactly once. A slow cycle can
            # overshoot its target, and skipping it would decide late at the
            # next offset. `already_fired` stops a second firing within the
            # grace window.
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
    # Never sleep past the ordinary cadence, so settlement and reconciliation
    # do not wait for the next offset.
    return float(min(min(candidates), horizon))


def entry_cycle(offset: Optional[int], config: Config) -> bool:
    """Return whether this cycle may open a position."""
    if offset is None:
        return False
    entries = config.entry_offsets
    return entries is None or int(offset) in tuple(entries)


def _write_predictions(writer: PgWriter, items: list) -> None:
    for fields in items:
        writer.write_prediction(**fields)


async def run_cycle(args, config: Config, writer: PgWriter, model,
                    kalshi: Optional[KalshiClient]) -> list[Decision]:
    """Run one cycle.

    On an entry cycle the decision goes first and bookkeeping runs after the
    orders. Bookkeeping reads only the past, and earlier cycles in the window
    have already done it. Sizing uses `account.bankroll`, the venue balance
    adopted last cycle. If no reconcile succeeded within
    `max_reconcile_age_seconds`, the cycle reconciles first. Circuit breakers
    always run before the decision.
    """
    state._RUNNING_CONFIG = config
    now = datetime.now(timezone.utc)
    window_open, elapsed = current_window(now, config)
    offset = decision_offset(elapsed, config, forced=args.offset)
    settle_time = window_open + pd.Timedelta(minutes=config.window_minutes)

    # Time per phase. The interval from book read to order decides fills.
    phase: dict[str, float] = {}
    _mark = time.perf_counter()

    def _phase(name: str) -> None:
        nonlocal _mark
        now_ = time.perf_counter()
        phase[name] = now_ - _mark
        _mark = now_

    fast = entry_cycle(offset, config)
    if (fast and kalshi is not None and getattr(args, 'reconcile', False)
            and (state._LAST_RECONCILED is None
                 or time.monotonic() - state._LAST_RECONCILED > config.max_reconcile_age_seconds)):
        logger.warning('the venue has not been reconciled within %.0fs; '
                       'reconciling before deciding', config.max_reconcile_age_seconds)
        fast = False

    # Bars before the book, so only the book's age is paid at the touch. An
    # entry cycle prefers the trade stream, which keeps REST off this path.
    bars = await (fetch_bars(config, prefer_stream=True) if fast
                  else fetch_bars(config))
    _phase('bars')
    if not bars:
        logger.error('no bars, nothing to do this cycle')
        return []
    if not fast:
        await _db(record_minute_prices, writer, bars)
    deferred_prices = fast

    halted = await _db(check_circuit_breakers, writer, config, now=now)
    if halted:
        # Keep the offset: `decide()` refuses every row with `Reason.HALTED`,
        # and scores and quotes keep accruing for `market_benchmark`.
        logger.error('account halted (%s); scoring and recording continue, '
                     'no new entries', halted)

    # Bars must be fresh and complete for every symbol. A short universe changes
    # every cross-asset feature, so the whole cycle abstains.
    stale = dict(stale_symbols(bars, config, now=now))
    stale.update({sym: why for sym, why in short_symbols(bars, config).items()
                  if sym not in stale})
    if stale:
        for symbol, why in sorted(stale.items()):
            logger.error('%s: %s', symbol, why)
        logger.error('the universe is %d of %d symbols this cycle, so no decision '
                     'is made — a short universe redefines every cross-asset '
                     'feature rather than simply omitting one row',
                     len(config.symbols) - len(stale), len(config.symbols))
        offset, fast = None, False
    if offset is not None:
        try:
            await prewarm_fit_lookup()
        except Exception as exc:                              # noqa: BLE001
            logger.warning('implied-vol prewarm: %s', str(exc)[:120])
    _phase('local')

    if not fast:
        if deferred_prices:
            await _db(record_minute_prices, writer, bars)
        await bookkeeping(args, config, writer, kalshi, bars, phase=_phase)

    if offset is None:
        # Between decisions, refresh market metadata (status, shard,
        # floor_strike), so a decision never pays to resolve it.
        if kalshi is not None and elapsed >= 1:
            try:
                await resolve_markets(kalshi, config.symbols, settle_time,
                                      refresh=True)
            except Exception as exc:                          # noqa: BLE001
                logger.debug('market refresh: %s', str(exc)[:120])
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

    # The book: the stream cache, else one REST read per symbol. Each quote is
    # stamped at its own read.
    quotes: dict = {}
    if kalshi is not None:
        try:
            quotes = await fetch_quotes(kalshi, list(config.symbols), settle_time)
        except Exception as exc:              # noqa: BLE001 - the cycle still settles
            logger.error('could not read the book (%s); no decision this cycle', exc)
            quotes, offset = {}, None
    read_at = pd.Timestamp.now(tz='UTC')
    quote_times = {
        symbol: (pd.Timestamp(entry[0].book_time, unit='s', tz='UTC')
                 if getattr(entry[0], 'book_time', None) is not None else read_at)
        for symbol, entry in quotes.items()}
    _phase('quotes')
    remember_exchange_index(quotes)
    if quotes:
        logger.info('book: %s', ', '.join(
            f'{s} {getattr(q, "book_source", None) or "?"}'
            for s, (q, _) in sorted(quotes.items())))
    if offset is None:
        if fast:
            await bookkeeping(args, config, writer, kalshi, bars, phase=_phase)
            await _db(record_minute_prices, writer, bars)
        return []

    # Columns `_attach_book_features` fills later, declared so `score_live`
    # does not warn about them.
    from core.book_features import (
        CROSS_VENUE as _CV, IMPLIED_VOL as _IV,
        MARKET_PRICE as _MP, MARKET_STATE as _MS)
    from core.index_features import INDEX_PROXY as _IP
    deferred = tuple(_MS) + tuple(_MP) + tuple(_CV) + tuple(_IV) + tuple(_IP)
    scored = score_live(bars, model.scoring, config, deferred=deferred,
                        window_open=window_open, offset=offset,
                        groups=model.groups or None)

    scored['ask_up'] = [
        quotes[s][0].ask_for('up') if s in quotes else np.nan for s in scored['symbol']]
    scored['ask_down'] = [
        quotes[s][0].ask_for('down') if s in quotes else np.nan for s in scored['symbol']]
    scored['market_ticker'] = [
        quotes[s][1] if s in quotes else None for s in scored['symbol']]
    scored['exchange_index'] = [
        quotes[s][0].exchange_index if s in quotes else 0 for s in scored['symbol']]
    # `Quote.mid` is None on a one-sided book.
    scored['market_mid'] = [
        quotes[s][0].mid if s in quotes else np.nan for s in scored['symbol']]

    # Book features, from the snapshot the order is priced against.
    _attach_book_features(scored, quotes, _stream_cache(), config=config,
                          groups=model.groups or None)
    scored = prepare_init_score(scored, model)
    _phase('score')
    # Liveness, once the cycle has scored.
    heartbeat.touch('trade')

    _warn_unscoreable_features(scored, model,
                               offsets=tuple(config.decision_offsets or ()))
    for column, values in model.score(scored).items():
        scored[column] = values

    # The venue publishes its settlement strike as `floor_strike` when the
    # window opens. Record it; ours is a Coinbase proxy for BRTI.
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
        # Record the venue strike; never feed it to the model. Training builds
        # `strike`, `displacement` and the features derived from them from
        # Coinbase bars, so the venue strike would put the BRTI basis into
        # them. The venue strike is for settlement only.
        scored.loc[index, 'strike_source'] = 'venue'
        scored.loc[index, 'venue_strike'] = venue_strike
        scored.loc[index, 'venue_strike_drift_bps'] = drift_bps

    if 'strike_source' not in scored.columns:
        scored['strike_source'] = 'bars'
    scored['strike_source'] = scored['strike_source'].fillna('bars')

    # Depth at the touch caps the stake.
    for index, row in scored.iterrows():
        quote = quotes.get(row['symbol'], (None, None))[0]
        if quote is None:
            continue
        for side in ('up', 'down'):
            depth = quote.depth_dollars(side)
            if depth is not None:
                scored.loc[index, f'depth_{side}'] = depth

    account = await _db(writer.account)
    if account is None:
        raise RuntimeError('no account row; main() must call ensure_account first')
    bankroll = float(account.bankroll)

    # Seed exposure from what this window already holds, so the per-window
    # limits survive a new cycle, a new offset and a restart.
    entered, staked, n_entered = await _db(
        writer.entries_for_window,
        window_open.to_pydatetime() if hasattr(window_open, 'to_pydatetime') else window_open)
    base_exposure = WindowExposure(stake=staked, positions=n_entered,
                                   symbols_entered=entered)
    if entered:
        logger.info('window %s already holds %s ($%.2f); those symbols will refuse',
                    window_open, sorted(entered), staked)
    exposure = base_exposure
    decisions: list[Decision] = []
    rows: dict[str, object] = {}
    pending: list[tuple[Decision, object]] = []

    # Live prices come from the book or not at all. Without a quote, `decide()`
    # would price against our own baseline.
    require_quote = args.mode == 'live'
    # decide() compares this against `require_model_version`.
    scored['model_version'] = getattr(model, 'version', None)
    for _, row in scored.sort_values('symbol').iterrows():
        remaining = (settle_time - pd.Timestamp.now(tz='UTC')).total_seconds()
        if remaining < config.min_remaining_seconds:
            logger.warning(
                '%s: %.0fs left of the window, under the %ds floor — the sigma on '
                'this row is for %s and the clock has moved past it',
                row['symbol'], remaining, config.min_remaining_seconds, settle_time)
            break
        quote = quotes.get(row['symbol'], (None, None))[0]
        decision = decide(
            row, config, bankroll=bankroll, exposure=exposure,
            require_quote=require_quote, halted=bool(halted))
        decisions.append(decision)
        rows[decision.symbol] = row
        logger.log(decision_log_level(decision), decision.describe())
        if decision.traded:
            # Reserve exposure before sending: all orders are in flight at once.
            pending.append((decision, row))
            exposure = exposure.with_(decision)

    # The decision instant, which every archived quote age is measured to.
    decided_at = pd.Timestamp.now(tz='UTC')
    model_version = getattr(model, 'version', None)

    def record_for(decision: Decision) -> dict:
        row = rows[decision.symbol]
        return dict(
            symbol=decision.symbol, window_open=window_open, settle_time=settle_time,
            offset_minutes=offset,
            decision_time=quote_times.get(decision.symbol, read_at),
            strike=float(row['strike']), last_price=float(row['last_price']),
            displacement=float(row['displacement']),
            sigma_remaining=_finite(row.get('sigma_remaining')),
            z_score=_finite(row.get('z_score')),
            baseline_probability=float(row['baseline_probability']),
            model_probability=float(row['model_probability']),
            # The market mid and both asks on every row, refused ones too, so
            # the market benchmark sample is unselected.
            market_probability=_finite(row.get('market_mid')),
            market_ask_up=_finite(row.get('ask_up')),
            market_ask_down=_finite(row.get('ask_down')),
            price_source=decision.price_source,
            reason=decision.reason.value, traded=decision.traded,
            side=decision.side.value if decision.side else None,
            price=_finite(decision.price), effective_cost=_finite(decision.effective_cost),
            edge=_finite(decision.edge), contracts=decision.contracts or None,
            model_version=model_version,
        )

    # Send every order at once. Each traded row's prediction is written first,
    # so its ticket and position carry the prediction id.
    async def _one(index: int):
        decision, row = pending[index]
        others = base_exposure
        for j, (other, _) in enumerate(pending):
            if j != index:
                others = others.with_(other)
        try:
            prediction_id = await _db(writer.write_prediction, **record_for(decision))
        except Exception:                                     # noqa: BLE001
            logger.exception('%s: could not write the prediction before the '
                             'order; the ticket will carry no prediction_id',
                             decision.symbol)
            prediction_id = None
        return await execute_entry(
            args, writer, kalshi, decision, row, config=config,
            quote=quotes.get(decision.symbol, (None, None))[0],
            quote_time=quote_times.get(decision.symbol, read_at),
            others=others, bankroll=bankroll, halted=bool(halted),
            require_quote=require_quote, prediction_id=prediction_id)

    outcomes = await asyncio.gather(*(_one(i) for i in range(len(pending))),
                                    return_exceptions=True)
    if pending:
        _phase('orders')
    finals: dict[str, Decision] = {}
    kills: list[tuple[Decision, OrderResult, bool]] = []
    for (decision, _), outcome in zip(pending, outcomes):
        if isinstance(outcome, BaseException):
            logger.error('%s: the order path raised (%s); see the traceback',
                         decision.symbol, outcome, exc_info=outcome)
            continue
        final_decision, final, killed = outcome
        finals[decision.symbol] = final_decision
        for d, r in killed:
            kills.append((d, r, r is final))

    if phase and pending:
        names = list(phase)
        after = names[names.index('quotes') + 1:] if 'quotes' in phase else names
        logger.info('cycle latency: %s | %.2fs between the book and the orders',
                    ' '.join(f'{k} {v:.2f}s' for k, v in phase.items()),
                    sum(phase[k] for k in after))

    # ---- after the orders: everything that only records -------------------
    decisions = [finals.get(d.symbol, d) for d in decisions]
    # Write every row. Traded rows upsert on (symbol, window, offset), which
    # keeps their id and records a resend's final price.
    records = [record_for(d) for d in decisions]
    try:
        await _db(_write_predictions, writer, records)
    except Exception:                                         # noqa: BLE001
        logger.exception('could not write this cycle\'s predictions')

    if kills:
        await asyncio.gather(*(
            diagnose_and_note(kalshi, writer, d, r, write_note=last)
            for d, r, last in kills), return_exceptions=True)

    # Archive the book already read, after the orders.
    if os.getenv('RECORD_TOUCH', '1') == '1' and quotes:
        await _record_touch(scored, quotes, window_open, offset, config, kalshi,
                            decided_at=decided_at)
        _phase('touch')

    if fast:
        await bookkeeping(args, config, writer, kalshi, bars, phase=_phase)
        await _db(record_minute_prices, writer, bars)

    await _db(_write_equity_point, writer, now)
    return decisions


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def config_from_args(args) -> Config:
    """Build the run's configuration from the command line.

    `entry_offsets` is set here, not defaulted in `Config`, because
    `scripts.evaluate` must keep measuring every offset.
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
        # Not clamped: a silent clamp would run a limit nobody asked for.
        overrides['max_daily_loss_fraction'] = float(daily)
    peer = getattr(args, 'require_peer_book', None)
    if peer is not None:
        overrides['require_peer_book'] = bool(peer)
    pinned = getattr(args, 'require_model_version', None)
    if pinned:
        overrides['require_model_version'] = str(pinned)
    gate = getattr(args, 'min_edge_pp', None)
    if gate is not None:
        # Not clamped at zero: a negative gate opens the order path for a fill
        # test.
        overrides['min_edge_pp'] = float(gate)
    cap = getattr(args, 'max_stake_dollars', None)
    if cap is not None:
        overrides['max_stake_dollars'] = float(cap)
    kelly = getattr(args, 'kelly_fraction', None)
    if kelly is not None:
        overrides['kelly_fraction'] = float(kelly)
    cap = getattr(args, 'max_positions_per_window', None)
    if cap is not None:
        overrides['max_positions_per_window'] = int(cap)
    # An explicit `--compound` must reach `cli_overrides`, or the artifact's
    # provenance wins in `config_for_artifact`.
    compound = getattr(args, 'compound', None)
    if compound is not None:
        overrides['compound'] = bool(compound)
    return config.with_overrides(**overrides) if overrides else config


async def main(argv: Optional[Sequence[str]] = None, *, gate=None) -> int:
    """Run the live loop. `argv` is explicit so `scripts.run_live` can compose it.

    `gate` is a `TradingGate` shared with the recorders in one process. The
    loop holds it across each cycle, so no recorder starts work during a
    decision.
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
        # Handled before the model loads, so a halt clears without an artifact.
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
    # The command-line config before artifact adoption. A hot reload
    # re-derives the trading config from it.
    base_config = config
    artifact = Path(args.model) if args.model else Path(MODELS_ROOT) / LIVE_MODEL
    signature = artifact_signature(artifact)

    # Load without the config first: the init source is not known until
    # `config_for_artifact` adopts it. The full `verify` runs after that.
    model = (load_live(config=None) if args.model is None
             else __import__('core.model', fromlist=['ForecastModel'])
             .ForecastModel.load(args.model, None))
    if model is not None:
        config = config_for_artifact(config, model, mode=args.mode)
        model.verify(config)
        stamp_version(model, artifact)
        pinned = getattr(config, 'require_model_version', None)
        if pinned and getattr(model, 'version', None) != pinned:
            logger.warning('the loaded model is %s but orders are pinned to %s: '
                           'every decision will be refused model_not_pinned',
                           getattr(model, 'version', None), pinned)
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
    warn_if_peer_cap_tight(config)

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
            # (window_open, offset) pairs already decided, so a late target
            # fires once. Pruned to the current and next window.
            fired_targets: set = set()
            # Per-window counters for the heartbeat line.
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
                            lag_seconds=hb_lag,
                            peer=take_peer_counts(hb_window)))
                        # Reload a newly promoted artifact at the first cycle of
                        # a window, when nothing is in flight.
                        model, config, signature = await reload_if_changed(
                            args, base_config, artifact, model, config, signature)
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
                    # One unscoreable cycle skips that cycle; it does not exit.
                    logger.error('cycle skipped, nothing scored: %s', exc)
                    decisions = []
                except Exception:                     # noqa: BLE001
                    # Any other failure costs only this cycle. CancelledError
                    # and KeyboardInterrupt still stop the loop.
                    logger.exception('cycle failed; continuing with the next')
                    decisions = []
                if decisions:
                    counts = rejection_histogram(decisions)
                    noteworthy = any(d.traded or d.reason in LOUD_REFUSALS
                                     for d in decisions)
                    logger.log(logging.INFO if noteworthy else logging.DEBUG,
                               'cycle: %s', counts[counts > 0].to_dict())
                    hb_decisions += len(decisions)
                    hb_traded += sum(1 for d in decisions if d.traded)
                    # Decision lag against the offset; the window's last value
                    # is reported.
                    hb_lag = (datetime.now(timezone.utc) - (
                        decisions[0].window_open.to_pydatetime()
                        + timedelta(minutes=int(decisions[0].offset)))
                    ).total_seconds()
                if not args.loop:
                    return 0
                # Record what this cycle decided, so the planner does not refire
                # it with a duplicate order id.
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


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
