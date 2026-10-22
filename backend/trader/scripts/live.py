"""The live engine: score the current window, price it against the book, act.

One script for the whole operational loop, because the previous incarnation of
this project had four (`signals`, `paper_engine`, `live_orchestrator`, and a
promotion cadence inside the orchestrator) and they disagreed about what had
already happened.

Each cycle, in this order:

1. **Fetch bars.** The last day of one-minute bars for the three symbols. On an
   entry cycle the newest minutes come from the Coinbase trade stream where it
   provably saw them all (`stream_bars`), and from REST otherwise.
2. **Bookkeeping** — reconcile with the venue, settle matured positions, record
   minute prices. On a cycle that can ENTER, this moves after the orders: it
   reads the past, every earlier cycle in the window has already done it, and
   on the decision path it was seconds of staleness paid at the touch.
3. **Score** the current window at the configured offset, through the same
   `core.dataset` path the backtest uses.
4. **Price** it. Live, against the stream book or one REST `/orderbook` read
   per symbol; without a venue, against the calibrated baseline, and the row
   says which.
5. **Decide** with `core.decide.decide` — the same function the backtest calls —
   and send every order at once, resending once after a kill.

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
from core.costs import trade_fee
from core.promotion import LIVE_MODEL, MODELS_ROOT, load_live
from data_collection.kalshi_client import KalshiClient

# The engine's parts live in `livetrade`. Everything `run_cycle` and `main` call
# is imported here, and so is every public name this module used to define, so
# `from scripts.live import X` keeps working. Module state is NOT re-exported:
# it has one owner each (`livetrade.state` for what is shared), and a copy here
# would silently go stale.
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
    book_feature_row, book_flow_row, config_offsets, cross_venue_row,
    gap_change,
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
    _record_touch, _stream_cache, cached_floor_strikes, depth_dollars,
    fetch_quotes, fresh_quote,
    ladder_from_cache, quote_age_seconds, quote_ladder, resolve_markets,
    stream_quote,
)
from livetrade.risk import (                                        # noqa: F401
    MIN_USABLE_BARS, check_circuit_breakers, short_symbols, stale_symbols,
)
from livetrade.util import _db

logger = logging.getLogger('live')

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
    # **Maker entry: rest at the bid instead of lifting the ask.**
    # Measured 2026-10-18 on 54,100 live-recorded rows, the SAME model with no
    # retraining: taker -$77.25 (-0.108pp/ct, t=-0.17) against maker +$2445.76
    # (+1.608pp/ct, t=+2.91), and +$3781.70 (t=+4.74) also excluding 35-65c.
    # The fee is only half of it; buying at the bid on a 1.000c spread is worth
    # as much again. OFF by default because every one of those numbers assumes
    # a resting order FILLS, and none ever has on this account — all 1,556
    # recorded fills are `is_taker: true`.
    # Must mirror the `retrain` service's own flag exactly, or the gates score a
    # policy the account does not run. It was added to the trader's override on
    # 2026-10-18 and `scripts/live.py` had no such argument, so the service
    # exited with `SystemExit 2` before placing anything — caught before the
    # container was recreated.
    parser.add_argument('--exclude-price-band', type=str, default=None,
                        metavar='LO,HI',
                        help='Refuse trades priced inside this band, e.g. '
                             '"0.35,0.65". Symmetric about 0.5. The fee '
                             '0.07*p(1-p) peaks mid-book while the measured '
                             'signal there is ~0.')
    parser.add_argument('--maker-entry', action='store_true',
                        help='Rest at the bid rather than lifting the ask. '
                             'decide() is priced at the resting price, so the '
                             'edge it reports is the one actually available.')
    parser.add_argument('--maker-time-in-force', default=None,
                        help="The venue's enum for an order that rests. "
                             'Omitted by default, which lets the venue apply '
                             'its own default. Not guessed here: a wrong value '
                             'may be rejected (harmless) or silently fill as a '
                             'TAKER (not).')
    parser.add_argument('--maker-cancel-seconds', type=float, default=60.0,
                        help='Pull an unfilled resting order after this many '
                             'seconds of its OWN age. The forecast is what '
                             'expires: sigma_remaining is computed for the '
                             'decision instant, so a fill arriving minutes '
                             'later takes an edge that has moved. 0 never '
                             'cancels, which is only reasonable while '
                             'measuring whether fills happen at all.')
    parser.add_argument('--loop', action='store_true',
                        help='Run every cycle-seconds until interrupted.')
    parser.add_argument('--cycle-seconds', type=int, default=60)
    # Size off the running balance rather than the starting bankroll.
    #
    # Measured on the promoted config over the out-of-sample span: return 1.319
    # -> 2.497 with FRACTIONAL drawdown essentially unchanged (0.242 -> 0.244),
    # because drawdown is a ratio and peak and trough scale together. Sharpe
    # falls 3.72 -> 3.45, so the extra return is bought with proportionally more
    # risk, and in dollars the same 24% is far more money.
    #
    # It does NOT affect how the edge is measured: `model_minus_market` is a
    # log-loss comparison on predictions, `realised_edge_pp` is per contract,
    # and daily return-on-cost is a ratio. All three are sizing-independent.
    # What it costs is the equity curve's slope no longer being the per-trade
    # edge.
    #
    # Explicit here beats the artifact's provenance — see `config_for_artifact`.
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
    # **Diagnostic levers.** Both exist so a forced run is visible in the deploy
    # config and in `ps`, and reverting it is deleting a line — the same reason
    # `--max-daily-loss-fraction` is a flag rather than an edit to the default.
    # Explicit here beats the artifact's provenance: `config_for_artifact` skips
    # any field in `cli_overrides`.
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
    parser.add_argument('--max-entries-per-symbol-window', type=int, default=None,
                        help='entries allowed per (symbol, window). Default 1 — '
                             'the four offsets are one bet seen four times, so '
                             'above 1 this is a real increase in correlated '
                             'exposure. Raise --max-positions-per-window with '
                             'it or the 3-slot window cap just reallocates.')
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
    Reason.MODEL_NOT_PINNED,
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
                      bankroll: float, lag_seconds: float,
                      peer: Optional[dict] = None) -> str:
    """The things worth checking, on one line.

    `peer` is `take_peer_counts` for the window: how many peer books were used
    of those read, and the refusals by reason.
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


def entry_cycle(offset: Optional[int], config: Config) -> bool:
    """Whether this cycle may open a position, so the decision goes first."""
    if offset is None:
        return False
    entries = config.entry_offsets
    return entries is None or int(offset) in tuple(entries)


def _write_predictions(writer: PgWriter, items: list) -> None:
    for fields in items:
        writer.write_prediction(**fields)


async def book_resting_fills(writer: PgWriter, kalshi, config: Config) -> int:
    """Book positions for resting orders the venue has filled.

    **The half the `resting` status was missing.** Marking a maker order
    `resting` instead of `killed` stopped it being written off — but nothing
    then moved it to `filled`, so the fill was simply lost. Measured
    2026-10-18: the venue held 35 contracts from 30 fills while our `positions`
    table showed 6. The cancel sweep said so every minute without anyone
    reading it: every DELETE came back `404 not_found`, because the order was
    gone from the book for the one reason the code never considered — it had
    already filled.

    That matters past the bookkeeping. `exposure`,
    `max_positions_per_window` and `max_window_exposure_fraction` are computed
    from OUR book, so a position it cannot see is a position no risk limit
    binds on.

    Idempotent: `open_position` is keyed on (symbol, window_open), so a fill
    seen twice books once.
    """
    if not getattr(config, 'maker_entry', False) or kalshi is None:
        return 0
    tickets = [t for t in writer.open_tickets(limit=100)
               if str(getattr(t, 'status', '')) == 'resting'
               and getattr(t, 'venue_order_id', None)]
    if not tickets:
        return 0
    try:
        raw = await kalshi.fills(limit=200)
    except Exception as exc:                                   # noqa: BLE001
        logger.info('could not read fills to reconcile resting orders (%s); '
                    'they stay resting and will be retried', str(exc)[:120])
        return 0
    by_order: dict[str, list[dict]] = {}
    for row in raw:
        oid = str(row.get('order_id') or '')
        if oid:
            by_order.setdefault(oid, []).append(row)

    booked = 0
    for ticket in tickets:
        rows = by_order.get(str(ticket.venue_order_id))
        if not rows:
            continue
        side = str(ticket.side)
        # The price for the side we BOUGHT. V2 quotes from the YES side, so a
        # NO fill's cost is 1 - the yes price; reading the wrong one books a
        # 30c purchase as 70c.
        prices, count = [], 0.0
        for row in rows:
            n = float(row.get('count_fp') or row.get('count') or 0) or 0.0
            yes = row.get('yes_price_dollars') or row.get('yes_price')
            try:
                yes = float(yes)
            except (TypeError, ValueError):
                continue
            if yes > 1.0:                      # integer cents, older encoding
                yes /= 100.0
            prices.append((yes if side == 'up' else 1.0 - yes) * n)
            count += n
        if count <= 0 or not prices:
            continue
        # **A sub-contract partial is not a fill yet.** `contracts` is an
        # integer column, so `int(round(count))` turned 0.4 filled contracts
        # into a position of 0 -- real money spent, nothing recorded, and a
        # ticket closed so the rest of the order could never be booked.
        # Measured live: position 1133 held `contracts=0, outlay=$0.07`
        # against a venue fill of 1 contract and -$0.70. Leaving the ticket
        # resting costs nothing: the next cycle sees the larger total, and
        # `cancel_stale_resting` retires it if it never completes.
        if int(round(count)) < 1:
            logger.info('%s window %s: %.2f contract(s) filled so far, under '
                        'one; leaving the ticket resting rather than booking '
                        'a zero-contract position',
                        ticket.symbol, ticket.window_open, count)
            continue
        price = sum(prices) / count
        fee = float(trade_fee(int(round(count)), price, config))
        outlay = count * price + fee
        # **`positions` requires the forecast columns, not just the money.**
        # model_probability, baseline_probability and edge are all NOT NULL, so
        # booking without them raises NotNullViolation and the whole reconcile
        # aborts — which is how the first version of this silently did nothing
        # while logging "the resting-order sweep raised" once a minute.
        #
        # The ticket carries the model's own number and the edge it was sized
        # on. The baseline comes from the prediction that produced the ticket;
        # where that is missing, the MARKET price is the honest stand-in,
        # because a resting order's price IS what the book offered.
        model_p = getattr(ticket, 'model_probability', None)
        edge = getattr(ticket, 'edge', None)
        baseline_p = None
        pid = getattr(ticket, 'prediction_id', None)
        if pid is not None:
            pred = writer.prediction(pid) if hasattr(writer, 'prediction') else None
            if pred is not None:
                model_p = model_p if model_p is not None else getattr(
                    pred, 'model_probability', None)
                baseline_p = getattr(pred, 'baseline_probability', None)
                edge = edge if edge is not None else getattr(pred, 'edge', None)
        if model_p is None:
            model_p = price
        if baseline_p is None:
            baseline_p = price
        if edge is None:
            edge = float(model_p) - price
        await _db(writer.open_position,
                  symbol=ticket.symbol, window_open=ticket.window_open,
                  settle_time=ticket.settle_time,
                  offset_minutes=ticket.offset_minutes, side=side,
                  contracts=int(round(count)), price=price,
                  outlay=outlay, fee=fee,
                  model_probability=float(model_p),
                  baseline_probability=float(baseline_p),
                  edge=float(edge),
                  prediction_id=pid)
        await _db(writer.adjust_account, bankroll_delta=-outlay, fees_delta=fee)
        await _db(writer.resolve_ticket, ticket.id, status='filled',
                  filled_contracts=int(round(count)), filled_price=price,
                  note=f'resting order filled at {price:.4f} ({count:.2f} '
                       f'contracts), booked by the reconcile')
        logger.info('%s: resting order filled %.2f @ %.4f — position booked',
                    ticket.symbol, count, price)
        booked += 1
    if booked:
        logger.info('booked %d position(s) from resting orders the venue filled',
                    booked)
    return booked


async def cancel_stale_resting(writer: PgWriter, kalshi, config: Config,
                               now=None) -> int:
    """Cancel resting orders that have outlived the forecast behind them.

    **An immediate-or-cancel order is over the instant it is sent; a resting one
    is not.** It sits until it fills, the market closes, or somebody cancels it
    — and on a fifteen-minute market an order left from offset 3 can fill at
    offset 14 against a barrier probability that has almost entirely resolved.
    That is the real objection to resting as a policy, and it is why
    `place_order` defaults to immediate_or_cancel. This is what makes resting
    safe to switch on: the order's life is bounded by the forecast's.

    **The leash is the order's own AGE, not an offset into the window.** This
    keyed on `window_open` and a fixed minute, which gave an order placed at
    +3m a six-minute life and one placed at +12m none at all — one flag meaning
    two different leashes depending on when it was sent. `created_at` gives
    every order the same one.

    Inert unless `maker_entry` is on, and a non-positive `maker_cancel_seconds`
    disables it — which is only reasonable while measuring whether fills happen
    at all, and leaves orders resting to settlement.
    """
    if not getattr(config, 'maker_entry', False):
        return 0
    cutoff = getattr(config, 'maker_cancel_seconds', None)
    if cutoff is None or float(cutoff) <= 0:
        return 0
    now = now or pd.Timestamp.now(tz='UTC')
    cancelled = 0
    vanished = 0
    for ticket in writer.open_tickets(limit=100):
        order_id = getattr(ticket, 'venue_order_id', None)
        if not order_id:
            continue
        sent = getattr(ticket, 'created_at', None)
        if sent is None:
            # No send time is not "old enough to cancel". Leaving it is the
            # conservative error: the reconcile still finds any position, and
            # cancelling an order we cannot age could kill a fresh one.
            continue
        sent = pd.Timestamp(sent)
        if sent.tzinfo is None:
            sent = sent.tz_localize('UTC')
        age = (now - sent).total_seconds()
        if age < float(cutoff):
            continue
        try:
            await kalshi.cancel_order(str(order_id))
        except Exception as exc:                               # noqa: BLE001
            # **404 means the venue has no such order, so close the ticket.**
            # These markets settle every 15 minutes and the venue cancels what
            # is still resting, so an order disappears server-side as a matter
            # of course. Logging and retrying leaves the ticket `resting`, and
            # `open_tickets` then returns it on every later cycle forever.
            #
            # This sweep runs BEFORE the decision, between the book read and
            # `decided_at`, so each dead ticket adds a round trip to the quote
            # age the decision is judged on -- the cost falls on the forecast,
            # not on the bookkeeping.
            #
            # The status is the venue's own word for these orders, and the
            # note keeps a cancel we performed separate from one we merely
            # found. `book_resting_fills` has already run, so a fill is booked
            # before we get here, and the reconcile owns the position either
            # way.
            if getattr(exc, 'status', None) == 404:
                await _db(writer.resolve_ticket, ticket.id, status='cancelled',
                          note='the venue reported no such order; closed by '
                               'the sweep, reconcile owns any position')
                vanished += 1
                continue
            # Any other failure is transient as far as we can tell, so leave
            # the ticket open and try again next cycle.
            logger.info('could not cancel %s (%s); reconcile will settle it',
                        order_id, str(exc)[:100])
            continue
        await _db(writer.resolve_ticket, ticket.id, status='cancelled',
                  note=f'resting order cancelled after {age:.0f}s unfilled '
                       f'(maker_cancel_seconds={cutoff})')
        cancelled += 1
    if cancelled:
        logger.info('cancelled %d resting order(s) unfilled past %.0fs',
                    cancelled, float(cutoff))
    if vanished:
        logger.info('closed %d ticket(s) the venue no longer holds', vanished)
    return cancelled


async def run_cycle(args, config: Config, writer: PgWriter, model,
                    kalshi: Optional[KalshiClient]) -> list[Decision]:
    """One cycle. On an ENTRY cycle the decision goes first and the
    bookkeeping after the orders; on every other cycle the reverse.

    **What an entry cycle defers, and why it is safe.** Venue reconciliation
    (four authenticated calls), `settle_due`, `settle_predictions`, the balance
    adoption, minute prices and the equity point all read the past; none of it
    changes what this decision should be. Positions settle at window
    boundaries, and entries are at +12m, so every cycle since the boundary has
    already settled and reconciled.

    **Sizing uses the last-known bankroll**: the venue balance the previous
    cycle adopted (at most a cycle old), less any stake debited since, which is
    what `account.bankroll` holds. Only a deposit or withdrawal in the last
    minute is unseen, and the ruin floor and breakers read the same figure. If
    the venue has not answered a reconcile within `max_reconcile_age_seconds`
    the cycle reconciles first, as before. The circuit breakers still run
    before the decision — they read our settled positions, not the venue.
    """
    state._RUNNING_CONFIG = config
    now = datetime.now(timezone.utc)
    window_open, elapsed = current_window(now, config)
    offset = decision_offset(elapsed, config, forced=args.offset)
    settle_time = window_open + pd.Timedelta(minutes=config.window_minutes)

    # Where the cycle spends its time, because the interval from reading the
    # book to sending the order decides whether a fill happens.
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

    # Bars, then the book: only the book's age is paid at the touch. On an
    # entry cycle the newest minutes come from the trade stream where it can
    # vouch for them, which takes the ~3s REST fetch off this path.
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
        # The offset is deliberately NOT cleared: `decide()` refuses every row
        # with `Reason.HALTED`, so quotes, scores and outcomes keep accruing for
        # `market_benchmark` while no money moves.
        logger.error('account halted (%s); scoring and recording continue, '
                     'no new entries', halted)

    # Fresh AND complete. A short universe redefines every cross-asset feature
    # rather than omitting one row, so the whole cycle abstains.
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
        # Between decisions: keep this window's market metadata current (status,
        # shard, floor_strike), so a decision never pays to resolve it.
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

    # **The book.** Stream cache where it can vouch for it, else one REST read
    # per symbol, concurrently. Each quote is stamped at its own read.
    quotes: dict = {}
    if kalshi is not None:
        try:
            quotes = await fetch_quotes(kalshi, list(config.symbols), settle_time)
        except Exception as exc:              # noqa: BLE001 - the cycle still settles
            logger.error('could not read the book (%s); no decision this cycle', exc)
            quotes, offset = {}, None
    read_at = pd.Timestamp.now(tz='UTC')
    # **Mark the open positions while the book is in hand.** These markets
    # settle every fifteen minutes, so anything still pending belongs to the
    # window just quoted -- no extra request, no second clock. The bid for the
    # side held, because that is where a sale lands; marking to our own
    # forecast would book conviction as profit, which is why `at risk` on the
    # dashboard has always been cost.
    #
    # Wrapped, and deliberately after the quotes rather than inside the
    # decision path: this is telemetry, and a cycle that cannot write telemetry
    # must still trade and settle.
    if quotes:
        try:
            marks = {}
            for symbol, entry in quotes.items():
                quote = entry[0]
                for side in ('up', 'down'):
                    price = quote.bid_for(side)
                    if price is not None:
                        marks[(symbol, side)] = float(price)
            by_position = {}
            for position in await _db(writer.open_positions):
                price = marks.get((str(position.symbol), str(position.side)))
                if price is not None:
                    by_position[(str(position.symbol), position.window_open)] = price
            if by_position:
                await _db(writer.mark_open_positions, by_position,
                          at=read_at.to_pydatetime())
        except Exception:                         # noqa: BLE001
            logger.exception('could not mark the open positions this cycle')
    quote_times = {
        symbol: (pd.Timestamp(entry[0].book_time, unit='s', tz='UTC')
                 if getattr(entry[0], 'book_time', None) is not None else read_at)
        for symbol, entry in quotes.items()}
    # Before this cycle's decisions, retire any resting order from an earlier
    # one that has outlived its forecast. Inert unless `maker_entry` is on.
    if getattr(config, 'maker_entry', False) and kalshi is not None:
        try:
            # Fills first: an order that filled must be booked, not cancelled.
            await book_resting_fills(writer, kalshi, config)
            await cancel_stale_resting(writer, kalshi, config, now=read_at)
        except Exception:                                      # noqa: BLE001
            logger.exception('the resting-order sweep raised; continuing — a '
                             'stale order is a worse outcome than a noisy log, '
                             'but not worse than skipping the cycle entirely')
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

    # Everything `_attach_book_features` fills after this returns, declared so
    # the unscoreable-feature warning does not fire for columns not yet filled.
    from core.book_features import (
        CROSS_VENUE as _CV, IMPLIED_VOL as _IV,
        MARKET_PRICE as _MP, MARKET_STATE as _MS)
    from core.book_flow import BOOK_FLOW as _BF
    from core.index_features import INDEX_PROXY as _IP
    # `_BF` is attached by `book_flow_row` below, after this call. Leaving it
    # out made `reindex_to_features` warn about all six on EVERY cycle --
    # noise that reports a gap where there is none, and that buries the real
    # signal. `_warn_unscoreable_features` runs after the attach and is the
    # check that can still see a genuinely empty book_flow.
    deferred = (tuple(_MS) + tuple(_MP) + tuple(_CV) + tuple(_IV)
                + tuple(_IP) + tuple(_BF))
    # **The venue's strike, from the cache, BEFORE the features are built.**
    # `build_dataset` already does this for research (`strike_source=venue`);
    # the live path could not, so a model trained on venue strikes would have
    # been scored live on bar-derived ones -- the C6 divergence, and a silent
    # one, because the drift warning fires above 25bp while the basis is ~1bp.
    # `cached_floor_strikes` reads what the non-decision cycles already
    # refreshed, so this costs no round trip in the decision path.
    venue_strikes = {}
    if getattr(config, 'strike_source', 'proxy') == 'venue':
        venue_strikes = cached_floor_strikes(config.symbols, settle_time)
        missing = [s for s in config.symbols if s not in venue_strikes]
        if missing:
            # Falling back to the bar strike for SOME symbols is the mixed
            # state C6 forbids, so say so rather than let it pass.
            logger.warning(
                'strike_source=venue but the venue has published no strike for '
                '%s; those symbols fall back to the bar-derived strike and are '
                'scored on a different reference than the model was fitted on',
                ', '.join(sorted(missing)))
    scored = score_live(bars, model.scoring, config, deferred=deferred,
                        window_open=window_open, offset=offset,
                        groups=model.groups or None,
                        strikes_by_symbol=venue_strikes or None)

    # **The price `decide()` is handed must be the one we will actually pay.**
    # A taker lifts the ask; a MAKER rests at the bid. Handing decide() the ask
    # while resting at the bid would understate every edge by a full spread and
    # refuse trades that are profitable — the mirror of the defect that had the
    # live path adding a half-spread to a real ask. `Config.maker_entry` is off
    # by default, so this is the ask unless asked otherwise.
    _price_at = 'bid_for' if getattr(config, 'maker_entry', False) else 'ask_for'
    scored['ask_up'] = [
        getattr(quotes[s][0], _price_at)('up') if s in quotes else np.nan
        for s in scored['symbol']]
    scored['ask_down'] = [
        getattr(quotes[s][0], _price_at)('down') if s in quotes else np.nan
        for s in scored['symbol']]
    if getattr(config, 'maker_entry', False):
        # What is queued AHEAD of us at that price. Not a cap on our size the
        # way the ask's depth is — a reason to expect not to fill.
        scored['rest_queue_up'] = [
            quotes[s][0].rest_size_for('up') if s in quotes else np.nan
            for s in scored['symbol']]
        scored['rest_queue_down'] = [
            quotes[s][0].rest_size_for('down') if s in quotes else np.nan
            for s in scored['symbol']]
    scored['market_ticker'] = [
        quotes[s][1] if s in quotes else None for s in scored['symbol']]
    scored['exchange_index'] = [
        quotes[s][0].exchange_index if s in quotes else 0 for s in scored['symbol']]
    # `Quote.mid` is None on a one-sided book, which is the honest answer.
    scored['market_mid'] = [
        quotes[s][0].mid if s in quotes else np.nan for s in scored['symbol']]

    # The book features, from the same snapshot the order is priced against.
    # Without these LightGBM substitutes learned defaults and the loop runs as
    # a different model from the one whose gates were measured.
    _attach_book_features(scored, quotes, _stream_cache(), config=config,
                          groups=model.groups or None)
    # Book DYNAMICS, from the minutes `scripts.record_book_minutes` sampled.
    # Separate from `_attach_book_features` because it reads a HISTORY rather
    # than the decision snapshot: `book_flow` differences the book over one and
    # three minutes, and decisions happen only at offsets 3/6/9/12, so the
    # decision cycle can never supply offset-1 on its own. All six of these
    # were NaN on every live decision until this existed, while the deployed
    # artifact was fitted on them.
    for _col, _values in book_flow_row(scored).items():
        scored[_col] = _values.to_numpy()
    scored = prepare_init_score(scored, model)
    _phase('score')
    # Liveness, at the point the cycle has demonstrably scored.
    heartbeat.touch('trade')

    _warn_unscoreable_features(scored, model,
                               offsets=tuple(config.decision_offsets or ()))
    for column, values in model.score(scored).items():
        scored[column] = values

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
        # **RECORDED, never fed to the model.**
        #
        # This used to overwrite `strike` and `displacement` in the scored frame
        # and then recompute exactly one of the ten features derived from them
        # (`z_score`), leaving `abs_z_score`, the three excursion-z columns,
        # `path_efficiency`, `displacement_vs_elapsed`, `touched_opposite` and
        # `peer_displacement` on the bar-derived basis. `abs_z_score ==
        # |z_score|` holds by construction in every training row, so the vector
        # handed to `predict` occupied a state the booster has never seen — and
        # a tree ensemble does not degrade gracefully outside its training
        # manifold, it lands in whatever leaf the splits reach.
        #
        # It fired on every cycle and every symbol, and was invisible: the only
        # warning triggers above 25bp while the Coinbase-vs-BRTI basis is of
        # order 1bp, which at offset 12 moves `z_score` ~0.13 z while its
        # siblings do not move at all.
        #
        # Recomputing all ten would fix the impossible state and not the skew:
        # training's displacement is `Coinbase_close / Coinbase_OHLC_mean - 1`
        # and the override makes it `Coinbase_close / BRTI - 1`, injecting the
        # cross-index basis into a feature fitted without it. The venue strike
        # is the right reference for SETTLEMENT — which is where `decide()` and
        # `settle_due` use it — not for features.
        scored.loc[index, 'strike_source'] = 'venue'
        scored.loc[index, 'venue_strike'] = venue_strike
        scored.loc[index, 'venue_strike_drift_bps'] = drift_bps

    if 'strike_source' not in scored.columns:
        scored['strike_source'] = 'bars'
    scored['strike_source'] = scored['strike_source'].fillna('bars')

    # Depth at the touch caps the stake: measured, unlike `max_stake_dollars`.
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

    # Seed exposure from what is already committed for THIS window, so
    # `ALREADY_ENTERED` / `POSITION_LIMIT` / `WINDOW_EXPOSURE` survive a new
    # cycle, a new offset and a process restart.
    wopen = (window_open.to_pydatetime()
             if hasattr(window_open, 'to_pydatetime') else window_open)
    # **Counts, not just symbols.** `entries_for_window` deduplicates by symbol,
    # which under-reports both the stake and the entry count once
    # `max_entries_per_symbol_window` exceeds 1 -- and the stake is what
    # `max_window_exposure_fraction` caps, so that under-report would quietly
    # raise the real limit. `entry_counts_for_window` pairs on `prediction_id`
    # instead, so each decision is counted exactly once.
    counts, staked, n_entered, sides = await _db(
        writer.entry_counts_for_window, wopen)
    entered = frozenset(counts)
    base_exposure = WindowExposure(stake=staked, positions=n_entered,
                                   symbols_entered=entered, entry_counts=counts,
                                   sides=sides)
    if entered:
        logger.info('window %s already holds %s ($%.2f); those symbols will refuse',
                    window_open, sorted(entered), staked)
    exposure = base_exposure
    decisions: list[Decision] = []
    rows: dict[str, object] = {}
    pending: list[tuple[Decision, object]] = []

    # Live prices from the book or not at all: without a quote `decide()` would
    # fall back to our own baseline and could return TRADED.
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
            # Reserved before the order goes out: every order is in flight at
            # once, so no later decision may assume an earlier one failed.
            pending.append((decision, row))
            exposure = exposure.with_(decision)

    # The instant the decisions were taken: what every quote's age is
    # measured to, for the archive.
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
            # The venue's belief (the mid) and what a trade would cost (both
            # asks), on EVERY row, refused ones included: the sample that
            # answers "does the model beat the market" must be unselected.
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

    # **Every order at once.** One symbol's round trip, kill or resend no
    # longer delays the next symbol's order. Each traded row's prediction is
    # written first (in a thread, all in parallel) so its ticket and position
    # carry the id of the prediction that produced them.
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
    # Every row, the traded ones again: an upsert on (symbol, window, offset)
    # that keeps their id and records a resend's final price.
    records = [record_for(d) for d in decisions]
    try:
        await _db(_write_predictions, writer, records)
    except Exception:                                         # noqa: BLE001
        logger.exception('could not write this cycle\'s predictions')

    if kills:
        await asyncio.gather(*(
            diagnose_and_note(kalshi, writer, d, r, write_note=last)
            for d, r, last in kills), return_exceptions=True)

    # The top of book we already have, written after the orders: it changes
    # only WHEN the archive is written, never what it contains.
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
    """The run's configuration, with everything that changes an answer explicit.

    `entry_offsets` is applied here rather than defaulted in `Config` because
    `scripts.evaluate` must keep measuring every offset — a narrowed library
    default would make the sweep unable to price the very cells that justify the
    narrowing.
    """
    config = DEFAULT_CONFIG.with_fee_assumptions(find_fee_config())
    overrides: dict = {}
    # **Everything the parser accepts must be mapped HERE.** This function
    # shadows `scripts._common.config_from_args`, so a flag added to the shared
    # parser does not reach live, and a flag added to live's parser does
    # nothing unless it is written below. Measured 2026-10-18: `--maker-entry`,
    # `--maker-time-in-force` and `--maker-cancel-seconds` all parsed cleanly
    # and were dropped on the floor, leaving live a plain taker paying full
    # fees while the weekly gates were scored fee-free. That is the "flag
    # declared and never read" pattern this repo has now recorded five times.
    band = getattr(args, 'exclude_price_band', None)
    if isinstance(band, str) and band.strip():
        try:
            lo_s, hi_s = band.split(',')
            overrides['exclude_price_band'] = (float(lo_s), float(hi_s))
        except (ValueError, AttributeError):
            raise SystemExit(
                f'--exclude-price-band wants "LO,HI" like "0.35,0.65"; '
                f'got {band!r}')
    elif band is not None and not isinstance(band, str):
        overrides['exclude_price_band'] = band
    if getattr(args, 'maker_entry', False):
        # Implies `assume_maker` in Config.__post_init__: a maker pays the
        # maker fee, and pricing the entry one way while charging the other
        # understates the net edge by the whole fee.
        overrides['maker_entry'] = True
    tif = getattr(args, 'maker_time_in_force', None)
    if tif:
        overrides['maker_time_in_force'] = str(tif)
    cancel = getattr(args, 'maker_cancel_seconds', None)
    if cancel is not None:
        overrides['maker_cancel_seconds'] = float(cancel)
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
    peer = getattr(args, 'require_peer_book', None)
    if peer is not None:
        overrides['require_peer_book'] = bool(peer)
    pinned = getattr(args, 'require_model_version', None)
    if pinned:
        overrides['require_model_version'] = str(pinned)
    gate = getattr(args, 'min_edge_pp', None)
    if gate is not None:
        # Not clamped at zero on purpose. A negative gate is the documented way
        # to force the order path open for a fill test, and a silent clamp would
        # leave the operator watching a loop that quietly refused to trade.
        overrides['min_edge_pp'] = float(gate)
    cap = getattr(args, 'max_stake_dollars', None)
    if cap is not None:
        overrides['max_stake_dollars'] = float(cap)
    kelly = getattr(args, 'kelly_fraction', None)
    if kelly is not None:
        overrides['kelly_fraction'] = float(kelly)
    per_symbol = getattr(args, 'max_entries_per_symbol_window', None)
    if per_symbol is not None:
        overrides['max_entries_per_symbol_window'] = int(per_symbol)
    cap = getattr(args, 'max_positions_per_window', None)
    if cap is not None:
        overrides['max_positions_per_window'] = int(cap)
    # **`--compound` was parsed and dropped on the floor.** Declared at the
    # parser since 2026-09-02 and read nowhere: `config.compound` stayed False,
    # so `decide()` sized off `starting_bankroll` and the running balance was
    # ignored. Measured 2026-09-13 on a $550 account — a trade that should have
    # been 9 contracts was 3, because the base in force was the fixed $200.
    #
    # `6b288fc9` is titled "Compound live, and let an explicit flag beat the
    # artifact's provenance" and added `compound` to the adoption loop in
    # `config_for_artifact`, which skips any field in `cli_overrides`. Nothing
    # ever put it there, so the flag could never win and the artifact's
    # provenance (False) always did. The provenance half was built; the flag
    # half was not.
    #
    # Same shape as `--dry-run`, which was "declared and never read, so
    # `--mode live --dry-run --place-orders` parsed cleanly and placed real
    # orders". A flag has to be honoured or be a usage error; silently ignored
    # is the one option that costs money.
    compound = getattr(args, 'compound', None)
    if compound is not None:
        overrides['compound'] = bool(compound)
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
    # What the command line asked for, before any artifact's economics are
    # adopted: a hot reload re-derives the trading config from this.
    base_config = config
    artifact = Path(args.model) if args.model else Path(MODELS_ROOT) / LIVE_MODEL
    signature = artifact_signature(artifact)

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
                            lag_seconds=hb_lag,
                            peer=take_peer_counts(hb_window)))
                        # The first cycle of a window: nothing is in flight and
                        # the next decision is minutes away, so a newly
                        # promoted artifact is picked up here, not by a restart.
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
                    # One unscoreable cycle is not a reason to exit. This used to
                    # be fatal: `score_live` raised on every cycle and the loop
                    # caught only KeyboardInterrupt, so the process died and
                    # `restart: unless-stopped` crash-looped it forever.
                    logger.error('cycle skipped, nothing scored: %s', exc)
                    decisions = []
                except Exception:                     # noqa: BLE001
                    # Any other failure costs this cycle, not the next one.
                    # Escaping to `supervise` meant a 5s-300s backoff and a
                    # fresh startup, skipping decisions for nothing.
                    # CancelledError and KeyboardInterrupt are not Exceptions
                    # and still stop the loop.
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


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
