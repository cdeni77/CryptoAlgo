#!/usr/bin/env python3
"""The live loop: scrape, build features, emit signals, retrain on a cadence.

    python -m scripts.live_orchestrator
    python -m scripts.live_orchestrator --run-once
    python -m scripts.live_orchestrator --retrain-only

Each cycle runs the same four steps, in order, as separate processes:

    run_pipeline              --backfill-only   fetch new bars, funding, OI
    migrate_to_research_store                   sync the scraper DB into Parquet
    build_features                              rebuild the panel
    signals                                     decide, using the promoted model

Retraining is on its own cadence and goes through `scripts.promote`, which is
the only thing that can install a model. The orchestrator does not decide whether
a candidate is good; it decides *when* to ask, and `core.promotion` answers with
the gate results. That separation is the change from the previous version, which
promoted any model that finished training and then checked its paper win rate
afterwards — a lagging measurement on a model already trading with real size.

The kill switch survives, and its job is now the one thing the gates cannot do:
notice a *promoted* model decaying in live paper trading. The gates measure a
candidate before it trades; this measures reality after. Both are needed, and
neither substitutes for the other.

Steps run as subprocesses on purpose. A segfault in LightGBM, an OOM during
feature building, or an exchange client wedging on a socket then kills one step
and not the loop, and the exit code says which step it was.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from core.model import MODELS_DIR
from core.pg_writer import PgWriter
from core.promotion import current_record

LOGGER = logging.getLogger('live_orchestrator')
STOP_REQUESTED = False
STATE_FILE = Path(os.getenv('ORCHESTRATOR_STATE_FILE', './data/orchestrator_state.json'))

# The promotion gates cannot be re-run cheaply enough to sit in the hot loop, so
# a retrain is scheduled rather than triggered. This is how long a promoted model
# is allowed to stay live before a fresh candidate is evaluated against it.
DEFAULT_RETRAIN_DAYS = 7


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchThresholds:
    """When a *live* model has decayed enough to stop trusting it.

    Deliberately different quantities from the promotion gates. Those are
    out-of-sample statistics on a candidate; these are realised outcomes on the
    model that is trading. A model can clear every gate and still degrade,
    because the market it was fitted to stopped existing.
    """

    min_win_rate: float
    min_profit_factor: float
    max_drawdown: float
    min_trades_per_week: float
    max_negative_expectancy_streak: int
    min_samples: int


def _monitoring_thresholds(lookback_days: int = 14) -> KillSwitchThresholds:
    thresholds = KillSwitchThresholds(
        min_win_rate=float(os.getenv('PAPER_MONITOR_MIN_WIN_RATE', '0.42')),
        min_profit_factor=float(os.getenv('PAPER_MONITOR_MIN_PROFIT_FACTOR', '0.9')),
        max_drawdown=float(os.getenv('PAPER_MONITOR_MAX_DRAWDOWN', '0.12')),
        min_trades_per_week=float(os.getenv('PAPER_MONITOR_MIN_TRADES_PER_WEEK', '4.5')),
        max_negative_expectancy_streak=int(os.getenv('PAPER_MONITOR_NEG_EXPECTANCY_STREAK', '2')),
        min_samples=int(os.getenv('PAPER_MONITOR_MIN_SAMPLES', '8')),
    )

    # The velocity check is only reachable above the sample gate, and the sample
    # gate already implies a floor: `min_samples` trades over `lookback_days`
    # is `min_samples * 7 / lookback_days` per week. A threshold at or below that
    # can never fire, which is what the old default of 2.0 against 8 trades over
    # 14 days (a floor of 4.0/week) quietly was. Say so rather than shipping a
    # knob that does nothing.
    implied_floor = thresholds.min_samples * 7 / max(lookback_days, 1)
    if thresholds.min_trades_per_week <= implied_floor:
        LOGGER.warning(
            'PAPER_MONITOR_MIN_TRADES_PER_WEEK=%.2f can never fire: passing the '
            'sample gate (%d trades in %dd) already implies >= %.2f/week. Raise it '
            'above %.2f, or rely on insufficient_samples to catch a quiet strategy.',
            thresholds.min_trades_per_week, thresholds.min_samples, lookback_days,
            implied_floor, implied_floor,
        )
    return thresholds


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _paper_kpis(writer: PgWriter, lookback_days: int = 14) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    closed = writer.get_closed_paper_positions_since(since)
    equity_points = writer.get_paper_equity_curve_since(since)

    pnls = [_as_float(p.realized_pnl) for p in closed]
    trades = len(pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    drawdown = 0.0
    if equity_points:
        peak = _as_float(equity_points[0].equity)
        for point in equity_points:
            equity = _as_float(point.equity)
            peak = max(peak, equity)
            if peak > 0:
                drawdown = max(drawdown, (peak - equity) / peak)

    return {
        'window_days': lookback_days,
        'trades': trades,
        'win_rate': (len([p for p in pnls if p > 0]) / trades) if trades else 0.0,
        'profit_factor': (
            gross_profit / gross_loss if gross_loss > 0
            else (float('inf') if gross_profit > 0 else 0.0)
        ),
        'drawdown': drawdown,
        'expectancy': (sum(pnls) / trades) if trades else 0.0,
        'trades_per_week': trades * (7 / lookback_days),
    }


def _quarantine_reasons(
    kpis: dict[str, Any],
    thresholds: KillSwitchThresholds,
    state: dict[str, Any],
) -> list[str]:
    """Why the live model should stop trading, or an empty list.

    Too few trades returns early with `insufficient_samples`, which is reported
    but does *not* quarantine: "we have not seen enough to judge" is not the same
    finding as "this is broken", and conflating them would halt every new model
    in its first week.
    """
    if kpis['trades'] < thresholds.min_samples:
        return [f"insufficient_samples:{kpis['trades']}<{thresholds.min_samples}"]

    reasons: list[str] = []
    if kpis['win_rate'] < thresholds.min_win_rate:
        reasons.append(f"win_rate_collapse:{kpis['win_rate']:.3f}<{thresholds.min_win_rate:.3f}")
    if kpis['profit_factor'] < thresholds.min_profit_factor:
        reasons.append(
            f"profit_factor_breach:{kpis['profit_factor']:.3f}<{thresholds.min_profit_factor:.3f}"
        )
    if kpis['drawdown'] > thresholds.max_drawdown:
        reasons.append(f"drawdown_breach:{kpis['drawdown']:.3f}>{thresholds.max_drawdown:.3f}")
    if kpis['trades_per_week'] < thresholds.min_trades_per_week:
        reasons.append(
            f"low_trade_velocity:{kpis['trades_per_week']:.2f}<{thresholds.min_trades_per_week:.2f}"
        )

    streak = int(state.get('negative_expectancy_streak', 0))
    streak = streak + 1 if kpis['expectancy'] < 0 else 0
    state['negative_expectancy_streak'] = streak
    if streak >= thresholds.max_negative_expectancy_streak:
        reasons.append(
            f'sustained_negative_expectancy:{streak}>='
            f'{thresholds.max_negative_expectancy_streak}'
        )

    return reasons


def evaluate_live_model(writer: Optional[PgWriter]) -> tuple[bool, dict[str, Any]]:
    """Check the promoted model against realised paper results.

    Returns (quarantined, record). Writes the verdict into the state file so the
    API can serve it without recomputing, and so a quarantine survives a restart.
    """
    if writer is None:
        return False, {}

    state = _load_state()
    thresholds = _monitoring_thresholds()
    kpis = _paper_kpis(writer, lookback_days=int(os.getenv('PAPER_MONITOR_LOOKBACK_DAYS', '14')))
    reasons = _quarantine_reasons(kpis, thresholds, state)
    quarantined = any(not r.startswith('insufficient_samples') for r in reasons)

    live = current_record(MODELS_DIR)
    record = {
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'version': live.version if live else None,
        'kpis': kpis,
        'thresholds': asdict(thresholds),
        'reasons': reasons,
        'status': 'quarantined' if quarantined else 'active',
    }
    state['paper_monitoring'] = record
    if quarantined:
        state['quarantined_version'] = record['version']
    else:
        state.pop('quarantined_version', None)
    _save_state(state)

    if quarantined:
        LOGGER.error('kill switch: live model %s quarantined (%s)',
                     record['version'], ', '.join(reasons))
    return quarantined, record


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        LOGGER.warning('unreadable state file %s, starting fresh', STATE_FILE)
        return {}


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True, default=str))
    os.replace(temporary, STATE_FILE)


def _retrain_due(retrain_every_days: int) -> bool:
    last = _load_state().get('last_retrain_attempt_at')
    if not last:
        return True
    try:
        return datetime.now(timezone.utc) >= datetime.fromisoformat(last) + timedelta(
            days=retrain_every_days
        )
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _run_step(name: str, command: Sequence[str], *, allow_codes: Sequence[int] = (0,)) -> int:
    """Run one step as its own process. Returns the exit code.

    `allow_codes` is what makes the promotion step work: `scripts.promote` exits
    2 when the gates block a candidate, which is a correct outcome and not a
    failure of the loop.
    """
    LOGGER.info('step %s: %s', name, ' '.join(command))
    code = subprocess.run(list(command), check=False).returncode
    if code not in allow_codes:
        raise RuntimeError(f'{name} failed with exit code {code}')
    LOGGER.info('step %s finished (exit %d)', name, code)
    return code


def _data_arguments(args: argparse.Namespace) -> list[str]:
    """The `scripts._common` argument surface, one place.

    Every research script takes the same data arguments, so the orchestrator
    passes the same set to all of them. A venue or cost config that differed
    between the feature build and the signal write would silently score a model
    on inputs it was not trained for.
    """
    flags = ['--venue', args.venue, '--min-quality', args.min_quality]
    if args.store:
        flags += ['--store', args.store]
    if args.reference_venue:
        flags += ['--reference-venue', args.reference_venue]
    if args.symbols:
        flags += ['--symbols', args.symbols]
    if args.cost_config:
        flags += ['--cost-config', args.cost_config]
    flags += ['--log-level', args.log_level]
    return flags


def _training_arguments(args: argparse.Namespace) -> list[str]:
    """Controls that only the training step needs.

    Deliberately separate from `_data_arguments`: a training window passed to the
    signal writer would truncate the panel it has to score the latest bar from.
    """
    flags: list[str] = []
    if args.train_window_days:
        flags += ['--train-window-days', str(args.train_window_days)]
    if args.recency_half_life_days is not None:
        flags += ['--recency-half-life-days', str(args.recency_half_life_days)]
    return flags


def _scrape(args: argparse.Namespace, window_hours: float) -> None:
    command = [
        sys.executable, '-m', 'scripts.run_pipeline',
        '--backfill-only', '--backfill-hours', str(window_hours),
        '--db-path', args.db_path,
    ]
    if args.include_oi:
        command.append('--include-oi')
    _run_step('scrape', command)


def _sync_store(args: argparse.Namespace) -> None:
    """Move what the scraper wrote into the research store.

    The scraper owns SQLite; research owns Parquet. Keeping the copy explicit is
    what makes a build reproducible: the store is bitemporal, so a later revision
    of a bar becomes a new row rather than overwriting the one a past model was
    trained on.
    """
    command = [
        sys.executable, '-m', 'scripts.migrate_to_research_store',
        '--db-path', args.db_path, '--venue', args.venue,
    ]
    if args.store:
        command += ['--store', args.store]
    _run_step('sync research store', command)


def _build_features(args: argparse.Namespace) -> None:
    _run_step('build features',
              [sys.executable, '-m', 'scripts.build_features', *_data_arguments(args)])


def _write_signals(args: argparse.Namespace) -> None:
    command = [
        sys.executable, '-m', 'scripts.signals',
        '--model', str(MODELS_DIR / 'forecast.joblib'),
        '--equity', str(args.equity),
        *_data_arguments(args),
    ]
    _run_step('signals', command)


def _attempt_promotion(args: argparse.Namespace, writer: Optional[PgWriter]) -> bool:
    """Train a candidate and let the gates decide. Returns True if installed.

    A blocked candidate is a normal outcome, not an error: the previous model
    stays live and the ledger records why the new one did not displace it.
    """
    state = _load_state()
    state['last_retrain_attempt_at'] = datetime.now(timezone.utc).isoformat()
    _save_state(state)

    run_id = None
    if writer:
        # All three of these used to describe something that did not happen:
        # `retrain_window_days` recorded 0 whenever the window was unset (the
        # default), `symbols_total` was a literal zero, and `artifacts_version`
        # was minted here — before promote ran — while `promotion.new_version()`
        # mints its own with a uuid suffix, so the recorded version could never
        # match a directory in models/promotions/. The real version is written by
        # `complete_model_run` from the promotion record.
        run_id = writer.create_model_run(
            retrain_window_days=int(args.train_window_days) or None,
            symbols_total=len(args.symbols.split(',')) if args.symbols else 0,
            artifacts_version=None,
        )

    command = [
        sys.executable, '-m', 'scripts.promote',
        '--models-dir', str(MODELS_DIR),
        '--periods', str(args.walk_forward_periods),
        '--equity', str(args.equity),
        *_data_arguments(args),
        *_training_arguments(args),
    ]
    try:
        code = _run_step('promote', command, allow_codes=(0, 2))
    except RuntimeError as exc:
        LOGGER.exception('promotion step crashed: %s', exc)
        if writer and run_id:
            writer.complete_model_run(run_id=run_id, success=False,
                                      symbols_trained=0, error=str(exc))
        return False

    installed = code == 0
    live = current_record(MODELS_DIR)
    if installed:
        state = _load_state()
        state['last_promoted_version'] = live.version if live else None
        state['last_promoted_at'] = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        LOGGER.info('promoted %s', live.version if live else 'candidate')
    else:
        LOGGER.warning(
            'candidate blocked by the gates; keeping %s live',
            live.version if live else 'the existing model',
        )

    if writer and run_id:
        symbols = live.provenance.get('symbols', []) if live else []
        writer.complete_model_run(
            run_id=run_id,
            success=installed,
            symbols_trained=len(symbols) if installed else 0,
            metrics=live.as_dict() if live and installed else None,
            error=None if installed else 'blocked by promotion gates',
            # The version promote actually produced, which is the one that names
            # a directory in models/promotions/.
            artifacts_version=live.version if live and installed else None,
        )
    return installed


def _run_cycle(args: argparse.Namespace, window_hours: float, *, quarantined: bool) -> None:
    _scrape(args, window_hours)
    _sync_store(args)
    _build_features(args)

    if quarantined:
        LOGGER.error('live model is quarantined: skipping the signal write')
        return
    if not (MODELS_DIR / 'forecast.joblib').exists():
        LOGGER.warning(
            'no promoted model at %s: skipping the signal write. Evaluate one with '
            'python -m scripts.promote', MODELS_DIR / 'forecast.joblib',
        )
        return
    _write_signals(args)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _handle_signal(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.info('received signal %s; stopping after the current step', signum)


def _sleep_until_next_aligned(align_minute: int, max_wait_seconds: int) -> None:
    now = datetime.now(timezone.utc)
    next_run = now.replace(minute=align_minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(hours=1)
    seconds = min((next_run - now).total_seconds(), max_wait_seconds)
    LOGGER.info('sleeping until %s UTC (~%ds)', next_run.strftime('%H:%M'), int(seconds))
    time.sleep(seconds)


def _setup_logging(level: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        root.addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data surface, matching scripts/_common.py so every step sees one dataset.
    parser.add_argument('--db-path', default=os.getenv('TRADER_DB_PATH', '/app/data/trading.db'))
    parser.add_argument('--store', default=os.getenv('RESEARCH_STORE') or None)
    parser.add_argument('--venue', default=os.getenv('TRADE_VENUE', 'coinbase'))
    parser.add_argument('--reference-venue', default=os.getenv('REFERENCE_VENUE', 'binance'))
    parser.add_argument('--symbols', default=os.getenv('SYMBOLS') or None)
    parser.add_argument('--min-quality', default='valid',
                        choices=['valid', 'suspicious', 'unvalidated', 'all'])
    parser.add_argument('--cost-config', default=os.getenv('COST_CONFIG') or None)

    # Cadence.
    parser.add_argument('--backfill-days', type=int,
                        default=int(os.getenv('INITIAL_BACKFILL_DAYS', '30')))
    parser.add_argument('--incremental-backfill-hours', type=int,
                        default=int(os.getenv('INCREMENTAL_BACKFILL_HOURS', '6')))
    parser.add_argument('--cycle-interval-seconds', type=int,
                        default=int(os.getenv('CYCLE_INTERVAL_SECONDS', '3600')))
    parser.add_argument('--cycle-align-minute', type=int,
                        default=int(os.getenv('CYCLE_ALIGN_MINUTE', '-1')))
    parser.add_argument('--retrain-every-days', type=int,
                        default=int(os.getenv('RETRAIN_EVERY_DAYS', str(DEFAULT_RETRAIN_DAYS))))
    parser.add_argument('--train-window-days', type=float,
                        default=float(os.getenv('TRAIN_WINDOW_DAYS', '0')),
                        help='Fit on the most recent N days only. 0 (the default) '
                             'uses all history and lets the recency half-life do '
                             'the weighting.')
    parser.add_argument('--recency-half-life-days', type=float,
                        default=(float(os.environ['RECENCY_HALF_LIFE_DAYS'])
                                 if os.getenv('RECENCY_HALF_LIFE_DAYS') else None),
                        help='Decay on training weights, in days. Governs how much '
                             'of a long history reaches the model at all.')
    parser.add_argument('--walk-forward-periods', type=int,
                        default=int(os.getenv('WALK_FORWARD_PERIODS', '6')))
    parser.add_argument('--equity', type=float, default=float(os.getenv('EQUITY', '100000')))

    # Modes.
    parser.add_argument('--run-once', action='store_true', help='One cycle, then exit')
    parser.add_argument('--retrain-only', action='store_true',
                        help='Evaluate a candidate and exit')
    parser.add_argument('--include-oi', action='store_true')
    parser.add_argument('--log-level', default=os.getenv('LOG_LEVEL', 'INFO'))
    parser.add_argument('--log-file',
                        default=os.getenv('ORCHESTRATOR_LOG_FILE',
                                          '/app/logs/live_orchestrator.log'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level, Path(args.log_file))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    writer = PgWriter() if os.environ.get('DATABASE_URL') else None
    if writer is None:
        LOGGER.warning('DATABASE_URL unset: signals and monitoring will not be persisted')

    if args.retrain_only:
        return 0 if _attempt_promotion(args, writer) else 2

    # Hours, not days. `ceil(hours/24)` made every value from 1 to 24 fetch a
    # full day, so the documented "last 6h each cycle" fetched 24h — four times
    # the API calls for the same data.
    incremental_hours = max(1, int(args.incremental_backfill_hours))
    LOGGER.info('live orchestrator starting (venue=%s, retrain every %dd)',
                args.venue, args.retrain_every_days)

    try:
        quarantined, _ = evaluate_live_model(writer)
        _run_cycle(args, args.backfill_days * 24, quarantined=quarantined)

        cycle = 1
        while not STOP_REQUESTED:
            if _retrain_due(args.retrain_every_days):
                _attempt_promotion(args, writer)

            if args.run_once:
                break

            if 0 <= args.cycle_align_minute < 60:
                _sleep_until_next_aligned(args.cycle_align_minute, args.cycle_interval_seconds)
            else:
                LOGGER.info('sleeping %ds', args.cycle_interval_seconds)
                time.sleep(args.cycle_interval_seconds)
            if STOP_REQUESTED:
                break

            cycle += 1
            LOGGER.info('cycle #%d', cycle)
            quarantined, _ = evaluate_live_model(writer)
            _run_cycle(args, incremental_hours, quarantined=quarantined)

    except Exception as exc:  # noqa: BLE001 - the loop reports and exits nonzero
        LOGGER.exception('orchestrator failed: %s', exc)
        return 1

    LOGGER.info('orchestrator stopped cleanly')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
