"""What stops the live loop from deciding: circuit breakers and bar-feed guards."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.pg_writer import PgWriter

logger = logging.getLogger('live')

# The longest lookback any feature needs, plus the mean it is taken over.
# `volume_z_15` is a 1,440-minute z-score of a 15-minute rolling mean, so
# bit-parity needs 1,440 + 15 = 1,455 contiguous bars; `log_rv_1440` needs
# 1,441. `FETCH_MINUTES` (1,500) leaves a 45-bar margin.
MIN_USABLE_BARS = 1_455


def check_circuit_breakers(writer: PgWriter, config: Config,
                           *, now: datetime) -> Optional[str]:
    """Halt the account on a bad day or a bad streak. Returns the reason, or None.

    Writes `Account.halted` / `halted_reason`, which the dashboard shows. The
    ruin floor in `decide` is the only other live limit, and it fires only after
    half the account is gone.

    A halt is sticky. Clearing it is a manual decision (`--clear-halt`), because
    a breaker that resets itself at midnight is a speed bump.
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

    # Peak-to-current drawdown on realised equity. The daily rule cannot see a
    # run-up given back within one UTC day. Realised basis on both sides:
    # `realised_high_water()` is the running maximum of the series
    # `account.realized_pnl` holds, derived from settlements, not stored.
    peak_realised = writer.realised_high_water()
    start = float(config.starting_bankroll)
    peak_equity = start + max(peak_realised, 0.0)
    current_equity = start + float(account.realized_pnl or 0.0)
    if peak_equity > 0:
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= config.max_drawdown_fraction:
            # Only a halt once it is also a loss of the starting stake. Above
            # water this is a signal, not a stop: the stake is already guarded
            # by `ruin_floor_fraction` and the daily rule.
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


def short_symbols(bars: dict, config: Config) -> dict[str, str]:
    """Symbols whose history is too SHORT to reproduce the training features.

    `get_candles_range` can return a short or holed series after only logging an
    ERROR. Below ~721 bars `log_rv_1440` goes NaN and the cycle abstains on its
    own; between 721 and 1,454 the features are confidently wrong (e.g.
    `volume_z_15` 46x off at 1,200 bars) — exactly where a partial fetch lands.
    `Dataset.coverage()` cannot see it, because a short grid spans only what was
    returned.

    Counts rows, not span: a 1,500-minute span can hold 400 bars, and it is the
    row count the rolling windows consume.
    """
    reasons: dict[str, str] = {}
    for symbol in config.symbols:
        frame = bars.get(symbol)
        if frame is None or not len(frame):
            reasons[symbol] = 'no bars returned'
            continue
        need = int(getattr(config, 'min_usable_bars', MIN_USABLE_BARS))
        usable = int(pd.to_numeric(frame['close'], errors='coerce').notna().sum())
        if usable < need:
            reasons[symbol] = (
                f'{usable} usable bars, under the {need} needed to reproduce '
                f'the training features (volume_z_15 needs 1,455)')
    return reasons


def stale_symbols(bars: dict[str, pd.DataFrame], config: Config,
                  *, now: datetime) -> dict[str, str]:
    """Symbols whose feed is too old, or absent, with the reason.

    * **An old feed.** The forward-fill inside `build_windows` cannot tell a
      minute in which nothing traded from one not yet fetched, so a stale feed
      becomes a fabricated `last_price` the barrier treats as a measurement.
    * **A missing symbol.** The universe is part of the feature definition: a
      missing symbol silently redefines the others' `cross_asset` features
      (`beta_1440` moved 7.7x with no error and no NaN).
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
