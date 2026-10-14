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

# The longest lookback any feature needs. `volume_z_15` is a 1,440-minute
# z-score of a 15-minute rolling mean, so parity with training needs
# 1,440 + 15 = 1,455 contiguous bars.
MIN_USABLE_BARS = 1_455


def check_circuit_breakers(writer: PgWriter, config: Config,
                           *, now: datetime) -> Optional[str]:
    """Halt the account on a bad day or a bad streak. Returns the reason, or None.

    Writes `Account.halted` and `halted_reason`. A halt is sticky: only
    `--clear-halt` clears it.
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

    # Peak-to-current drawdown on realised equity. Both sides use the realised
    # basis: `realised_high_water()` is the running maximum of
    # `account.realized_pnl`.
    peak_realised = writer.realised_high_water()
    start = float(config.starting_bankroll)
    peak_equity = start + max(peak_realised, 0.0)
    current_equity = start + float(account.realized_pnl or 0.0)
    if peak_equity > 0:
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown >= config.max_drawdown_fraction:
            # Halt only below the starting stake. Above it, the ruin floor and
            # the daily rule guard the stake, so this is only a warning.
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
    """Symbols whose history is too short to reproduce the training features.

    A short history gives finite but wrong features, so the cycle must refuse
    it. Counts rows, not span, because the rolling windows consume rows.
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

    A stale feed becomes a forward-filled `last_price` that the barrier treats
    as a measurement. A missing symbol changes the `cross_asset` features of
    the others.
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
