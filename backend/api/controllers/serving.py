"""Read the serving store and shape it for the dashboard.

The API serves measurements, never substitutes. Every value is one of three
states: a measured value, an explicit null with a `reason`, or an empty
collection. Nothing plausible is invented for a missing value.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models.serving import (
    Account, CalibrationBin, EquityPoint, MinutePrice, ModelRun, OrderTicket,
    Outcome, Position, Prediction, VenueBalance, VenueSettlement,
)

# Decision reasons, in the order the gates apply. The dashboard shows them in
# this order.
FUNNEL_ORDER = (
    'traded', 'not_finite', 'probability_invalid', 'no_quote', 'no_peer_book',
    'price_out_of_band', 'disagreement_implausible', 'edge_below_gate',
    'edge_uncertain', 'members_disagree', 'below_min_contracts',
    'fee_ceiling', 'window_exposure', 'position_limit', 'already_entered',
    'offset_not_traded', 'bankroll_floor', 'halted', 'model_not_pinned',
)


def _missing(reason: str) -> dict[str, Any]:
    return {'value': None, 'reason': reason}


def _measured(value: Any) -> dict[str, Any]:
    return {'value': value, 'reason': None}


#: The time column each serving table is filtered on for the dashboard epoch.
#: Named explicitly: a row is filtered by when it happened, not when it was
#: written.
RESET_COLUMN = {
    'Prediction': 'decision_time',
    'Position': 'window_open',
    'EquityPoint': 'timestamp',
    'OrderTicket': 'created_at',
    'VenueSettlement': 'settled_time',
    'VenueFill': 'created_time',
    'VenueBalance': 'timestamp',
}


def reset_epoch(db: Session):
    """The instant the dashboard reads from, or None for all of history.

    The epoch hides rows; it never deletes them. `predictions` is the only
    out-of-sample test of the model against the price, and the venue ledger
    holds windows Kalshi no longer serves. Research and promotion ignore the
    epoch. Clearing the column restores the full view.
    """
    row = db.execute(select(Account).order_by(Account.id)).scalars().first()
    return getattr(row, 'reset_at', None) if row is not None else None


def since_reset(query, model, epoch):
    """Apply the dashboard epoch to `query`, or return it unchanged.

    Every time-series query in this module goes through here; a test asserts it.
    """
    if epoch is None:
        return query
    column = getattr(model, RESET_COLUMN[model.__name__], None)
    if column is None:
        return query
    return query.where(column >= epoch)


def account_state(db: Session, *, starting_bankroll: float = 100.0) -> dict[str, Any]:
    epoch = reset_epoch(db)
    row = db.execute(select(Account).order_by(Account.id)).scalars().first()
    open_rows = db.execute(
        since_reset(select(Position), Position, epoch).where(Position.outcome == Outcome.PENDING.value)
    ).scalars().all()
    staked = sum(p.outlay for p in open_rows)

    if row is None:
        return {
            'configured': False,
            'mode': 'paper',
            'starting_bankroll': starting_bankroll,
            'bankroll': _missing('no account row yet — the paper engine has not run'),
            'equity': _missing('no account row yet — the paper engine has not run'),
            'staked': _measured(staked),
            'open_positions': len(open_rows),
            'realized_pnl': _missing('no account row yet'),
            'fees_paid': _missing('no account row yet'),
            'halted': False, 'halted_reason': None, 'updated_at': None,
        }
    return {
        'configured': True,
        # 'paper' or 'live'. Every response with a number from this row
        # carries it.
        'mode': row.mode,
        'starting_bankroll': row.starting_bankroll,
        'bankroll': _measured(row.bankroll),
        # Equity is bankroll plus open stake at cost. Never mark an open binary
        # at our own forecast: that books unearned edge as profit.
        'equity': _measured(row.bankroll + staked),
        'staked': _measured(staked),
        'open_positions': len(open_rows),
        'realized_pnl': _measured(row.realized_pnl),
        'fees_paid': _measured(row.fees_paid),
        'halted': bool(row.halted),
        'halted_reason': row.halted_reason,
        'updated_at': row.updated_at,
    }


def equity_curve(db: Session, *, days: int = 30) -> list[dict[str, Any]]:
    epoch = reset_epoch(db)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        since_reset(select(EquityPoint), EquityPoint, epoch).where(EquityPoint.timestamp >= since)
        .order_by(EquityPoint.timestamp)
    ).scalars().all()
    return [{
        'timestamp': r.timestamp, 'equity': r.equity, 'bankroll': r.bankroll,
        'staked': r.staked, 'open_positions': r.open_positions,
        'realized_pnl': r.realized_pnl,
    } for r in rows]


def live_windows(db: Session) -> list[dict[str, Any]]:
    """The most recent decision point per symbol — the barrier state, now."""
    epoch = reset_epoch(db)
    latest = db.execute(
        since_reset(select(Prediction.symbol, func.max(Prediction.decision_time)), Prediction, epoch)
        .group_by(Prediction.symbol)
    ).all()
    out = []
    for symbol, when in latest:
        row = db.execute(
            since_reset(select(Prediction), Prediction, epoch)
            .where(Prediction.symbol == symbol, Prediction.decision_time == when)
        ).scalars().first()
        if row is None:
            continue
        out.append(_prediction_payload(row))
    return sorted(out, key=lambda r: r['symbol'])


def _prediction_payload(row: Prediction) -> dict[str, Any]:
    return {
        'symbol': row.symbol,
        'window_open': row.window_open, 'settle_time': row.settle_time,
        'offset_minutes': row.offset_minutes, 'decision_time': row.decision_time,
        'strike': row.strike, 'last_price': row.last_price,
        'displacement': row.displacement,
        'sigma_remaining': row.sigma_remaining, 'z_score': row.z_score,
        'baseline_probability': row.baseline_probability,
        'model_probability': row.model_probability,
        'market_probability': row.market_probability,
        # 'quote' when the venue's ask priced this, 'baseline' when the
        # calibrated barrier stood in for an unobserved market.
        'price_source': row.price_source,
        'reason': row.reason, 'traded': bool(row.traded), 'side': row.side,
        'price': row.price, 'effective_cost': row.effective_cost,
        'edge': row.edge, 'contracts': row.contracts,
        'model_version': row.model_version,
    }


def recent_predictions(db: Session, *, limit: int = 100,
                       traded_only: bool = False) -> list[dict[str, Any]]:
    epoch = reset_epoch(db)
    query = since_reset(select(Prediction), Prediction, epoch).order_by(Prediction.decision_time.desc()).limit(limit)
    if traded_only:
        query = since_reset(select(Prediction), Prediction, epoch).where(Prediction.traded.is_(True)) \
            .order_by(Prediction.decision_time.desc()).limit(limit)
    return [_prediction_payload(r) for r in db.execute(query).scalars().all()]


def funnel(db: Session, *, days: int = 7) -> list[dict[str, Any]]:
    epoch = reset_epoch(db)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        since_reset(select(Prediction.reason, func.count(Prediction.id)), Prediction, epoch)
        .where(Prediction.window_open >= since)
        .group_by(Prediction.reason)
    ).all()
    counts = {reason: int(n) for reason, n in rows}
    total = sum(counts.values())
    if total == 0:
        # Nothing decided in this span: return no rows, not a zero-count row.
        return []
    return [{
        'reason': reason, 'count': counts.get(reason, 0),
        'share': counts.get(reason, 0) / total if total else None,
    } for reason in FUNNEL_ORDER if reason in counts]


def positions(db: Session, *, open_only: bool = False,
              limit: int = 100) -> list[dict[str, Any]]:
    epoch = reset_epoch(db)
    query = since_reset(select(Position), Position, epoch)
    if open_only:
        query = query.where(Position.outcome == Outcome.PENDING.value) \
            .order_by(Position.settle_time)
    else:
        query = query.order_by(Position.window_open.desc()).limit(limit)
    return [{
        'id': r.id, 'symbol': r.symbol, 'window_open': r.window_open,
        'settle_time': r.settle_time, 'offset_minutes': r.offset_minutes,
        'side': r.side, 'contracts': r.contracts, 'price': r.price,
        'outlay': r.outlay, 'fee': r.fee,
        'model_probability': r.model_probability,
        'baseline_probability': r.baseline_probability, 'edge': r.edge,
        'outcome': r.outcome, 'settled_up': r.settled_up, 'payout': r.payout,
        'pnl': r.pnl, 'settled_at': r.settled_at,
    } for r in db.execute(query).scalars().all()]


def model_state(db: Session) -> dict[str, Any]:
    row = db.execute(
        select(ModelRun).order_by(ModelRun.created_at.desc())
    ).scalars().first()
    if row is None:
        return {
            'present': False,
            'reason': 'no promotion attempt recorded — run `python -m scripts.promote`',
        }
    return {
        'present': True, 'reason': None,
        'version': row.version, 'created_at': row.created_at,
        'installed': bool(row.installed), 'forced': bool(row.forced),
        'force_reason': row.force_reason,
        'folds': row.folds, 'windows_evaluated': row.windows_evaluated,
        'log_loss_skill': row.log_loss_skill,
        'log_loss_skill_se': row.log_loss_skill_se,
        'folds_positive': row.folds_positive,
        'calibration_error': row.calibration_error,
        'residual_scale': row.residual_scale,
        'control_gain_share': row.control_gain_share,
        'sharpe': row.sharpe, 'total_return': row.total_return,
        'gates': row.gates or [],
        'failed_gates': (row.failed_gates or '').split(', ') if row.failed_gates else [],
        'provenance': row.provenance or {},
    }


def model_history(db: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    """Every promotion attempt, blocked ones included: the trial count."""
    rows = db.execute(
        select(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit)
    ).scalars().all()
    return [{
        'version': r.version, 'created_at': r.created_at,
        'installed': bool(r.installed), 'forced': bool(r.forced),
        'log_loss_skill': r.log_loss_skill, 'folds_positive': r.folds_positive,
        'windows_evaluated': r.windows_evaluated,
        'failed_gates': (r.failed_gates or '').split(', ') if r.failed_gates else [],
        'force_reason': r.force_reason,
    } for r in rows]


def calibration(db: Session, *, version: Optional[str] = None) -> dict[str, Any]:
    """The reliability table for model and baseline, side by side."""
    if version is None:
        latest = db.execute(
            select(ModelRun.version).order_by(ModelRun.created_at.desc())
        ).scalars().first()
        version = latest
    if version is None:
        return {'version': None, 'bins': [],
                'reason': 'no model run recorded, so no reliability table exists'}
    rows = db.execute(
        select(CalibrationBin).where(CalibrationBin.model_version == version)
        .order_by(CalibrationBin.source, CalibrationBin.bin_low)
    ).scalars().all()
    if not rows:
        return {'version': version, 'bins': [],
                'reason': f'no reliability table stored for {version}'}
    return {
        'version': version, 'reason': None,
        'bins': [{
            'source': r.source, 'bin_low': r.bin_low, 'bin_high': r.bin_high,
            'predicted': r.predicted, 'observed': r.observed, 'count': r.count,
        } for r in rows],
    }


def minute_prices(db: Session, symbol: str, *, minutes: int = 240) -> list[dict[str, Any]]:
    """Recent one-minute bars for one symbol, for the window chart."""
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rows = db.execute(
        select(MinutePrice)
        .where(MinutePrice.symbol == symbol, MinutePrice.minute >= since)
        .order_by(MinutePrice.minute)
    ).scalars().all()
    return [{
        'minute': r.minute, 'open': r.open, 'high': r.high,
        'low': r.low, 'close': r.close,
    } for r in rows]


def window_strikes(db: Session, *, minutes: int = 240) -> list[dict[str, Any]]:
    """The strike of each recent window, per symbol.

    The strike resets every window, so the chart draws one segment per window.
    """
    epoch = reset_epoch(db)
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rows = db.execute(
        since_reset(select(Prediction.symbol, Prediction.window_open, Prediction.settle_time,
               func.min(Prediction.strike)), Prediction, epoch)
        .where(Prediction.window_open >= since)
        .group_by(Prediction.symbol, Prediction.window_open, Prediction.settle_time)
        .order_by(Prediction.window_open)
    ).all()
    return [{
        'symbol': symbol, 'window_open': window_open, 'settle_time': settle_time,
        'strike': float(strike),
    } for symbol, window_open, settle_time, strike in rows]


def tickets(db: Session, *, open_only: bool = True, limit: int = 50) -> list[dict[str, Any]]:
    """Live order tickets: the record of each live decision, dry run or filled."""
    epoch = reset_epoch(db)
    query = since_reset(select(OrderTicket), OrderTicket, epoch).order_by(OrderTicket.created_at.desc()).limit(limit)
    if open_only:
        query = (since_reset(select(OrderTicket), OrderTicket, epoch).where(OrderTicket.status == 'new')
                 .order_by(OrderTicket.created_at.desc()).limit(limit))
    return [{
        'id': r.id, 'symbol': r.symbol, 'window_open': r.window_open,
        'settle_time': r.settle_time, 'offset_minutes': r.offset_minutes,
        'market_ticker': r.market_ticker, 'side': r.side, 'contracts': r.contracts,
        'limit_price': r.limit_price, 'max_price': r.max_price,
        'expected_cost': r.expected_cost, 'model_probability': r.model_probability,
        'edge': r.edge, 'status': r.status,
        'filled_contracts': r.filled_contracts, 'filled_price': r.filled_price,
        'filled_at': r.filled_at, 'note': r.note,
    } for r in db.execute(query).scalars().all()]


# ---------------------------------------------------------------------------
# The venue's ledger.
#
# Everything above reports our own arithmetic. Below reads `venue_fills`,
# `venue_settlements` and `venue_balance`, which the trader fills from the
# venue. Where the two disagree, the venue is right.
#
# * Per-settlement P&L is a stored column; this module only sums and orders.
# * A `None` is counted in `incomplete`, never added as zero.
# * The public tape is not used: it has no account attribution.


def _venue_settlement_payload(row: VenueSettlement) -> dict[str, Any]:
    return {
        'ticker': row.ticker,
        'event_ticker': row.event_ticker,
        'market_result': row.market_result,
        'yes_contracts': row.yes_contracts,
        'no_contracts': row.no_contracts,
        'contracts': (row.yes_contracts or 0.0) + (row.no_contracts or 0.0),
        'cost': (None if row.yes_cost is None and row.no_cost is None
                 else (row.yes_cost or 0.0) + (row.no_cost or 0.0)),
        'revenue': row.revenue,
        'fee_cost': row.fee_cost,
        'pnl': row.pnl,
        'settled_time': row.settled_time,
        # Ours beside theirs. A null `our_pnl` means we have no record of
        # buying this market.
        'our_pnl': row.our_pnl,
        'pnl_gap': (None if row.pnl is None or row.our_pnl is None
                    else row.pnl - row.our_pnl),
        'won': _venue_won(row),
        'synced_at': row.synced_at,
    }


def _venue_won(row: VenueSettlement) -> Optional[bool]:
    """Did we win, per the venue's resolution of the market?

    Reads `market_result` against the side held, not the sign of the P&L: a
    winner bought near $1 can net negative after the fee. Mirrors
    `core.venue_ledger.won`, because the API cannot import the trader.
    """
    if not row.market_result:
        return None
    yes_n = row.yes_contracts or 0.0
    no_n = row.no_contracts or 0.0
    if yes_n > 0 and no_n > 0:
        return None
    if yes_n > 0:
        return row.market_result == 'yes'
    if no_n > 0:
        return row.market_result == 'no'
    return None


def venue_settlements(db: Session, *, days: Optional[int] = None,
                      limit: int = 200) -> list[dict[str, Any]]:
    """Settled markets as the venue paid them, newest first."""
    epoch = reset_epoch(db)
    query = since_reset(select(VenueSettlement), VenueSettlement, epoch)
    if days is not None:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = query.where(VenueSettlement.settled_time >= since)
    query = query.order_by(VenueSettlement.settled_time.desc()).limit(limit)
    return [_venue_settlement_payload(r) for r in db.execute(query).scalars().all()]


def venue_account_state(db: Session) -> dict[str, Any]:
    """The account as the venue's ledger reports it, with ours beside it.

    Returns `available: False` with a reason, not zeros, when the ledger is
    empty, as for a paper account.
    """
    epoch = reset_epoch(db)
    rows = db.execute(since_reset(select(VenueSettlement), VenueSettlement, epoch)).scalars().all()
    balance = db.execute(
        since_reset(select(VenueBalance), VenueBalance, epoch).order_by(VenueBalance.timestamp.desc())
    ).scalars().first()
    account = db.execute(select(Account).order_by(Account.id)).scalars().first()

    if not rows and balance is None:
        return {
            'available': False,
            'reason': ('no venue ledger stored — this account has never traded '
                       'live, or `python -m scripts.sync_venue` has not run'),
            'mode': account.mode if account is not None else 'paper',
            'settlements': 0,
            'realized_pnl': _missing('no venue settlements stored'),
            'fees': _missing('no venue settlements stored'),
            'balance': _missing('no venue balance sampled'),
            'win_rate': _missing('no venue settlements stored'),
            'wins': 0, 'losses': 0, 'undecided': 0, 'incomplete': 0,
            'contracts': 0.0,
            'first_settled': None, 'last_settled': None,
            'our_realized_pnl': (_measured(account.realized_pnl)
                                 if account is not None
                                 else _missing('no account row yet')),
            'pnl_gap': _missing('no venue settlements stored'),
            'balance_drift': _missing('no venue balance sampled'),
            'balance_at': None,
            'exchange_index': None,
        }

    realized = 0.0
    fees = 0.0
    contracts = 0.0
    have_pnl = have_fees = False
    wins = losses = undecided = incomplete = 0
    stamps: list[datetime] = []
    for row in rows:
        if row.pnl is None:
            incomplete += 1
        else:
            realized += float(row.pnl)
            have_pnl = True
        if row.fee_cost is not None:
            fees += float(row.fee_cost)
            have_fees = True
        contracts += (row.yes_contracts or 0.0) + (row.no_contracts or 0.0)
        verdict = _venue_won(row)
        if verdict is True:
            wins += 1
        elif verdict is False:
            losses += 1
        else:
            undecided += 1
        if row.settled_time is not None:
            stamps.append(row.settled_time)

    decided = wins + losses
    ours = float(account.realized_pnl) if account is not None else None
    return {
        'available': True,
        'reason': None,
        # Carried here too, so a live figure never renders as a paper one.
        'mode': account.mode if account is not None else 'paper',
        'settlements': len(rows),
        'realized_pnl': (_measured(realized) if have_pnl
                         else _missing('every settlement was missing a field')),
        'fees': (_measured(fees) if have_fees
                 else _missing('the venue served no fee on any settlement')),
        'balance': (_measured(balance.balance) if balance is not None
                    else _missing('no venue balance sampled')),
        'win_rate': (_measured(wins / decided) if decided
                     else _missing('nothing has settled with a named result')),
        'wins': wins, 'losses': losses, 'undecided': undecided,
        # Settlements excluded from `realized_pnl` for a missing field. Non-zero
        # means the total is short by an unknown amount.
        'incomplete': incomplete,
        'contracts': contracts,
        'first_settled': min(stamps) if stamps else None,
        'last_settled': max(stamps) if stamps else None,
        'our_realized_pnl': (_measured(ours) if ours is not None
                             else _missing('no account row yet')),
        'pnl_gap': (_measured(realized - ours) if have_pnl and ours is not None
                    else _missing('one of the two figures is unavailable')),
        'balance_drift': (_measured(balance.drift)
                          if balance is not None and balance.drift is not None
                          else _missing('no drift recorded on the last sample')),
        'balance_at': balance.timestamp if balance is not None else None,
        'exchange_index': balance.exchange_index if balance is not None else None,
    }


def venue_equity_curve(db: Session, *, days: int = 30) -> dict[str, Any]:
    """The account chart: realised P&L stepped by settlement, cash sampled beside it.

    `points` is cumulative realised P&L, one step per settled market. It is
    built from settlement P&L, not balance differences, because a deposit and a
    profit look the same in the balance.

    `balances` is the venue's cash, sampled every cycle. An entry debits it and
    settlement credits it.

    Both are windowed by `days`. The cumulative P&L starts at zero at the left
    edge; `pnl_before_window` holds the excluded total.
    """
    epoch = reset_epoch(db)
    since = datetime.now(timezone.utc) - timedelta(days=days)

    earlier = db.execute(
        select(func.sum(VenueSettlement.pnl))
        .where(VenueSettlement.settled_time < since,
               VenueSettlement.pnl.is_not(None))
    ).scalar()

    rows = db.execute(
        since_reset(select(VenueSettlement), VenueSettlement, epoch)
        .where(VenueSettlement.settled_time >= since,
               VenueSettlement.pnl.is_not(None))
        .order_by(VenueSettlement.settled_time)
    ).scalars().all()

    points: list[dict[str, Any]] = []
    total = 0.0
    for row in rows:
        total += float(row.pnl)
        points.append({
            'timestamp': row.settled_time,
            'ticker': row.ticker,
            'pnl': float(row.pnl),
            'cumulative_pnl': total,
        })

    balances = db.execute(
        since_reset(select(VenueBalance), VenueBalance, epoch).where(VenueBalance.timestamp >= since)
        .order_by(VenueBalance.timestamp)
    ).scalars().all()

    return {
        'days': days,
        'points': points,
        'balances': [{
            'timestamp': b.timestamp, 'balance': b.balance,
            'our_bankroll': b.our_bankroll, 'drift': b.drift,
            'exchange_index': b.exchange_index,
        } for b in balances],
        # P&L settled before the window opened. Null when nothing did.
        'pnl_before_window': None if earlier is None else float(earlier),
        'realized_pnl_in_window': total if points else None,
    }
