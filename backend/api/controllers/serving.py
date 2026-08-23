"""Read the serving store and shape it for the dashboard.

**The API serves measurements, never substitutes.** A missing value comes back
as null with a reason beside it. The previous version of this surface reported
`pr_auc` as `holdout_auc - 0.06`, `precision_at_threshold` as
`holdout_auc - 0.04`, and — when the artifact it wanted was absent, which was
always — a hardcoded table of six feature importances. All of it rendered
identically to real data, which is the whole problem: a fabricated number and a
measured one look the same in a chart.

So every response here distinguishes three states, and the frontend renders them
differently: a measured value, an explicit null with `reason`, and an empty
collection. There is no fourth state where something plausible is invented.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from models.serving import (
    Account, CalibrationBin, EquityPoint, ModelRun, Outcome, Position, Prediction,
)

# Reasons a decision was refused, in funnel order. The dashboard shows them in
# this order because it is the order the gates apply, and the shape of the funnel
# is the most informative thing this system can report: `edge_below_gate`
# dominating means the forecast does not cover the fee, so it declines. On the
# previous perp system that single number said what no Sharpe ratio said.
FUNNEL_ORDER = (
    'traded', 'not_finite', 'price_out_of_band', 'disagreement_implausible',
    'edge_below_gate', 'below_min_contracts', 'fee_ceiling', 'window_exposure',
    'position_limit', 'already_entered', 'bankroll_floor',
)


def _missing(reason: str) -> dict[str, Any]:
    return {'value': None, 'reason': reason}


def _measured(value: Any) -> dict[str, Any]:
    return {'value': value, 'reason': None}


def account_state(db: Session, *, starting_bankroll: float = 100.0) -> dict[str, Any]:
    row = db.execute(select(Account).order_by(Account.id)).scalars().first()
    open_rows = db.execute(
        select(Position).where(Position.outcome == Outcome.PENDING.value)
    ).scalars().all()
    staked = sum(p.outlay for p in open_rows)

    if row is None:
        return {
            'configured': False,
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
        'starting_bankroll': row.starting_bankroll,
        'bankroll': _measured(row.bankroll),
        # Equity is bankroll plus open stake carried at COST. Deliberately not
        # marked to the model's own probability: marking an open binary at our
        # own forecast books the edge we believe in as profit we have not
        # received, which is how a losing system draws a rising equity curve.
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
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(EquityPoint).where(EquityPoint.timestamp >= since)
        .order_by(EquityPoint.timestamp)
    ).scalars().all()
    return [{
        'timestamp': r.timestamp, 'equity': r.equity, 'bankroll': r.bankroll,
        'staked': r.staked, 'open_positions': r.open_positions,
        'realized_pnl': r.realized_pnl,
    } for r in rows]


def live_windows(db: Session) -> list[dict[str, Any]]:
    """The most recent decision point per symbol — the barrier state, now."""
    latest = db.execute(
        select(Prediction.symbol, func.max(Prediction.decision_time))
        .group_by(Prediction.symbol)
    ).all()
    out = []
    for symbol, when in latest:
        row = db.execute(
            select(Prediction)
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
        'reason': row.reason, 'traded': bool(row.traded), 'side': row.side,
        'price': row.price, 'effective_cost': row.effective_cost,
        'edge': row.edge, 'contracts': row.contracts,
        'model_version': row.model_version,
    }


def recent_predictions(db: Session, *, limit: int = 100,
                       traded_only: bool = False) -> list[dict[str, Any]]:
    query = select(Prediction).order_by(Prediction.decision_time.desc()).limit(limit)
    if traded_only:
        query = select(Prediction).where(Prediction.traded.is_(True)) \
            .order_by(Prediction.decision_time.desc()).limit(limit)
    return [_prediction_payload(r) for r in db.execute(query).scalars().all()]


def funnel(db: Session, *, days: int = 7) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        select(Prediction.reason, func.count(Prediction.id))
        .where(Prediction.window_open >= since)
        .group_by(Prediction.reason)
    ).all()
    counts = {reason: int(n) for reason, n in rows}
    total = sum(counts.values())
    return [{
        'reason': reason, 'count': counts.get(reason, 0),
        'share': counts.get(reason, 0) / total if total else None,
    } for reason in FUNNEL_ORDER if reason in counts or reason == 'traded']


def positions(db: Session, *, open_only: bool = False,
              limit: int = 100) -> list[dict[str, Any]]:
    query = select(Position)
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
    """Every attempt, blocked ones included. The trial count.

    A project that hides its failures cannot compute its own multiple-testing
    correction, so this list is deliberately complete rather than filtered to
    what was installed.
    """
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
    """The reliability table for model and baseline, side by side.

    The one diagnostic that cannot be faked by a good average: a model can hit
    the base rate exactly while being wrong at every level of confidence. Since
    this system only trades its confident predictions, a miscalibration in the
    0.85-0.95 band matters far more than the headline number — so both curves are
    returned and the frontend draws them against the diagonal.
    """
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
