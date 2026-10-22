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
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from models.serving import (
    Account, CalibrationBin, EquityPoint, MinutePrice, ModelRun, OrderTicket,
    Outcome, Position, Prediction, VenueBalance, VenuePosition, VenueSettlement,
)

# Reasons a decision was refused, in funnel order. The dashboard shows them in
# this order because it is the order the gates apply, and the shape of the funnel
# is the most informative thing this system can report: `edge_below_gate`
# dominating means the forecast does not cover the fee, so it declines. On the
# previous perp system that single number said what no Sharpe ratio said.
FUNNEL_ORDER = (
    'traded', 'not_finite', 'probability_invalid', 'no_quote', 'no_peer_book',
    'price_out_of_band', 'disagreement_implausible', 'edge_below_gate',
    'edge_uncertain', 'members_disagree', 'below_min_contracts',
    'fee_ceiling', 'window_exposure', 'position_limit', 'already_entered',
    # The window is held on the OTHER side. Distinct from `already_entered`,
    # which means the entries are used up: this one means the model changed
    # its mind, and buying the opposite side nets against the open position
    # rather than hedging it -- measured -$6.29 on three markets, the whole
    # stake, before the constraint existed.
    'opposite_side',
    'offset_not_traded', 'bankroll_floor', 'halted', 'model_not_pinned',
)


def _missing(reason: str) -> dict[str, Any]:
    return {'value': None, 'reason': reason}


def _measured(value: Any) -> dict[str, Any]:
    return {'value': value, 'reason': None}


#: The time column each serving table is filtered on for the dashboard epoch.
#: Named explicitly rather than guessed: `Prediction` alone carries four
#: DateTimes, and filtering a decision on `created_at` instead of
#: `decision_time` would hide rows by when they were WRITTEN rather than when
#: they happened.
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

    **This hides; it does not delete.** `predictions` is the only
    out-of-sample measurement of whether the model beats the PRICE --
    `PgWriter.scored_against_market` reads nothing else, and both
    `model_minus_market` and `calibration_vs_market` are computed from it --
    and `venue_fills`/`venue_settlements` hold windows Kalshi itself no longer
    serves. Truncating for a clean chart would trade the one number that is not
    self-graded for a cosmetic reset.

    Research never consults this: `scripts/evaluate` and `scripts/promote` read
    the store and the full `predictions` table regardless, so a reset cannot
    silently shrink the sample a gate is computed on.

    Clearing the column restores the full view.
    """
    row = db.execute(select(Account).order_by(Account.id)).scalars().first()
    return getattr(row, 'reset_at', None) if row is not None else None


def since_reset(query, model, epoch):
    """Apply the dashboard epoch to `query`, or return it unchanged.

    A named function so the seam can be TESTED. Eighteen call sites each
    remembering a filter is the shape of defect this codebase keeps producing --
    `entry_offsets` reaching `promote` and not `evaluate` inverted a gate, and
    three scripts silently ignoring the fold geometry cut a different
    experiment. `tests/test_serving_reset.py` asserts every serving query goes
    through here.
    """
    if epoch is None:
        return query
    # `.get`, not `[]`: an unlisted model used to raise KeyError and 500 the
    # whole surface rather than simply going unfiltered.
    column = getattr(model, RESET_COLUMN.get(model.__name__, ''), None)
    if column is None:
        return query
    return query.where(column >= epoch)


def model_performance(db: Session, *, all_time: bool = False) -> dict[str, Any]:
    """Settled P&L per model version, newest first, plus the total.

    Attribution runs `positions.prediction_id -> predictions.model_version`;
    positions carry no model of their own. A position whose prediction was
    deleted, or which predates `model_version` being recorded, lands in
    `unattributed` rather than being dropped -- a per-model table that does
    not add up to the account is worse than one with a remainder.

    `all_time` ignores the dashboard epoch. Every other serving query is
    epoch-filtered; this one asks "how did each model do", and a model that
    traded before the last reset still did so.

    `win_rate` is not a quality measure here. A model trading longshots wins
    seldom and profitably -- 20261022T171846Z settled 22% and made money --
    so read `pnl_per_position` beside it.
    """
    epoch = reset_epoch(db)
    if all_time:
        epoch = None

    # Outer join: a position whose prediction is gone keeps its P&L and lands
    # in `unattributed`, rather than vanishing from a table that is supposed
    # to reconcile with the account.
    rows = since_reset(select(Position), Position, epoch).join(
        Prediction, Position.prediction_id == Prediction.id, isouter=True
    ).add_columns(Prediction.model_version)
    settled, unattributed = {}, {'positions': 0, 'pnl': 0.0}
    for position, version in db.execute(rows).all():
        if position.settled_at is None:
            continue
        pnl = float(position.pnl or 0.0)
        if not version:
            unattributed['positions'] += 1
            unattributed['pnl'] += pnl
            continue
        bucket = settled.setdefault(str(version), {
            'model_version': str(version), 'positions': 0, 'pnl': 0.0,
            'wins': 0, 'contracts': 0, 'first_window': None,
            'last_window': None})
        bucket['positions'] += 1
        bucket['pnl'] += pnl
        bucket['wins'] += 1 if position.settled_up else 0
        bucket['contracts'] += int(position.contracts or 0)
        when = position.window_open
        if when is not None:
            if bucket['first_window'] is None or when < bucket['first_window']:
                bucket['first_window'] = when
            if bucket['last_window'] is None or when > bucket['last_window']:
                bucket['last_window'] = when

    models = []
    for bucket in settled.values():
        n = bucket['positions']
        models.append({
            **bucket,
            'pnl': round(bucket['pnl'], 4),
            'pnl_per_position': round(bucket['pnl'] / n, 6) if n else None,
            'pnl_per_contract': (round(bucket['pnl'] / bucket['contracts'], 6)
                                 if bucket['contracts'] else None),
            'win_rate': round(bucket['wins'] / n, 4) if n else None,
        })
    models.sort(key=lambda m: (m['first_window'] is None, m['first_window']),
                reverse=True)

    total = round(sum(m['pnl'] for m in models) + unattributed['pnl'], 4)
    return {
        'all_time': bool(all_time),
        'epoch': epoch.isoformat() if epoch is not None else None,
        'models': models,
        'unattributed': {'positions': unattributed['positions'],
                         'pnl': round(unattributed['pnl'], 4)},
        'total_pnl': total,
        'total_positions': (sum(m['positions'] for m in models)
                            + unattributed['positions']),
    }


def account_state(db: Session, *, starting_bankroll: float = 100.0) -> dict[str, Any]:
    epoch = reset_epoch(db)
    row = db.execute(select(Account).order_by(Account.id)).scalars().first()
    open_rows = db.execute(
        since_reset(select(Position), Position, epoch).where(Position.outcome == Outcome.PENDING.value)
    ).scalars().all()
    staked = sum(p.outlay for p in open_rows)

    # **Marked to the venue's bid, where there is one.** `staked` is cost and
    # stays cost; this is a second, different number -- what the book says the
    # open contracts would fetch right now. A position with no mark yet falls
    # back to its cost, so `mark_value` is never short by a whole position; it
    # is counted in `unmarked` instead, so the surface can say how much of the
    # figure is actually live.
    # **The CROSS-SHARD total, because that is the account.** `account.bankroll`
    # holds the trading shard only -- correct for sizing, since that is the only
    # cash a crypto order can draw on -- but showing it as "account value" hides
    # whatever sits on the other shards. Measured 2026-10-20: the page read
    # $494.27 against a real $495.37, and the missing $1.10 was simply invisible.
    # Sizing still reads `bankroll`; only the DISPLAY totals use this.
    latest_balance = db.execute(
        since_reset(select(VenueBalance), VenueBalance, epoch)
        .order_by(VenueBalance.timestamp.desc()).limit(1)
    ).scalars().first()
    venue_total = (float(latest_balance.total_balance)
                   if latest_balance is not None
                   and latest_balance.total_balance is not None else None)

    # **The venue's own figures for the open book, where it has them.**
    #
    # Two of ours were wrong and in opposite directions. `Position.fee` is set
    # at DECISION time, where `assume_maker` makes it 0.00, and is only
    # corrected at settlement -- so an open position was shown fee-free for its
    # whole life: $0.0000 against the venue's $0.0414 on three markets,
    # measured 2026-10-20. And we mark at the BID, which is what a sale would
    # fetch; Kalshi marks higher, so the same book read $3.436 to us and $3.540
    # to it, and the page showed "$0.54 still open" beside an "open p&l" of
    # $0.36 with nothing explaining the difference.
    #
    # The cost basis agreed exactly, which is the reassuring part: the fills
    # are right, only the charges and the marking convention were ours.
    venue_open = db.execute(select(VenuePosition)).scalars().all()
    venue_cost = sum(float(r.market_exposure or 0.0) for r in venue_open)
    venue_fees = sum(float(r.fees_paid or 0.0) for r in venue_open)

    marked = 0.0
    unmarked = 0
    oldest_mark = None
    open_contracts = sum(float(p.contracts or 0) for p in open_rows)
    # **The entry fee is already spent and is not in `outlay`.** `staked` is
    # `contracts * price` exactly -- the ORM comment claiming otherwise was
    # wrong and is corrected -- so `marked - staked` overstates the open
    # position by whatever was charged to open it.
    for position in open_rows:
        if position.mark_price is None:
            marked += float(position.outlay or 0.0)
            unmarked += 1
            continue
        marked += float(position.mark_price) * float(position.contracts or 0)
        if position.mark_at is not None:
            oldest_mark = (position.mark_at if oldest_mark is None
                           else min(oldest_mark, position.mark_at))

    # The venue's where it has one, ours otherwise. `portfolio_value` is a
    # TOTAL and arrives with the balance sample, so it is only usable when the
    # set of open positions it describes is the one we are reporting -- which
    # is why it is taken from the same row as the cash.
    venue_mark = (float(latest_balance.portfolio_value)
                  if latest_balance is not None
                  and getattr(latest_balance, 'portfolio_value', None) is not None
                  and venue_open else None)
    open_worth = venue_mark if venue_mark is not None else marked
    open_cost = venue_cost if venue_open else staked
    open_fees = venue_fees if venue_open else sum(
        float(p.fee or 0.0) for p in open_rows)

    if row is None:
        return {
            'configured': False,
            'mode': 'paper',
            'starting_bankroll': starting_bankroll,
            'bankroll': _missing('no account row yet — the paper engine has not run'),
            'equity': _missing('no account row yet — the paper engine has not run'),
            'staked': _measured(staked),
            'account_value': _missing('no account row yet'),
            'total_balance': _missing('no account row yet'),
            'tradeable_balance': _missing('no account row yet'),
            'stranded': _missing('no account row yet'),
            'mark_value': _measured(open_worth),
            'unrealized_pnl': _measured(round(open_worth - open_cost - open_fees, 6)),
            'open_fees': _measured(open_fees),
            'unmarked_positions': unmarked,
            'marked_at': oldest_mark,
            'open_positions': len(open_rows),
            'open_contracts': open_contracts,
            'unrealized_pp_per_contract': _missing('no account row yet'),
            'realized_pnl': _missing('no account row yet'),
            'fees_paid': _missing('no account row yet'),
            'halted': False, 'halted_reason': None, 'updated_at': None,
        }
    return {
        'configured': True,
        # 'paper' or 'live'. Surfaced on every response that carries a number
        # from this row, because a live account that renders identically to a
        # paper one is the worst failure this surface could permit.
        'mode': row.mode,
        'starting_bankroll': row.starting_bankroll,
        'bankroll': _measured(row.bankroll),
        # **Equity carries open positions at COST.** Deliberately not marked to
        # the model's own probability: marking an open binary at our own
        # forecast books the edge we believe in as profit we have not received,
        # which is how a losing system draws a rising equity curve. Left on
        # cost so the figure cannot move on an opinion.
        'equity': _measured(row.bankroll + staked),
        'staked': _measured(staked),
        # **Account value, marked to the VENUE's bid.** The same total as
        # `equity` except that open contracts are carried at what someone will
        # actually pay for them rather than at what we paid. That is not an
        # opinion, so it is safe to show moving -- and it is the number a
        # person means by "what is my account worth right now".
        # Falls back to the shard when no cross-shard total has been sampled
        # yet, which is only true on a brand-new store -- and says so, rather
        # than silently reporting a smaller number as the whole account.
        'account_value': (_measured(venue_total + open_worth) if venue_total is not None
                          else _measured(row.bankroll + open_worth)),
        # Carried so the surface can show what is spendable where, and name the
        # part that is stranded off the trading shard.
        'total_balance': (_measured(venue_total) if venue_total is not None
                          else _missing('no cross-shard balance sampled yet')),
        'tradeable_balance': _measured(row.bankroll),
        'stranded': (_measured(round(venue_total - float(row.bankroll), 6))
                     if venue_total is not None
                     else _missing('no cross-shard balance sampled yet')),
        'mark_value': _measured(open_worth),
        # Marked minus cost AND minus the fee already paid to open it: the
        # open P&L, which is what moves between cycles. Every term is the
        # venue's where the venue has one, so this equals the "still open"
        # part of the headline instead of sitting $0.18 under it.
        'unrealized_pnl': _measured(round(open_worth - open_cost - open_fees, 6)),
        'open_fees': _measured(open_fees),
        # Ours at the bid, kept beside the venue's so the difference between
        # "what it is worth" and "what a sale would fetch" stays visible rather
        # than being quietly chosen for the reader.
        'mark_value_bid': _measured(marked),
        # How much of `mark_value` is still cost rather than a live quote, and
        # how old the oldest quote in it is. Without these a stale mark is
        # indistinguishable from a fresh one.
        'unmarked_positions': unmarked,
        'marked_at': oldest_mark,
        'open_positions': len(open_rows),
        'open_contracts': open_contracts,
        # **Per CONTRACT, in probability points.** The size-independent figure,
        # and the same unit `realised_edge_pp` gates on -- so the open book and
        # the settled record can be read against each other. A dollar total
        # cannot: it moves with how much was staked, not with how well the
        # position is doing.
        'unrealized_pp_per_contract': (
            _measured(round((marked - staked - open_fees) / open_contracts * 100.0, 4))
            if open_contracts else _missing('nothing open')),
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
        # Side-relative where a side was chosen; a refused row has none and
        # keeps its P(up), which is the only meaningful reading for it.
        'baseline_probability': _for_side(row.baseline_probability,
                                          getattr(row, 'side', None)),
        'model_probability': _for_side(row.model_probability,
                                       getattr(row, 'side', None)),
        # The MARKET's number stays P(up) always: it is the venue's mid, not a
        # claim about the side we took, and `model_minus_market` compares the
        # two in up terms.
        'market_probability': row.market_probability,
        # **The same quantity read for the SIDE TAKEN, for display only.**
        # `model_probability` above is already side-relative, so plotting it on
        # one axis against a P(up) market put the two on OPPOSITE axes for
        # every down row: measured 2026-10-20, a down trade showed model 0.7951
        # against a market caret at 0.2050 — they sum to 1.0 — so a +0.51pp
        # edge was drawn as a 59-point gap, with the caret on the wrong side of
        # the break-even hairline. A third of the table is down rows.
        #
        # Added rather than converting the field above, because the gate reads
        # that one and must keep comparing in up terms.
        'market_probability_side': _for_side(row.market_probability,
                                             getattr(row, 'side', None)),
        # 'quote' when the venue's own ask priced this, 'baseline' when the
        # calibrated barrier stood in for a market that was not observed.
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


def funnel(db: Session, *, days: int = 7) -> dict[str, Any]:
    """Why decisions were refused, and over what span.

    **It returns the span it actually covers, not the one it was asked for.**
    This applied BOTH a `days` bound and the dashboard epoch, then the endpoint
    echoed back `days: 7` regardless -- so a panel headed "7 days" was summing
    2.2 days of rows, with 69% of the period missing and nothing saying so.
    """
    epoch = reset_epoch(db)
    requested = datetime.now(timezone.utc) - timedelta(days=days)
    # The later of the two bounds is the one that binds, and it is the one the
    # caller is told about.
    since = requested if epoch is None else max(requested, epoch)
    rows = db.execute(
        since_reset(select(Prediction.reason, func.count(Prediction.id)), Prediction, epoch)
        .where(Prediction.window_open >= since)
        .group_by(Prediction.reason)
    ).all()
    counts = {reason: int(n) for reason, n in rows}
    total = sum(counts.values())
    covered = (datetime.now(timezone.utc) - since).total_seconds() / 86400.0
    if total == 0:
        # Nothing decided in this span. Returning a single zero-count `traded`
        # row drew a chart axis with one empty bar, which reads as "we looked and
        # the answer was zero" rather than "we have not looked yet".
        return {'stages': [], 'requested_days': days,
                'covered_days': round(covered, 2), 'epoch': None, 'total': 0}
    return {
        'stages': [{
            'reason': reason, 'count': counts.get(reason, 0),
            'share': counts.get(reason, 0) / total if total else None,
            # `traded` is in this list because the funnel has to add to the
            # whole population to be read as shares -- but it is not a refusal,
            # and a chart headed "why it declined" should not lead with it.
            'declined': reason != 'traded',
        } for reason in FUNNEL_ORDER if reason in counts],
        'requested_days': days,
        'covered_days': round(covered, 2),
        'epoch': epoch.isoformat() if epoch is not None else None,
        'total': total,
    }



def _for_side(probability, side):
    """A stored P(up) read as the probability of the SIDE actually taken.

    **The column is P(up) and must stay P(up).** `scripts/promote.py` scores
    `model_minus_market` by comparing `model_probability` against
    `market_probability` and `outcome`, all in up terms, so flipping the stored
    value would break the gate. But a DOWN row displayed as P(up) is nonsense
    to read: measured 2026-10-17, every one of the 80 DOWN trades since the
    start line satisfied `edge == (1 - model_probability) - effective_cost`,
    so the dashboard showed e.g. "paid 0.920, model probability 0.054" for a
    trade whose real forecast was 0.946 and whose edge was +2.07pp.

    The conversion belongs here, at the point of DISPLAY, and nowhere else.
    """
    if probability is None or side is None:
        return probability
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return probability
    return value if str(side) == 'up' else 1.0 - value
def positions(db: Session, *, open_only: bool = False,
              limit: int = 100) -> list[dict[str, Any]]:
    epoch = reset_epoch(db)
    query = since_reset(select(Position), Position, epoch)
    if open_only:
        query = query.where(Position.outcome == Outcome.PENDING.value) \
            .order_by(Position.settle_time)
    else:
        query = query.order_by(Position.window_open.desc()).limit(limit)

    # **The venue's fee, per position, so these rows sum to the open P&L
    # above.** `Position.fee` is 0.00 on anything still open -- it is set at
    # decision time where `assume_maker` makes it zero, and corrected only at
    # settlement -- so a row's P&L was GROSS while the panel's total was net,
    # and four rows summing to $0.88 sat under a headline of $0.83.
    #
    # The venue charges per MARKET and a market can hold several of our
    # entries, so the charge is split by contracts, the same rule settlement
    # already uses.
    venue_fee_for: dict[tuple, float] = {}
    if open_only:
        charged = {r.ticker: float(r.fees_paid or 0.0)
                   for r in db.execute(select(VenuePosition)).scalars().all()}
        if charged:
            # Epoch-filtered like everything else the dashboard shows. A
            # position still open across a reset loses its fee line here,
            # which is the consistent answer: a reset says "show me from
            # here", and a pre-reset fee against post-reset P&L reads as a
            # cost with no trade behind it.
            rows = db.execute(
                since_reset(select(OrderTicket), OrderTicket, epoch)
                .where(OrderTicket.market_ticker.in_(charged))
            ).scalars().all()
            by_ticker: dict[str, list] = {}
            for ticket in rows:
                by_ticker.setdefault(str(ticket.market_ticker), []).append(ticket)
            for ticker, tickets in by_ticker.items():
                fee = charged.get(ticker, 0.0)
                total = sum(float(t.contracts or 0.0) for t in tickets)
                for ticket in tickets:
                    share = ((float(ticket.contracts or 0.0) / total) if total > 0
                             else 1.0 / len(tickets))
                    venue_fee_for[(ticket.symbol, ticket.window_open,
                                   ticket.offset_minutes)] = fee * share

    def open_fee(row) -> float:
        """The venue's charge where it has one, ours otherwise."""
        charge = venue_fee_for.get(
            (row.symbol, row.window_open, row.offset_minutes))
        return charge if charge is not None else float(row.fee or 0.0)

    return [{
        'id': r.id, 'symbol': r.symbol, 'window_open': r.window_open,
        'settle_time': r.settle_time, 'offset_minutes': r.offset_minutes,
        'side': r.side, 'contracts': r.contracts, 'price': r.price,
        'outlay': r.outlay, 'fee': r.fee,
        # Side-relative for display; the column itself stays P(up).
        'model_probability': _for_side(r.model_probability, r.side),
        'baseline_probability': _for_side(r.baseline_probability, r.side),
        'edge': r.edge,
        'outcome': r.outcome, 'settled_up': r.settled_up, 'payout': r.payout,
        'pnl': r.pnl, 'settled_at': r.settled_at,
        # **What the position is worth NOW, from the venue's bid.** Only
        # meaningful while open: once settled, `pnl` is the answer and a mark
        # beside it would be a second, contradictory one.
        'mark_price': r.mark_price,
        'mark_at': r.mark_at,
        'mark_value': (None if r.mark_price is None
                       else round(float(r.mark_price) * float(r.contracts or 0), 6)),
        # Marked minus what we paid. Unrealised: it is not money until the
        # market settles, and a binary goes to the full win or the full loss.
        'unrealized_pnl': (
            None if r.mark_price is None or r.outcome != Outcome.PENDING.value
            else round(float(r.mark_price) * float(r.contracts or 0)
                       - float(r.outlay or 0.0) - open_fee(r), 6)),
        # What opening it actually cost in fees: the venue's figure while it
        # is open, ours only as a fallback.
        'open_fee': (round(open_fee(r), 6)
                     if r.outcome == Outcome.PENDING.value else None),
    } for r in db.execute(query).scalars().all()]


def model_state(db: Session) -> dict[str, Any]:
    """The artifact that is TRADING, not the newest attempt recorded.

    **This took the newest row by `created_at`.** On 2026-10-20 that was
    `20261018T091125Z` -- a candidate that failed four gates and was never
    installed -- while every live prediction carried `20261018T064404Z`. So the
    Model and Calibration pages showed the gates, the calibration curve and the
    headline statistics of a model that had never placed an order, with nothing
    on screen saying so. Someone asking "is the thing trading my money any
    good?" was reading a different model's report card.

    The installed one is what trades, so the installed one is what is served.
    The newest attempt is still returned alongside as `latest_candidate`, since
    a blocked candidate is information -- it is the trial count, and the reason
    the ledger keeps failures -- it just is not the live model.
    """
    row = db.execute(
        select(ModelRun).where(ModelRun.installed.is_(True))
        .order_by(ModelRun.created_at.desc())
    ).scalars().first()
    newest = db.execute(
        select(ModelRun).order_by(ModelRun.created_at.desc())
    ).scalars().first()
    if row is None:
        # Nothing installed: fall back to the newest attempt rather than an
        # empty page, and say which it is so it cannot be read as the live one.
        row = newest
    if row is None:
        return {
            'present': False,
            'reason': 'no promotion attempt recorded — run `python -m scripts.promote`',
        }
    return {
        'present': True, 'reason': None,
        'version': row.version, 'created_at': row.created_at,
        'installed': bool(row.installed), 'forced': bool(row.forced),
        # **Whether this is the artifact actually placing orders.** False means
        # nothing has ever been installed and the page is showing a candidate.
        'is_live': bool(row.installed),
        # A newer attempt that did NOT install, if there is one. Surfaced so a
        # blocked candidate is visible without being mistaken for the live
        # model — the ledger keeps failures because they are the trial count.
        'latest_candidate': (
            None if newest is None or newest.version == row.version else {
                'version': newest.version,
                'created_at': newest.created_at,
                'installed': bool(newest.installed),
                'forced': bool(newest.forced),
                'failed_gates': ((newest.failed_gates or '').split(', ')
                                 if newest.failed_gates else []),
            }),
        'force_reason': row.force_reason,
        'folds': row.folds, 'windows_evaluated': row.windows_evaluated,
        'log_loss_skill': row.log_loss_skill,
        'log_loss_skill_se': row.log_loss_skill_se,
        'folds_positive': row.folds_positive,
        'calibration_error': row.calibration_error,
        'residual_scale': row.residual_scale,
        'control_gain_share': row.control_gain_share,
        'sharpe': row.sharpe, 'total_return': row.total_return,
        # **`measured` distinguishes "cannot compute yet" from "computed and
        # lost".** The market comparisons read NaN until the live loop has
        # recorded enough quotes, and both arrived as `passed: false` -- one
        # red row for two different claims. Derived here from the value's
        # presence so it works on records written before `promote.py` started
        # sending it; newer ones carry it and a `note` explaining the gate.
        'gates': [
            {**g, 'measured': g.get('measured', g.get('value') is not None)}
            for g in (row.gates or [])
        ],
        # Computed every run, written to the ledger, and never surfaced until
        # now. The page tells the reader to discount skill by the trial count;
        # these are what performs that discount.
        'trial_count': (row.provenance or {}).get('trial_count'),
        'deflated_sharpe': (row.provenance or {}).get('deflated_sharpe'),
        'gate_spec_hash': (row.provenance or {}).get('gate_spec_hash'),
        # The alpha actually BAKED INTO the artifact, which is a different
        # quantity from the `residual_scale` gate: measured 2026-10-20 the gate
        # read 0.9404 across folds while the deployed model applied 0.0151 --
        # 60x apart, same name, nothing on screen distinguishing them.
        'deployed_residual_scale': ((row.provenance or {}).get('model') or {}).get('residual_scale'),
        'report_stats': (row.provenance or {}).get('report') or {},
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
        'log_loss_skill': r.log_loss_skill,
        # **`folds_positive` is meaningless without its denominator.** A count
        # of agreeing folds only compares across runs at a FIXED fold count:
        # "5 of 10" is a 62% event under the null where "5 of 6" is a 10.9%
        # one. The ledger showed `agree 6` (6/6, perfect) sorted below
        # `agree 25` (25/29), which is backwards. `core/metrics.py` retired the
        # gate for exactly this reason; the column reintroduced it.
        'folds_positive': r.folds_positive, 'folds': r.folds,
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
        # **The INSTALLED artifact, for the reason `model_state` documents** —
        # this defaulted to the newest attempt and drew the reliability curve
        # of a blocked candidate under the heading of the live model.
        version = db.execute(
            select(ModelRun.version).where(ModelRun.installed.is_(True))
            .order_by(ModelRun.created_at.desc())
        ).scalars().first()
        if version is None:
            version = db.execute(
                select(ModelRun.version).order_by(ModelRun.created_at.desc())
            ).scalars().first()
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
    """Recent one-minute bars for one symbol, for the window chart.

    A barrier problem's natural picture is the price path against the line it has
    to finish above, and the prediction rows carry only four points per window —
    enough to decide with, far too sparse to draw.
    """
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

    The chart draws these as horizontal segments spanning their own window, which
    is what makes "above or below" readable at a glance — a single line across the
    whole chart would be wrong, because the strike is reset every fifteen minutes.
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
    """Live order tickets.

    This repository places orders through the Kalshi client when told to twice;
    a ticket is the record of the decision either way, so a dry run and a fill
    are distinguishable after the fact.
    """
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
        'expected_cost': r.expected_cost,
        'model_probability': _for_side(r.model_probability, r.side),
        'edge': r.edge, 'status': r.status,
        'filled_contracts': r.filled_contracts, 'filled_price': r.filled_price,
        'filled_at': r.filled_at, 'note': r.note,
    } for r in db.execute(query).scalars().all()]


# ---------------------------------------------------------------------------
# The venue's ledger.
#
# Everything above this line reports our own arithmetic: a bankroll debited at
# the price `decide()` sized at, a fee predicted from the published schedule, and
# a payout decided by an OHLC mean of Coinbase standing in for sixty seconds of
# CF Benchmarks BRTI. Live, all three are estimates of numbers Kalshi already
# holds, and where they disagree the venue is right.
#
# These read `venue_fills`, `venue_settlements` and `venue_balance`, which the
# trader fills from `/portfolio/fills`, `/portfolio/settlements` and
# `/portfolio/balance`. Two rules follow from "the API serves measurements, never
# substitutes":
#
# * Per-settlement P&L is NOT computed here. It is a stored column, written by
#   the one process that reads the venue, so the arithmetic has one definition
#   rather than one per container. What happens here is summation and ordering.
# * A `None` is counted, not coerced. A settlement whose revenue the venue did
#   not serve has an unknown P&L; adding it as zero would understate a loss, and
#   an equity curve must never err in that direction. `incomplete` carries the
#   count so the frontend can say the total is short.
#
# The public tape (`/historical/trades`) is deliberately absent. It carries every
# print in a market by anyone, with no account attribution, so it cannot produce
# a portfolio — summing it sums the exchange.


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
        # Ours beside theirs, on the same row. The gap is a settlement our
        # Coinbase proxy called differently or a fee we mispriced; a null means
        # the venue settled a market we have no record of buying.
        'our_pnl': row.our_pnl,
        'pnl_gap': (None if row.pnl is None or row.our_pnl is None
                    else row.pnl - row.our_pnl),
        'won': _venue_won(row),
        'synced_at': row.synced_at,
    }


def _venue_won(row: VenueSettlement) -> Optional[bool]:
    """Did we win, per the venue's own resolution of the market?

    Read off `market_result` against the side we held, not off the sign of the
    P&L. Those come apart exactly where this system trades: a winner bought at
    97c and charged a fee can net negative, and calling that a loss would put the
    win rate at odds with the venue's settlement record.

    Mirrors `core.venue_ledger.won` — deliberately, since the API cannot import
    the trader. It is a classification of two stored fields rather than
    arithmetic, which is the line this file draws.
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


#: The venue settles on a UTC clock, but Kalshi's own app rolls the trading day
#: over at MIDNIGHT EASTERN, and that is the boundary the operator reconciles
#: against. Measured 2026-10-19: settlements since 00:00 UTC summed to $10.08
#: while the account showed $7.07 -- the difference was two markets from the
#: previous evening ET. Grouping by UTC would make every day on this calendar
#: disagree with the venue's own by a few trades.
EASTERN = 'America/New_York'


def venue_daily_pnl(db: Session, *, days: int = 120) -> dict[str, Any]:
    """Realised P&L per EASTERN calendar day, from the venue's own settlements.

    **Epoch-scoped, like every other figure on this page.** It was briefly
    served unfiltered on the reasoning that a calendar exists to show history;
    that put days from a different configuration beside days from this one,
    under one month total, with nothing on the surface saying where one ended.
    The dashboard epoch is exactly the line that separates them, so the
    calendar honours it and the page means one thing throughout.

    Days with no settlement are omitted rather than sent as zero -- a day the
    system did not trade is not a break-even day, and drawing it flat would put
    a false point on anyone's mental equity curve.
    """
    epoch = reset_epoch(db)
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute(
        since_reset(select(VenueSettlement), VenueSettlement, epoch)
        .where(VenueSettlement.settled_time >= since)
        .order_by(VenueSettlement.settled_time)
    ).scalars().all()

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.settled_time is None:
            continue
        stamp = row.settled_time
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        key = stamp.astimezone(ZoneInfo(EASTERN)).date().isoformat()
        day = buckets.setdefault(key, {
            'date': key, 'pnl': 0.0, 'markets': 0, 'contracts': 0.0,
            'fees': 0.0, 'wins': 0, 'losses': 0, 'incomplete': 0,
        })
        day['markets'] += 1
        day['contracts'] += (row.yes_contracts or 0.0) + (row.no_contracts or 0.0)
        if row.fee_cost is not None:
            day['fees'] += float(row.fee_cost)
        if row.pnl is None:
            # Counted, never treated as break-even: the venue left a field
            # absent, so the day's total is short by an unknown amount.
            day['incomplete'] += 1
            continue
        day['pnl'] += float(row.pnl)
        verdict = _venue_won(row)
        if verdict is True:
            day['wins'] += 1
        elif verdict is False:
            day['losses'] += 1

    # **A day is the change in ACCOUNT VALUE across it, and nothing else.**
    #
    # This was built three times the hard way and each attempt failed the same
    # reader test. First as realised P&L on markets that settled in the day,
    # which disagreed with the Kalshi app. Then as the cash change, which sawed
    # up and down as the trader opened and closed windows. Then as cash plus a
    # reconstruction of what was held at each boundary -- which produced a
    # correction nobody could read, because a single net number covering two
    # different midnights does not mean anything you can say in a sentence.
    #
    # Account value is `cash + what the open book is worth`, and the venue
    # serves both on `/portfolio/balance` -- the second as `portfolio_value`,
    # its OWN mark. Sampled every second by `scripts/mark_positions.py`,
    # because Kalshi has no history endpoint of any kind (probed exhaustively
    # 2026-10-20: summary, pnl, history, balance_history, statements, ledger
    # are all 404). A day's P&L is then the last sample of the day minus the
    # last sample of the day before. No boundary correction exists, because a
    # window open at midnight is simply inside both measurements.
    #
    # Fees, settlement, netting and the whole fee schedule are already in those
    # two numbers. Nothing here models any of them.
    #
    # Deposits and withdrawals would corrupt this, being cash that is not
    # strategy -- `incomplete` carries the warning and the dashboard says so.
    # None have landed inside any sampled day on this account.
    #
    # **One row per day, selected in the database.** The sampler writes a row a
    # second -- 86,400 a day -- so loading the range into Python and keeping
    # the last of each day, which is what this did first, would pull millions
    # of rows to compute a few dozen numbers. `DISTINCT ON` does the same work
    # at the index. The bucketing is `AT TIME ZONE` in Postgres so the day
    # boundary is the same one the settlement buckets above use; computing it
    # in two places in two languages is how they drift apart.
    eastern = ZoneInfo(EASTERN)
    closes = db.execute(text('''
        SELECT DISTINCT ON (day) day, timestamp, balance, total_balance, portfolio_value,
               first_value(balance)         OVER w AS open_balance,
               first_value(total_balance)   OVER w AS open_total,
               first_value(portfolio_value) OVER w AS open_portfolio
        FROM (
            SELECT (timestamp AT TIME ZONE :tz)::date AS day,
                   timestamp, balance, total_balance, portfolio_value
            FROM venue_balance
            WHERE timestamp >= :since AND timestamp >= :epoch
        ) rows
        WINDOW w AS (PARTITION BY day ORDER BY timestamp
                     ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        ORDER BY day, timestamp DESC
    '''), {'tz': EASTERN, 'since': since,
           # No epoch means the whole ledger, expressed as a floor nothing is
           # below rather than as a NULL branch -- `:epoch::timestamptz` reads
           # as a bind parameter to SQLAlchemy and will not parse.
           'epoch': epoch or datetime(1970, 1, 1, tzinfo=timezone.utc)},
    ).mappings().all()

    close_value: dict[str, float] = {}
    #: Only used for the FIRST day in range, which has no previous close.
    open_value: dict[str, float] = {}
    closed_at: dict[str, Any] = {}
    _mark_only: dict[str, float] = {}
    ordered_days: list[str] = []
    marked_days: set[str] = set()
    for row in closes:
        # **The trading shard, consistently -- NOT the cross-shard total.**
        # Preferring the total and falling back to the shard mixed two bases
        # within one subtraction: 2026-10-18 opened on a row with
        # `total_balance` NULL (472.28, the shard) and closed on one that had
        # it (486.81, the total), inventing the $1.11 that permanently sits on
        # the other shards. These are DIFFERENCES, and the stranded cash does
        # not move, so the shard gives the same answer with no NULLs in it.
        cash = row['balance']
        if cash is None:
            continue
        # **Null is not zero here.** Rows written before the sampler existed
        # carry no mark, and treating that as a flat book would report every
        # open position at a day's end as a total loss and then as a windfall
        # the next morning. Those days are measured on cash alone and flagged.
        mark = row['portfolio_value']
        key = row['day'].isoformat()
        ordered_days.append(key)
        closed_at[key] = row['timestamp']
        close_value[key] = float(cash) + (0.0 if mark is None else float(mark))
        open_cash = (row['open_total'] if row['open_total'] is not None
                     else row['open_balance'])
        if open_cash is not None:
            open_value[key] = float(open_cash) + (
                0.0 if row['open_portfolio'] is None
                else float(row['open_portfolio']))
        if mark is not None:
            marked_days.add(key)
            _mark_only[key] = float(mark)
    ordered_days.sort()
    newest = ordered_days[-1] if ordered_days else None
    first_day = ordered_days[0] if ordered_days else None

    def day_change(key: str) -> Optional[float]:
        """Close minus the previous day's close."""
        if key not in close_value:
            return None
        position = ordered_days.index(key)
        if position == 0:
            # **The first day in range falls back to its settled P&L.**
            #
            # Differencing it against its own first sample was tried and is
            # worse: the epoch lands mid-day AND mid-trade -- measured, three
            # positions costing $5.56 were already open at 2026-10-18 12:33 --
            # so that day's opening sample is a balance with their cost
            # already deducted and nothing recorded for what they were worth.
            # The day then reported a gain it had not made.
            #
            # Settled P&L is the honest answer for a day whose opening
            # position is unknown, and `partial` says it is not comparable
            # with the days around it.
            return None
        return round(close_value[key] - close_value[ordered_days[position - 1]], 4)

    days_out = [buckets[k] for k in sorted(buckets)]
    for day in days_out:
        day['cash_change'] = day_change(day['date'])
        # Retained as a field so the payload shape does not change under the
        # frontend, but there is no boundary correction any more: the account
        # value already contains whatever was open at each end.
        day['open_value'] = 0.0
        # Partial when the day is still running, or when it is the first day in
        # range and has no previous close to difference against.
        day['partial'] = day['date'] in (newest, first_day)
        # Measured on cash alone, because no mark was recorded that day. Worth
        # saying: it is right only if the book was flat at the boundary.
        day['unmarked'] = day['date'] not in marked_days
        day['pp_per_contract'] = (
            round(day['pnl'] / day['contracts'] * 100.0, 3)
            if day['contracts'] else None)
        day['pnl'] = round(day['pnl'], 4)
        day['fees'] = round(day['fees'], 4)

    # **The first day also gets credited with whatever it ended still holding.**
    #
    # It is measured as settled P&L, which counts nothing for a position that
    # had not resolved by midnight. The NEXT day is measured as a change in
    # account value, and its opening value does include that position -- so
    # the open book at that one boundary was being subtracted from day two
    # without ever having been added to day one. Measured 2026-10-20: $0.32,
    # showing up as +$0.53 on 10-19 and -$0.21 on 10-20, and leaving the
    # headline 32c above the settled total on a completely flat book.
    #
    # Adding the closing unrealised to the first day cancels it exactly.
    if days_out and ordered_days:
        boundary = ordered_days[0]
        at = closed_at.get(boundary)
        first = next((d for d in days_out if d['date'] == boundary), None)
        if first is not None and at is not None and boundary in marked_days:
            held = db.execute(
                since_reset(select(Position), Position, epoch)
                .where(Position.window_open <= at)
                .where((Position.settled_at.is_(None))
                       | (Position.settled_at > at))
            ).scalars().all()
            cost = sum(float(p.outlay or 0.0) + float(p.fee or 0.0) for p in held)
            # What the open book was worth at that midnight, minus what it
            # cost to put on: the unrealised P&L carried into the next day.
            carried = round(_mark_only.get(boundary, 0.0) - cost, 4)
            first['carried_open'] = carried
            if first['cash_change'] is None:
                first['cash_change'] = round(first['pnl'] + carried, 4)

    # The sum of exactly what the cells show, so the header can never disagree
    # with the squares underneath it.
    shown_total = round(sum(
        (d['cash_change'] if d['cash_change'] is not None else d['pnl'])
        for d in days_out), 4)

    return {
        'timezone': EASTERN,
        # So the surface can say what it is showing rather than imply it is
        # everything. Null when no reset is set, which means the full ledger.
        'epoch': epoch.isoformat() if epoch is not None else None,
        'days': days_out,
        'first': days_out[0]['date'] if days_out else None,
        'last': days_out[-1]['date'] if days_out else None,
        # The SHOWN total -- the sum of the cells. `settled_total` is the old
        # quantity (realised P&L on markets that resolved), kept because the
        # promotion gates and the research surface still speak in it.
        'total': shown_total,
        'settled_total': round(sum(d['pnl'] for d in days_out), 4),
    }


def venue_account_state(db: Session) -> dict[str, Any]:
    """The account as the venue's ledger reports it, with ours beside it.

    Returns `available: False` with a reason rather than zeros when the ledger is
    empty — which is the normal state of a paper account, since a paper account
    has no venue ledger at all. A dashboard that renders an unsynced ledger as
    `$0.00 realised` is claiming a measurement it does not have.
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
            'total_balance': _missing('no venue balance sampled'),
            'stranded': _missing('no venue balance sampled'),
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
    # Settled at the venue with no position of ours behind it. Joined through
    # `order_tickets`, which is the only thing that knows which market one of
    # our rows was sent to.
    booked = {
        str(t.market_ticker)
        for t in db.execute(
            since_reset(select(OrderTicket), OrderTicket, epoch)
            .where(OrderTicket.market_ticker.is_not(None))
        ).scalars().all()
    }
    unbooked = 0
    unbooked_pnl = 0.0
    for row in rows:
        if str(row.ticker) in booked:
            continue
        unbooked += 1
        if row.pnl is not None:
            unbooked_pnl += float(row.pnl)
    unbooked_pnl = round(unbooked_pnl, 4)

    return {
        'available': True,
        'reason': None,
        # Carried on this response too, not just on `/account`. A live figure
        # that renders identically to a paper one is the worst failure this
        # surface could permit, and this is the response that holds real money.
        'mode': account.mode if account is not None else 'paper',
        'settlements': len(rows),
        'realized_pnl': (_measured(realized) if have_pnl
                         else _missing('every settlement was missing a field')),
        'fees': (_measured(fees) if have_fees
                 else _missing('the venue served no fee on any settlement')),
        # **Shard, then total.** `balance` is the trading shard -- the only cash
        # an order on a crypto market can draw on, and what Kelly sizes off.
        # `total_balance` is the cross-shard figure the venue's own app puts on
        # screen. Showing one without the other is why the dashboard read
        # $482.46 against an app saying $483.57 with drift at exactly 0.00.
        'balance': (_measured(balance.balance) if balance is not None
                    else _missing('no venue balance sampled')),
        'total_balance': (_measured(balance.total_balance)
                          if balance is not None and balance.total_balance is not None
                          else _missing('the venue served no cross-shard total; '
                                        'rows written before 2026-10-18 have none')),
        'win_rate': (_measured(wins / decided) if decided
                     else _missing('nothing has settled with a named result')),
        'wins': wins, 'losses': losses, 'undecided': undecided,
        # How many settlements are excluded from `realized_pnl` because the venue
        # left a field absent. Non-zero means the total is short, and by an
        # unknown amount rather than by zero.
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
        # Cash the trading shard cannot reach. Derived here rather than stored
        # so it can never disagree with the two figures it is the difference of.
        'stranded': (_measured(float(balance.total_balance) - float(balance.balance))
                     if balance is not None and balance.total_balance is not None
                     else _missing('no cross-shard total on the last sample')),
        'exchange_index': balance.exchange_index if balance is not None else None,
        # **The one check that still means anything.**
        #
        # `pnl_gap` compares two figures that are now both set from the
        # venue's own `realized_pnl`, so it is Kalshi against Kalshi and reads
        # 0.00 by construction. What it cannot be is a fee we mispriced or a
        # settlement our proxy called differently -- nothing is priced or
        # called on our side any more.
        #
        # The failure that survives is a market the venue settled and we never
        # booked at all: an order POST that timed out after being accepted
        # leaves a real position with no row of ours, and it is the one
        # discrepancy that costs money silently. Measured 2026-10-20, there
        # was one. Counted here rather than inferred from a dollar gap,
        # because a market we never booked can settle at exactly break-even
        # and move no total at all.
        'unbooked': unbooked,
        'unbooked_pnl': (_measured(unbooked_pnl) if unbooked
                         else _measured(0.0)),
    }


def venue_equity_curve(db: Session, *, days: int = 30) -> dict[str, Any]:
    """The account chart: realised P&L stepped by settlement, cash sampled beside it.

    Two series, because they answer different questions and neither one answers
    both.

    `pnl` is cumulative realised P&L, one step per settled market, and it is what
    "how is the portfolio doing" means. It is built from settlement P&L and **not**
    from balance differences, which is the load-bearing choice here: nothing in
    the ledger distinguishes a deposit from a profit, so a balance-difference
    curve reports the first deposit as the best day the strategy ever had.

    `balance` is the venue's cash, sampled every cycle. It is the real money and
    it makes the chart live between settlements, but on its own it sawtooths —
    entering a position debits it, settling credits it back — so it is the second
    series rather than the first.

    Both are windowed by `days`, and the cumulative series is therefore
    **cumulative within the window**: it starts at zero at the left edge rather
    than carrying the account's lifetime total. `pnl_before_window` says what was
    excluded, so a caller can shift the series onto a lifetime scale and a reader
    is never shown a partial total labelled as the whole.
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
        # What settled before the window opened, so the series can be read on a
        # lifetime scale. Null when nothing did.
        'pnl_before_window': None if earlier is None else float(earlier),
        'realized_pnl_in_window': total if points else None,
    }
