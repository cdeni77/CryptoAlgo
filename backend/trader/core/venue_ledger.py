"""P&L from the venue's ledger, not from our own arithmetic.

Live, the venue is the account of record. This module turns `/portfolio/fills`
and `/portfolio/settlements` into serving-store rows. Our own figures are kept
alongside, and the gap between them is a measurement.

`/historical/trades` is not used: it is the public tape with no account
attribution, so summing it sums the exchange.

Rules:

* A missing field is not a zero. An unparsed settlement has P&L `None`.
* Live and historical tiers overlap, so the same `trade_id` can arrive twice.
  The client deduplicates; the store's unique keys are the backstop.
* The ledger cannot tell a deposit from profit, so the curve uses settlement
  P&L, not balance changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


def is_ours(ticker: str, series: Optional[Iterable[str]] = None) -> bool:
    """Is this market one the loop trades?

    The account may hold unrelated markets, which must not enter the equity curve.
    Every writer of this ledger uses this filter, including `scripts/sync_venue.py`.

    Matches the series prefix from `SERIES_BY_SYMBOL` plus a hyphen
    (`KXBTC15M-26SEP041130-30`), so `KXBTC15MINI` does not match `KXBTC15M`.
    `series` is injectable because the caller's binding of the series map is the authority.
    """
    if series is None:
        from core.config import SERIES_BY_SYMBOL
        series = SERIES_BY_SYMBOL.values()
    ticker = (ticker or '').strip().upper()
    return any(ticker.startswith(f'{s.upper()}-') for s in series if s)


def fill_row(fill: Any) -> dict[str, Any]:
    """A `Fill` as a `venue_fills` row.

    Only names columns: `kalshi_client.parse_fill` owns the wire translation.
    """
    return {
        'trade_id': fill.trade_id,
        'order_id': fill.order_id,
        'ticker': fill.ticker,
        'side': fill.side,
        'action': fill.action,
        'contracts': float(fill.contracts),
        'price': fill.price,
        'is_taker': fill.is_taker,
        'created_time': fill.created_time,
        'raw': fill.raw,
    }


def settlement_row(settlement: Any, *, position: Any = None) -> dict[str, Any]:
    """A `Settlement` as a `venue_settlements` row, with our own figure beside it.

    `our_pnl` is `None` when no position of ours maps to the ticker. That means
    the venue settled something we have no record of buying.
    """
    return {
        'ticker': settlement.ticker,
        'event_ticker': settlement.event_ticker,
        'market_result': settlement.market_result,
        'yes_contracts': float(settlement.yes_contracts),
        'no_contracts': float(settlement.no_contracts),
        'yes_cost': settlement.yes_cost,
        'no_cost': settlement.no_cost,
        'revenue': settlement.revenue,
        'fee_cost': settlement.fee_cost,
        'pnl': settlement.pnl,
        'settled_time': settlement.settled_time,
        'position_id': getattr(position, 'id', None) if position is not None else None,
        'our_pnl': getattr(position, 'pnl', None) if position is not None else None,
        'raw': settlement.raw,
    }


def won(*, market_result: Optional[str], yes_contracts: float,
        no_contracts: float) -> Optional[bool]:
    """Did we win this market, per the venue's own resolution?

    Reads `market_result`, not the P&L's sign: a winner bought at 97c can net
    negative after the fee.

    Returns None when the venue named no result, or when the row holds neither side.
    """
    if not market_result:
        return None
    if yes_contracts > 0 and no_contracts > 0:
        # Both sides of one market violates one-entry-per-window. No single answer.
        return None
    if yes_contracts > 0:
        return market_result == 'yes'
    if no_contracts > 0:
        return market_result == 'no'
    return None


@dataclass(frozen=True)
class LedgerSummary:
    """The account, as the venue's ledger reports it.

    `incomplete` counts settlements whose P&L is unknown because the venue left a
    field absent. It is surfaced, not absorbed into the totals.
    """

    settlements: int
    realized_pnl: Optional[float]
    fees: Optional[float]
    revenue: Optional[float]
    cost: Optional[float]
    contracts: float
    wins: int
    losses: int
    undecided: int
    incomplete: int
    first_settled: Optional[datetime]
    last_settled: Optional[datetime]

    @property
    def win_rate(self) -> Optional[float]:
        decided = self.wins + self.losses
        return self.wins / decided if decided else None


def summarise(rows: Sequence[Any]) -> LedgerSummary:
    """Total up settlement rows. Rows may be ORM objects or plain dicts.

    Sums skip and count a `None`, never coerce it to zero. With every row
    incomplete the totals are `None`, not `0.0`.
    """
    def field(row: Any, name: str) -> Any:
        return row.get(name) if isinstance(row, dict) else getattr(row, name, None)

    realized = 0.0
    fees = 0.0
    revenue = 0.0
    cost = 0.0
    contracts = 0.0
    have_pnl = have_fees = have_revenue = have_cost = False
    wins = losses = undecided = incomplete = 0
    stamps: list[datetime] = []

    for row in rows:
        pnl = field(row, 'pnl')
        if pnl is None:
            incomplete += 1
        else:
            realized += float(pnl)
            have_pnl = True

        fee = field(row, 'fee_cost')
        if fee is not None:
            fees += float(fee)
            have_fees = True
        rev = field(row, 'revenue')
        if rev is not None:
            revenue += float(rev)
            have_revenue = True
        for name in ('yes_cost', 'no_cost'):
            value = field(row, name)
            if value is not None:
                cost += float(value)
                have_cost = True

        yes_n = float(field(row, 'yes_contracts') or 0.0)
        no_n = float(field(row, 'no_contracts') or 0.0)
        contracts += yes_n + no_n

        verdict = won(market_result=field(row, 'market_result'),
                      yes_contracts=yes_n, no_contracts=no_n)
        if verdict is True:
            wins += 1
        elif verdict is False:
            losses += 1
        else:
            undecided += 1

        when = field(row, 'settled_time')
        if when is not None:
            stamps.append(when)

    return LedgerSummary(
        settlements=len(rows),
        realized_pnl=realized if have_pnl else None,
        fees=fees if have_fees else None,
        revenue=revenue if have_revenue else None,
        cost=cost if have_cost else None,
        contracts=contracts,
        wins=wins, losses=losses, undecided=undecided, incomplete=incomplete,
        first_settled=min(stamps) if stamps else None,
        last_settled=max(stamps) if stamps else None,
    )


def cumulative_curve(rows: Iterable[Any], *,
                     starting_balance: Optional[float] = None) -> list[dict[str, Any]]:
    """Cumulative realised P&L, one point per settlement, oldest first.

    Built from settlement P&L, not balance changes: a balance curve counts a
    deposit as profit. Rows with P&L `None` get no point.

    `starting_balance` shifts the curve onto an equity scale. Without it the
    series starts at zero and reads as P&L.
    """
    points: list[dict[str, Any]] = []
    total = 0.0

    def field(row: Any, name: str) -> Any:
        return row.get(name) if isinstance(row, dict) else getattr(row, name, None)

    ordered = sorted(
        (r for r in rows if field(r, 'settled_time') is not None),
        key=lambda r: field(r, 'settled_time'),
    )
    for row in ordered:
        pnl = field(row, 'pnl')
        if pnl is None:
            continue
        total += float(pnl)
        points.append({
            'timestamp': field(row, 'settled_time'),
            'ticker': field(row, 'ticker'),
            'pnl': float(pnl),
            'cumulative_pnl': total,
            'equity': None if starting_balance is None else starting_balance + total,
        })
    return points


def balance_check(*, venue_balance: float, settlements: Sequence[Any],
                  fills: Sequence[Any]) -> dict[str, Any]:
    """Does the ledger's own cash flow explain the balance? Reports, never corrects.

    `net_flow` is settlement revenue in, fill cost and fees out. The venue balance
    also holds the starting cash and any deposits, so the two are not expected to
    be equal. A difference that grows between syncs signals a double-counted fee
    or an unseen fill.
    """
    def field(row: Any, name: str) -> Any:
        return row.get(name) if isinstance(row, dict) else getattr(row, name, None)

    revenue = sum(float(field(r, 'revenue') or 0.0) for r in settlements)
    fees = sum(float(field(r, 'fee_cost') or 0.0) for r in settlements)
    spent = 0.0
    unpriced = 0
    for fill in fills:
        price = field(fill, 'price')
        if price is None:
            unpriced += 1
            continue
        spent += float(price) * float(field(fill, 'contracts') or 0.0)

    net_flow = revenue - spent - fees
    implied_start = venue_balance - net_flow
    if unpriced:
        logger.warning(
            '%d fill(s) carried no price, so the cash-flow check is short by '
            'whatever they cost. This is a parse gap, not a free trade.', unpriced)
    return {
        'venue_balance': venue_balance,
        'revenue': revenue,
        'spent': spent,
        'fees': fees,
        'net_flow': net_flow,
        # The starting balance implied if no money was ever deposited or withdrawn.
        # Drift between syncs is the alarm.
        'implied_starting_balance': implied_start,
        'fills_without_price': unpriced,
        'settlements': len(settlements),
        'fills': len(fills),
    }
