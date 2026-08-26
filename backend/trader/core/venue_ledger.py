"""P&L from the venue's ledger, not from our own arithmetic.

**The venue is the account of record, and until now nothing here acted like it.**
The paper engine debits a bankroll at the price `decide()` sized at and the fee it
predicted from the published schedule, then credits a payout decided by an OHLC
mean of Coinbase standing in for sixty seconds of CF Benchmarks BRTI. Live, every
one of those three is an estimate of a number someone else already holds: the
fill price, the fee charged, and the settlement value. `adopt_venue_balance` was
the only place that admitted it, and it admitted it one number at a time.

This module is the other direction. It takes `/portfolio/fills` and
`/portfolio/settlements` — what the venue recorded — and produces the rows the
serving store keeps and the dashboard draws. Our own figures are carried
*alongside* rather than replaced, because the gap between them is a measurement
in its own right: a drift that grows is a mispriced fee, a partial fill nobody
booked, or a settlement our Coinbase proxy got wrong.

**What this deliberately does not use: `/historical/trades`.** That endpoint is
the public tape — every print in a market, by anyone, with no account attribution
at all. It is the natural thing to reach for and it cannot compute a portfolio:
summing it sums the exchange. Its two honest jobs are marking an open position at
a price the market printed (our own forecast must never do that job) and checking
that a fill printed where the venue said it did, joined on `trade_id`. Both live
in `KalshiClient.market_trades`; neither is P&L.

Three shapes of failure are handled explicitly, because each one has a plausible
wrong answer:

* **A missing field is not a zero.** A settlement whose revenue did not parse has
  an unknown P&L, and `None` travels through to the column. Treating it as
  break-even understates a loss and flatters the curve, which is the one
  direction of error an equity curve must never make.
* **The tiers overlap.** Live and historical fills both cover the cutoff, so the
  same `trade_id` arrives twice; counted twice it doubles a cost basis.
  Deduplication happens in the client, and the store's unique keys are the
  backstop.
* **Deposits are invisible.** Nothing in the ledger distinguishes a $50 deposit
  from $50 of profit, so the curve is built from settlement P&L rather than from
  balance changes. `balance_check` exists to say when the two disagree, which is
  the only way a deposit — or an unrecorded fill — becomes visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


def fill_row(fill: Any) -> dict[str, Any]:
    """A `Fill` as a `venue_fills` row.

    Flat and dumb on purpose: the translation from the wire lives in
    `kalshi_client.parse_fill`, so this only names columns. A second place that
    reinterprets `side` or re-derives a price is a second place to get it wrong.
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

    `pnl` is stored rather than computed on read — see `VenueSettlement` for why —
    and `our_pnl` is whatever our own books said about the same market, or `None`
    when no position of ours maps to this ticker. That null is informative: it
    means the venue settled something we have no record of buying, which is what
    an order POST that timed out after being accepted leaves behind.
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
    """Did *we* win this market, per the venue's own resolution?

    Classification from stored fields, not from the P&L's sign. Those come apart
    at the edge that matters: a winner bought at 97c and charged a fee can net out
    negative, and calling that a loss would make the win rate disagree with the
    venue's own settlement record. The system buys favourites, so that edge is
    where most of the trades are.

    `None` when the venue did not name a result, or when the row holds neither
    side — a market we did not trade has no answer, not a losing one.
    """
    if not market_result:
        return None
    if yes_contracts > 0 and no_contracts > 0:
        # Both sides of the same market. Never observed — one entry per window is
        # an invariant — and it has no single answer, so it does not get a guess.
        return None
    if yes_contracts > 0:
        return market_result == 'yes'
    if no_contracts > 0:
        return market_result == 'no'
    return None


@dataclass(frozen=True)
class LedgerSummary:
    """The account, as the venue's ledger reports it.

    `incomplete` is the count of settlements whose P&L could not be computed
    because the venue left a field absent. It is surfaced rather than absorbed:
    a total that quietly excludes four rows is a different number from a total
    over everything, and only one of them is the account.
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

    Sums skip a `None` and count it, rather than coercing it to zero. With every
    row incomplete the totals are `None` rather than `0.0` — "we cannot say" and
    "you made nothing" are different claims and a dashboard renders them
    differently.
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

    **Built from settlement P&L rather than from balance changes**, and that is
    the load-bearing choice. A balance-difference curve is tempting — it needs no
    arithmetic and always agrees with the venue — but nothing in the ledger
    distinguishes a deposit from a profit, so the first time money is added the
    curve reports it as the best day the strategy ever had. Settlement P&L cannot
    make that error.

    Rows whose P&L is `None` do not get a point. Carrying the running total
    forward across a gap is right — the total genuinely has not changed by a known
    amount — and inventing a step of zero would draw a flat segment that looks
    like a measurement.

    `starting_balance` shifts the curve onto an equity scale when given. Without
    it the series starts at zero and reads as P&L, which is the honest default:
    the venue's ledger does not say what the account held before its first
    settlement, and back-projecting today's balance through the P&L assumes no
    deposit ever happened.
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
    """Does the ledger's own cash flow explain the balance? Report, never correct.

    Two independent readings of the same account: the balance the venue serves,
    and what the fills and settlements say should have happened to it. They agree
    only if every fill and settlement was seen, every fee was read as the venue
    charged it, and nothing was deposited or withdrawn.

    So a gap does not identify its own cause, and this makes no attempt to guess.
    It is a smoke alarm for the two errors that would otherwise be invisible: a
    fee counted twice — `Settlement.pnl` subtracts `fee_cost` on the assumption
    that `revenue` is gross, and if that is backwards every settled trade is
    understated by its fee — and a fill the ledger never saw.

    `net_flow` is what the ledger says the account gained: settlement revenue in,
    fill cost and fees out. It is compared against the balance the venue reports,
    which also contains whatever the account started with and any deposit since,
    so the two are **not** expected to be equal. The number worth watching is
    whether the difference stays put or grows.
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
        # What the account must have held before any of this, if no money was ever
        # deposited or withdrawn. A figure that drifts between syncs is the alarm;
        # a figure that simply is not the starting bankroll usually means a deposit.
        'implied_starting_balance': implied_start,
        'fills_without_price': unpriced,
        'settlements': len(settlements),
        'fills': len(fills),
    }
