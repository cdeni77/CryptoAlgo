"""Make our books say what Kalshi's books say, market by market.

**Kalshi is the bookkeeper. This script is not an adjustment, it is a copy.**
Everything our own ledger computes about money -- the fill price, the fee, the
settlement value, the payout -- is an estimate of a number the venue already
holds, and every accounting bug this system has had was one of those estimates
drifting. The cure is to stop estimating after the fact.

**The source is `/portfolio/positions?settlement_status=all`, not
`/portfolio/settlements`.** That is not a preference; the settlements feed is
measurably wrong in one case. When both sides of a market are held, Kalshi nets
the offsetting contracts and credits that cash when the netting happens, so the
settlement row reports `revenue: 0` for the netted part. Measured 2026-10-20,
three markets (1 YES/4 NO, 2/2, 2/2) reported revenue 0 against $1, $2 and $2
genuinely owed -- a $5.00 hole in one day's realised P&L, and the entire
disagreement between the dashboard and the venue's own app that day. The
positions feed is correct on exactly those three:

    ticker                     traded   realized   implies revenue
    KXETH15M-26OCT200715-15      1.51      -0.51      1.00  (owed 1.00)
    KXSOL15M-26OCT200745-45      2.64      -0.64      2.00  (owed 2.00)
    KXETH15M-26OCT200800-00      2.14      -0.14      2.00  (owed 2.00)

**`realized_pnl_dollars` is GROSS of fees.** The venue carries the fee in a
separate `fees_paid_dollars`, so the figure our books want is the difference.
Reading the gross as net understates every loss and overstates every win by the
fee, which is the same class of bug this is meant to end.

**Multi-entry splits one market across several of our positions.** The venue
knows a market, not a decision offset, so a market's P&L is divided between our
positions in proportion to contracts -- the same rule the fee already uses. One
position in a market gets the whole thing, which is every row before
multi-entry existed.

Always `--dry-run` first. It prints every market it would change and the total
correction, and writes nothing.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger('reconcile')

#: Below this a difference is rounding in the venue's own dollar strings rather
#: than a booking error, and rewriting the row would churn without informing.
TOLERANCE = 0.0001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true',
                        help='print what would change and write nothing')
    parser.add_argument('--limit', type=int, default=None,
                        help='stop after this many corrected markets')
    parser.add_argument('--verbose', action='store_true',
                        help='print every market, not only the ones that differ')
    parser.add_argument('--no-book-unmatched', action='store_true',
                        help='do not create positions for markets the venue '
                             'settled and we never booked')
    return parser


def _money(row: dict, name: str) -> float:
    try:
        return float(row.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


async def venue_markets(client) -> dict[str, dict]:
    """Every market this account has ever traded, keyed by ticker.

    Paged: ~9 pages at 200 a page as of 2026-10-20. `settlement_status=all` is
    what reaches back past the open ones -- the default returns only what is
    currently held, which is how this was first written and why it found
    nothing to reconcile.
    """
    out: dict[str, dict] = {}
    cursor: Optional[str] = None
    while True:
        params = {'limit': 200, 'settlement_status': 'all'}
        if cursor:
            params['cursor'] = cursor
        payload = await client._request(            # noqa: SLF001
            'GET', '/portfolio/positions', params=params)
        rows = payload.get('market_positions') or []
        for row in rows:
            ticker = row.get('ticker')
            if ticker:
                out[str(ticker)] = row
        cursor = payload.get('cursor')
        if not cursor or not rows:
            return out


def venue_pnl(row: dict) -> float:
    """What the venue says this market made us, net of its own fee."""
    return _money(row, 'realized_pnl_dollars') - _money(row, 'fees_paid_dollars')


def plan(writer, markets: dict[str, dict]) -> list[dict]:
    """What would change, without changing it."""
    from core.pg_writer import OrderTicket, Position

    changes: list[dict] = []
    with writer._session() as session:               # noqa: SLF001
        for ticker, row in sorted(markets.items()):
            # Open markets have nothing realised yet, and writing a 0 over a
            # pending position would book it as a settled break-even.
            if abs(float(row.get('position_fp') or 0.0)) > 0:
                continue
            tickets = (session.query(OrderTicket)
                       .filter(OrderTicket.market_ticker == ticker).all())
            if not tickets:
                continue
            keys = {(t.symbol, t.window_open, t.offset_minutes) for t in tickets}
            positions = []
            for symbol, window_open, offset in keys:
                found = (session.query(Position)
                         .filter(Position.symbol == symbol,
                                 Position.window_open == window_open,
                                 Position.offset_minutes == offset)
                         .one_or_none())
                if found is not None:
                    positions.append(found)
            if not positions:
                continue
            theirs = venue_pnl(row)
            ours = sum(float(p.pnl or 0.0) for p in positions)
            if abs(theirs - ours) <= TOLERANCE:
                continue
            total = sum(float(p.contracts or 0.0) for p in positions)
            for position in positions:
                share = ((float(position.contracts or 0.0) / total) if total > 0
                         else 1.0 / len(positions))
                changes.append({
                    'ticker': ticker,
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'window_open': position.window_open,
                    'offset': position.offset_minutes,
                    'ours': float(position.pnl or 0.0),
                    'theirs': round(theirs * share, 6),
                    'share': share,
                })
    return changes


def apply(writer, changes: list[dict]) -> int:
    from core.pg_writer import Position

    written = 0
    with writer._session() as session:               # noqa: SLF001
        for change in changes:
            position = session.get(Position, change['position_id'])
            if position is None:
                continue
            position.pnl = change['theirs']
            written += 1
        session.commit()
    return written


def plan_settlements(writer) -> list[dict]:
    """Rows in `venue_settlements` whose stored revenue is short.

    Same defect, other table. `revenue` was copied from the settlements feed,
    which reports 0 for the netted part of a market where both sides were
    held, so the stored `pnl` is short by that amount. The dashboard compares
    this table against our positions and calls the difference `pnl_gap` -- so
    leaving it unrepaired turns a fixed bug into a permanent false alarm, with
    the sign inverted now that the positions are right.

    A binary pays $1 a contract on the winning side, so the payout is derived
    rather than fetched.
    """
    from core.pg_writer import VenueSettlement

    changes: list[dict] = []
    with writer._session() as session:                # noqa: SLF001
        for row in session.query(VenueSettlement).all():
            result = (row.market_result or '').strip().lower()
            if result == 'yes':
                payout = float(row.yes_contracts or 0.0)
            elif result == 'no':
                payout = float(row.no_contracts or 0.0)
            else:
                continue                              # the venue named no result
            stored = float(row.revenue) if row.revenue is not None else None
            if stored is not None and abs(payout - stored) <= TOLERANCE:
                continue
            cost = float(row.yes_cost or 0.0) + float(row.no_cost or 0.0)
            fee = float(row.fee_cost or 0.0)
            changes.append({
                'id': row.id,
                'ticker': row.ticker,
                'was': stored,
                'revenue': round(payout, 6),
                'pnl': round(payout - cost - fee, 6),
            })
    return changes


def apply_settlements(writer, changes: list[dict]) -> int:
    from core.pg_writer import VenueSettlement

    with writer._session() as session:                # noqa: SLF001
        for change in changes:
            row = session.get(VenueSettlement, change['id'])
            if row is None:
                continue
            row.revenue = change['revenue']
            row.pnl = change['pnl']
        session.commit()
    return len(changes)


async def run(args=None) -> int:
    from core.pg_writer import PgWriter
    from data_collection.kalshi_client import KalshiClient

    args = args or build_parser().parse_args([])
    url = os.getenv('DATABASE_URL')
    if not url:
        logger.error('DATABASE_URL is unset')
        return 2
    writer = PgWriter(database_url=url)

    # Not `live=True`: this reads the venue and writes only our own database.
    client = KalshiClient()
    async with client:
        markets = await venue_markets(client)
    logger.info('the venue knows %d markets on this account', len(markets))

    # The venue's settlement rows first, because `pnl_gap` compares them
    # against the positions below and repairing only one side would report a
    # gap that is now entirely our own doing.
    settlement_changes = plan_settlements(writer)
    for change in settlement_changes:
        logger.info('%-28s venue_settlements revenue %s -> %.4f',
                    change['ticker'],
                    'null' if change['was'] is None else f"{change['was']:.4f}",
                    change['revenue'])
    if settlement_changes and not args.dry_run:
        apply_settlements(writer, settlement_changes)
        logger.info('repaired %d settlement row(s)', len(settlement_changes))

    changes = plan(writer, markets)
    if args.limit:
        changes = changes[:args.limit]
    if not changes:
        if not settlement_changes:
            logger.info('every booked market already agrees with the venue')
        if not args.no_book_unmatched:
            book_unmatched(args.dry_run)
        return 0

    delta = sum(c['theirs'] - c['ours'] for c in changes)
    for change in changes:
        logger.info('%-28s offset %-3s  ours %+9.4f -> venue %+9.4f  (%+.4f)',
                    change['ticker'], change['offset'], change['ours'],
                    change['theirs'], change['theirs'] - change['ours'])
    logger.info('%d position(s) across %d market(s); net correction %+.4f',
                len(changes), len({c['ticker'] for c in changes}), delta)

    if args.dry_run:
        logger.info('dry run — nothing written')
        if not args.no_book_unmatched:
            book_unmatched(True)
        return 0
    written = apply(writer, changes)
    logger.info('wrote %d position(s). `correct_outcomes` restates '
                'account.realized_pnl from these rows, so the headline follows '
                'on its own rather than being set here.', written)
    if not args.no_book_unmatched:
        book_unmatched(False)
    return 0


def book_unmatched(dry_run: bool) -> None:
    """Create positions for markets the venue settled and we never booked.

    **Delegated, not reimplemented.** `rebase_books` already does this, and it
    knows the awkward parts: a market held on BOTH sides is not one position
    and is skipped rather than guessed at, and the forecast columns are written
    NaN rather than a plausible 0.5 -- these rows record a fill that happened,
    not a decision we made, and a fabricated probability would quietly enter
    the research sample as if it were one.

    Run last. Repricing the existing rows first means the unmatched set is only
    what is genuinely missing, rather than everything that merely disagreed.
    """
    from scripts import rebase_books

    argv = ['--book-unmatched'] + (['--dry-run'] if dry_run else [])
    rebase_books.main(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
