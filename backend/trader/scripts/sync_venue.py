"""Pull the venue's own ledger into the serving store, and total it up.

The venue is the account of record. This reads `/portfolio/fills` and
`/portfolio/settlements` from both the live and historical tiers, writes
`venue_fills` and `venue_settlements`, and prints the venue's P&L beside ours.

The live loop already stores what each reconcile returns. This covers a gap
while the loop was down and anything older than the live tier's cutoff.

Read-only against the venue. The client is built without `live=True`, so it
cannot place an order.

    python -m scripts.sync_venue                  # everything both tiers hold
    python -m scripts.sync_venue --days 7         # just the last week
    python -m scripts.sync_venue --dry-run        # read and total, write nothing
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from core import venue_ledger
from core.pg_writer import PgWriter
from data_collection.kalshi_client import KalshiClient, KalshiError

logger = logging.getLogger('sync_venue')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--days', type=int, default=None,
                        help='Only fills and settlements newer than this many '
                             'days. Default: everything the venue will serve, '
                             'because a P&L over a truncated ledger is not the '
                             "account's P&L.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Read and total, but write nothing to the store.')
    parser.add_argument('--database-url', type=str, default=None,
                        help='Override DATABASE_URL.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def _money(value: Optional[float]) -> str:
    """Format a dollar figure, or a dash when the venue did not provide one.

    Never `$0.00` for a missing value: a missing figure is not zero.
    """
    return '—' if value is None else f'${value:,.2f}'


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-14s %(message)s',
        datefmt='%H:%M:%S', stream=sys.stdout,
    )

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)
             if args.days else None)

    client = KalshiClient()          # not live: this cannot place an order
    if not client.configured:
        print('Kalshi credentials are not configured. Set KALSHI_KEY_ID and '
              'either KALSHI_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH.')
        return 2

    async with client:
        cutoff = await client.historical_cutoff()
        if cutoff:
            for name, when in sorted(cutoff.items()):
                print(f'  live/historical cutoff · {name}: {when.isoformat()}')
        else:
            print('  live/historical cutoff: unreadable — querying both tiers')

        try:
            fills = await client.all_fills(since=since)
            settlements = await client.all_settlements(since=since)
            balance = await client.balance()
        except KalshiError as exc:
            print(f'\nThe venue refused: {exc}')
            return 1

    print(f'\n  {len(fills)} fill(s), {len(settlements)} settlement(s) read')

    writer = PgWriter(args.database_url or os.getenv('DATABASE_URL'))
    # The ownership filter the live loop applies: only markets this system
    # trades enter the equity curve.
    foreign_s = [s for s in settlements if not venue_ledger.is_ours(getattr(s, 'ticker', ''))]
    foreign_f = [f for f in fills if not venue_ledger.is_ours(getattr(f, 'ticker', ''))]
    if foreign_s or foreign_f:
        print(f'  skipping {len(foreign_s)} settlement(s) and {len(foreign_f)} fill(s) '
              f'on markets this loop does not trade')
    settlements = [s for s in settlements if venue_ledger.is_ours(getattr(s, 'ticker', ''))]
    fills = [f for f in fills if venue_ledger.is_ours(getattr(f, 'ticker', ''))]

    settlement_rows = []
    for settled in settlements:
        # Our own position for the same market, joined through `order_tickets`,
        # the only place a venue ticker is recorded.
        position = None if args.dry_run else writer.position_for_ticker(settled.ticker)
        settlement_rows.append(venue_ledger.settlement_row(settled, position=position))
    fill_rows = [venue_ledger.fill_row(f) for f in fills]

    if args.dry_run:
        print('  --dry-run: nothing written')
    else:
        written_fills = writer.upsert_venue_fills(fill_rows)
        written_settlements = writer.upsert_venue_settlements(settlement_rows)
        print(f'  stored {written_fills} fill(s), '
              f'{written_settlements} settlement(s)')

    summary = venue_ledger.summarise(settlement_rows)
    print('\n  the venue\'s ledger')
    print(f'    settled markets   {summary.settlements}')
    print(f'    realised P&L      {_money(summary.realized_pnl)}')
    print(f'    revenue           {_money(summary.revenue)}')
    print(f'    cost              {_money(summary.cost)}')
    print(f'    fees              {_money(summary.fees)}')
    print(f'    contracts         {summary.contracts:,.0f}')
    win_rate = summary.win_rate
    print(f'    won / lost        {summary.wins} / {summary.losses}'
          + (f'  ({win_rate * 100:.1f}%)' if win_rate is not None else ''))
    if summary.undecided:
        print(f'    unresolved        {summary.undecided} '
              '(the venue named no result)')
    if summary.incomplete:
        # Report incomplete rows; a total that silently excludes rows is wrong.
        print(f'    INCOMPLETE        {summary.incomplete} settlement(s) had a '
              'field missing and are excluded from the P&L above')

    # Ours beside theirs. They are not expected to be identical.
    account = writer.account()
    if account is not None:
        print('\n  our own books, for comparison')
        print(f'    mode              {account.mode}')
        print(f'    bankroll          ${account.bankroll:,.2f}')
        print(f'    realised P&L      ${account.realized_pnl:,.2f}')
        print(f'    fees paid         ${account.fees_paid:,.2f}')
        if summary.realized_pnl is not None:
            gap = summary.realized_pnl - float(account.realized_pnl)
            print(f'    P&L gap           {gap:+,.2f}  (venue minus ours)')

            # Report how much of the gap is coverage: our store is reset between
            # experiments and often covers a shorter window than the venue.
            ours_from = writer.first_position_time()
            if ours_from is not None and summary.first_settled is not None:
                theirs = summary.first_settled
                if theirs < ours_from - timedelta(hours=1):
                    print(f'      Coverage differs: the venue settles from '
                          f'{theirs:%Y-%m-%d %H:%M} and our books only from '
                          f'{ours_from:%Y-%m-%d %H:%M}, so these totals are '
                          f'over different periods and the gap is mostly that. '
                          f'Wiping the store between experiments does this and '
                          f'is deliberate.')
            if abs(gap) > 0.01:
                print('      Beyond coverage, a gap is a settlement our '
                      'Coinbase proxy called differently, a fee we mispriced, '
                      'or a fill we never booked. The venue is right.')

    check = venue_ledger.balance_check(
        venue_balance=balance, settlements=settlement_rows, fills=fill_rows)
    print('\n  cash-flow check')
    print(f'    venue balance     {_money(check["venue_balance"])}')
    print(f'    revenue in        {_money(check["revenue"])}')
    print(f'    spent on fills    {_money(check["spent"])}')
    print(f'    fees out          {_money(check["fees"])}')
    print(f'    net flow          {check["net_flow"]:+,.2f}')
    print(f'    implied start     {_money(check["implied_starting_balance"])}')
    print('      What the account must have held before any of this, if nothing '
          'was ever deposited.')
    print('      Watch whether it MOVES between syncs: a stable figure that is '
          'not the starting')
    print('      bankroll is usually a deposit; a drifting one is an unrecorded '
          'fill or a')
    print('      double-counted fee.')
    if check['fills_without_price']:
        print(f'    {check["fills_without_price"]} fill(s) carried no price, so '
              'the spend above is short by whatever they cost.')

    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
