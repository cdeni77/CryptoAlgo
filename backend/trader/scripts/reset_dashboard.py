"""Restart the dashboard's P&L view without deleting anything.

Sets `account.reset_at`; the API shows only rows on or after it. Nothing is
deleted: `predictions` is the only out-of-sample record of model against
market and cannot be regenerated, and the venue no longer serves old fills and
settlements. `--undo` clears the marker.

`scripts/evaluate` and `scripts/promote` never read this column, so a reset
cannot shrink a gate's sample.

`--rebase` also sets the stored `starting_bankroll` to the current bankroll,
so the new equity curve starts at 0%. It does not change position sizing:
`decide()` takes `starting_bankroll` from the loop's config, not this row.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger('reset-dashboard')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--at', type=str, default=None,
                        help='ISO instant to reset to (default: now)')
    parser.add_argument('--rebase', action='store_true',
                        help='also set starting_bankroll to the current '
                             'bankroll, so the curve starts flat. Changes '
                             'position sizing.')
    parser.add_argument('--undo', action='store_true',
                        help='clear the marker; the full history returns')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    url = os.getenv('DATABASE_URL')
    if not url:
        logger.error('DATABASE_URL is unset. This must run where the serving '
                     'store is reachable — in the compose service, not on the '
                     'host.')
        return 1

    from core.pg_writer import Account, PgWriter

    writer = PgWriter(database_url=url)
    with writer._session() as session:                    # noqa: SLF001
        row = session.query(Account).order_by(Account.id).first()
        if row is None:
            logger.error('no account row yet; nothing to reset')
            return 1

        logger.info('account: mode=%s starting_bankroll=%.2f bankroll=%.2f '
                    'reset_at=%s', row.mode, row.starting_bankroll,
                    row.bankroll, row.reset_at)

        if args.undo:
            logger.info('clearing reset_at — the full history returns')
            if not args.dry_run:
                row.reset_at = None
                session.commit()
            return 0

        when = (datetime.fromisoformat(args.at) if args.at
                else datetime.now(timezone.utc))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)

        logger.info('setting reset_at = %s', when.isoformat())
        if args.rebase:
            logger.info('rebasing the STORED starting_bankroll %.2f -> %.2f '
                        '(dashboard return only; sizing reads the loop\'s '
                        '--bankroll flag, and it compounds off the balance)',
                        row.starting_bankroll, row.bankroll)
        logger.info('NOTHING is deleted; --undo restores the view')

        if args.dry_run:
            logger.info('--dry-run: no change written')
            return 0

        row.reset_at = when
        if args.rebase:
            row.starting_bankroll = float(row.bankroll)
            row.realized_pnl = 0.0
            row.fees_paid = 0.0
        session.commit()
    logger.info('done — reload the dashboard')
    return 0


if __name__ == '__main__':
    sys.exit(main())
