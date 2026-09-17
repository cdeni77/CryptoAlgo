"""Restart the dashboard's P&L view without destroying the evidence.

**Nothing is deleted.** Sets `account.reset_at`, and the API shows only rows on
or after it. The reason is specific rather than cautious:

  * `predictions` is the ONLY out-of-sample measurement of whether the model
    beats the PRICE. `PgWriter.scored_against_market` reads nothing else, and
    both `model_minus_market` and `calibration_vs_market` -- the two gates this
    whole stack exists to satisfy -- are computed from it. There are 21,956
    scored rows and they cannot be regenerated.
  * `venue_fills` and `venue_settlements` are the venue's own ledger for windows
    Kalshi itself no longer serves; the live endpoints refuse to look back past
    a moving ~3-month cutoff.

So a truncate would buy a clean chart with the one number that is not
self-graded. `--undo` clears the marker and the full history returns.

`scripts/evaluate` and `scripts/promote` never read this column: research always
sees everything, so a reset cannot silently shrink the sample a gate is computed
on.

`--rebase` additionally sets the stored `starting_bankroll` to the CURRENT
bankroll, so the new equity curve starts flat at 0% instead of carrying the old
return forward.

**It does NOT change position sizing**, which is worth stating because the
opposite is the obvious guess. `decide()` reads
`sizing_base = bankroll if config.compound else config.starting_bankroll`, and
`config.starting_bankroll` comes from the loop's own `--bankroll` flag, not from
this row -- the two have been out of step for some time, the row reading 100.00
against a configured 500.00. The live loop also runs `--compound`, so it sizes
off the running balance either way. This row feeds the dashboard's return
calculation and nothing else.
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
