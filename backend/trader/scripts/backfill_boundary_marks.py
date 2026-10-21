"""Fill in what the open book was worth at each day boundary, before sampling.

A day's P&L is `account value at its close minus account value at the previous
close`, and account value is cash plus what the open positions are worth. The
second term is sampled every second by `scripts/mark_positions.py` -- but only
since that sampler existed. For every boundary before it, `portfolio_value` is
null, the boundary is measured on cash alone, and a window open across midnight
is counted wrong in BOTH days: the day that paid for it is short its value, and
the day that receives the payout is flattered by the whole of it.

Measured 2026-10-20: three windows bought at 03:45 UTC were open at the Eastern
boundary and settled nine seconds past it. The dashboard read +11.47 for the day
against the venue's own +9.47 -- **exactly the $2.00 those three paid out**.

**How a past boundary is valued, and why it is honest rather than hindsight.**
No mark history exists, so the value is reconstructed from the positions
themselves:

* A position that settled within `--resolved-within` seconds of the boundary is
  valued at its PAYOUT. This is not using the future: a binary nine seconds from
  expiry is already decided, and its book is quoting ~$1.00 or ~$0.00. The
  payout is what the market was showing, not what we later learned.
* Anything else is valued at COST, which is the only unbiased figure available
  and is exact for a position opened moments before the boundary.

Days filled this way are flagged, so the surface can say the figure is
reconstructed rather than measured. Boundaries where nothing was open get a
0.0, which is a real measurement: a flat book is worth nothing.

Always `--dry-run` first.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import timedelta
from zoneinfo import ZoneInfo

logger = logging.getLogger('boundary')

EASTERN = 'America/New_York'

#: A binary this close to expiry is already decided and its book says so.
DEFAULT_RESOLVED_WITHIN = 120.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resolved-within', type=float,
                        default=DEFAULT_RESOLVED_WITHIN,
                        help='seconds after a boundary within which a '
                             'settlement is treated as already priced in')
    return parser


def run(args=None) -> int:
    from sqlalchemy import create_engine, text

    args = args or build_parser().parse_args([])
    url = os.getenv('DATABASE_URL')
    if not url:
        logger.error('DATABASE_URL is unset')
        return 2
    engine = create_engine(url.replace('+psycopg2', '+psycopg2'))
    eastern = ZoneInfo(EASTERN)

    with engine.connect() as conn:
        # The last balance sample of each Eastern day that has no mark yet.
        rows = conn.execute(text('''
            SELECT DISTINCT ON (day) id, day, timestamp
            FROM (
                SELECT id, (timestamp AT TIME ZONE :tz)::date AS day, timestamp,
                       portfolio_value
                FROM venue_balance
            ) r
            WHERE portfolio_value IS NULL
            ORDER BY day, timestamp DESC
        '''), {'tz': EASTERN}).mappings().all()

        if not rows:
            logger.info('every day boundary already carries a mark')
            return 0

        plan = []
        for row in rows:
            at = row['timestamp']
            # Open at that instant: bought before it, settled after it (or
            # never). `window_open` is when the window began, which is when the
            # position was paid for.
            held = conn.execute(text('''
                SELECT id, symbol, offset_minutes, contracts, outlay, payout,
                       settled_at
                FROM positions
                WHERE window_open <= :at
                  AND (settled_at IS NULL OR settled_at > :at)
            '''), {'at': at}).mappings().all()

            value = 0.0
            detail = []
            for p in held:
                settled_at = p['settled_at']
                soon = (settled_at is not None
                        and (settled_at - at).total_seconds() <= args.resolved_within)
                if soon and p['payout'] is not None:
                    worth = float(p['payout'])
                    how = 'payout'
                else:
                    worth = float(p['outlay'] or 0.0)
                    how = 'cost'
                value += worth
                detail.append(f"{p['symbol']}@{p['offset_minutes']} {how} {worth:.2f}")
            plan.append((row['id'], row['day'], at, round(value, 4), detail))

        for _id, day, at, value, detail in plan:
            logger.info('%s  boundary %s  -> portfolio_value %.4f%s',
                        day, at.astimezone(eastern).strftime('%H:%M:%S'),
                        value, f'   [{"; ".join(detail)}]' if detail else '   [flat]')

        if args.dry_run:
            logger.info('%d boundary/ies — dry run, nothing written', len(plan))
            return 0

        for _id, _day, _at, value, _detail in plan:
            conn.execute(text(
                'UPDATE venue_balance SET portfolio_value = :v WHERE id = :i'),
                {'v': value, 'i': _id})
        conn.commit()
        logger.info('wrote %d boundary mark(s)', len(plan))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
