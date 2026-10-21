"""Re-base our own books onto the venue's ledger, which is the account of record.

**This corrects arithmetic; it does not invent money.** Two defects left our
figures diverging from Kalshi's, both fixed in the code on 2026-10-20:

  * `settle_position` computed `pnl = payout - outlay`, and `outlay` is
    `contracts * price` exactly, so the fee was never subtracted. Our books
    reported GROSS where the venue reports net -- measured over 419 settled
    markets, 409 differed from the venue by precisely the fee and nothing else.
  * `Position.fee` was 0.0000 on every row, because `maker_entry` implies
    `assume_maker` and that estimate is made at DECISION time, before it is
    known whether the order rests or crosses. About 19% of fills cross.

The code fix only applies to settlements from here on. This back-fills what
already settled, so `pnl_gap` starts from a clean baseline and any NEW
divergence is visible rather than buried under a known constant.

**Reconciled PER MARKET, not by overwriting the account total.**
`PgWriter.correct_outcomes` -- which the live loop runs periodically --
restates the invariant `realized_pnl == sum(positions.pnl where settled_at >=
epoch)`. An earlier version of this tool set `account.realized_pnl` to the
venue's figure directly; the next cycle recomputed it from the positions and
the gap came straight back, to the same cent. The invariant is right, so the
fix has to be in the positions it sums.

**What it does NOT fix:**

  * `uq_position_window` is UNIQUE on (symbol, window_open), so under
    multi-entry only the FIRST entry of a window is ever booked. The venue
    holds the rest. That needs a schema change, not a back-fill.
  * We settle from an OHLC mean of Coinbase standing in for CF Benchmarks
    BRTI. The two disagree on roughly 3% of windows, all of them near-ties,
    and that is irreducible without changing the label source.

So a small drifting gap is expected and healthy. A gap that GROWS is the
signal -- an unrecorded fill, a partial, or a fee we mispriced.

    python -m scripts.rebase_books --dry-run
    python -m scripts.rebase_books
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger('rebase-books')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true',
                        help='print what would change and write nothing')
    parser.add_argument('--book-unmatched', action='store_true',
                        help='create positions for settled markets we hold no '
                             'position for, from the venue\'s own record. '
                             'Recording a fill that happened, not inventing '
                             'one — see the note in this script.')
    parser.add_argument('--since', default=None,
                        help='ISO instant to re-base from (default: the '
                             'dashboard epoch, account.reset_at)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    url = os.getenv('DATABASE_URL')
    if not url:
        logger.error('DATABASE_URL is unset. Run this inside the compose '
                     'service, where the serving store is reachable.')
        return 1

    import pandas as pd
    from sqlalchemy import select

    from core.pg_writer import (Account, Outcome, OrderTicket, PgWriter,
                                Position, VenueSettlement)

    writer = PgWriter(database_url=url)
    with writer._session() as session:  # noqa: SLF001 - a maintenance tool
        account = session.query(Account).order_by(Account.id).first()
        if account is None:
            logger.error('no account row')
            return 1
        since = (pd.Timestamp(args.since) if args.since
                 else pd.Timestamp(account.reset_at) if account.reset_at
                 else None)
        if since is None:
            logger.error('no --since and no account.reset_at to fall back on')
            return 1
        if since.tzinfo is None:
            since = since.tz_localize('UTC')
        logger.info('re-basing from %s', since.isoformat())

        # ticker -> the venue's own fee, via the ticket that names the market.
        fee_by_ticker = {
            row.ticker: float(row.fee_cost or 0.0)
            for row in session.execute(select(VenueSettlement)).scalars()
            if row.fee_cost is not None
        }
        ticker_by_window: dict = {}
        for row in session.execute(select(OrderTicket)).scalars():
            if not row.market_ticker:
                continue
            # Keyed on the DECISION where possible, since a window can hold one
            # ticket per offset; the (symbol, window) key is kept as a fallback
            # for rows written before the offset was part of the identity.
            ticker_by_window[(row.symbol, row.window_open, row.offset_minutes)] = \
                row.market_ticker
            ticker_by_window.setdefault((row.symbol, row.window_open),
                                        row.market_ticker)

        # **Filtered on `settled_at`, not on `window_open`.** The venue side
        # below is bounded by `settled_time`, and `correct_outcomes` restates
        # `realized_pnl == sum(positions.pnl where settled_at >= epoch)` -- so
        # bounding ours on the window instead put a position that OPENED before
        # the epoch and SETTLED after it on one side of the comparison only.
        # Measured: that single row left a -0.083 residual that no amount of
        # repricing could close, because the two sides were summing different
        # sets.
        positions = [
            p for p in session.execute(select(Position)).scalars()
            if p.settled_at is not None and pd.Timestamp(p.settled_at) >= since
        ]

        by_ticker_row = {
            row.ticker: row
            for row in session.execute(select(VenueSettlement)).scalars()
        }
        pnl_by_ticker = {
            row.ticker: float(row.pnl)
            for row in session.execute(select(VenueSettlement)).scalars()
            if row.pnl is not None
        }
        # Several positions can share a market once multi-entry is on, so the
        # venue's per-market P&L is split across them BY CONTRACT. With one
        # position per market -- every row before 2026-10-20 -- this is a plain
        # assignment.
        rows_by_ticker: dict[str, list] = {}
        for position in positions:
            ticker = ticker_by_window.get(
                (position.symbol, position.window_open, position.offset_minutes))
            if ticker is None:
                ticker = ticker_by_window.get(
                    (position.symbol, position.window_open))
            if ticker is not None:
                rows_by_ticker.setdefault(ticker, []).append(position)

        repriced = 0
        fee_total = 0.0
        pnl_before = 0.0
        pnl_after = 0.0
        for ticker, group in rows_by_ticker.items():
            venue_pnl_row = pnl_by_ticker.get(ticker)
            if venue_pnl_row is None:
                for position in group:
                    pnl_before += float(position.pnl or 0.0)
                    pnl_after += float(position.pnl or 0.0)
                continue
            venue_fee = fee_by_ticker.get(ticker, 0.0)
            settlement = by_ticker_row.get(ticker)
            venue_ct = 0.0
            venue_cost = 0.0
            if settlement is not None:
                venue_ct = (float(settlement.yes_contracts or 0.0)
                            + float(settlement.no_contracts or 0.0))
                venue_cost = (float(settlement.yes_cost or 0.0)
                              + float(settlement.no_cost or 0.0))
            # **Equal shares when the group holds no contracts at all.**
            # `share = contracts / total` gives 0 for a zero-contract position,
            # so the venue's P&L for that market was never assigned to
            # anything and the gap could not close. Zero-contract positions are
            # real and are their own bug: `int(round(count))` rounds a
            # sub-contract partial fill down to 0, so position 1133 recorded
            # `contracts=0, outlay=$0.07` against a venue fill of 1 contract
            # and -$0.70.
            total_ct = sum(float(p.contracts or 0.0) for p in group)
            for position in group:
                share = (float(position.contracts or 0.0) / total_ct
                         if total_ct > 0 else 1.0 / len(group))
                old_pnl = float(position.pnl or 0.0)
                new_pnl = venue_pnl_row * share
                new_fee = venue_fee * share
                pnl_before += old_pnl
                pnl_after += new_pnl
                fee_total += new_fee
                # **Size is corrected too, but only when the market maps to
                # ONE position.** With several there is no way to attribute
                # the venue's contract count between them, so they are left
                # alone; with one it is unambiguous and the venue is the
                # account of record. This is what makes `pnl / contracts` --
                # the per-contract figure the promotion gate reads -- agree
                # with the venue rather than divide by a rounded-down zero.
                if len(group) == 1 and venue_ct > 0:
                    if abs(float(position.contracts or 0) - venue_ct) > 1e-9:
                        logger.info('  size %s: %s -> %g contracts (venue)',
                                    ticker, position.contracts, venue_ct)
                        if not args.dry_run:
                            position.contracts = int(round(venue_ct))
                            position.outlay = venue_cost
                            if venue_ct:
                                position.price = venue_cost / venue_ct
                if (abs(new_pnl - old_pnl) > 1e-9
                        or abs(float(position.fee or 0.0) - new_fee) > 1e-9):
                    repriced += 1
                    if not args.dry_run:
                        position.fee = new_fee
                        position.pnl = new_pnl

        logger.info('%d settled position(s) in range, %d repriced',
                    len(positions), repriced)
        logger.info('  fees recovered      $%.4f', fee_total)
        logger.info('  our P&L   %+.4f -> %+.4f  (delta %+.4f)',
                    pnl_before, pnl_after, pnl_after - pnl_before)

        # --- markets the venue settled and we never booked --------------
        #
        # **This records a fill that HAPPENED; it does not invent one.** The
        # case it was written for: KXSOL15M-26OCT180845-45, where the venue
        # holds three maker fills totalling 4 contracts at 0.80 and our ticket
        # says `killed`. The order filled, for more contracts than the ticket
        # asked, and we booked a kill — so no position was ever created and
        # $0.80 of real settled P&L sat outside our books.
        #
        # Every field comes from the venue's settlement row rather than from a
        # model or an assumption: side from `market_result` and which side
        # carries contracts, size and cost from that side's totals, payout from
        # `revenue`. `offset_minutes` is taken from the ticket that named the
        # market, or 0 when there is none -- it is an identity field here, not
        # a claim about when the decision was made.
        if args.book_unmatched:
            booked = 0
            booked_pnl = 0.0
            # **"Already booked" means a position exists for that MARKET, not
            # that this script managed to map it.** Keying off `rows_by_ticker`
            # missed a position sitting at a different offset than its ticket
            # and offered to book it a second time -- the dry run put the gap
            # at -0.70 instead of 0.00, which is what caught it. The question
            # is about the market, so it is asked of the market.
            booked_windows = {
                (str(p.symbol), p.window_open)
                for p in session.execute(select(Position)).scalars()
            }
            for row in session.execute(select(VenueSettlement)).scalars():
                if row.pnl is None:
                    continue
                if row.settled_time is None or pd.Timestamp(row.settled_time) < since:
                    continue
                yes_ct = float(row.yes_contracts or 0.0)
                no_ct = float(row.no_contracts or 0.0)
                if yes_ct > 0 and no_ct > 0:
                    logger.warning('%s holds BOTH sides (%g yes, %g no); not '
                                   'bookable as one position, skipping',
                                   row.ticker, yes_ct, no_ct)
                    continue
                side = 'up' if yes_ct > 0 else 'down'
                contracts = yes_ct if yes_ct > 0 else no_ct
                if contracts <= 0:
                    continue
                cost = float((row.yes_cost if yes_ct > 0 else row.no_cost) or 0.0)
                ticket = session.execute(
                    select(OrderTicket).where(OrderTicket.market_ticker == row.ticker)
                ).scalars().first()
                if ticket is None:
                    logger.warning('%s has no ticket; cannot place it in a '
                                   'window, skipping', row.ticker)
                    continue
                if (str(ticket.symbol), ticket.window_open) in booked_windows:
                    continue
                won = (row.market_result or '').strip().lower() == (
                    'yes' if side == 'up' else 'no')
                logger.info('  book %s %s %g @ %.4f -> pnl %+.4f',
                            row.ticker, side, contracts,
                            cost / contracts if contracts else 0.0,
                            float(row.pnl))
                booked += 1
                booked_pnl += float(row.pnl)
                pnl_after += float(row.pnl)
                fee_total += float(row.fee_cost or 0.0)
                if not args.dry_run:
                    session.add(Position(
                        symbol=ticket.symbol, window_open=ticket.window_open,
                        settle_time=ticket.settle_time,
                        offset_minutes=ticket.offset_minutes,
                        side=side, contracts=int(round(contracts)),
                        price=cost / contracts if contracts else 0.0,
                        outlay=cost, fee=float(row.fee_cost or 0.0),
                        model_probability=float('nan'),
                        baseline_probability=float('nan'),
                        edge=float('nan'),
                        outcome=(Outcome.WON.value if won else Outcome.LOST.value),
                        settled_up=(row.market_result or '').strip().lower() == 'yes',
                        payout=float(row.revenue or 0.0),
                        pnl=float(row.pnl),
                        settled_at=row.settled_time,
                    ))
            logger.info('  %d unmatched market(s) booked, %+.4f', booked, booked_pnl)

        venue_pnl = sum(
            float(r.pnl) for r in session.execute(select(VenueSettlement)).scalars()
            if r.pnl is not None and r.settled_time is not None
            and pd.Timestamp(r.settled_time) >= since)
        venue_fees = sum(
            float(r.fee_cost) for r in session.execute(select(VenueSettlement)).scalars()
            if r.fee_cost is not None and r.settled_time is not None
            and pd.Timestamp(r.settled_time) >= since)

        logger.info('  venue P&L %+.4f   gap after repricing %+.4f',
                    venue_pnl, venue_pnl - pnl_after)
        logger.info('')
        # **Set to the repriced SUM, not to the venue's figure.**
        # `correct_outcomes` restates `realized_pnl == sum(positions.pnl)` on
        # its own schedule, so anything else here survives at most one cycle.
        # Writing the sum means this tool and that invariant agree, and the sum
        # equals the venue's total precisely because the positions above were
        # reconciled to it market by market.
        logger.info('account.realized_pnl %+.4f -> %+.4f',
                    float(account.realized_pnl), pnl_after)
        logger.info('account.fees_paid    %.4f -> %.4f',
                    float(account.fees_paid), fee_total)
        if args.dry_run:
            logger.info('\n--dry-run: nothing written')
            return 0
        account.realized_pnl = pnl_after
        account.fees_paid = fee_total
        session.commit()
        logger.info('\ndone — and it survives correct_outcomes, because the '
                    'positions now sum to this')
    return 0


if __name__ == '__main__':
    sys.exit(main())
