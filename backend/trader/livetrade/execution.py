"""Turning a decision into an order: price it, send it, book what filled.

A position is booked only from what the venue says filled. A kill is diagnosed
from the book and the venue's own order record, and a taker kill may be resent
once against a fresh book.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.costs import trade_fee
from core.decide import Decision, Side, WindowExposure, decide
from core.pg_writer import PgWriter
from data_collection.kalshi_client import KalshiClient, KalshiError
from livetrade.quotes import fresh_quote
from livetrade.util import _db

logger = logging.getLogger('live')


async def execute_entry(args, writer: PgWriter, kalshi, decision: Decision, row, *,
                        config: Config, quote, quote_time,
                        others: WindowExposure, bankroll: float, halted: bool,
                        require_quote: bool, prediction_id: Optional[int] = None):
    """Send one entry, and resend once if an immediate_or_cancel took nothing.

    The resend re-reads the touch and re-runs `decide()` on it with the SAME
    model probability, against the window's exposure from every OTHER order
    this cycle, so it can never exceed a limit the first attempt respected. It
    goes out only while the kill is `kill_retry_window_seconds` fresh, only on
    the same side, under `<key>-r1`. A partial fill is never topped up.

    Returns `(final decision, final result, [(decision, killed result), ...])`.
    """
    result = await place_and_book(args, writer, kalshi, decision, row,
                                  config=config, quote_time=quote_time,
                                  prediction_id=prediction_id)
    kills = [(decision, result)] if result.status == 'killed' else []
    final_decision, final = decision, result
    attempt = 0
    while (final.status == 'killed' and final.filled == 0
           and attempt < int(config.kill_retries) and quote is not None):
        age = time.monotonic() - (final.answered_at or 0.0)
        if age > config.kill_retry_window_seconds:
            logger.info('%s: killed %.1fs ago, past the %.1fs resend window',
                        decision.symbol, age, config.kill_retry_window_seconds)
            break
        attempt += 1
        fresh = await fresh_quote(kalshi, quote, config)
        if fresh is None or not fresh.tradeable():
            logger.info('%s: no fresh book to re-price against; not resending',
                        decision.symbol)
            break
        repriced = dict(row)
        repriced['ask_up'] = fresh.ask_for('up')
        repriced['ask_down'] = fresh.ask_for('down')
        for side in ('up', 'down'):
            depth = fresh.depth_dollars(side)
            if depth is not None:
                repriced[f'depth_{side}'] = depth
        again = decide(repriced, config, bankroll=bankroll, exposure=others,
                       require_quote=require_quote, halted=halted)
        logger.info('%s: killed at %.4f; the book now says %s -> %s',
                    decision.symbol, decision.price,
                    f'{again.price:.4f}' if np.isfinite(again.price) else 'n/a',
                    again.describe())
        if not again.traded or again.side is not decision.side:
            break
        stamp = pd.Timestamp(fresh.book_time, unit='s', tz='UTC') \
            if fresh.book_time is not None else pd.Timestamp.now(tz='UTC')
        retried = await place_and_book(
            args, writer, kalshi, again, repriced, config=config,
            quote_time=stamp, attempt=attempt, prediction_id=prediction_id,
            note=(f'resend {attempt} after a kill at {decision.price:.4f}: '
                  f'{again.contracts} @ {again.price:.4f} '
                  f'({fresh.book_source} book)'))
        if retried.status == 'killed':
            kills.append((again, retried))
        final_decision, final = again, retried
    return final_decision, final, kills


def order_limit_price(decision: Decision, *, config: Optional[Config] = None) -> float:
    """The worst price still worth paying for this decision.

    **Pay down to the gate, not to a fraction.** `min_edge_pp` is the threshold
    that admitted this trade, so any fill leaving at least that much is one the
    system has already said yes to — while no fill at all earns zero. Everything
    above the gate is therefore spendable.

    Crossing further costs little: the order is `immediate_or_cancel` with
    `taker_at_cross`, so it fills against resting size from the touch outward
    and stops. The limit bounds the worst fill; it does not set the price.

    **The gate is the one in force** (`config`), not `DEFAULT_CONFIG`'s: a lower
    default gate would spend more than the rule allows.

    `max_slippage_cents` is a rail against a book so thin the order walks it
    absurdly far.
    """
    config = config if config is not None else DEFAULT_CONFIG
    edge = decision.edge if np.isfinite(decision.edge) else 0.0
    spendable = max(0.0, edge - config.min_edge_pp / 100.0)
    allowance = min(spendable, config.max_slippage_cents / 100.0)
    # Never below the price we must actually pay: a limit below the ask cannot
    # fill.
    # No grid snapping here. `place_order` puts the limit on the venue's grid,
    # rounding the way that can still FILL (`ceil` for a bid, `floor` for an
    # ask); snapping down first would turn a sub-cent allowance into zero. The
    # bound that matters is the gate in `decide()`, which a fill one tick above
    # `max_price` still clears by `min_edge_pp - 1c`.
    return float(max(min(0.99, decision.price + allowance), decision.price))


def kill_diagnosis(*, side: str, sent_yes_price: float, book: dict) -> dict:
    """Name which of the three things a non-fill was.

    A miss cannot be attributed without the book as it stood at the kill, and
    with it there are only three possibilities:

      * our limit never crossed         -> a price error, ours
      * it crossed and nothing rested   -> a size race, the venue's
      * it crossed with size resting    -> the order itself is malformed

    Everything here is **YES-denominated**, because the venue's book is two BID
    stacks and every previous confusion in this area came from mixing
    denominations. `sent_yes_price` is the price actually on the wire — what
    `place_order` puts in the body, and what the venue echoes back as
    `yes_price_dollars`.

    `side='up'` sends a YES **bid**, which fills at or ABOVE the YES ask, and
    crosses the NO stack. `side='down'` sends a YES **ask**, which fills at or
    BELOW the YES bid, and crosses the YES stack. Getting that comparison
    backwards is the inverted-comparison bug, so both directions are asserted.
    """
    up = str(side) == 'up'
    need = float(book.get('yes_ask', float('nan')) if up
                 else book.get('yes_bid', float('nan')))
    size = float(book.get('no_bid_size', float('nan')) if up
                 else book.get('yes_bid_size', float('nan')))
    price = float(sent_yes_price)
    if not np.isfinite(need):
        crossed = False
        line = (f'no book on the {"ask" if up else "bid"} side at the kill, so '
                f'the {price:.4f} YES {"bid" if up else "ask"} had nothing to '
                f'cross. A settled or halted market looks like this.')
    else:
        crossed = bool(price >= need - 1e-9) if up else bool(price <= need + 1e-9)
        if not crossed:
            line = (f'our {price:.4f} YES {"bid" if up else "ask"} did not cross: '
                    f'it needed {need:.4f} {"or more" if up else "or less"}. '
                    f'The price was ours to get right.')
        elif not np.isfinite(size) or size <= 0:
            line = (f'our {price:.4f} YES {"bid" if up else "ask"} crossed '
                    f'{need:.4f} but nothing was resting there — a size race, '
                    f'not a price error.')
        else:
            line = (f'our {price:.4f} YES {"bid" if up else "ask"} crossed '
                    f'{need:.4f} with {size:.2f} resting. Not price, not size: '
                    f'the order itself is being refused.')
    return {'crossed': crossed, 'needed': need, 'size_available': size,
            'line': line}


def sent_yes_price(order: dict, decision, limit: float) -> float:
    """The YES price that actually went on the wire.

    The venue echoes it as `yes_price_dollars`, which beats recomputing: it is
    what the venue read, after `place_order`'s rounding onto the tick grid. The
    fallback reproduces that conversion — a DOWN order sends `1 - limit`.
    """
    raw = order.get('yes_price_dollars') if isinstance(order, dict) else None
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return float(limit) if decision.side is Side.UP else 1.0 - float(limit)


async def diagnose_kill(kalshi, decision, order: dict, limit: float) -> str:
    """Read the book and the venue's own order record, right after a kill.

    Both are readable in the second after the kill; the POST reply's shape is
    not the GET order record's, and the recorded ladder is sampled too far from
    the order to attribute a miss.

    Never raises. A diagnosis that breaks the loop is worse than no diagnosis,
    and it runs inside the path that has just failed to buy something.
    """
    bits: list[str] = []
    price = sent_yes_price(order, decision, limit)
    try:
        book = await kalshi.orderbook(decision.market_ticker)
        verdict = kill_diagnosis(side=decision.side.value,
                                 sent_yes_price=price, book=book)
        bits.append(verdict['line'])
        bits.append(f'book at kill: {json.dumps(book, sort_keys=True, default=str)}')
    except Exception as exc:            # noqa: BLE001 - diagnosis must not raise
        bits.append(f'orderbook read failed: {type(exc).__name__}: {exc}')
    try:
        oid = str(order.get('order_id') or '')
        # Filtered by TICKER, so the read is exact (a market holds at most a
        # handful of our orders). The listing can race the POST and miss the
        # order; no retry or sleep, because the POST reply already carries the
        # counts and this is corroboration, not worth stalling the loop for.
        candidates = await kalshi.orders(ticker=decision.market_ticker)
        record = next((o for o in candidates
                       if str(o.get('order_id')) == oid), None)
        if record is None:
            bits.append(
                f'venue has no order {oid[:18]} on {decision.market_ticker} '
                f'among {len(candidates)} it lists there — the POST reply gave '
                f'us an id the listing does not show, which is worth knowing')
        else:
            bits.append(f'venue order record: '
                        f'{json.dumps(record, sort_keys=True, default=str)}')
    except Exception as exc:            # noqa: BLE001
        bits.append(f'order record read failed: {type(exc).__name__}: {exc}')
    for line in bits:
        logger.warning('kill diagnosis | %s', line)
    return ' || '.join(bits)


def filled_from_order(order: dict, requested: int) -> tuple[int, float]:
    """Contracts actually filled, and at what price, from the venue's reply.

    Never `count`, which is the size REQUESTED: a killed order, a partial fill
    and an HTTP 200 with an empty body must not book a full fill.

    Returns `(0, nan)` for anything that is not a confirmed fill. Assuming a fill
    is the one error here that cannot be reconciled later: the position is
    invented, `settle_due` settles it, and the PnL is fiction.
    """
    status = str(order.get('status', '') or '').strip().lower()
    dead = status in ('canceled', 'cancelled', 'killed', 'rejected', 'expired')

    # Counts arrive as fixed-point decimal strings ("10.00"), which `int(str)`
    # rejects, so parse as float. Float also because the venue fills
    # fractionally (one order came back 0.43 + 0.57).
    def count(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # `fill_count_fp` first: it is the only count a V2 order carries. Under
    # `immediate_or_cancel` a partial fill is `status='canceled'` with a
    # non-zero fill count, so status alone cannot decide.
    filled = None
    for key in ('fill_count_fp', 'taker_fill_count', 'filled_count', 'fill_count'):
        if order.get(key) is not None:
            filled = count(order[key])
            if filled is None:
                return 0, float('nan')
            break

    # Never infer a fill from `remaining` on a dead order: remaining reaches zero
    # when the order leaves the book, not only when it trades. It only means
    # anything while the order is live.
    if filled is None and not dead:
        for key in ('remaining_count_fp', 'remaining_count'):
            if order.get(key) is not None:
                remaining = count(order[key])
                if remaining is None:
                    return 0, float('nan')
                filled = float(requested) - remaining
                break

    if filled is None:
        # No count at all. Only 'executed'/'filled' justifies believing the whole
        # order traded; anything else (including an empty body) is unknown, and
        # unknown must not become a position.
        if status in ('executed', 'filled'):
            filled = float(requested)
        else:
            return 0, float('nan')

    filled = max(0, min(int(round(filled)), requested))
    if filled <= 0:
        return 0, float('nan')

    # The price actually paid: `taker_fill_cost_dollars` is the money that left
    # the account, so cost / count is the true average fill, including any
    # improvement on the limit.
    price = float('nan')
    for key in ('taker_fill_cost_dollars', 'maker_fill_cost_dollars',
                'fill_cost_dollars'):
        cost = count(order.get(key))
        if cost is not None and cost > 0:
            paid = cost / float(filled)
            # That cost is denominated in the side actually BOUGHT, while this
            # function's contract is YES-denominated (the caller inverts for
            # DOWN). A NO buy at 7c has `taker_fill_cost_dollars=0.07`; returning
            # it unchanged would have the caller invert it and book 93c.
            outcome = str(order.get('outcome_side', '') or '').strip().lower()
            price = (1.0 - paid) if outcome == 'no' else paid
            break

    if not np.isfinite(price):
        for key in ('average_fill_price_dollars', 'avg_price_dollars',
                    'average_fill_price', 'yes_price_dollars', 'yes_price',
                    'no_price'):
            raw = order.get(key)
            if raw is None:
                continue
            value = count(raw)
            if value is None:
                continue
            # Integer-cent fields on the same keys as dollar ones; the venue serves
            # both shapes. Above 1.0 it can only be cents.
            price = value / 100.0 if value > 1.0 else value
            if key == 'no_price':
                price = 1.0 - price
            break
    return filled, price


class OrderResult(NamedTuple):
    """What one order attempt did, for the caller to act on after the fact."""

    booked: bool
    # 'filled', 'partial', 'killed', 'refused', 'unknown', 'stale', 'unresolved',
    # 'dry_run', 'paper'
    status: str
    filled: int = 0
    order: Optional[dict] = None
    ticket_id: Optional[int] = None
    limit: float = float('nan')
    # time.monotonic() when the venue answered, for the kill-retry window.
    answered_at: Optional[float] = None


def client_order_id_for(decision: Decision, attempt: int = 0) -> str:
    """One key per (symbol, window, offset); a re-priced resend adds `-r<n>`."""
    key = (f'{decision.symbol}-{decision.window_open:%Y%m%d%H%M}'
           f'-{decision.offset:02d}')
    return key if attempt <= 0 else f'{key}-r{attempt}'


async def lookup_after_ambiguity(kalshi, client_order_id: str, ticker: Optional[str],
                                 *, attempts: int = 2, pause: float = 1.0):
    """The venue's record of an order whose POST we never saw answered.

    Returns `(found, record)`: `found` False means the lookup itself failed and
    nothing is known. A listing that does not show the order is asked twice,
    because the order may simply not be listed yet.
    """
    finder = getattr(kalshi, 'order_by_client_id', None)
    if finder is None:
        return False, None
    record = None
    for attempt in range(attempts):
        try:
            record = await finder(client_order_id, ticker=ticker)
        except Exception as exc:                              # noqa: BLE001
            logger.error('could not look up order %s after an ambiguous POST '
                         '(%s)', client_order_id, str(exc)[:160])
            return False, None
        if record is not None:
            return True, record
        if attempt + 1 < attempts:
            await asyncio.sleep(pause)
    return True, None


async def place_and_book(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                         decision: Decision, row, *,
                         config: Config = DEFAULT_CONFIG,
                         quote_time: Optional[pd.Timestamp] = None,
                         attempt: int = 0,
                         note: Optional[str] = None,
                         prediction_id: Optional[int] = None) -> OrderResult:
    """Record the ticket, place the order when asked to twice, book the fill.

    **A position is written only when money actually moved.** An unresolved
    market or a dry run books nothing, so `settle_due` cannot settle a holding
    the venue never heard of.

    **`quote_time` is when the book was read**, checked against
    `max_quote_age_seconds`. None makes no staleness claim.

    The ticket write happens BEFORE the POST, per order, because it is what
    `entries_for_window` counts: a crash between sending and booking must still
    block the window. It runs in a thread so the other orders keep moving.

    A POST that times out, or that the venue answers with a 5xx, may have been
    accepted. The order is looked up by `client_order_id` and booked from the
    venue's own record — filled, partial or nothing. Only if that lookup fails
    is the ticket left `unknown`, which still blocks the window.
    """
    placing = bool(args.place_orders) and kalshi is not None and not args.dry_run
    limit = order_limit_price(decision, config=config)

    if placing and quote_time is not None:
        age = (pd.Timestamp.now(tz='UTC') - pd.Timestamp(quote_time)).total_seconds()
        if age > config.max_quote_age_seconds:
            await _db(
                writer.write_ticket,
                symbol=decision.symbol, window_open=decision.window_open,
                settle_time=decision.settle_time, offset_minutes=decision.offset,
                market_ticker=decision.market_ticker, side=decision.side.value,
                contracts=decision.contracts, limit_price=decision.price,
                max_price=limit, expected_cost=decision.stake,
                model_probability=decision.model_probability, edge=decision.edge,
                prediction_id=prediction_id, status='skipped',
                note=(f'stale quote: {age:.0f}s old, over the '
                      f'{config.max_quote_age_seconds}s limit'))
            logger.warning(
                '%s window %s: quote is %.0fs old, over the %ds limit. '
                'Refusing rather than trading a book that may have moved.',
                decision.symbol, decision.window_open, age,
                config.max_quote_age_seconds)
            return OrderResult(False, 'stale', limit=limit)

    if placing and not decision.market_ticker:
        logger.error(
            '%s window %s: no market resolved, so nothing can be bought. '
            'Abstaining rather than booking a position against our own baseline.',
            decision.symbol, decision.window_open)
        return OrderResult(False, 'unresolved', limit=limit)

    ticket_id = await _db(
        writer.write_ticket,
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        market_ticker=decision.market_ticker, side=decision.side.value,
        contracts=decision.contracts, limit_price=decision.price,
        # The worst price still worth paying: down to the gate, never past it.
        max_price=limit,
        expected_cost=decision.stake, model_probability=decision.model_probability,
        edge=decision.edge, prediction_id=prediction_id,
    )
    if attempt > 0 or note:
        # `write_ticket` is one row per (symbol, window): a resend reuses the
        # killed ticket, which must stop reading as dead before the POST so a
        # crash mid-flight still blocks the window.
        await _db(writer.resolve_ticket, ticket_id, status='new',
                  note=(note or f'resend {attempt}')[:400])

    filled = decision.contracts
    placed_price = decision.price
    venue_fill_price: Optional[float] = None
    order: Optional[dict] = None
    answered_at: Optional[float] = None
    # The market's own exchange, not the order body's default of 0: a mismatch
    # returns `404 market_not_found`, which names the market, not the mismatch.
    exchange_index = int(getattr(row, 'get', lambda *_: 0)('exchange_index', 0) or 0)
    key = client_order_id_for(decision, attempt)
    if placing:
        try:
            # The offset is in this key, so a kill at one offset does not
            # lock every later offset of the window out with a 409.
            # **A maker order RESTS; it does not cross.** `time_in_force=None`
            # omits the field so the venue applies its own default, which is
            # the documented way to rest. The enum is configurable rather than
            # guessed: a wrong value may be rejected (harmless) or silently
            # fill as a taker (not), and nothing in this repo has ever recorded
            # a working one — all 1,556 fills are `is_taker: true`.
            _maker = bool(getattr(config, 'maker_entry', False))
            _tif = (getattr(config, 'maker_time_in_force', None) if _maker
                    else 'immediate_or_cancel')
            order = await kalshi.place_order(
                ticker=decision.market_ticker, side=decision.side.value,
                exchange_index=exchange_index,
                contracts=decision.contracts, limit_price=limit,
                client_order_id=key, time_in_force=_tif, rest=_maker)
            answered_at = time.monotonic()
        except (KalshiError, asyncio.TimeoutError, OSError) as exc:
            ambiguous = (not isinstance(exc, KalshiError)
                         or bool(getattr(exc, 'ambiguous', False)))
            if not ambiguous:
                await _db(writer.resolve_ticket, ticket_id, status='skipped',
                          note=str(exc)[:400])
                logger.error('order refused, no position recorded: %s', exc)
                return OrderResult(False, 'refused', ticket_id=ticket_id, limit=limit)
            # The request may well have reached the venue. Ask it, by our key.
            found, record = await lookup_after_ambiguity(
                kalshi, key, decision.market_ticker)
            if not found:
                await _db(writer.resolve_ticket, ticket_id, status='unknown',
                          note=str(exc)[:400])
                logger.error(
                    'the order request to %s failed in flight (%s) and the venue '
                    'could not be asked about %s. Not booking and not retrying; '
                    'the next reconcile will report a position we do not hold.',
                    decision.market_ticker, exc, key)
                return OrderResult(False, 'unknown', ticket_id=ticket_id, limit=limit)
            if record is None:
                await _db(writer.resolve_ticket, ticket_id, status='skipped',
                          note=(f'POST failed in flight ({str(exc)[:200]}); the '
                                f'venue lists no order {key}'))
                logger.error('the order POST to %s failed in flight (%s) and the '
                             'venue lists no order %s; nothing was bought',
                             decision.market_ticker, exc, key)
                return OrderResult(False, 'refused', ticket_id=ticket_id, limit=limit)
            logger.warning('the order POST to %s failed in flight (%s); the venue '
                           'has %s as %s — booking what it says',
                           decision.market_ticker, exc, key, record.get('status'))
            order = record
            answered_at = time.monotonic()
        except ValueError as exc:
            await _db(writer.resolve_ticket, ticket_id, status='skipped',
                      note=str(exc)[:400])
            logger.error('order refused, no position recorded: %s', exc)
            return OrderResult(False, 'refused', ticket_id=ticket_id, limit=limit)

        filled, fill_price = filled_from_order(order, decision.contracts)
        # V2 quotes everything from the YES side, so a DOWN fill comes back as
        # the price we SOLD yes at; what we paid for the NO contract is 1 - that.
        if np.isfinite(fill_price) and decision.side is not Side.UP:
            fill_price = 1.0 - fill_price
        if filled <= 0:
            # **A RESTING order has not failed; it has not happened YET.**
            #
            # `filled_from_order` reads the POST reply, and for an
            # immediate_or_cancel that is the whole story: it crossed or it
            # died, in one round trip. A `good_till_canceled` order necessarily
            # comes back with zero fills because it is sitting ON the book —
            # the fill arrives seconds later. Booking that as 'killed' is how
            # 2026-10-18's first three maker fills became UNRECORDED positions:
            # the venue held 1 ETH @ 0.79 and 2 SOL @ 0.80, both `is_taker:
            # false`, while our own `positions` table had neither. The ticket
            # notes even said so — "venue has no order ... among 0 it lists
            # there" — because a filled order is not an OPEN order.
            #
            # That matters past the bookkeeping: `exposure`,
            # `max_positions_per_window` and `max_window_exposure_fraction` are
            # all computed from OUR book, so positions it cannot see are
            # positions the risk limits do not bind on.
            #
            # So: a maker order that the venue accepted (it returned an id) and
            # did not explicitly kill is RESTING. The reconcile books the fill
            # if one comes; `cancel_stale_resting` retires it if none does.
            status_text = str(order.get('status', '') or '').strip().lower()
            dead = status_text in ('canceled', 'cancelled', 'killed',
                                   'rejected', 'expired')
            order_id = order.get('order_id')
            if _maker and order_id and not dead:
                logger.info(
                    '%s window %s: resting %d @ %.4f, order %s — not a kill; '
                    'the reconcile books any fill and the sweep retires it '
                    'after %ss unfilled',
                    decision.symbol, decision.window_open, decision.contracts,
                    limit, str(order_id)[:18],
                    getattr(config, 'maker_cancel_seconds', None))
                await _db(writer.resolve_ticket, ticket_id, status='resting',
                          note=(f"resting at {limit:.4f} under "
                                f"time_in_force={_tif!r}; POST carried no fill, "
                                f"which for a maker order means it is on the "
                                f"book rather than dead")[:4000],
                          venue_order_id=order_id)
                return OrderResult(False, 'resting', 0, order, ticket_id,
                                   limit, answered_at)
            logger.warning(
                '%s window %s: the order did not fill (status %r). No position.',
                decision.symbol, decision.window_open, order.get('status'))
            logger.warning('kill diagnosis | post reply: %s',
                           json.dumps(order, sort_keys=True, default=str))
            await _db(writer.resolve_ticket, ticket_id, status='killed',
                      note=(f"status={order.get('status')!r} "
                            f"order_id={str(order.get('order_id', ''))[:60]} "
                            f"key={key}")[:4000],
                      venue_order_id=order.get('order_id'))
            return OrderResult(False, 'killed', 0, order, ticket_id, limit, answered_at)
        if np.isfinite(fill_price):
            placed_price = fill_price
            venue_fill_price = float(fill_price)
        await _db(writer.resolve_ticket,
                  ticket_id, status='filled', filled_contracts=filled,
                  filled_price=placed_price,
                  note=str(order.get('order_id', ''))[:200],
                  venue_order_id=order.get('order_id'))
        if filled < decision.contracts:
            logger.warning('%s window %s: partial fill %d of %d',
                           decision.symbol, decision.window_open,
                           filled, decision.contracts)
        logger.info('filled %d @ %.4f (order %s)', filled, placed_price,
                    order.get('order_id'))
    elif args.mode == 'live':
        # A real book was read and priced; nothing was bought.
        logger.info('dry run: ticket %d written, no order placed, no position',
                    ticket_id)
        return OrderResult(False, 'dry_run', ticket_id=ticket_id, limit=limit)

    # **Book the fill, not the intention.** Outlay and fee come from the price
    # the venue reports when it reports one; otherwise pro-rated from the
    # decision, which in paper mode already carries the half-spread.
    ratio = (filled / decision.contracts) if decision.contracts else 0.0
    if filled and venue_fill_price is not None:
        fee = float(trade_fee(filled, venue_fill_price, config))
        outlay = float(filled) * float(venue_fill_price) + fee
    else:
        outlay = decision.stake * ratio
        fee = decision.fee * ratio
    await _db(
        writer.open_position,
        symbol=decision.symbol, window_open=decision.window_open,
        settle_time=decision.settle_time, offset_minutes=decision.offset,
        side=decision.side.value, contracts=filled, price=placed_price,
        outlay=outlay, fee=fee,
        model_probability=decision.model_probability,
        baseline_probability=decision.baseline_probability, edge=decision.edge,
        prediction_id=prediction_id,
    )
    # Relative, in one statement: a read-then-write across two transactions loses
    # one of two overlapping debits, and nothing enforces a single writer.
    await _db(writer.adjust_account, bankroll_delta=-outlay, fees_delta=fee)
    status = 'paper' if not placing else (
        'filled' if filled >= decision.contracts else 'partial')
    return OrderResult(True, status, int(filled), order, ticket_id, limit, answered_at)


async def act_on(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                 decision: Decision, row, *,
                 config: Config = DEFAULT_CONFIG,
                 quote_time: Optional[pd.Timestamp] = None,
                 prediction_id: Optional[int] = None) -> bool:
    """`place_and_book` for one decision, diagnosing a kill straight away.

    Returns whether a position was booked. The cycle itself calls
    `place_and_book` for every order at once and diagnoses afterwards.
    """
    result = await place_and_book(args, writer, kalshi, decision, row,
                                  config=config, quote_time=quote_time,
                                  prediction_id=prediction_id)
    if result.status == 'killed':
        await diagnose_and_note(kalshi, writer, decision, result)
    return result.booked


async def diagnose_and_note(kalshi, writer: PgWriter, decision: Decision,
                            result: OrderResult, *, write_note: bool = True) -> None:
    """Read the book and the venue's record for a kill, and keep it on the ticket.

    `write_note=False` when a resend has since changed the ticket: the
    diagnosis is logged and the ticket's newer status is left alone.
    """
    try:
        diagnosis = await diagnose_kill(kalshi, decision, result.order or {},
                                        result.limit)
        if write_note and result.ticket_id is not None:
            order = result.order or {}
            await _db(writer.resolve_ticket, result.ticket_id, status='killed',
                      note=(f"status={order.get('status')!r} "
                            f"order_id={str(order.get('order_id', ''))[:60]} || "
                            f"{diagnosis}")[:4000],
                      venue_order_id=order.get('order_id'))
    except Exception:                                         # noqa: BLE001
        logger.exception('kill diagnosis failed')
