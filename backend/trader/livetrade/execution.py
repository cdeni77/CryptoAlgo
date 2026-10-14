"""Turn a decision into an order: price it, send it, book what filled.

A position is booked only from what the venue says filled. A kill is diagnosed
from the book and the venue's order record, and may be resent once against a
fresh book.
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

    The resend re-runs `decide()` on a fresh touch with the same model
    probability and the exposure of every other order this cycle, so it never
    exceeds a limit the first attempt respected. It goes out only within
    `kill_retry_window_seconds`, only on the same side, under `<key>-r1`. A
    partial fill is never topped up.

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
    """Return the worst price still worth paying for this decision.

    Edge above the `min_edge_pp` gate in force is spendable, because any fill
    that keeps the gate is already accepted. The order is `immediate_or_cancel`,
    so the limit bounds the worst fill and does not set the price.
    `max_slippage_cents` caps the allowance on a thin book.
    """
    config = config if config is not None else DEFAULT_CONFIG
    edge = decision.edge if np.isfinite(decision.edge) else 0.0
    spendable = max(0.0, edge - config.min_edge_pp / 100.0)
    allowance = min(spendable, config.max_slippage_cents / 100.0)
    # Never below the price we must pay: a limit below the ask cannot fill.
    # No grid snapping here. `place_order` rounds onto the venue grid in the
    # direction that can still fill (`ceil` for a bid, `floor` for an ask).
    # Snapping down here would turn a sub-cent allowance into zero.
    return float(max(min(0.99, decision.price + allowance), decision.price))


def kill_diagnosis(*, side: str, sent_yes_price: float, book: dict) -> dict:
    """Classify a non-fill from the book at the kill.

    The three cases are: the limit never crossed (a price error), it crossed
    with nothing resting (a size race), or it crossed with size resting (the
    order is malformed).

    All prices are YES-denominated. `sent_yes_price` is the price on the wire.
    `side='up'` sends a YES bid, which must be at or above the YES ask.
    `side='down'` sends a YES ask, which must be at or below the YES bid.
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
    """Return the YES price that went on the wire.

    Prefers the venue's echo `yes_price_dollars`, which includes the tick
    rounding. The fallback sends `1 - limit` for a DOWN order.
    """
    raw = order.get('yes_price_dollars') if isinstance(order, dict) else None
    try:
        if raw is not None:
            return float(raw)
    except (TypeError, ValueError):
        pass
    return float(limit) if decision.side is Side.UP else 1.0 - float(limit)


async def diagnose_kill(kalshi, decision, order: dict, limit: float) -> str:
    """Read the book and the venue's order record right after a kill.

    Never raises, because it runs inside the order path.
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
        # Filter by ticker so the read is exact. The listing can race the POST
        # and miss the order. No retry: this is corroboration only.
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
    """Return contracts filled and the YES-denominated price from the venue reply.

    Never reads `count`, which is the requested size. Returns `(0, nan)` for
    anything that is not a confirmed fill, because an assumed fill invents a
    position that later settles.
    """
    status = str(order.get('status', '') or '').strip().lower()
    dead = status in ('canceled', 'cancelled', 'killed', 'rejected', 'expired')

    # Counts arrive as fixed-point decimal strings ("10.00") and can be
    # fractional, so parse as float.
    def count(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # `fill_count_fp` first: it is the only count a V2 order carries. Under
    # `immediate_or_cancel` a partial fill has `status='canceled'` and a
    # non-zero fill count, so status alone cannot decide.
    filled = None
    for key in ('fill_count_fp', 'taker_fill_count', 'filled_count', 'fill_count'):
        if order.get(key) is not None:
            filled = count(order[key])
            if filled is None:
                return 0, float('nan')
            break

    # Never infer a fill from `remaining` on a dead order: it reaches zero when
    # the order leaves the book, traded or not.
    if filled is None and not dead:
        for key in ('remaining_count_fp', 'remaining_count'):
            if order.get(key) is not None:
                remaining = count(order[key])
                if remaining is None:
                    return 0, float('nan')
                filled = float(requested) - remaining
                break

    if filled is None:
        # No count at all. Only 'executed' or 'filled' means the whole order
        # traded. Anything else is unknown and must not become a position.
        if status in ('executed', 'filled'):
            filled = float(requested)
        else:
            return 0, float('nan')

    filled = max(0, min(int(round(filled)), requested))
    if filled <= 0:
        return 0, float('nan')

    # The price paid: fill cost / count is the true average fill, including
    # any improvement on the limit.
    price = float('nan')
    for key in ('taker_fill_cost_dollars', 'maker_fill_cost_dollars',
                'fill_cost_dollars'):
        cost = count(order.get(key))
        if cost is not None and cost > 0:
            paid = cost / float(filled)
            # The cost is in the side bought; this function returns YES terms
            # and the caller inverts for DOWN. So a NO cost is inverted here.
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
            # The venue serves both dollar and integer-cent values on these
            # keys. Above 1.0 it can only be cents.
            price = value / 100.0 if value > 1.0 else value
            if key == 'no_price':
                price = 1.0 - price
            break
    return filled, price


class OrderResult(NamedTuple):
    """The outcome of one order attempt."""

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
    """Return one key per (symbol, window, offset); a resend adds `-r<n>`."""
    key = (f'{decision.symbol}-{decision.window_open:%Y%m%d%H%M}'
           f'-{decision.offset:02d}')
    return key if attempt <= 0 else f'{key}-r{attempt}'


async def lookup_after_ambiguity(kalshi, client_order_id: str, ticker: Optional[str],
                                 *, attempts: int = 2, pause: float = 1.0):
    """Return the venue's record of an order whose POST got no answer.

    Returns `(found, record)`. `found` is False when the lookup itself failed.
    Asks twice, because the order may not be listed yet.
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
    """Record the ticket, place the order if both flags allow, and book the fill.

    A position is written only when money moved; an unresolved market or a dry
    run books nothing. `quote_time` is when the book was read; None skips the
    staleness check.

    The ticket is written before the POST, because `entries_for_window` counts
    it: a crash between sending and booking must still block the window.

    A POST that times out or gets a 5xx may have been accepted. The order is
    then looked up by `client_order_id` and booked from the venue's record. If
    that lookup fails, the ticket stays `unknown`, which still blocks the window.
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
        # The worst price still worth paying.
        max_price=limit,
        expected_cost=decision.stake, model_probability=decision.model_probability,
        edge=decision.edge, prediction_id=prediction_id,
    )
    if attempt > 0 or note:
        # `write_ticket` is one row per (symbol, window). A resend reuses the
        # killed ticket and marks it live before the POST, so a crash still
        # blocks the window.
        await _db(writer.resolve_ticket, ticket_id, status='new',
                  note=(note or f'resend {attempt}')[:400])

    filled = decision.contracts
    placed_price = decision.price
    venue_fill_price: Optional[float] = None
    order: Optional[dict] = None
    answered_at: Optional[float] = None
    # Use the market's own exchange index. A mismatch returns
    # `404 market_not_found`.
    exchange_index = int(getattr(row, 'get', lambda *_: 0)('exchange_index', 0) or 0)
    key = client_order_id_for(decision, attempt)
    if placing:
        try:
            # The key carries the offset, so a kill at one offset does not
            # block later offsets with a 409.
            order = await kalshi.place_order(
                ticker=decision.market_ticker, side=decision.side.value,
                exchange_index=exchange_index,
                contracts=decision.contracts, limit_price=limit,
                client_order_id=key)
            answered_at = time.monotonic()
        except (KalshiError, asyncio.TimeoutError, OSError) as exc:
            ambiguous = (not isinstance(exc, KalshiError)
                         or bool(getattr(exc, 'ambiguous', False)))
            if not ambiguous:
                await _db(writer.resolve_ticket, ticket_id, status='skipped',
                          note=str(exc)[:400])
                logger.error('order refused, no position recorded: %s', exc)
                return OrderResult(False, 'refused', ticket_id=ticket_id, limit=limit)
            # The request may have reached the venue. Ask it, by our key.
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
        # V2 quotes from the YES side. A DOWN fill is the YES sale price, so
        # the NO price paid is 1 minus it.
        if np.isfinite(fill_price) and decision.side is not Side.UP:
            fill_price = 1.0 - fill_price
        if filled <= 0:
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

    # Book the fill, not the intention. Outlay and fee use the venue's fill
    # price when known; otherwise they pro-rate the decision.
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
    # Adjust relatively, in one statement. A read-then-write can lose one of
    # two overlapping debits.
    await _db(writer.adjust_account, bankroll_delta=-outlay, fees_delta=fee)
    status = 'paper' if not placing else (
        'filled' if filled >= decision.contracts else 'partial')
    return OrderResult(True, status, int(filled), order, ticket_id, limit, answered_at)


async def act_on(args, writer: PgWriter, kalshi: Optional[KalshiClient],
                 decision: Decision, row, *,
                 config: Config = DEFAULT_CONFIG,
                 quote_time: Optional[pd.Timestamp] = None,
                 prediction_id: Optional[int] = None) -> bool:
    """Run `place_and_book` for one decision and diagnose a kill at once.

    Returns whether a position was booked.
    """
    result = await place_and_book(args, writer, kalshi, decision, row,
                                  config=config, quote_time=quote_time,
                                  prediction_id=prediction_id)
    if result.status == 'killed':
        await diagnose_and_note(kalshi, writer, decision, result)
    return result.booked


async def diagnose_and_note(kalshi, writer: PgWriter, decision: Decision,
                            result: OrderResult, *, write_note: bool = True) -> None:
    """Diagnose a kill and write the result on the ticket.

    Pass `write_note=False` when a resend has changed the ticket since.
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
