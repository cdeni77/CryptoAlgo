"""The live book, folded from a stream, and dated on every read.

**A cached book that quietly stops updating is worse than a REST call**, because
it looks healthy. So there is no way to read a ladder here without also reading
its age: `Ladder` carries `age_seconds` and `stale`, and an unknown ticker
returns None rather than an empty book. An empty ladder and no ladder are
different claims, and conflating them would let a dead subscription read as a
market with nothing resting in it.

**Sequence gaps are detected per CONNECTION, not per market.** Measured against
the live venue: Kalshi's `seq` is contiguous across a whole subscription
(1..34,956, every step +1, three markets on one `sid`) and is NOT contiguous
within any one market (BTC reads 1, 9, 10, ...). Checking per market — which is
what "snapshot then incremental updates" naturally suggests — would flag every
single delta as a gap and mark every book permanently corrupt. So one missed
`seq` means every book on that connection is suspect, and the repair is to
resubscribe all of them.

Pure and synchronous on purpose. This holds the state a wrong answer would come
from, so it must be exhaustively testable without a network.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from data_collection.stream.base import BookEvent, Level

logger = logging.getLogger(__name__)

# **Anything smaller than this is floating-point residue, not resting size.**
#
# Sizes arrive as 2-decimal fixed-point strings and a signed delta is applied by
# addition, so a level driven to exactly zero accumulates error instead: measured
# on a real capture, three BTC levels held 2.4e-12, 3.5e-14 and 4.9e-13 after
# being emptied. `size > 0` kept all three, and because they sat ABOVE the real
# touch the cache reported a phantom best bid of 0.59 against a true 0.56 — a
# three-cent error, on the one number every book feature is built from, with
# nothing raised anywhere.
#
# A tenth of the smallest representable size. No real level can be this small.
MIN_SIZE = 1e-3


@dataclass(frozen=True)
class Ladder:
    yes: list[Level]
    no: list[Level]
    age_seconds: float
    stale: bool


@dataclass
class _Book:
    yes: dict[float, float] = field(default_factory=dict)
    no: dict[float, float] = field(default_factory=dict)
    last_received: float = 0.0
    gapped: bool = False


class BookCache:
    """Per-market ladders folded from a normalized event stream.

    `max_age_seconds` defaults to 10, and the number is derived rather than
    picked. The tape measurement in the latency work puts the market's
    information gain below the noise floor out to about 30 seconds, so anything
    under that costs nothing. Meanwhile a book more than a few seconds old means
    the transport is sick, not that the market is quiet. Ten seconds sits inside
    the free region with room, and a breach reads as a fault rather than as a
    slow market.
    """

    def __init__(self, max_age_seconds: float = 10.0,
                 now: Callable[[], float] = time.time) -> None:
        self.max_age_seconds = max_age_seconds
        self._now = now
        self._books: dict[str, _Book] = {}
        # Sequence state is per CONNECTION — keyed by venue, because there is
        # one socket per venue. This is a grouping, not a per-venue rule.
        self._last_seq: dict[str, int] = {}

    # -- writing ------------------------------------------------------------

    def apply(self, event: BookEvent) -> None:
        gap = self._note_sequence(event)
        book = self._books.get(event.market_ticker)
        if book is None:
            if event.is_delta:
                # Folding a delta into nothing would invent a whole book out of
                # whichever level happened to change first.
                logger.debug('%s: delta before snapshot, dropped',
                             event.market_ticker)
                return
            book = self._books[event.market_ticker] = _Book()

        if event.is_snapshot:
            book.yes = {p: s for p, s in event.yes if s > MIN_SIZE}
            book.no = {p: s for p, s in event.no if s > MIN_SIZE}
            book.gapped = False
        else:
            if gap:
                book.gapped = True
            for side, levels in (('yes', event.yes), ('no', event.no)):
                target = getattr(book, side)
                for price, size in levels:
                    # `absolute` rides on the event precisely so this line never
                    # has to know which venue sent it.
                    if not event.absolute:
                        size = target.get(price, 0.0) + size
                    if size > MIN_SIZE:
                        target[price] = size
                    else:
                        target.pop(price, None)

        book.last_received = event.received

    def _note_sequence(self, event: BookEvent) -> bool:
        """Advance the connection's sequence; True if a frame was missed.

        A gap marks EVERY book on the connection, because the missed frame could
        have belonged to any of them.
        """
        if event.seq is None:
            return False
        previous = self._last_seq.get(event.venue)
        self._last_seq[event.venue] = event.seq
        if previous is None or event.seq == previous + 1:
            return False
        if event.seq <= previous:
            # A resubscribe restarts the sequence at 1. That is not a gap; the
            # snapshots that follow rebuild every book anyway.
            logger.info('%s: sequence restarted at %s (was %s)',
                        event.venue, event.seq, previous)
            return False
        logger.warning('%s: sequence gap %s -> %s; every book on this '
                       'connection is suspect', event.venue, previous, event.seq)
        for book in self._books.values():
            book.gapped = True
        return True

    # -- reading ------------------------------------------------------------

    def ladder(self, ticker: str) -> Optional[Ladder]:
        book = self._books.get(ticker)
        if book is None:
            return None
        age = self._now() - book.last_received
        return Ladder(yes=sorted(book.yes.items()), no=sorted(book.no.items()),
                      age_seconds=age, stale=age > self.max_age_seconds)

    def gapped(self, ticker: str) -> bool:
        book = self._books.get(ticker)
        return bool(book and book.gapped)

    def any_gapped(self) -> bool:
        """Whether the connection needs resubscribing."""
        return any(b.gapped for b in self._books.values())

    def tickers(self) -> list[str]:
        return sorted(self._books)

    def forget(self, ticker: str) -> None:
        """Drop a settled market, so a closed book cannot be sampled as live."""
        self._books.pop(ticker, None)

    def reset_sequence(self, venue: str) -> None:
        """Forget the sequence after a reconnect, so the restart is not a gap."""
        self._last_seq.pop(venue, None)
