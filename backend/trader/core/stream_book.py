"""The live book, folded from a stream, and dated on every read.

Every ladder read carries its age. An unknown ticker returns None, not an empty book.
Sequence gaps are per connection, not per market: Kalshi's `seq` is contiguous
across a subscription, not within a market. One missed `seq` marks every book on
that connection, and the repair is to resubscribe all of them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from data_collection.stream.base import BookEvent, Level

logger = logging.getLogger(__name__)

# Sizes below this are floating-point residue from summed signed deltas, not resting size.
# A residue level above the real touch would report a phantom best bid.
# A tenth of the smallest representable size (sizes have 2 decimals).
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
    venue: str = ''
    # Set when the connection that built this book is gone. Only a fresh
    # snapshot on the new subscription makes the book current again.
    awaiting_snapshot: bool = False


class BookCache:
    """Per-market ladders folded from a normalized event stream.

    `max_age_seconds` is 10: the market gains no measurable information under
    about 30s, and a book older than a few seconds means a sick transport.
    """

    def __init__(self, max_age_seconds: float = 10.0,
                 now: Callable[[], float] = time.time) -> None:
        self.max_age_seconds = max_age_seconds
        self._now = now
        self._books: dict[str, _Book] = {}
        # Sequence state is per connection, keyed by venue (one socket per venue).
        self._last_seq: dict[str, int] = {}
        # When each connection last delivered any frame. A live, gap-free
        # connection vouches for every book on it: a quiet book is unchanged, not stale.
        self._last_frame: dict[str, float] = {}

    # -- writing ------------------------------------------------------------

    def apply(self, event: BookEvent) -> None:
        gap = self._note_sequence(event)
        book = self._books.get(event.market_ticker)
        if book is None:
            if event.is_delta:
                # A delta without a snapshot would invent a book from one level.
                logger.debug('%s: delta before snapshot, dropped',
                             event.market_ticker)
                return
            book = self._books[event.market_ticker] = _Book(venue=event.venue)

        if event.is_snapshot:
            book.yes = {p: s for p, s in event.yes if s > MIN_SIZE}
            book.no = {p: s for p, s in event.no if s > MIN_SIZE}
            book.gapped = False
            book.awaiting_snapshot = False
        else:
            if gap:
                book.gapped = True
            for side, levels in (('yes', event.yes), ('no', event.no)):
                target = getattr(book, side)
                for price, size in levels:
                    # `absolute` rides on the event, so this line stays venue-blind.
                    if not event.absolute:
                        size = target.get(price, 0.0) + size
                    if size > MIN_SIZE:
                        target[price] = size
                    else:
                        target.pop(price, None)

        book.last_received = event.received
        self._last_frame[event.venue] = max(
            self._last_frame.get(event.venue, 0.0), event.received)

    def _note_sequence(self, event: BookEvent) -> bool:
        """Advance the connection's sequence; True if a frame was missed.

        A gap marks every book on the connection: the missed frame could
        belong to any of them.
        """
        if event.seq is None:
            return False
        previous = self._last_seq.get(event.venue)
        self._last_seq[event.venue] = event.seq
        if previous is None or event.seq == previous + 1:
            return False
        if event.seq <= previous:
            # A resubscribe restarts the sequence at 1. That is not a gap:
            # the snapshots that follow rebuild every book.
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

    def connection_age(self, ticker: str) -> Optional[float]:
        """Seconds since the connection carrying `ticker` delivered a frame."""
        book = self._books.get(ticker)
        if book is None or book.venue not in self._last_frame:
            return None
        return self._now() - self._last_frame[book.venue]

    def current(self, ticker: str, max_age_seconds: float) -> Optional[Ladder]:
        """The ladder, only if it can be priced against right now.

        Requires: present, not gapped, not stale, and a connection frame within
        `max_age_seconds`. Returns None otherwise, and the caller reads the venue.
        """
        book = self._books.get(ticker)
        if book is None or book.gapped or book.awaiting_snapshot:
            return None
        ladder = self.ladder(ticker)
        age = self.connection_age(ticker)
        if ladder is None or ladder.stale or age is None or age > max_age_seconds:
            return None
        return Ladder(yes=ladder.yes, no=ladder.no, age_seconds=age, stale=False)

    def gapped(self, ticker: str) -> bool:
        book = self._books.get(ticker)
        return bool(book and (book.gapped or book.awaiting_snapshot))

    def any_gapped(self) -> bool:
        """Whether the connection needs resubscribing."""
        return any(b.gapped for b in self._books.values())

    def tickers(self) -> list[str]:
        return sorted(self._books)

    def forget(self, ticker: str) -> None:
        """Drop a settled market, so a closed book cannot be sampled as live."""
        self._books.pop(ticker, None)

    def invalidate(self, venue: str) -> None:
        """Mark the venue's books as awaiting a snapshot after the connection is lost.

        Not `gapped`, which makes the recorder rebuild: the new subscription's
        snapshots rebuild these books.
        """
        for book in self._books.values():
            if book.venue == venue:
                book.awaiting_snapshot = True
        self._last_frame.pop(venue, None)

    def reset_sequence(self, venue: str) -> None:
        """Forget the sequence after a reconnect, so the restart is not a gap."""
        self._last_seq.pop(venue, None)
