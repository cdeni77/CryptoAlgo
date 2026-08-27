"""What every venue's stream reduces to, so nothing downstream knows the venue.

**What a size MEANS travels on the event, not in the reader.** Kalshi sends a
signed change (`delta_fp: "-5.00"`); Polymarket sends a resulting size. An
adapter cannot convert the first into the second, because that needs the current
resting size and only the cache holds it. So `absolute` says which convention
this event uses and `BookCache` branches on that rather than on `venue`.
Branching on the venue would work today and quietly make every future venue an
edit to the cache.

An absolute size of 0.0 is a REMOVAL, not a price of zero. Dropping it would
leave a level resting in the book forever after the venue said it was gone.

**`no` is NO-denominated on BOTH venues.** Kalshi's orderbook is two bid stacks
and serves it that way; Polymarket serves YES-denominated asks and its adapter
converts at this boundary. Storing them as served put a 0.51 YES ask in the
column holding a 0.51 NO bid — same name, opposite meaning, and no exception
anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, Sequence

Level = tuple[float, float]
KINDS = ('snapshot', 'delta')


@dataclass(frozen=True)
class BookEvent:
    venue: str
    market_ticker: str
    kind: str
    received: float          # time.time() at receipt, for staleness
    seq: int | None          # None where the venue publishes no sequence
    yes: list[Level]         # YES bids
    no: list[Level]          # NO bids — NO-denominated on BOTH venues
    absolute: bool = True    # False: sizes are signed changes to apply

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}, got {self.kind!r}')
        # A snapshot IS the book. A "signed snapshot" has no meaning, and
        # honouring the flag would fold a whole book into whatever was there.
        if self.kind == 'snapshot' and not self.absolute:
            object.__setattr__(self, 'absolute', True)

    @property
    def is_snapshot(self) -> bool:
        return self.kind == 'snapshot'

    @property
    def is_delta(self) -> bool:
        return self.kind == 'delta'


class VenueStream(Protocol):
    """Connect, subscribe, yield normalized events."""

    async def connect(self) -> None: ...

    async def subscribe(self, tickers: Sequence[str]) -> None: ...

    def events(self) -> AsyncIterator[BookEvent]: ...

    async def close(self) -> None: ...
