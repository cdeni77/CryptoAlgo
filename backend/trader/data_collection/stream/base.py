"""The venue-neutral book event that every stream adapter produces.

`absolute` says what a size means. Kalshi sends a signed change; Polymarket
sends the resulting size. Only the cache holds the current size, so the adapter
cannot convert. `BookCache` branches on `absolute`, not on `venue`.

An absolute size of 0.0 removes the level.

`no` holds NO bids, NO-denominated, on both venues. The Polymarket adapter
converts its YES-denominated asks at this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

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
        # A snapshot replaces the book, so its sizes are always absolute.
        if self.kind == 'snapshot' and not self.absolute:
            object.__setattr__(self, 'absolute', True)

    @property
    def is_snapshot(self) -> bool:
        return self.kind == 'snapshot'

    @property
    def is_delta(self) -> bool:
        return self.kind == 'delta'

