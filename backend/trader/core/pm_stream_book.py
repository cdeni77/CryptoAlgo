"""The Polymarket CLOB order book, folded from its websocket.

The counterpart to `core/stream_book.py`, and deliberately not a copy: the two
venues send different things, and every difference below was captured off the
live socket rather than read from documentation — which is the lesson the Kalshi
port paid for.

**`price_change.size` is ABSOLUTE, not a signed change.** Kalshi sends
`delta_fp`, so a level driven to zero accumulates float residue; measured, three
BTC levels held 2.4e-12, 3.5e-14 and 4.9e-13, and sitting above the real touch
they made the cache report a best bid of 0.59 against a true 0.56.
`stream_book.MIN_SIZE` exists for that. Polymarket sends the new size at a
price, so a removal arrives as a clean `0.00`:

    0.003  before 1554.88  ->  msg 1553.87
    0.002  before 5115.37  ->  msg  115.37
    0.003  before 1525.88  ->  msg    0.00

No residue, and no MIN_SIZE guard — assuming symmetry with Kalshi here would
have added a guard against a problem this venue does not have while missing the
one it does.

**Bids arrive ASCENDING and asks DESCENDING**, so the touch is the LAST entry on
each side. Reading the first entry gives the far end of the book, which is a
well-formed number and completely wrong.

**One `price_change` message carries changes for more than one asset**, each
entry with its own `asset_id`. Folding a message wholesale into the subscribed
book mixes two markets.

**The ladder is returned in KALSHI'S denomination.** Polymarket serves bids and
asks on one token, so its asks are YES-denominated, while Kalshi's `no_levels`
holds NO-side prices. Storing them as served puts a 0.51 YES ask in the column
holding a 0.51 NO bid — same name, opposite meaning, wrong by the spread with
imbalance inverted, and no exception anywhere. This repo has already been bitten
by exactly that once.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PmBookCache:
    """One book per asset id, folded from `book` and `price_change` frames."""

    def __init__(self) -> None:
        self._bids: dict[str, dict[float, float]] = {}
        self._asks: dict[str, dict[float, float]] = {}
        self._stamp: dict[str, pd.Timestamp] = {}

    # -- folding ---------------------------------------------------------
    def apply(self, event: dict) -> None:
        kind = str(event.get('event_type') or '')
        if kind == 'book':
            self._snapshot(event)
        elif kind == 'price_change':
            self._change(event)

    def _snapshot(self, event: dict) -> None:
        asset = str(event.get('asset_id') or '')
        if not asset:
            return
        # REPLACE, never merge: a later snapshot is the venue restating the
        # whole side, and merging would keep levels it has dropped.
        self._bids[asset] = {float(l['price']): float(l['size'])
                             for l in event.get('bids') or []
                             if float(l.get('size') or 0) > 0}
        self._asks[asset] = {float(l['price']): float(l['size'])
                             for l in event.get('asks') or []
                             if float(l.get('size') or 0) > 0}
        self._stamp[asset] = self._time(event)

    def _change(self, event: dict) -> None:
        stamp = self._time(event)
        for row in event.get('price_changes') or []:
            asset = str(row.get('asset_id') or '')
            if not asset:
                continue
            try:
                price, size = float(row['price']), float(row['size'])
            except (KeyError, TypeError, ValueError):
                continue
            side = str(row.get('side') or '').upper()
            book = self._bids if side == 'BUY' else self._asks
            levels = book.setdefault(asset, {})
            if size > 0:
                levels[price] = size
            else:
                # Absolute, so zero means gone. Deleted rather than kept at
                # size 0, which would sit above the touch and misreport it.
                levels.pop(price, None)
            self._stamp[asset] = stamp

    @staticmethod
    def _time(event: dict) -> pd.Timestamp:
        raw = event.get('timestamp')
        try:
            return pd.Timestamp(int(raw), unit='ms', tz='UTC')
        except (TypeError, ValueError):
            return pd.Timestamp.now(tz='UTC')

    # -- reading ---------------------------------------------------------
    def best_bid(self, asset: str) -> Optional[float]:
        levels = self._bids.get(asset) or {}
        return max(levels) if levels else None

    def best_ask(self, asset: str) -> Optional[float]:
        levels = self._asks.get(asset) or {}
        return min(levels) if levels else None

    def size_at(self, asset: str, side: str, price: float) -> Optional[float]:
        book = self._bids if side == 'bid' else self._asks
        return (book.get(asset) or {}).get(price)

    def age_seconds(self, asset: str, *, now=None) -> Optional[float]:
        stamp = self._stamp.get(asset)
        if stamp is None:
            return None
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
        return float((now - stamp).total_seconds())

    def ladder(self, asset: str) -> tuple[list, list]:
        """(yes_levels, no_levels) best-first, in KALSHI'S denomination.

        The YES ask `p` is the NO bid `1 - p`. See the module docstring: storing
        Polymarket's asks as served would put a YES ask in the column holding a
        NO bid.
        """
        bids = self._bids.get(asset) or {}
        asks = self._asks.get(asset) or {}
        yes = [[p, s] for p, s in sorted(bids.items(), reverse=True)]
        no = [[round(1.0 - p, 6), s] for p, s in sorted(asks.items())]
        return yes, no
