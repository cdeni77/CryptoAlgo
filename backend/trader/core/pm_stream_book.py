"""The Polymarket CLOB order book, folded from its websocket.

Venue rules:
- `price_change.size` is the absolute size at a price, not a signed change.
  A removal arrives as a clean 0, so no MIN_SIZE guard is needed.
- Bids arrive ascending and asks descending, so the touch is the last entry.
- One `price_change` message can carry changes for several assets.
  Each entry carries its own `asset_id`.
- `ladder` returns Kalshi's denomination: `no_levels` holds NO bids, not YES asks.
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
        # Replace, never merge: a snapshot restates the whole side.
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
                # Size is absolute, so zero means gone. Delete the level:
                # a size-0 level would sit above the touch.
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
        """(yes_levels, no_levels) best-first, in Kalshi's denomination.

        The YES ask `p` is the NO bid `1 - p`.
        """
        bids = self._bids.get(asset) or {}
        asks = self._asks.get(asset) or {}
        yes = [[p, s] for p, s in sorted(bids.items(), reverse=True)]
        no = [[round(1.0 - p, 6), s] for p, s in sorted(asks.items())]
        return yes, no
