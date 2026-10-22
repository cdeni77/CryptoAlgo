"""CF Benchmarks' index values over Kalshi's `cfbenchmarks_value` channel.

This is the number the markets settle on, pushed about once a second, and the
venue computes the settlement quantity itself: `avg_60s_data` is its own
60-second average, not an estimate of one.

**Subscribe with `index_ids`, not `market_tickers`.** The orderbook channel
takes markets; this one takes indices, and a `market_tickers` subscription is
accepted and then silent.

Two shapes in the payload are the kind that fail without raising:

* `msg.data` is a JSON **string**, not an object, so `msg['data']['value']`
  raises and `msg.get('data', {}).get('value')` quietly returns None. The
  orderbook channel has the same class of defect -- `yes_dollars_fp` against
  REST's `orderbook_fp.yes_dollars` -- which yielded an empty book and no
  exception.
* `avg_60s_data` needs **sixty seconds of connection** before it means
  anything. `window_size` climbs 0, 8, 16 ... 60 and then holds, and while it
  fills, `value` is an average over however many samples have arrived: a
  plausible number, no error, and wrong. At `window_size = 0` it equals the
  instantaneous value exactly, which is what makes it look correct.

  Measured 2026-10-22:

      t+s   window_size   avg_60s          spot
        1       0         82833.38000000   82833.38   <- identical, empty
       33      32         82813.46687500   82802.52
       65      60         82808.05766667   82804.75   <- full

  So `IndexValue.settles` is False until the window is full, and it matters on
  every RECONNECT rather than only at startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

#: The venue's index id for each spot product we trade.
INDEX_BY_SYMBOL = {
    'BTC-USD': 'BRTI',
    'ETH-USD': 'ETHUSD_RTI',
    'SOL-USD': 'SOLUSD_RTI',
}
SYMBOL_BY_INDEX = {v: k for k, v in INDEX_BY_SYMBOL.items()}

#: Samples in a full 60-second window. The venue reports the count, so this is
#: a comparison and not an assumption.
FULL_WINDOW = 60

CHANNEL = 'cfbenchmarks_value'


@dataclass(frozen=True)
class IndexValue:
    """One index print, with the venue's own 60-second average beside it."""

    symbol: str
    index_id: str
    #: Epoch seconds of the print itself, from the nested `data` payload.
    at: float
    #: The instantaneous index value.
    value: float
    #: The venue's 60-second average ending at `avg_window_end`, or None.
    avg_60s: Optional[float]
    #: How many samples that average covers. Below `FULL_WINDOW` it is partial.
    window_size: int
    #: Epoch seconds, exclusive, of the averaging window's end.
    avg_window_end: Optional[float]

    @property
    def settles(self) -> bool:
        """Whether `avg_60s` is the settlement quantity rather than a partial.

        The guard that makes a reconnect safe. A partial average is not a
        worse estimate of the settlement value, it is a different quantity.
        """
        return (self.avg_60s is not None
                and self.window_size >= FULL_WINDOW)


def _number(raw) -> Optional[float]:
    try:
        out = float(raw)
    except (TypeError, ValueError):
        return None
    return out if out == out else None          # reject NaN


def parse_value_frame(payload: dict) -> Optional[IndexValue]:
    """A `cfbenchmarks_value` frame as an `IndexValue`, or None.

    Returns None for every other frame type, including the `subscribed`
    acknowledgement, so a caller can hand it the whole stream.
    """
    if not isinstance(payload, dict) or payload.get('type') != CHANNEL:
        return None
    msg = payload.get('msg')
    if not isinstance(msg, dict):
        return None
    index_id = str(msg.get('index_id') or '')
    symbol = SYMBOL_BY_INDEX.get(index_id)
    if symbol is None:
        return None

    # `data` arrives as a JSON STRING. Accept an object too: the venue has
    # changed an encoding before, and reading one shape against the other is
    # how the orderbook channel came to report an empty book in silence.
    inner = msg.get('data')
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except (ValueError, TypeError):
            inner = None
    if not isinstance(inner, dict):
        inner = {}

    value = _number(inner.get('value'))
    if value is None:
        return None
    at_ms = _number(inner.get('time')) or _number(msg.get('received_at'))
    if at_ms is None:
        return None

    avg = msg.get('avg_60s_data')
    avg = avg if isinstance(avg, dict) else {}
    end_ms = _number(avg.get('window_end_ts_exclusive'))
    try:
        window_size = int(avg.get('window_size') or 0)
    except (TypeError, ValueError):
        window_size = 0

    return IndexValue(
        symbol=symbol, index_id=index_id,
        at=at_ms / 1000.0, value=value,
        avg_60s=_number(avg.get('value')),
        window_size=window_size,
        avg_window_end=None if end_ms is None else end_ms / 1000.0,
    )


def subscribe_params(symbols) -> dict:
    """The subscribe frame's params. `index_ids`, never `market_tickers`."""
    ids = [INDEX_BY_SYMBOL[s] for s in symbols if s in INDEX_BY_SYMBOL]
    return {'channels': [CHANNEL], 'index_ids': ids}
