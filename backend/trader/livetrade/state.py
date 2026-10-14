"""Module-level state that more than one live module reads or writes.

Each value has exactly one home. State used by a single module (the bar cache,
the market cache, the gap history, the implied-vol memo, the last exchange
shard) lives in that module instead. Read and assign these through the module
(`state._RUNNING_CONFIG = ...`), never via `from livetrade.state import ...`,
which would copy the value at import time.
"""

from __future__ import annotations

from typing import Optional

from core.config import Config

# The configuration the running loop is trading under. Set by `run_cycle`; read
# by `quotes.fetch_quotes` (stream quote age) and `features.config_offsets`.
_RUNNING_CONFIG: Optional[Config] = None

# time.monotonic() of the last reconciliation the venue answered. Set by
# `ledger.bookkeeping`; read by `run_cycle`, which sizes an entry off the
# balance it adopted and reconciles first when it is too old.
_LAST_RECONCILED: Optional[float] = None
