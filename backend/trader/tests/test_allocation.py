"""Kalshi shards its exchange by category and balances are local to a shard.

The KX*15M crypto series live on `exchange_index` 2. Money on shard 0 cannot
buy a contract on shard 2 — it is refused `insufficient_balance` against a
reported total that includes it. A standing target allocation makes the venue
rebalance every ~10 seconds, so a settlement landing on the wrong shard heals
itself instead of bouncing orders mid-session.
"""
from __future__ import annotations

import pytest

from scripts.set_allocation import allocation_payload


def test_everything_on_the_crypto_shard():
    payload = allocation_payload({2: 100})
    assert payload == {'allocations': [{'exchange_index': 2, 'percent': 100}]}


def test_percentages_must_total_one_hundred():
    """The venue rejects a partial allocation, and a silently-accepted 90 would
    strand the remainder on whichever shard it happened to sit."""
    with pytest.raises(ValueError, match='100'):
        allocation_payload({2: 90})


def test_a_split_is_allowed_when_it_totals_one_hundred():
    payload = allocation_payload({0: 25, 2: 75})
    assert sum(a['percent'] for a in payload['allocations']) == 100
