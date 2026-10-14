"""Tests for the Kalshi exchange shard allocation payload.

Balances are local to a shard, and the KX*15M series trade on `exchange_index` 2. A
standing allocation makes the venue move settled cash back to that shard.
"""
from __future__ import annotations

import pytest

from scripts.set_allocation import allocation_payload


def test_everything_on_the_crypto_shard():
    payload = allocation_payload({2: 100})
    assert payload == {'allocations': [{'exchange_index': 2, 'percent': 100}]}


def test_percentages_must_total_one_hundred():
    """The payload must total 100%, otherwise the remainder stays stranded on another shard."""
    with pytest.raises(ValueError, match='100'):
        allocation_payload({2: 90})


def test_a_split_is_allowed_when_it_totals_one_hundred():
    payload = allocation_payload({0: 25, 2: 75})
    assert sum(a['percent'] for a in payload['allocations']) == 100
