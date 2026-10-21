"""A rate-limited hour must be retried, not recorded as absent data.

`/cfbenchmarks/*` costs 50 tokens against a 200/s refill, so 429 is the
ordinary failure here. Dropping on it left `minute_bars` under
`venue='cfbenchmarks'` at ~7% coverage, which reads as "the venue serves
little history" rather than "we asked too fast".
"""

from __future__ import annotations

import inspect

from scripts import backfill_index as bi


def test_the_request_sits_inside_a_bounded_retry():
    src = inspect.getsource(bi.run)
    assert "'/cfbenchmarks/history/values'" in src
    assert 'for attempt in range(6)' in src
    assert "DROPPED['hours']" in src
    assert 'DROPPED after 6 attempts' in src


def test_the_drop_counter_starts_empty():
    assert bi.DROPPED['hours'] == 0


def test_the_rate_default_respects_the_endpoint_cost():
    """50 tokens against a 200/s refill is ~4 req/s; 3.0 leaves headroom."""
    assert 0 < bi.DEFAULT_RATE <= 4.0
