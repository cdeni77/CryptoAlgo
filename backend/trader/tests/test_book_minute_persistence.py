"""The <1s sampler must be KEPT, not discarded.

The ladder recorder samples 25s into the minute, so `decision_age` scores its
rows negative -- seen after the decision -- and refuses 97.8-99.8% of them.
Those three recorders are 140,646 rows contributing 1.2% of usable quotes.
This sampler reads the same in-process cache the decision reads, under a
second past the minute, and until now wrote nothing.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from core.quotes import QUOTE_SOURCE_PRIORITY
from scripts import record_book_minutes as rbm

UTC = dt.timezone.utc
OPEN = dt.datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
YES = [(0.60, 1200.0), (0.59, 800.0)]
NO = [(0.39, 900.0), (0.38, 500.0)]


def setup_function(_):
    rbm._PENDING.clear()                                      # noqa: SLF001


def test_a_sample_is_buffered_with_its_own_source_and_real_age():
    rbm._buffer_row('KXBTC15M-X', 'BTC-USD', OPEN, 3, 0.42, YES, NO)  # noqa: SLF001
    assert len(rbm._PENDING) == 1                             # noqa: SLF001
    row = rbm._PENDING[0]                                     # noqa: SLF001
    assert row['source'] == rbm.SOURCE == 'live_minute'
    assert row['quote_age_seconds'] == 0.42
    assert row['offset_minutes'] == 3
    # available_time is the capture, event_time the minute it describes
    assert (row['available_time'] - row['event_time']).total_seconds() == 0.42


def test_it_carries_sizes_which_is_the_whole_point():
    """Candles have no sizes and the venue serves no historical book, so this
    is the only route to depth a decision could have had."""
    rbm._buffer_row('KXBTC15M-X', 'BTC-USD', OPEN, 6, 0.3, YES, NO)   # noqa: SLF001
    row = rbm._PENDING[0]                                     # noqa: SLF001
    assert row['yes_bid'] == 0.60
    assert row['yes_bid_size'] == 1200.0
    assert row['depth_bid_total'] == 2000.0
    assert row['levels_bid'] == 2.0


def test_the_buffer_is_bounded_so_an_unwritable_store_cannot_kill_the_process():
    for i in range(rbm.MAX_PENDING + 50):
        rbm._buffer_row('KX-X', 'BTC-USD', OPEN, i % 15, 0.2, YES, NO)  # noqa: SLF001
    assert len(rbm._PENDING) == rbm.MAX_PENDING               # noqa: SLF001


def test_buffering_never_raises():
    """A sampler that can break the trading loop is worse than one that
    misses a minute."""
    rbm._buffer_row(None, None, 'not-a-time', 'x', 'y', None, None)   # noqa: SLF001
    assert rbm._PENDING == []                                 # noqa: SLF001


def test_a_failing_store_keeps_the_rows_rather_than_losing_them():
    class Broken:
        def write(self, *a, **k):
            raise RuntimeError('disk full')

    rbm._buffer_row('KX-X', 'BTC-USD', OPEN, 3, 0.2, YES, NO)         # noqa: SLF001
    assert rbm.flush(Broken()) == 0
    assert len(rbm._PENDING) == 1, 'a failed write must not drop the sample'


def test_a_successful_write_clears_only_what_it_wrote():
    seen = {}

    class Store:
        def write(self, dataset, frame):
            seen['dataset'] = dataset
            seen['n'] = len(frame)

    rbm._buffer_row('KX-X', 'BTC-USD', OPEN, 3, 0.2, YES, NO)         # noqa: SLF001
    assert rbm.flush(Store()) == 1
    assert seen == {'dataset': 'venue_depth', 'n': 1}
    assert rbm._PENDING == []                                 # noqa: SLF001


def test_it_outranks_every_late_observer_but_not_the_decision_instant():
    p = list(QUOTE_SOURCE_PRIORITY)
    assert p.index('live_touch') < p.index('live_minute')
    for later in ('live_ws', 'live', 'live_ws_replay', 'candles', 'backfill'):
        assert p.index('live_minute') < p.index(later), later


def test_a_decision_instant_replay_outranks_every_late_observer():
    """`live_ws_replay_t3` is the archive refolded at m+3s. It is a
    reconstruction, so it sits below the live captures -- but above anything
    sampled 25s into the minute, which holds a LATER book than the decision."""
    p = list(QUOTE_SOURCE_PRIORITY)
    assert p.index('live_touch') < p.index('live_ws_replay_t3')
    assert p.index('live_minute') < p.index('live_ws_replay_t3')
    for later in ('live_ws', 'live', 'live_ws_replay', 'candles', 'backfill'):
        assert p.index('live_ws_replay_t3') < p.index(later), later
