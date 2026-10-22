"""A recorder must not hold rows longer than its own freshness limit allows.

`venue_ladder` wrote an unbroken minute grid and still reported stale, because
the flush was triggered by row COUNT alone: the newest stored row sawtoothed
with the row rate and crossed the 900s limit once per batch. The lag also grew
as the venue went quiet, so the alarm was loudest when there was least to see.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.freshness import EXPECTED_MAX_AGE, MAX_BUFFER_SECONDS, max_buffer_seconds

#: Every recorder that buffers rows and is judged against a freshness limit.
RECORDERS = {
    'record_ladder.py': 'venue_ladder',
    'record_trades.py': 'venue_trades',
    'record_pm_ladder.py': 'pm_ladder',
    'record_implied_vol.py': 'venue_implied_vol',
}

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / 'scripts'


@pytest.mark.parametrize('dataset', sorted(set(RECORDERS.values())))
def test_the_buffer_deadline_is_inside_the_freshness_limit(dataset):
    """Leaves room for store_sync and the freshness snapshot on top."""
    limit = EXPECTED_MAX_AGE[dataset]
    assert max_buffer_seconds(dataset) <= limit / 4.0


def test_an_unlisted_dataset_still_gets_a_deadline():
    """A dataset with no limit must not buffer for ever."""
    assert max_buffer_seconds('not_a_dataset') == MAX_BUFFER_SECONDS


@pytest.mark.parametrize('name,dataset', sorted(RECORDERS.items()))
def test_every_recorder_flushes_on_time_not_only_on_count(name, dataset):
    source = (SCRIPTS / name).read_text()
    assert f"max_buffer_seconds('{dataset}')" in source, (
        f'{name} must take its deadline from the dataset it writes')
    assert 'time.monotonic() - last_write >= max_buffer' in source, (
        f'{name} still flushes on row count alone')
    # The clock has to be reset, or the recorder writes on every cycle once
    # the first deadline passes.
    tree = ast.parse(source)
    resets = sum(1 for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, 'id', None) == 'last_write' for t in n.targets))
    assert resets >= 2, f'{name} never resets last_write after a write'
