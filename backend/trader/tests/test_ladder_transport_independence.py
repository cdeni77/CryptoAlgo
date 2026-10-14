"""Tests that the REST and stream ladder rows are independent observers.

The stream row is built before the REST call, so a REST failure cannot discard
it. Otherwise the stream can never be shown to cover a REST outage.
"""

from __future__ import annotations

import inspect

import scripts.record_ladder as rl


def _cycle_source() -> str:
    return inspect.getsource(rl.run)


def test_the_stream_sample_is_taken_before_the_rest_call():
    src = _cycle_source()
    ws_at = src.index('paired = ws_row(')
    rest_at = src.index("f\"/markets/{market['ticker']}/orderbook\"")
    assert ws_at < rest_at, (
        'a REST failure must not be able to discard the stream sample — that '
        'is what made ws_only = 0 over seven days'
    )


def test_the_rest_failure_path_comes_after_the_stream_append():
    """The `except ... continue` after a REST failure comes after the stream row is
    appended.
    """
    src = _cycle_source()
    append_at = src.index('rows.append(paired)')
    except_at = src.index('except Exception as exc:      # noqa: BLE001')
    assert append_at < except_at


def test_an_empty_rest_ladder_is_no_longer_silent():
    src = _cycle_source()
    tail = src[src.index('if not yes and not no:'):]
    assert 'logger.debug' in tail[:400], (
        'a run of empty ladders is a venue problem worth seeing'
    )
