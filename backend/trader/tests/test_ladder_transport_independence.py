"""The two transports must be independent observers.

`venue_ladder` carries both a REST poll and a websocket fold for the same
minute, and the whole point is to compare them: the retirement gate for the
REST sampler needs evidence that the stream covers a REST outage.

It could not. Both `continue`s in the cycle — the request failure and the empty
ladder — sat BEFORE the websocket row was built, so a REST failure discarded a
perfectly good cached book too. Measured over seven days: `rest_only = 30`,
**`ws_only = 0`**. The archive's websocket coverage was a strict subset of
REST's by construction, so the evidence the migration is gated on could never
have existed.
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
    """Specifically the `except ... continue`, which is the one that fired."""
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
