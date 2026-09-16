"""The market comparison must score the offset that TRADES, not all four.

`PgWriter.scored_against_market` returns every scored row, and the live loop
records a prediction at +3m, +6m, +9m and +12m while `--entry-offsets 12` means
exactly one of them can open a position. Pooling all four measures a policy
nobody runs -- and it does not merely add noise, it inverts the verdict.

Measured 2026-09-16 on 5,468 live rows:

    offset 12 (what trades)   model_minus_market  +0.000921   PASS
    all four pooled           model_minus_market  -0.000311   FAIL

`scripts/promote.py` was fixed for this and carries the note. `scripts/evaluate.py`
called the same helper WITHOUT `entry_offsets` for as long as it had existed, so
the two disagreed about the same candidate on the gate the whole stack exists to
satisfy, and `evaluate` read as "the model loses to the price".
"""

from __future__ import annotations

import inspect

from scripts.promote import market_measurement


def _rows(offset, market, model, outcomes):
    """(symbol, window_open, offset, market, baseline, model, outcome)."""
    return [('BTC-USD', f'{offset}-{i}', offset, market, 0.5, model, o)
            for i, o in enumerate(outcomes)]


def test_pooling_the_untraded_offsets_can_invert_the_verdict():
    """The defect itself, on numbers that make it unambiguous.

    At +12m the model is confident and right; at +3m it is equally confident
    and wrong. Scored where it trades it beats the price; pooled across offsets
    it loses to it. Same model, same rows, opposite verdict.
    """
    from core.metrics import market_gate_values

    traded = _rows(12, 0.5, 0.9, [1] * 50)
    untraded = _rows(3, 0.5, 0.9, [0] * 50)

    at_12 = market_gate_values(traded)['model_minus_market']
    pooled = market_gate_values(traded + untraded)['model_minus_market']

    assert at_12 > 0, 'model beats the price at the offset it trades'
    assert pooled < 0, 'pooling the untraded offset inverts it'
    assert at_12 > pooled


def test_evaluate_forwards_entry_offsets_to_the_market_gate():
    """A seam test, because the bug was a MISSING ARGUMENT at one call site.

    Testing `market_measurement` alone passed the whole time this was broken --
    the helper was always correct. What was wrong was that `evaluate` did not
    tell it which offsets trade. So the thing under test is the call, not the
    callee.
    """
    import scripts.evaluate as ev

    src = inspect.getsource(ev)
    assert 'market_measurement(' in src, 'call site vanished; update this test'
    call = src[src.index('gates = evaluate_gates('):]
    call = call[:call.index('\n\n')] if '\n\n' in call else call
    assert 'entry_offsets' in call, (
        'evaluate() must forward entry_offsets to market_measurement, or the '
        'market gate pools offsets the policy never trades')


def test_the_helper_accepts_the_argument_evaluate_now_passes():
    sig = inspect.signature(market_measurement)
    assert 'entry_offsets' in sig.parameters
    assert sig.parameters['entry_offsets'].kind is inspect.Parameter.KEYWORD_ONLY
