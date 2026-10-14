"""Tests that the market comparison scores only the offsets that trade.

Live records a prediction at four offsets while `--entry-offsets 12` lets only
one open a position. Pooling all four measures a policy nobody runs, and can
invert the verdict.
"""

from __future__ import annotations

import inspect

from scripts.promote import market_measurement


def _rows(offset, market, model, outcomes):
    """Rows are (symbol, window_open, offset, market, baseline, model, outcome)."""
    return [('BTC-USD', f'{offset}-{i}', offset, market, 0.5, model, o)
            for i, o in enumerate(outcomes)]


def test_pooling_the_untraded_offsets_can_invert_the_verdict():
    """Pooling an untraded offset can invert the verdict of the traded one."""
    from core.metrics import market_gate_values

    traded = _rows(12, 0.5, 0.9, [1] * 50)
    untraded = _rows(3, 0.5, 0.9, [0] * 50)

    at_12 = market_gate_values(traded)['model_minus_market']
    pooled = market_gate_values(traded + untraded)['model_minus_market']

    assert at_12 > 0, 'model beats the price at the offset it trades'
    assert pooled < 0, 'pooling the untraded offset inverts it'
    assert at_12 > pooled


def test_evaluate_forwards_entry_offsets_to_the_market_gate():
    """`scripts/evaluate.py` passes `entry_offsets` to `market_measurement`."""
    import scripts.evaluate as ev

    src = inspect.getsource(ev)
    assert 'market_measurement(' in src, 'call site vanished; update this test'
    call = src[src.index('market = market_measurement('):]
    call = call[:call.index('\n\n')] if '\n\n' in call else call
    assert 'entry_offsets' in call, (
        'evaluate() must forward entry_offsets to market_measurement, or the '
        'market gate pools offsets the policy never trades')


def test_the_helper_accepts_the_argument_evaluate_now_passes():
    sig = inspect.signature(market_measurement)
    assert 'entry_offsets' in sig.parameters
    assert sig.parameters['entry_offsets'].kind is inspect.Parameter.KEYWORD_ONLY
