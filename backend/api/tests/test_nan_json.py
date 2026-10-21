"""A NaN must become null, not a 500.

FastAPI's default encoder is `json.dumps(allow_nan=False)`, so one non-finite
float raises and the WHOLE response is lost. Measured 2026-10-18, 74 of 492
predictions carried `model_probability = NaN`, which made
`/predictions?limit=250` -- what the Decisions page requests -- return 500
while `limit=200` returned 200. The page rendered blank.
"""
from __future__ import annotations

import json
import math

from jsonsafe import SafeJSONResponse, _json_safe


def test_bare_nan_becomes_none():
    assert _json_safe(float('nan')) is None
    assert _json_safe(float('inf')) is None
    assert _json_safe(float('-inf')) is None


def test_ordinary_values_are_untouched():
    assert _json_safe(0.0) == 0.0
    assert _json_safe(-1.5) == -1.5
    assert _json_safe('nan') == 'nan'      # the string is not the float
    assert _json_safe(None) is None
    assert _json_safe(True) is True


def test_zero_survives():
    """A losing settlement has revenue exactly 0. Turning that into null would
    flatter the equity curve, which is the one direction it must never err."""
    assert _json_safe(0.0) == 0.0
    assert _json_safe({'revenue': 0.0})['revenue'] == 0.0


def test_nested_structures_are_walked():
    payload = {
        'predictions': [
            {'model_probability': float('nan'), 'edge': 0.012, 'reason': 'not_finite'},
            {'model_probability': 0.51, 'edge': float('-inf'), 'reason': None},
        ],
        'measured': {'value': float('nan'), 'reason': 'no quote'},
    }
    clean = _json_safe(payload)
    assert clean['predictions'][0]['model_probability'] is None
    assert clean['predictions'][0]['edge'] == 0.012
    # The reason survives, which is what makes the null readable rather than
    # just absent.
    assert clean['predictions'][0]['reason'] == 'not_finite'
    assert clean['predictions'][1]['edge'] is None
    assert clean['measured']['value'] is None


def test_the_response_class_actually_renders_valid_json():
    """The end-to-end property: it must parse, and as null -- not the `NaN`
    literal `allow_nan=True` would emit, which `JSON.parse` rejects."""
    body = SafeJSONResponse({'p': float('nan'), 'q': 1.0}).body
    assert b'NaN' not in body
    assert json.loads(body) == {'p': None, 'q': 1.0}


def test_a_realistic_prediction_row_round_trips():
    row = {'symbol': 'BTC-USD', 'model_probability': float('nan'),
           'baseline_probability': 0.6831, 'traded': False, 'reason': 'not_finite'}
    parsed = json.loads(SafeJSONResponse({'predictions': [row]}).body)
    got = parsed['predictions'][0]
    assert got['model_probability'] is None
    assert got['baseline_probability'] == 0.6831
    assert got['reason'] == 'not_finite'
    assert not math.isnan(got['baseline_probability'])
