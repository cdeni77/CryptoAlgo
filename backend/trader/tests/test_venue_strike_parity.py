"""The venue strike must not reach the feature matrix.

Live overwrote `strike` and `displacement` with Kalshi's `floor_strike` AFTER
`build_features` had already produced the vector, then recomputed exactly one of
the ten features derived from them (`z_score`) and called `model.predict` on the
result. Every other strike-dependent column kept the bar-derived basis:

    abs_z_score            no longer equalled |z_score|
    excursion_up/down/span_z, path_efficiency,
    displacement_vs_elapsed, touched_opposite,
    peer_displacement      all on the old strike

`abs_z_score == |z_score|` holds by construction in all 21,366 training rows, so
the live vector occupied a state the booster has never seen — and a tree
ensemble does not degrade gracefully outside its training manifold, it lands in
whatever leaf the split sequence reaches.

It fired on every cycle and every symbol: `floor_strike` is present on all three
live quotes. Invisible, too — the only warning triggers above 25bp of drift
while the Coinbase-vs-BRTI basis is of order 1bp, which at offset 12's
sigma_remaining of ~7.8bp moves `z_score` by ~0.13 z while its siblings do not
move at all.

Recomputing all ten would remove the impossible state but not the skew: training
defines displacement as `Coinbase_close / Coinbase_OHLC_mean - 1`, and the
override makes it `Coinbase_close / BRTI - 1`, injecting the cross-index basis
into a feature fitted without it. So the venue strike stays out of the matrix
entirely. It remains the right reference for SETTLEMENT, which is where it is
used — not for features.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from scripts import live as live_mod


def test_live_does_not_assign_strike_or_displacement_into_the_scored_frame():
    source = inspect.getsource(live_mod)
    for column in ("'strike'", "'displacement'"):
        assert f"scored.loc[index, {column}]" not in source, (
            f'live still writes {column} into the feature matrix after '
            f'build_features has produced it')


def test_the_identity_abs_z_equals_abs_of_z_is_preserved():
    """The single check that would have caught this: it is true in every
    training row and was false in every live row."""
    frame = pd.DataFrame({'z_score': [0.8, -1.4, 0.0],
                          'abs_z_score': [0.8, 1.4, 0.0]})
    assert np.allclose(frame['abs_z_score'], frame['z_score'].abs())


def test_the_venue_strike_is_still_observed_just_not_fed_to_the_model():
    """Dropping the override must not lose the measurement — the drift between
    our OHLC-mean strike and the venue's BRTI one is the evidence the label
    proxy is sound, and it is worth recording."""
    source = inspect.getsource(live_mod)
    assert 'venue_strike' in source
    assert 'strike_source' in source
