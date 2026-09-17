"""The "different model" warning must not fire every window.

`_warn_unscoreable_features` is the ONLY signal that the live model is not the
one whose gates were measured — LightGBM substitutes a learned default for a
NaN and nothing raises.

It fired **96 times in 24 hours**: once per window, always
`venue_gap_change_5`, which is a first difference against the previous offset
and therefore NaN at a window's first offset by construction. A genuine failure
— a dead stream leaving `imbalance_5c`, `depth_ratio` and `book_convexity`
empty — appends three names to the identical line at the identical rate, so the
signal was indistinguishable from the noise from the day the feature shipped.

Expected absences are still reported at debug rather than dropped: the day one
of them stops being expected, silence would be worse than noise.
"""

from __future__ import annotations

import logging

import pandas as pd

from scripts.live import _warn_unscoreable_features

OFFSETS = (3, 6, 9, 12)


class _Model:
    features = ('venue_gap_change_5', 'imbalance_5c', 'depth_ratio')


def _scored(offset, **cols):
    row = {'offset': offset, 'venue_gap_change_5': float('nan'),
           'imbalance_5c': 0.2, 'depth_ratio': 1.1}
    row.update(cols)
    return pd.DataFrame([row])


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_first_offset_of_a_window_does_not_warn(caplog):
    caplog.set_level(logging.DEBUG, logger='live')
    _warn_unscoreable_features(_scored(3), _Model(), offsets=OFFSETS)
    assert not _warnings(caplog), (
        'gap_change has nothing to difference against at the first offset; '
        'that is arithmetic, not a fault')


def test_it_is_still_reported_at_debug(caplog):
    caplog.set_level(logging.DEBUG, logger='live')
    _warn_unscoreable_features(_scored(3), _Model(), offsets=OFFSETS)
    assert any('as expected' in r.getMessage() for r in caplog.records)


def test_the_same_feature_at_a_later_offset_DOES_warn(caplog):
    """At +12m there is a previous offset, so an empty gap_change is real."""
    caplog.set_level(logging.DEBUG, logger='live')
    _warn_unscoreable_features(_scored(12), _Model(), offsets=OFFSETS)
    assert _warnings(caplog)


def test_a_dead_stream_still_warns_at_the_first_offset(caplog):
    """The case the saturation was hiding."""
    caplog.set_level(logging.DEBUG, logger='live')
    dead = _scored(3, imbalance_5c=float('nan'), depth_ratio=float('nan'))
    _warn_unscoreable_features(dead, _Model(), offsets=OFFSETS)
    warned = _warnings(caplog)
    assert warned, 'three empty book features is a real fault'
    msg = warned[0].getMessage()
    assert 'imbalance_5c' in msg and 'depth_ratio' in msg
    assert 'venue_gap_change_5' not in msg, (
        'the expected one must not pad the list it is read from')


def test_a_missing_column_always_warns(caplog):
    caplog.set_level(logging.DEBUG, logger='live')
    frame = _scored(3).drop(columns=['depth_ratio'])
    _warn_unscoreable_features(frame, _Model(), offsets=OFFSETS)
    assert _warnings(caplog)
