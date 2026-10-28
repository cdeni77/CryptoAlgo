"""A cold feature cache must abstain, not trade a substituted model.

The stream cache is process-local, so a restart or an outage empties it and
the book features the artifact was FITTED on arrive all-NaN. LightGBM then
substitutes its learned default, which the log already calls "a different
model from the one whose gates were measured" -- and the loop traded it
anyway.

Measured over 2026-10-14..27, windows grouped by time since a resume:

    window 1 (0-15 min)   23 windows   -$127.26   -14.12 pp/ct
    warm (>2h)           786 windows   + $65.93   + 0.26 pp/ct

The worst first-windows lost 80-93% of stake per contract across four
separate dates, so it is a pattern and not one bad session. Refusing mirrors
`require_peer_book`, which already abstains rather than scoring a row whose
peer book is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.decide import Reason, decide
from core.config import Config


def _row(**kw) -> pd.Series:
    base = dict(
        symbol='BTC-USD', window_open=pd.Timestamp('2026-10-27 21:30:00+00:00'),
        settle_time=pd.Timestamp('2026-10-27 21:45:00+00:00'), offset=3,
        model_probability=0.80, baseline_probability=0.78,
        market_probability=0.70, ask_up=0.72, ask_down=0.30,
        strike=80000.0, last_price=80100.0, sigma_remaining=0.002,
    )
    base.update(kw)
    return pd.Series(base)


def test_a_row_flagged_unscoreable_is_refused():
    config = Config(min_edge_pp=0.0)
    decision = decide(_row(features_unavailable=True), config, bankroll=1000.0)
    assert not decision.traded
    assert decision.reason is Reason.FEATURES_UNAVAILABLE


def test_an_unflagged_row_is_unaffected():
    """The flag is the only thing this adds; absent it nothing changes."""
    config = Config(min_edge_pp=0.0)
    flagged = decide(_row(features_unavailable=False), config, bankroll=1000.0)
    bare = decide(_row(), config, bankroll=1000.0)
    assert flagged.reason is bare.reason
    assert flagged.traded == bare.traded


def test_the_backtest_never_sets_the_flag():
    """A backtest has full history, so the condition cannot arise there.

    Guards against the refusal leaking into the gates as a policy the account
    does not run.
    """
    import inspect

    from core import backtest

    assert 'features_unavailable' not in inspect.getsource(backtest)


def test_a_three_minute_lookback_at_offset_3_is_not_cold():
    """`spread_change_3` and `mid_change_3` are empty at +3 in EVERY window.

    A three-minute lookback reaches past the window open at the first offset,
    warm or cold. Treating that as a cold cache would refuse every +3 decision
    and drop the offset -- which was measured and rejected.
    """
    from livetrade.features import EXPECTED_NAN, _warn_unscoreable_features

    assert {'spread_change_3', 'mid_change_3'} <= set(EXPECTED_NAN)

    class _Model:
        features = ('spread_change_3', 'mid_change_3', 'z_score')

    frame = pd.DataFrame({'offset': [3, 3], 'z_score': [0.4, 0.5],
                          'spread_change_3': [np.nan, np.nan],
                          'mid_change_3': [np.nan, np.nan]})
    assert _warn_unscoreable_features(frame, _Model(), offsets=(3, 6, 9, 12)) == []


def test_a_genuinely_cold_column_is_still_reported():
    """The warmup case must still abstain: an unexpected empty column."""
    from livetrade.features import _warn_unscoreable_features

    class _Model:
        features = ('mid_change_1', 'z_score')

    frame = pd.DataFrame({'offset': [3, 3], 'z_score': [0.4, 0.5],
                          'mid_change_1': [np.nan, np.nan]})
    assert _warn_unscoreable_features(frame, _Model(), offsets=(3, 6, 9, 12)) == ['mid_change_1']
