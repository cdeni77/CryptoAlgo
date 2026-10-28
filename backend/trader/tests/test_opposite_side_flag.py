"""Taking the other side is an EXIT priced at the spread, not a hedge.

Each matched pair settles at $1 whichever way the market resolves, so holding
one side at `p` and buying the other at `q` locks in `p + q - 1` the moment the
second order fills. Measured on KXSOL15M-26OCT200745: 2 contracts at 0.55 and 2
at 0.77 recorded exactly -$0.64, which is 2 x (0.55 + 0.77 - 1).

The original ban cited -$6.29 across three such markets. That figure was a
settlement-revenue artifact; reconciled against the venue's own per-market
ledger the three lost **-$1.29**. So the ban rests on thinner evidence than
recorded, which is why the behaviour is now a flag rather than a constant.

Default OFF: live is unchanged until a backtest says otherwise.
"""

from __future__ import annotations

import pandas as pd

from core.config import Config
from core.decide import Reason, WindowExposure, decide


def _row(**kw) -> pd.Series:
    base = dict(
        symbol='BTC-USD', window_open=pd.Timestamp('2026-10-28 03:00:00+00:00'),
        settle_time=pd.Timestamp('2026-10-28 03:15:00+00:00'), offset=6,
        model_probability=0.80, baseline_probability=0.78,
        market_probability=0.72, ask_up=0.73, ask_down=0.29,
        strike=80000.0, last_price=79900.0, sigma_remaining=0.002,
    )
    base.update(kw)
    return pd.Series(base)


def _holding_down(config: Config) -> WindowExposure:
    """A barrier already held on `down`, built through the real accessor."""
    first = decide(_row(offset=3, model_probability=0.30, baseline_probability=0.32,
                        market_probability=0.36, ask_up=0.39, ask_down=0.65),
                   config, bankroll=1000.0)
    assert first.traded and str(first.side.value) == 'down', first.reason
    return WindowExposure().with_(first)


def test_the_opposite_side_is_refused_by_default():
    """Live must not change: the flag is off unless a run asks for it."""
    config = Config(min_edge_pp=0.0, max_entries_per_symbol_window=4)
    assert config.allow_opposite_side is False
    out = decide(_row(), config, bankroll=1000.0,
                 exposure=_holding_down(config))
    assert not out.traded
    assert out.reason is Reason.OPPOSITE_SIDE


def test_the_flag_lets_the_model_reverse():
    config = Config(min_edge_pp=0.0, max_entries_per_symbol_window=4,
                    allow_opposite_side=True)
    out = decide(_row(), config, bankroll=1000.0,
                 exposure=_holding_down(config))
    assert out.reason is not Reason.OPPOSITE_SIDE


def test_the_same_side_is_unaffected_either_way():
    """The flag must only touch the opposite-side branch."""
    for allow in (False, True):
        config = Config(min_edge_pp=0.0, max_entries_per_symbol_window=4,
                        allow_opposite_side=allow)
        out = decide(_row(model_probability=0.30, baseline_probability=0.32,
                          market_probability=0.36, ask_up=0.39, ask_down=0.65),
                     config, bankroll=1000.0, exposure=_holding_down(config))
        assert out.reason is not Reason.OPPOSITE_SIDE, allow
