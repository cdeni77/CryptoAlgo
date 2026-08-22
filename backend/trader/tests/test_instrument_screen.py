"""The screen that should have run first: can an instrument pay for its trading?

Two gates that pull opposite ways. `required_ic = cost / sigma_h` falls as
`1/sqrt(h)` because cost is fixed per round trip. Effective observations fall as
`1/h` because a label spanning h bars overlaps its h-1 neighbours. Reporting
either alone recommends a horizon the other forbids, which is how this repo
simultaneously held that h=96h was the only affordable hold and that h=96h had too
few observations to fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config, find_cost_config
from scripts.instrument_screen import _effective_observations, _verdict, screen


def _bars(symbol: str, days: int, price: float, vol: float, seed: int) -> pd.DataFrame:
    index = pd.date_range('2025-01-01', periods=days * 24, freq='h', tz='UTC')
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, len(index))))
    return pd.DataFrame({
        'venue': 'test', 'symbol': symbol, 'event_time': index,
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': 1_000.0,
    })


class _Store:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def read(self, dataset: str, *, venue: str | None = None, **_):
        return self._frame


@pytest.fixture(scope='module')
def config() -> Config:
    path = find_cost_config()
    if path is None:
        pytest.skip('no cost config on the search path')
    return Config().with_cost_assumptions(path)


def test_required_ic_falls_with_the_horizon_and_the_sample_falls_with_it(config):
    """The whole reason the screen reports both.

    Cost is charged once per round trip and dispersion grows as sqrt(h), so a
    longer hold is cheaper in IC terms — by exactly the factor that makes the
    sample smaller. A screen reporting one of these is worse than no screen: it
    recommends a horizon with confidence the other number withdraws.
    """
    frame = screen(_Store(_bars('BIP-20DEC30-CDE', 300, 60_000.0, 0.004, 1)),
                   config, venue='test', horizons=[1, 4, 24, 96])
    by_h = frame.set_index('horizon')

    required = [float(by_h.loc[h, 'required_ic']) for h in (1, 4, 24, 96)]
    assert required == sorted(required, reverse=True), (
        f'required IC must fall with the hold, got {required}'
    )
    # And roughly as 1/sqrt(h): 96x the hold is ~9.8x cheaper.
    assert 6.0 < required[0] / required[3] < 15.0, required

    obs = [float(by_h.loc[h, 'effective_obs']) for h in (1, 4, 24, 96)]
    assert obs == sorted(obs, reverse=True), f'sample must fall with the hold, got {obs}'

    # The win-rate ceiling rises with the hold for the same reason: the toll
    # becomes a smaller share of the move.
    ceilings = [float(by_h.loc[h, 'ceiling_win_rate']) for h in (1, 4, 24, 96)]
    assert ceilings == sorted(ceilings), ceilings
    assert ceilings[0] < 0.75, 'a short hold cannot have a high ceiling'


def test_the_verdict_needs_both_gates_not_either(config):
    """A horizon that pays but cannot be fitted is not a pass."""
    frame = pd.DataFrame([
        # Cheap to forecast, far too few observations.
        {'symbol': 'A', 'horizon': 96, 'required_ic': 0.02, 'ceiling_win_rate': 0.95,
         'round_trip_bps': 27.0, 'effective_obs': 20.0},
        # Plenty of observations, needs impossible skill.
        {'symbol': 'A', 'horizon': 1, 'required_ic': 0.40, 'ceiling_win_rate': 0.53,
         'round_trip_bps': 27.0, 'effective_obs': 9_000.0},
    ])
    summary, any_pass = _verdict(frame, limit=0.05, min_obs=200.0)

    assert not any_pass, 'neither horizon clears both gates'
    rows = summary.set_index('horizon')
    assert bool(rows.loc[96, 'cost_ok']) and not bool(rows.loc[96, 'sample_ok'])
    assert bool(rows.loc[1, 'sample_ok']) and not bool(rows.loc[1, 'cost_ok'])

    # And a cell clearing both does pass, so the gate is not vacuously strict.
    frame.loc[len(frame)] = {'symbol': 'A', 'horizon': 24, 'required_ic': 0.03,
                             'ceiling_win_rate': 0.9, 'round_trip_bps': 27.0,
                             'effective_obs': 1_000.0}
    _, now_passes = _verdict(frame, limit=0.05, min_obs=200.0)
    assert now_passes


def test_the_recency_decay_caps_the_sample_however_deep_the_store_is():
    """More history stops helping past roughly three half-lives.

    This is the number that decides whether a deeper scrape can change a fit, and
    getting it wrong recommends fetching data that arrives pre-discounted to
    nothing. At a 50-day half-life, tripling the history from one year to three
    moves the effective sample by almost nothing.
    """
    def obs(days: int, half_life: float) -> float:
        index = pd.date_range('2020-01-01', periods=days * 24, freq='h', tz='UTC')
        return _effective_observations(index, 96, half_life)

    short, long = obs(365, 50.0), obs(1_095, 50.0)
    assert long / short < 1.05, (
        f'at H=50d, 3x the history moved the sample {long / short:.2f}x — the '
        f'decay is not capping it'
    )

    # Raising the half-life is the lever that does work.
    assert obs(1_095, 365.0) / obs(1_095, 50.0) > 3.0

    # And with no decay the sample is just uniqueness, which scales with history.
    assert obs(1_095, 0.0) / obs(365, 0.0) == pytest.approx(3.0, rel=0.05)
