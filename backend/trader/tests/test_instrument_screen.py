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


def test_the_verdict_needs_every_gate_not_any_of_them(config):
    """A horizon that pays but cannot be fitted or verified is not a pass.

    Three conditions, and all must hold: the forecast requirement is reachable,
    the effective sample is large enough to fit, and the requirement sits outside
    its own measurement noise. The third was added after the ladder showed a
    ridge reaching test IC +0.083 at h=96h against a standard error of 0.25 —
    nominally twice the required IC, and a third of one standard error from zero.
    """
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

    # And a cell clearing all three does pass, so the gate is not vacuously
    # strict. Required IC 0.05 against a standard error of ~0.032 at 1,000
    # observations: affordable, fittable, and outside its own noise.
    frame.loc[len(frame)] = {'symbol': 'A', 'horizon': 24, 'required_ic': 0.05,
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


def test_pooled_observations_are_discounted_by_the_universe_s_own_correlation():
    """18 instruments moving together are not 18 sources of evidence.

    The screen originally summed per-instrument observations and gated on that,
    which passed h=96h with 279. At the measured pairwise correlation of 0.658,
    `N / (1 + (N-1) rho)` turns 18 instruments into 1.48 effective names, so the
    honest figure is 23 — and the implied IC standard error is 0.21 against a
    required IC of 0.045. A horizon whose requirement sits inside its own
    measurement noise cannot be validated at all: a model that clears the hurdle
    is indistinguishable from one that does not.

    That is the vise this screen exists to show. Short holds are measurable and
    unaffordable; long holds are affordable and unmeasurable.
    """
    import pandas as pd

    from scripts.instrument_screen import PAIRWISE_CORRELATION, _verdict

    assert 0.0 < PAIRWISE_CORRELATION < 1.0

    # h=96h as measured: cheap to forecast, 279 pooled observations over 18 names.
    frame = pd.DataFrame([
        {'symbol': f'S{i}', 'horizon': 96, 'required_ic': 0.045,
         'ceiling_win_rate': 0.95, 'round_trip_bps': 29.0,
         'effective_obs': 279.0 / 18}
        for i in range(18)
    ])
    summary, any_pass = _verdict(frame, limit=0.05, min_obs=200.0)
    row = summary.iloc[0]

    assert row['effective_obs'] == pytest.approx(279.0, rel=0.01)
    assert row['effective_names'] < 2.0, row['effective_names']
    assert row['effective_obs_adj'] < 40.0, (
        f'pooled 279 discounted to {row["effective_obs_adj"]:.0f}; if this is '
        f'still in the hundreds the correlation discount is not being applied'
    )
    # The economics pass and the measurability does not, which is the whole point.
    assert bool(row['cost_ok'])
    assert not bool(row['measurable']), (
        f'required IC {row["required_ic_median"]:.3f} vs standard error '
        f'{row["ic_standard_error"]:.3f} — a requirement inside its own noise '
        f'must not be reported as verifiable'
    )
    assert not any_pass


def test_the_tradeable_universe_is_a_rule_not_a_symbol_list():
    """Three thresholds must reproduce the five, from the data, every run.

    `--exclude BIP,ETP` needs someone to remember why each name is on the list;
    a rule re-derives its own answer and fails loudly when the data moves. This is
    the same reason `--min-history-days` is preferred over `--exclude` for the
    ragged-listing problem.

    Two of the three thresholds are derived. `max_round_trip_bps=35` is the fee
    schedule: 27bp is the cheapest contract on this venue, so 35 keeps the book
    within ~30% of it. `min_history_days=231` is the span that covers three
    falling quarters plus the rally rather than the rally alone.

    The third is a choice, and the test says so: at `max_gap_over_cost=0.40` this
    admits ADP at 0.37 and rejects LCP at 0.41 — a 2.5% margin on a median, well
    inside its own noise. The two are interchangeable on that metric and the
    threshold was set knowing where they fell. Asserted anyway, because a silent
    drift in either direction should be visible.
    """
    import pandas as pd

    from core.config import Config, find_cost_config
    from core.targets import round_trip_cost_series
    from core.datastore import ResearchStore

    path = find_cost_config()
    if path is None:
        pytest.skip('no cost config on the search path')
    config = Config().with_cost_assumptions(path)

    bars = ResearchStore().read('bars', venue='coinbase')
    if bars.empty:
        pytest.skip('no bars in the store')

    rows = []
    for symbol, group in bars.groupby('symbol'):
        frame = group.set_index('event_time').sort_index()
        if len(frame) < 500:
            continue
        round_trip = float(
            round_trip_cost_series(symbol, frame['close'], config).median()) * 10_000
        gap = float((frame['open'].shift(-1) / frame['close'] - 1.0)
                    .abs().median()) * 10_000
        rows.append({
            'code': symbol.split('-')[0],
            'days': (frame.index[-1] - frame.index[0]).total_seconds() / 86_400.0,
            'round_trip_bps': round_trip,
            'gap_over_cost': gap / round_trip if round_trip else float('inf'),
        })
    frame = pd.DataFrame(rows)

    survivors = sorted(frame[(frame.round_trip_bps <= 35.0)
                             & (frame.gap_over_cost <= 0.40)
                             & (frame.days >= 231.0)].code)
    assert survivors == ['ADP', 'BIP', 'ETP', 'SLP', 'XPP'], (
        f'the rule now selects {survivors}. Either the store changed or a '
        f'threshold drifted — check which before changing this assertion.'
    )

    # Each exclusion for the stated reason, so a rewrite cannot keep the answer
    # while losing the mechanism.
    by_code = frame.set_index('code')
    for code in ('SHP', 'AVP', 'POP'):
        if code in by_code.index:
            assert by_code.loc[code, 'round_trip_bps'] > 35.0, code
    for code in ('LCP', 'LNP', 'DOP', 'BCP', 'SUP', 'XLP', 'NER', 'PEP', 'OND'):
        if code in by_code.index:
            assert by_code.loc[code, 'gap_over_cost'] > 0.40, code
    if 'HYP' in by_code.index:
        assert by_code.loc['HYP', 'days'] < 231.0

    # And the boundary is genuinely tight, which is the caveat worth keeping live.
    if 'LCP' in by_code.index:
        margin = by_code.loc['LCP', 'gap_over_cost'] - by_code.loc['ADP', 'gap_over_cost']
        assert margin < 0.10, (
            f'ADP and LCP are separated by {margin:.3f} on fill uncertainty; the '
            f'0.40 threshold is doing real work at a boundary inside its own noise'
        )


def test_pruning_protects_the_data_that_cannot_be_refetched(tmp_path):
    """Bars come back from a scrape; funding and open interest never do.

    Both are single-value snapshots on the product endpoint — no range
    parameters, no cursor — so they accumulate forward only and no request
    recovers a deleted row. `.gitignore` un-ignores those two datasets for
    exactly that reason.

    So a universe prune must refuse them by default. A tool that treats a
    regenerable partition and an irreplaceable one the same way is one flag away
    from destroying the only data in the store that cannot be rebuilt.
    """
    import pathlib
    import subprocess
    import sys

    root = tmp_path / 'research'
    for dataset in ('bars', 'funding', 'open_interest'):
        for symbol in ('BIP-20DEC30-CDE', 'LCP-20DEC30-CDE'):
            d = root / dataset / 'venue=coinbase' / f'symbol={symbol}' / 'month=2026-01'
            d.mkdir(parents=True)
            (d / 'data.parquet').write_bytes(b'x' * 32)

    def run(*extra):
        return subprocess.run(
            [sys.executable, '-m', 'scripts.prune_universe',
             '--store', str(root), '--db-path', str(tmp_path / 'none.db'),
             '--keep', 'BTC', *extra],
            capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parents[1]),
        )

    # Dry run changes nothing at all.
    out = run()
    assert out.returncode == 0, out.stderr
    assert 'dry run' in out.stdout
    assert (root / 'bars/venue=coinbase/symbol=LCP-20DEC30-CDE').exists()

    # Applying removes the out-of-universe bars and keeps the snapshots.
    out = run('--apply')
    assert out.returncode == 0, out.stderr
    assert not (root / 'bars/venue=coinbase/symbol=LCP-20DEC30-CDE').exists(), \
        'out-of-universe bars should be removed'
    assert (root / 'funding/venue=coinbase/symbol=LCP-20DEC30-CDE').exists(), \
        'funding is irreplaceable and must survive without an explicit override'
    assert (root / 'open_interest/venue=coinbase/symbol=LCP-20DEC30-CDE').exists()
    assert 'PROTECTED' in out.stdout

    # The kept instrument is untouched throughout.
    assert (root / 'bars/venue=coinbase/symbol=BIP-20DEC30-CDE').exists()

    # And the override does delete them, so the protection is a default and not
    # a dead branch.
    out = run('--apply', '--include-irreplaceable')
    assert out.returncode == 0, out.stderr
    assert not (root / 'funding/venue=coinbase/symbol=LCP-20DEC30-CDE').exists()
