"""How much of a long history actually reaches the model.

Three knobs governed this and none of them was connected:

- `recency_half_life_days` defaulted to 50 and had no override path at all. No
  script wired `CLI_PARAMS` into its parser, `Config.from_env` had no callers,
  and `_common.build_config` constructed a bare `Config()`. So the decay was a
  hardcoded constant that nothing could see or change.
- `train_lookback_days` (with `--train-lookback-days` and `TRAIN_WINDOW_DAYS`)
  was declared in `Config` and referenced by nothing outside `config.py`.
- `live_orchestrator` defaulted `--train-window-days` to 90 and wrote
  `retrain_window_days=90` into Postgres on every run, but never passed it to
  `scripts.promote` — which had no such flag either. The run table described a
  90-day fit that never happened.

The consequence was quantitative, not cosmetic. A 50-day half-life caps the
weighted sample at about `24 * H / ln 2` bar-equivalents however far back the
store goes, so at a 96h horizon it saturates near 18 effective observations
whether you hold one year of history or five — while `preflight` reported the
unweighted count (456 at five years) and recommended scraping more.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.cv import average_uniqueness, effective_sample_size
from core.dataset import Dataset
from scripts._common import add_data_arguments, build_config
from scripts.preflight import MIN_EFFECTIVE_OBSERVATIONS, _weighted_effective

HOURS = 24


def _parser() -> argparse.ArgumentParser:
    return add_data_arguments(argparse.ArgumentParser())


def _dataset(days: int, symbols=('BIP', 'ETP'), horizon_bars: int = 24) -> Dataset:
    times = pd.date_range('2021-01-01', periods=days * HOURS, freq='h', tz='UTC')
    index = pd.MultiIndex.from_product([list(symbols), times],
                                       names=['symbol', 'event_time'])
    rng = np.random.default_rng(0)
    features = pd.DataFrame({'f0': rng.normal(size=len(index))}, index=index)
    targets = pd.DataFrame({'price': rng.normal(size=len(index)) * 1e-3}, index=index)
    bars = {
        s: pd.DataFrame({'close': 100.0, 'volume': 1_000.0}, index=times)
        for s in symbols
    }
    funding = {s: pd.DataFrame({'rate': 2e-5}, index=times) for s in symbols}
    return Dataset(
        features=features, targets=targets, bars=bars, funding=funding,
        profiles={}, venue='coinbase', reference_venue=None, as_of=None,
        horizon_bars=horizon_bars,
    )


# ---------------------------------------------------------------------------
# The override paths exist
# ---------------------------------------------------------------------------


def test_the_recency_half_life_is_reachable_from_the_command_line():
    """The regression test for the whole class: a knob nobody can turn.

    `recency_half_life_days` decides how much of the history reaches the model,
    and it was a hardcoded 50.0 — the CliParam existed but no script wired
    `CLI_PARAMS` into a parser, so there was no way to set it short of editing
    the source.
    """
    args = _parser().parse_args(['--recency-half-life-days', '365'])
    assert build_config(args).recency_half_life_days == 365.0

    # And zero has to mean off, not "fall back to the default".
    args = _parser().parse_args(['--recency-half-life-days', '0'])
    assert build_config(args).recency_half_life_days == 0.0


def test_omitting_the_flag_keeps_the_config_default():
    args = _parser().parse_args([])
    assert build_config(args).recency_half_life_days == Config().recency_half_life_days


def test_every_config_field_the_flags_claim_to_set_is_actually_read():
    """A flag that parses and reaches nothing is worse than no flag.

    `CLI_PARAMS` used to declare 22 of them, plus 5 environment variables, and
    nothing called the machinery that would have wired them — including
    `LEVERAGE`, which an operator could lower in docker-compose while the book
    kept trading at 4x. It has been deleted; the surface is
    `scripts/_common.py:add_data_arguments`, and this asserts that what it sets
    is genuinely consumed somewhere.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    sources = {
        path: path.read_text()
        for directory in ('core', 'scripts', 'data_collection')
        for path in (root / directory).rglob('*.py')
        if '__pycache__' not in str(path) and path.name != 'config.py'
    }

    # Fields `build_config` sets from a flag.
    common = (root / 'scripts' / '_common.py').read_text()
    assigned = set(re.findall(r'replace\(config, (\w+)=', common))
    assert assigned, 'no flag sets a Config field; the surface has moved'

    unread = [
        field for field in assigned
        if not any(re.search(rf'\b{field}\b', text) for text in sources.values())
    ]
    assert not unread, (
        f'these flags set a Config field nothing reads: {sorted(unread)}'
    )


# ---------------------------------------------------------------------------
# The training window
# ---------------------------------------------------------------------------


def test_a_window_truncates_every_frame_together():
    """Features, targets, bars and funding must be cut on the same boundary.

    Truncating the panel but not the bars would leave the backtest simulating
    over a span the model never saw.
    """
    dataset = _dataset(days=400)
    windowed = dataset.trailing(90)

    times = windowed.features.index.get_level_values('event_time')
    span_days = (times.max() - times.min()).total_seconds() / 86_400
    assert 89 <= span_days <= 90

    assert len(windowed.features) < len(dataset.features)
    assert len(windowed.targets) == len(windowed.features)
    for symbol in windowed.bars:
        assert windowed.bars[symbol].index.min() >= times.min()
        assert windowed.funding[symbol].index.min() >= times.min()


def test_a_window_says_what_it_dropped():
    """A silent truncation is how a 90-day fit gets reported as a 400-day one."""
    windowed = _dataset(days=400).trailing(90)

    assert any('training window' in w for w in windowed.warnings)


@pytest.mark.parametrize('window', [None, 0, -5])
def test_no_window_is_a_no_op(window):
    dataset = _dataset(days=120)

    assert dataset.trailing(window) is dataset


def test_a_window_longer_than_the_history_is_a_no_op():
    dataset = _dataset(days=30)

    assert dataset.trailing(365) is dataset


def test_the_window_flag_reaches_the_parser():
    args = _parser().parse_args(['--train-window-days', '180'])

    assert args.train_window_days == 180.0


# ---------------------------------------------------------------------------
# What the orchestrator hands to the training step
# ---------------------------------------------------------------------------


def test_the_orchestrator_passes_the_training_controls_to_promote():
    """The original bug: recorded in Postgres, never passed to the trainer."""
    from scripts.live_orchestrator import _training_arguments

    args = argparse.Namespace(train_window_days=730.0, recency_half_life_days=365.0)
    flags = _training_arguments(args)

    assert flags == ['--train-window-days', '730.0', '--recency-half-life-days', '365.0']


def test_the_training_controls_stay_out_of_the_data_arguments():
    """The signal writer must score the latest bar from the full panel.

    A training window leaking into `_data_arguments` would truncate the features
    that `scripts.signals` needs to decide on the most recent timestamp.
    """
    from scripts.live_orchestrator import _data_arguments

    args = argparse.Namespace(
        venue='coinbase', min_quality='valid', store=None, reference_venue='binance',
        symbols=None, cost_config=None, log_level='INFO',
        train_window_days=730.0, recency_half_life_days=365.0,
    )
    flags = _data_arguments(args)

    assert '--train-window-days' not in flags
    assert '--recency-half-life-days' not in flags


def test_the_orchestrator_defaults_to_all_history():
    """A 90-day default that now actually bites would cap a five-year scrape.

    The old default was 90 and inert. Making it live without changing it would
    have silently thrown away every year of history the user scraped.
    """
    import sys

    from scripts.live_orchestrator import parse_args

    argv = sys.argv
    sys.argv = ['live_orchestrator']
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    assert not args.train_window_days, 'the default window must be unbounded'
    assert args.recency_half_life_days is None, 'the default must defer to Config'


# ---------------------------------------------------------------------------
# The number preflight has to report
# ---------------------------------------------------------------------------


def _hourly(days: int) -> pd.DatetimeIndex:
    return pd.date_range('2021-01-01', periods=days * HOURS, freq='h', tz='UTC')


def test_recency_weighting_saturates_the_sample():
    """More history stops helping past roughly three half-lives.

    This is the measurement that made the "scrape 2.2 years" advice wrong: at a
    96h horizon with the default 50-day half-life, one year and five years of
    history give training the same ~18 effective observations, while the
    unweighted count grows fivefold.
    """
    horizon, half_life = 96, 50.0

    one_year = _weighted_effective(_hourly(365), horizon, half_life)
    five_years = _weighted_effective(_hourly(1825), horizon, half_life)

    assert five_years - one_year < 2, (
        f'five years bought {five_years - one_year:.1f} effective observations'
    )

    # While the unweighted count says the opposite.
    assert (effective_sample_size(_hourly(1825), horizon)
            > 4 * effective_sample_size(_hourly(365), horizon))


def test_the_saturation_point_matches_the_closed_form():
    """Weights sum to about `24 * H / ln 2` bar-equivalents, divided by horizon."""
    horizon, half_life = 24, 180.0
    predicted = 24.0 * half_life / math.log(2) / horizon

    measured = _weighted_effective(_hourly(3650), horizon, half_life)

    assert measured == pytest.approx(predicted, rel=0.02)


def test_no_decay_reduces_to_the_unweighted_count():
    index = _hourly(200)

    assert (_weighted_effective(index, 24, 0.0)
            == pytest.approx(effective_sample_size(index, 24)))


def test_the_weighted_count_never_exceeds_the_unweighted_one():
    index = _hourly(500)
    for horizon in (8, 24, 96):
        for half_life in (0.0, 50.0, 365.0):
            assert (_weighted_effective(index, horizon, half_life)
                    <= effective_sample_size(index, horizon) + 1e-6)


def test_uniqueness_is_the_unweighted_building_block():
    """Guards the identity the whole calculation rests on."""
    index = _hourly(100)

    assert (float(average_uniqueness(index, 24).sum())
            == pytest.approx(effective_sample_size(index, 24)))


def test_the_default_settings_cannot_reach_the_threshold_at_a_long_horizon():
    """Documents why the profile default horizon was the wrong lever to keep.

    At 96h with a 50-day half-life the ceiling is far below the gate threshold,
    so no scrape of any length makes that combination trainable. This is the
    fact `preflight` now has to state rather than recommending more data.
    """
    ceiling = 24.0 * 50.0 / math.log(2) / 96

    assert ceiling < MIN_EFFECTIVE_OBSERVATIONS
    assert _weighted_effective(_hourly(3650), 96, 50.0) < MIN_EFFECTIVE_OBSERVATIONS


# ---------------------------------------------------------------------------
# What preflight tells you to do about it
# ---------------------------------------------------------------------------


class _Resolved:
    """The two attributes `_effective_sample` reads off a Dataset."""

    def __init__(self, days: int, horizon_bars: int):
        times = pd.date_range('2021-01-01', periods=days * HOURS, freq='h', tz='UTC')
        self.resolved_index = pd.MultiIndex.from_product(
            [['BIP'], times], names=['symbol', 'event_time'])
        self.horizon_bars = horizon_bars


def _check(days: int, horizon: int, half_life: float):
    from dataclasses import replace

    from scripts.preflight import _effective_sample

    return _effective_sample(_Resolved(days, horizon),
                             replace(Config(), recency_half_life_days=half_life))


def test_a_failing_check_returns_a_check_not_none():
    """Regression: the failing branch fell off the end of the function.

    Every test here exercised `_weighted_effective` directly, so a missing
    `return` on the path that actually fires — the one that tells you what to do
    about a short sample — went unnoticed until the diff was read by eye.
    """
    check = _check(days=92, horizon=96, half_life=50.0)

    assert check is not None
    assert check.passed is False
    assert isinstance(check.fix, str) and check.fix


def test_the_advice_names_the_half_life_when_that_is_the_binding_limit():
    """Five years at 96h/50d: more data cannot help, and it has to say so."""
    check = _check(days=1825, horizon=96, half_life=50.0)

    assert not check.passed
    assert 'half-life is the binding limit' in check.fix
    assert 'more history will not help' in check.fix
    assert 'scrape about' not in check.fix, 'recommended a scrape that cannot help'


def test_the_advice_recommends_a_scrape_when_history_is_what_is_short():
    """60 days at 8h: the decay is not the constraint, the span is."""
    check = _check(days=60, horizon=8, half_life=50.0)

    assert not check.passed
    assert 'scrape about' in check.fix
    assert 'binding limit' not in check.fix


def test_the_suggested_horizon_is_always_shorter_than_the_current_one():
    """Sizing the suggestion off the raw timestamp count advised 219h at 96h.

    Lengthening the horizon makes the sample smaller, so advice to lengthen it
    is not a slightly wrong number — it is the opposite of the fix.
    """
    for days, horizon, half_life in ((1825, 96, 50.0), (92, 96, 50.0), (365, 24, 50.0)):
        check = _check(days, horizon, half_life)
        if check.passed or '--horizon' not in check.fix:
            continue
        suggested = int(check.fix.split('--horizon ')[1].split(')')[0])
        assert suggested < horizon, (
            f'{days}d at {horizon}h suggested --horizon {suggested}'
        )


def test_a_long_enough_half_life_clears_the_gate_on_five_years():
    """The combination the measurements point at, pinned as a fact."""
    check = _check(days=1825, horizon=96, half_life=730.0)

    assert check.passed, check.detail


def test_the_detail_reports_both_numbers():
    """The unweighted count is still worth seeing next to the weighted one."""
    check = _check(days=1825, horizon=96, half_life=50.0)

    assert '18 effective observations' in check.detail
    assert 'uniqueness 456' in check.detail
    assert 'half-life 50d' in check.detail


def test_a_symbol_with_no_funding_survives_the_window():
    """The loader stores an empty DataFrame when funding is missing everywhere.

    An empty frame carries a RangeIndex, and `RangeIndex >= Timestamp` raises
    rather than selecting nothing — so the naive filter would take down every
    windowed run that included one instrument without funding data.
    """
    dataset = _dataset(days=400)
    dataset.funding['ETP'] = pd.DataFrame()

    windowed = dataset.trailing(90)

    assert windowed.funding['ETP'].empty
    assert not windowed.funding['BIP'].empty
    assert len(windowed.funding['BIP']) < len(dataset.bars['BIP'])
