"""Simulation stack invariants.

These are the tests that let a result be trusted. Each technique removes one
excuse a single backtest number leaves open, and each is checked against a case
where the right answer is known by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.metrics import evaluate_gates
from core.simulation import (
    RUIN_DRAWDOWN,
    SimulationReport,
    bootstrap_trades,
    capacity_curve,
    cost_stress,
    fit_regime_parameters,
    parameter_surface,
    politis_white_block_length,
    simulate_regime_path,
    stationary_bootstrap_indices,
    synthetic_panel,
)

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


# ---------------------------------------------------------------------------
# Block length
# ---------------------------------------------------------------------------


def test_block_length_tracks_dependence():
    """Resampling independently would destroy the clustering that makes drawdowns."""
    rng = np.random.default_rng(0)
    independent = rng.normal(0, 1, 500)
    autocorrelated = np.zeros(500)
    for i in range(1, 500):
        autocorrelated[i] = 0.8 * autocorrelated[i - 1] + rng.normal(0, 1)

    assert politis_white_block_length(independent) == pytest.approx(1.0, abs=0.5)
    assert politis_white_block_length(autocorrelated) > 4.0


def test_block_length_handles_degenerate_input():
    assert politis_white_block_length([]) == 1.0
    assert politis_white_block_length([1.0] * 50) == 1.0


def test_resample_preserves_length_and_stays_in_range():
    indices = stationary_bootstrap_indices(100, 5.0, np.random.default_rng(1))

    assert len(indices) == 100
    assert indices.min() >= 0
    assert indices.max() < 100


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_separates_a_real_edge_from_a_lucky_one():
    """The distinction a point-estimate Sharpe cannot make."""
    rng = np.random.default_rng(0)
    solid = rng.normal(0.010, 0.02, 200)
    lucky = rng.normal(0.001, 0.05, 40)

    strong = bootstrap_trades(solid, n_resamples=400, seed=3)
    weak = bootstrap_trades(lucky, n_resamples=400, seed=3)

    assert strong.probability_positive > 0.9
    assert weak.probability_positive < 0.5
    assert strong.sharpe.p05 > 0
    assert weak.sharpe.p05 < 0


def test_bootstrap_reports_risk_of_ruin():
    """Ruin is a drawdown a levered account does not come back from."""
    rng = np.random.default_rng(1)
    volatile = rng.normal(0.002, 0.15, 60)

    result = bootstrap_trades(volatile, n_resamples=400, seed=5)

    assert 0.0 < result.risk_of_ruin <= 1.0
    assert result.max_drawdown.p95 > RUIN_DRAWDOWN / 2


def test_bootstrap_needs_enough_trades():
    result = bootstrap_trades([0.01, 0.02, -0.01], n_resamples=100)

    assert result.n_resamples == 0
    assert result.risk_of_ruin == 1.0        # unknown is treated as unsafe


# ---------------------------------------------------------------------------
# Regimes and synthetic panels
# ---------------------------------------------------------------------------


def test_regime_fit_recovers_two_volatility_states():
    """A single-regime generator makes synthetic data uniformly easier than reality."""
    n = 3_000
    rng = np.random.default_rng(5)
    volatility = np.where(np.arange(n) % 600 < 150, 0.04, 0.008)
    returns = volatility * rng.standard_t(5, n) / np.sqrt(5 / 3)

    parameters = fit_regime_parameters(returns)

    assert parameters.violent_vol > parameters.quiet_vol * 1.5
    assert 0.1 < parameters.violent_share < 0.5
    assert parameters.tail_df < 15, 'fat tails should be detected, not assumed away'


def test_regime_fit_falls_back_on_short_series():
    parameters = fit_regime_parameters([0.01] * 20)

    assert parameters.quiet_vol > 0
    assert parameters.violent_vol > parameters.quiet_vol


def test_simulated_path_has_the_requested_scale():
    parameters = fit_regime_parameters(
        np.random.default_rng(2).normal(0, 0.02, 2_000)
    )
    path = simulate_regime_path(parameters, 5_000, np.random.default_rng(3))

    assert np.isfinite(path).all()
    assert 0.5 < path.std() / parameters.quiet_vol < 5.0


def test_synthetic_panel_preserves_cross_correlation():
    """Generating instruments independently removes the market factor.

    That flatters any strategy whose survival depends on diversification, which
    is most of them.
    """
    index = pd.date_range('2026-01-01', periods=1_200, freq='1h', tz='UTC')
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.01, len(index))

    bars = {}
    for symbol in ('BIP', 'ETP', 'SLP'):
        returns = 0.7 * market + 0.3 * rng.normal(0, 0.01, len(index))
        close = 100 * np.exp(np.cumsum(returns))
        bars[symbol] = pd.DataFrame(
            {'open': np.r_[close[0], close[:-1]], 'high': close * 1.004,
             'low': close * 0.996, 'close': close, 'volume': 1e5},
            index=index,
        )

    def mean_correlation(frames: dict[str, pd.DataFrame]) -> float:
        matrix = pd.DataFrame(
            {s: f['close'].pct_change() for s, f in frames.items()}
        ).corr().to_numpy()
        return float(matrix[np.triu_indices(len(frames), 1)].mean())

    synthetic = synthetic_panel(bars, seed=11)

    assert abs(mean_correlation(synthetic) - mean_correlation(bars)) < 0.2


def test_synthetic_bars_stay_internally_consistent():
    """A high below the close would make every barrier test meaningless."""
    index = pd.date_range('2026-01-01', periods=600, freq='1h', tz='UTC')
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    bars = {'SLP': pd.DataFrame(
        {'open': np.r_[close[0], close[:-1]], 'high': close * 1.005,
         'low': close * 0.995, 'close': close, 'volume': 1e5},
        index=index,
    )}

    for frame in synthetic_panel(bars, seed=2).values():
        assert (frame['high'] >= frame['close']).all()
        assert (frame['low'] <= frame['close']).all()
        assert (frame['close'] > 0).all()


def test_synthetic_panel_of_nothing_is_empty():
    assert synthetic_panel({}, seed=1) == {}


# ---------------------------------------------------------------------------
# Stress
# ---------------------------------------------------------------------------


def test_cost_stress_scales_the_per_contract_commission(config):
    """Otherwise stressing fees barely moves the contracts priced per contract.

    The commission is a fixed number of dollars, so doubling only the percentage
    leg understates the stress on exactly the instruments whose cost the
    commission dominates — the ones with the least notional per contract.
    """
    seen: list[float] = []

    def run(cfg: Config) -> float:
        seen.append(cfg.per_contract_fee_usd)
        return 1.0

    cost_stress(run, config)

    base = config.per_contract_fee_usd
    assert base > 0.0, 'baseline schedule charges nothing per contract'
    assert base in seen, 'baseline schedule missing'
    assert base * 2.0 in seen, 'the commission was not scaled by the fees_2x scenario'


def test_cost_stress_scales_a_per_symbol_schedule_too(config):
    """A venue that bills by instrument group must be stressed per symbol.

    Coinbase's app was measured billing one rate, so `per_contract_fee_by_symbol`
    is empty for it — but `per_contract_fee` prefers that dict whenever it is
    populated, so a schedule that fills it in would otherwise re-run the
    baseline under every fee scenario.
    """
    from dataclasses import replace

    grouped = replace(config, per_contract_fee_by_symbol={'BIP': 0.75, 'DOP': 0.10})
    seen: list[float] = []

    def run(cfg: Config) -> float:
        seen.append(cfg.per_contract_fee_by_symbol['BIP'])
        return 1.0

    cost_stress(run, grouped)

    assert 0.75 in seen, 'baseline schedule missing'
    assert 1.5 in seen, 'per-symbol commission was not scaled by the fees_2x scenario'


def test_stress_survival_requires_every_scenario_positive():
    good = cost_stress(lambda cfg: 0.5, Config())
    bad = cost_stress(
        lambda cfg: -0.2 if cfg.slippage_bps > 2.0 else 0.5, Config()
    )

    assert good.survives
    assert not bad.survives
    assert bad.worst < 0


# ---------------------------------------------------------------------------
# Surface and capacity
# ---------------------------------------------------------------------------


def test_surface_distinguishes_a_plateau_from_a_spike():
    """A real edge is insensitive to small parameter changes; an overfit is not."""
    plateau = parameter_surface(lambda p: 1.0, {'a': 1.0, 'b': 2.0})
    spike = parameter_surface(
        lambda p: 1.0 if p == {'a': 1.0, 'b': 2.0} else 0.1, {'a': 1.0, 'b': 2.0}
    )

    assert plateau.is_plateau
    assert not spike.is_plateau
    assert spike.retention == 0.0


def test_capacity_curve_finds_where_the_edge_dies():
    """A per-unit edge is not a fundable one."""
    result = capacity_curve(lambda equity: 1.5 - equity / 1e6, [1e4, 1e5, 1e6, 5e6])

    assert result.capacity == 1e6
    assert result.curve[5e6] < 0


def test_capacity_of_a_strategy_that_never_works_is_none():
    assert capacity_curve(lambda e: -1.0, [1e4, 1e5]).capacity is None


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_unmeasured_gates_block_promotion():
    """"We did not run that test" is not evidence of safety."""
    promoted, gates = evaluate_gates(SimulationReport(oos_trades=150).measurements())

    assert not promoted
    assert sum(1 for gate in gates if not gate.passed) >= 8


def test_report_flattens_into_gate_measurements():
    from core.metrics import summarise_paths

    report = SimulationReport(
        bootstrap=bootstrap_trades(
            np.random.default_rng(0).normal(0.01, 0.02, 200), n_resamples=200
        ),
        synthetic=summarise_paths([0.8, 0.9, 1.1, -0.2, 0.5]),
        per_period=summarise_paths([0.7, 0.8, 0.6, 0.9]),
        pbo=0.2,
        deflated_sharpe=1.4,
        oos_trades=180,
    )

    measurements = report.measurements()

    assert measurements['bootstrap_positive_fraction'] is not None
    assert measurements['synthetic_positive_fraction'] == pytest.approx(0.8)
    assert measurements['oos_trades'] == 180.0
    # Still blocked: stress and surface were not run.
    assert not evaluate_gates(measurements)[0]


def test_synthetic_panel_keeps_every_instrument_its_own_span():
    """A ragged panel must not be truncated to its shortest member.

    Joint generation needs a rectangular array, and the first attempt at that was
    `n = min(lengths)`. It worked and cost most of the evidence: on a store holding
    398 days across 18 CDE contracts the shortest was HYP at 75 days, so every
    synthetic panel, bootstrap and cost-stress figure was computed on 54 of 398
    days — and those 54 were a single directional quarter, the worst possible
    window to conclude from.

    Generating over the union span and restricting each instrument to its own
    listed window gives the ragged shape the real panel already has, so nothing
    downstream changes.
    """
    index = pd.date_range('2026-01-01', periods=2_000, freq='1h', tz='UTC')
    bars = {}
    for i, (symbol, start) in enumerate((('LONG', 0), ('MID', 1_200), ('SHORT', 1_900))):
        window = index[start:]
        rng = np.random.default_rng(i)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(window))))
        opens = np.concatenate([[close[0]], close[:-1]])
        bars[symbol] = pd.DataFrame(
            {'open': opens, 'high': np.maximum(opens, close) * 1.002,
             'low': np.minimum(opens, close) * 0.998, 'close': close,
             'volume': 1_000.0},
            index=window,
        )

    synthetic = synthetic_panel(bars, seed=11)

    for symbol, real in bars.items():
        assert synthetic[symbol].index.equals(real.index), (
            f'{symbol} was reindexed; the shortest common span would give '
            f'{min(len(b) for b in bars.values())} bars for all three'
        )
        assert np.isfinite(synthetic[symbol]['close']).all()
        # Compounding through months the contract did not exist would hand it a
        # drift it never had, so the path must start from the real first close.
        assert synthetic[symbol]['close'].iloc[0] == pytest.approx(
            float(real['close'].iloc[0]), rel=0.05)

    assert len(synthetic['LONG']) == 2_000
    assert len(synthetic['SHORT']) == 100
