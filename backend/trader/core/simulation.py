"""The simulation stack: what decides whether a strategy is allowed to trade.

A single backtest number is not evidence. It is one draw from a distribution
whose width nobody measured, chosen from however many configurations were tried,
on the one price path history happened to take. Each technique here removes one
of those excuses, and the promotion gates in `core.metrics` read their outputs.

    bootstrap_trades      Sharpe confidence interval, drawdown distribution and
                          risk of ruin, from resampling the trade sequence.
    synthetic_panels      Does the strategy survive paths that did not happen?
    cost_stress           Does it survive costs being worse than assumed?
    parameter_surface     Is the chosen configuration a plateau or a spike?
    capacity_curve        At what size does the edge disappear?

One honest limit, stated because the failure mode here is believing the
machinery: a synthetic generator contains only the structure calibrated into it.
Fit one with momentum and momentum strategies will work on it; fit one without
and nothing will. Synthetic panels test robustness and sizing. They are not
evidence of edge, and only genuinely unseen data is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config
from core.metrics import (
    DrawdownProfile,
    PathDistribution,
    drawdown_profile,
    sharpe_ratio,
    summarise_paths,
)

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 24 * 365

# Ruin is not "equity reached zero" — a 4x-levered account is finished long
# before that, because margin calls arrive first and nobody keeps trading a book
# down this far.
RUIN_DRAWDOWN = 0.5


# ---------------------------------------------------------------------------
# Stationary bootstrap
# ---------------------------------------------------------------------------


def politis_white_block_length(series: Sequence[float]) -> float:
    """Expected block length for a stationary bootstrap.

    Politis and White (2004), simplified: the block has to be long enough to
    carry the series' own dependence, so it is derived from the autocorrelation
    rather than picked. Independent draws want a block of one; strongly
    autocorrelated returns want many.

    Resampling individual trades independently would destroy the clustering that
    produces drawdowns, and a drawdown distribution is most of what this is for.
    """
    values = np.asarray([float(x) for x in series], dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 8:
        return 1.0

    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / n)
    if variance <= 1e-18:
        return 1.0

    # Sum autocorrelations up to the lag where they stop being distinguishable
    # from zero, then convert to a block length.
    threshold = 2.0 / np.sqrt(n)
    total = 0.0
    for lag in range(1, min(n // 4, 50) + 1):
        rho = float(np.dot(centred[:-lag], centred[lag:]) / (n * variance))
        if abs(rho) < threshold:
            break
        total += rho

    return float(min(max(1.0 + 2.0 * total, 1.0), max(n / 4.0, 1.0)))


def stationary_bootstrap_indices(
    n: int,
    block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One resample of length `n` with geometric block lengths.

    Blocks wrap around the end of the series, which is what makes the resample
    stationary — every observation has the same chance of appearing, including
    the ones near the boundaries.
    """
    if n <= 0:
        return np.zeros(0, dtype=int)
    probability = 1.0 / max(block_length, 1.0)

    indices = np.empty(n, dtype=int)
    position = int(rng.integers(0, n))
    for i in range(n):
        indices[i] = position
        if rng.random() < probability:
            position = int(rng.integers(0, n))
        else:
            position = (position + 1) % n
    return indices


@dataclass
class BootstrapResult:
    """What resampling the trade sequence says about the result's reliability.

    `probability_positive` is the gate input. A Sharpe of 1.2 whose bootstrap
    distribution straddles zero is one lucky ordering of trades, and this is what
    tells the difference.
    """

    n_resamples: int
    block_length: float
    sharpe: PathDistribution
    total_return: PathDistribution
    max_drawdown: PathDistribution
    probability_positive: float
    risk_of_ruin: float
    observed_sharpe: float

    def as_dict(self) -> dict[str, Any]:
        return {
            'n_resamples': self.n_resamples,
            'block_length': round(self.block_length, 2),
            'observed_sharpe': round(self.observed_sharpe, 4),
            'sharpe': self.sharpe.as_dict(),
            'total_return': self.total_return.as_dict(),
            'max_drawdown': self.max_drawdown.as_dict(),
            'probability_positive': round(self.probability_positive, 4),
            'risk_of_ruin': round(self.risk_of_ruin, 4),
        }

    def __str__(self) -> str:
        return (
            f"{self.n_resamples} resamples (block {self.block_length:.1f}) | "
            f"Sharpe median {self.sharpe.median:+.2f} "
            f"[p05 {self.sharpe.p05:+.2f}, p95 {self.sharpe.p95:+.2f}] | "
            f"P(positive) {self.probability_positive:.0%} | "
            f"maxDD p95 {self.max_drawdown.p95:.1%} | "
            f"ruin {self.risk_of_ruin:.1%}"
        )


def bootstrap_trades(
    trade_returns: Sequence[float],
    *,
    n_resamples: int = 2_000,
    initial_equity: float = 1.0,
    seed: int = 7,
    block_length: Optional[float] = None,
) -> BootstrapResult:
    """Resample the trade sequence to get a distribution instead of one number.

    Trade returns are per-trade fractions of notional. Each resample is compounded
    into an equity path, so the drawdown distribution reflects the real ordering
    risk: the same trades in a different order produce very different worst cases.
    """
    returns = np.asarray([float(x) for x in trade_returns], dtype=float)
    returns = returns[np.isfinite(returns)]
    empty = summarise_paths([])
    if returns.size < 8:
        return BootstrapResult(0, 1.0, empty, empty, empty, 0.0, 1.0, 0.0)

    block = block_length if block_length is not None else politis_white_block_length(returns)
    rng = np.random.default_rng(seed)

    sharpes: list[float] = []
    totals: list[float] = []
    drawdowns: list[float] = []
    ruined = 0

    for _ in range(n_resamples):
        sample = returns[stationary_bootstrap_indices(returns.size, block, rng)]
        equity = initial_equity * np.cumprod(1.0 + sample)
        profile = drawdown_profile(
            np.concatenate([[initial_equity], equity]), periods_per_year=len(sample)
        )

        # Per-trade Sharpe, annualised by trade count rather than by calendar
        # time: these are trades, not a time series, so the trade is the period.
        sigma = sample.std(ddof=1)
        sharpes.append(float(sample.mean() / sigma * np.sqrt(len(sample))) if sigma > 1e-12 else 0.0)
        totals.append(float(equity[-1] / initial_equity - 1.0))
        drawdowns.append(profile.max_drawdown)
        if profile.max_drawdown >= RUIN_DRAWDOWN:
            ruined += 1

    observed_sigma = returns.std(ddof=1)
    observed = (
        float(returns.mean() / observed_sigma * np.sqrt(returns.size))
        if observed_sigma > 1e-12 else 0.0
    )

    return BootstrapResult(
        n_resamples=n_resamples,
        block_length=block,
        sharpe=summarise_paths(sharpes),
        total_return=summarise_paths(totals),
        max_drawdown=summarise_paths(drawdowns),
        probability_positive=float(np.mean([s > 0 for s in sharpes])),
        risk_of_ruin=float(ruined / n_resamples),
        observed_sharpe=observed,
    )


# ---------------------------------------------------------------------------
# Synthetic panels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeParameters:
    """A two-state volatility model fitted to one instrument.

    Crypto returns are not a single distribution: they alternate between quiet
    and violent, and the violent state is where strategies die. A single-regime
    generator produces synthetic data that is uniformly easier than reality.
    """

    quiet_vol: float
    violent_vol: float
    stay_quiet: float
    stay_violent: float
    drift: float
    tail_df: float

    @property
    def violent_share(self) -> float:
        """Long-run fraction of time spent in the violent state."""
        quiet_exit = 1.0 - self.stay_quiet
        violent_exit = 1.0 - self.stay_violent
        total = quiet_exit + violent_exit
        return float(quiet_exit / total) if total > 0 else 0.5


def fit_regime_parameters(returns: Sequence[float], *, quantile: float = 0.75) -> RegimeParameters:
    """Fit two volatility states by thresholding realised volatility.

    A full hidden-Markov fit is not worth the fragility here: a threshold on
    rolling volatility separates the states well enough, and the transition
    probabilities come straight from how often the classification flips.
    Student-t degrees of freedom are backed out of the excess kurtosis, so the
    synthetic tails are as fat as the real ones rather than Gaussian.
    """
    series = pd.Series([float(x) for x in returns]).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 100:
        return RegimeParameters(0.01, 0.03, 0.95, 0.90, 0.0, 5.0)

    rolling = series.rolling(24, min_periods=8).std()
    threshold = float(rolling.quantile(quantile))
    violent = (rolling > threshold).fillna(False)

    quiet_returns = series[~violent]
    violent_returns = series[violent]

    transitions = violent.astype(int).diff().fillna(0)
    quiet_count = int((~violent).sum())
    violent_count = int(violent.sum())
    quiet_exits = int((transitions == 1).sum())
    violent_exits = int((transitions == -1).sum())

    # Excess kurtosis of a Student-t is 6/(df-4), so df = 4 + 6/excess.
    excess = float(series.kurtosis())
    tail_df = float(np.clip(4.0 + 6.0 / excess, 3.0, 30.0)) if excess > 0.1 else 30.0

    return RegimeParameters(
        quiet_vol=float(quiet_returns.std() or 0.01),
        violent_vol=float(violent_returns.std() or 0.03),
        stay_quiet=float(1.0 - quiet_exits / max(quiet_count, 1)),
        stay_violent=float(1.0 - violent_exits / max(violent_count, 1)),
        drift=float(series.mean()),
        tail_df=tail_df,
    )


def simulate_regime_path(
    parameters: RegimeParameters,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One synthetic return path with regime switching and fat tails."""
    volatility = np.empty(n)
    violent = rng.random() < parameters.violent_share
    for i in range(n):
        volatility[i] = parameters.violent_vol if violent else parameters.quiet_vol
        stay = parameters.stay_violent if violent else parameters.stay_quiet
        if rng.random() > stay:
            violent = not violent

    # Standardised Student-t, so the requested volatility is what comes out.
    df = parameters.tail_df
    shocks = rng.standard_t(df, n) / np.sqrt(df / (df - 2.0)) if df > 2 else rng.standard_normal(n)
    return parameters.drift + volatility * shocks


def synthetic_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    seed: int,
    correlation: Optional[np.ndarray] = None,
) -> dict[str, pd.DataFrame]:
    """One alternative history for the whole universe.

    Instruments are generated jointly through the empirical correlation matrix.
    Generating them independently would produce a universe with no market factor,
    which flatters any strategy that survives by being diversified.
    """
    rng = np.random.default_rng(seed)
    symbols = list(bars_by_symbol)
    if not symbols:
        return {}

    reference = bars_by_symbol[symbols[0]]
    n = len(reference)

    returns_frame = pd.DataFrame({
        symbol: bars['close'].pct_change() for symbol, bars in bars_by_symbol.items()
    }).dropna()
    if correlation is None:
        correlation = returns_frame.corr().to_numpy()

    parameters = {
        symbol: fit_regime_parameters(returns_frame[symbol]) for symbol in symbols
    }

    # Regime paths give each instrument its own volatility clustering; the
    # copula step then couples them so the cross-section is realistic.
    independent = np.column_stack([
        simulate_regime_path(parameters[symbol], n, rng) for symbol in symbols
    ])
    try:
        chol = np.linalg.cholesky(_nearest_positive_definite(correlation))
    except np.linalg.LinAlgError:
        chol = np.eye(len(symbols))
    standardised = (independent - independent.mean(0)) / np.where(
        independent.std(0) > 0, independent.std(0), 1.0
    )
    coupled = standardised @ chol.T
    coupled = coupled * independent.std(0) + independent.mean(0)

    out: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(symbols):
        bars = bars_by_symbol[symbol]
        start = float(bars['close'].iloc[0])
        close = start * np.exp(np.cumsum(coupled[:, i]))
        open_ = np.concatenate([[close[0]], close[:-1]])
        # Reuse the real intrabar range as a fraction, so highs and lows keep a
        # realistic relationship to the close rather than being invented.
        high_ratio = (bars['high'] / bars['close']).to_numpy()
        low_ratio = (bars['low'] / bars['close']).to_numpy()
        out[symbol] = pd.DataFrame(
            {
                'open': open_,
                'high': close * np.maximum(high_ratio, 1.0),
                'low': close * np.minimum(low_ratio, 1.0),
                'close': close,
                'volume': bars['volume'].to_numpy(),
            },
            index=bars.index,
        )
    return out


def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """Clip eigenvalues so a sample correlation matrix can be factorised.

    Sample correlations from short overlapping windows are routinely not positive
    definite, and Cholesky simply fails on them.
    """
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    rebuilt = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(rebuilt))
    return rebuilt / np.outer(scale, scale)


# ---------------------------------------------------------------------------
# Stress and sensitivity
# ---------------------------------------------------------------------------


@dataclass
class StressResult:
    """How the result holds up when the cost assumptions are wrong."""

    scenarios: dict[str, float]
    baseline: float

    @property
    def worst(self) -> float:
        return min(self.scenarios.values()) if self.scenarios else self.baseline

    @property
    def survives(self) -> bool:
        return self.worst > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            'baseline_sharpe': round(self.baseline, 4),
            'scenarios': {k: round(v, 4) for k, v in self.scenarios.items()},
            'worst': round(self.worst, 4),
            'survives': self.survives,
        }


def cost_stress(
    run: Callable[[Config], float],
    config: Config,
    *,
    scenarios: Optional[dict[str, dict[str, float]]] = None,
) -> StressResult:
    """Re-run under worse costs than assumed.

    The slippage model is calibrated from a spread assumption, not from measured
    book depth, so the honest question is not "what is the Sharpe" but "at what
    cost multiple does it stop working". A strategy that dies at 2x fees was
    never a strategy.
    """
    # `spread_bps`, not `slippage_bps`: `core/execution.py` prices fills from the
    # spread and never reads `slippage_bps`, so the old "3x slippage" scenario
    # multiplied a field nothing consumed and re-ran the baseline unchanged.
    scenarios = scenarios or {
        'fees_2x': {'fee_pct_per_side': 2.0, 'min_fee_per_contract': 2.0},
        'spread_3x': {'spread_bps': 3.0, 'slippage_bps': 3.0},
        'both': {
            'fee_pct_per_side': 2.0, 'min_fee_per_contract': 2.0,
            'spread_bps': 3.0, 'slippage_bps': 3.0,
        },
    }

    baseline = float(run(config))
    results: dict[str, float] = {}
    for name, multipliers in scenarios.items():
        stressed = config
        from dataclasses import replace as _replace
        changes = {
            field: getattr(config, field) * multiplier
            for field, multiplier in multipliers.items()
        }
        # The per-symbol schedule has to scale too, or stressing fees does
        # nothing on the contracts whose cost is set by the per-contract floor.
        if 'min_fee_per_contract' in multipliers and config.min_fee_per_contract_by_symbol:
            factor = multipliers['min_fee_per_contract']
            changes['min_fee_per_contract_by_symbol'] = {
                symbol: value * factor
                for symbol, value in config.min_fee_per_contract_by_symbol.items()
            }
        stressed = _replace(config, **changes)
        results[name] = float(run(stressed))

    return StressResult(scenarios=results, baseline=baseline)


@dataclass
class SurfaceResult:
    """Whether the chosen configuration sits on a plateau or a spike."""

    centre: float
    neighbours: dict[str, float]
    retention: float

    @property
    def is_plateau(self) -> bool:
        return self.retention >= 0.6

    def as_dict(self) -> dict[str, Any]:
        return {
            'centre': round(self.centre, 4),
            'neighbours': {k: round(v, 4) for k, v in self.neighbours.items()},
            'retention': round(self.retention, 3),
            'is_plateau': self.is_plateau,
        }


def parameter_surface(
    run: Callable[[dict[str, float]], float],
    parameters: dict[str, float],
    *,
    step: float = 0.2,
    keep_fraction: float = 0.7,
) -> SurfaceResult:
    """Perturb each parameter one step either way and re-run.

    A real edge is insensitive to small parameter changes, because it comes from
    something about the market. An overfit sits on a spike, because it comes from
    something about the sample. `retention` is the share of neighbours that keep
    at least `keep_fraction` of the centre's score.
    """
    centre = float(run(parameters))
    neighbours: dict[str, float] = {}

    for name, value in parameters.items():
        for direction, label in ((1 + step, 'up'), (1 - step, 'down')):
            perturbed = dict(parameters)
            perturbed[name] = value * direction
            neighbours[f'{name}_{label}'] = float(run(perturbed))

    if not neighbours or centre <= 0:
        return SurfaceResult(centre, neighbours, 0.0)

    kept = sum(1 for score in neighbours.values() if score >= keep_fraction * centre)
    return SurfaceResult(centre, neighbours, kept / len(neighbours))


@dataclass
class CapacityResult:
    """Sharpe as a function of capital, and where it stops paying."""

    curve: dict[float, float]

    @property
    def capacity(self) -> Optional[float]:
        """Largest equity whose Sharpe is still positive."""
        positive = [size for size, sharpe in sorted(self.curve.items()) if sharpe > 0]
        return max(positive) if positive else None

    def as_dict(self) -> dict[str, Any]:
        return {
            'curve': {str(int(k)): round(v, 4) for k, v in sorted(self.curve.items())},
            'capacity': self.capacity,
        }


def capacity_curve(
    run: Callable[[float], float],
    equities: Iterable[float],
) -> CapacityResult:
    """Re-run at increasing account sizes.

    Participation-rate slippage means the edge decays with size, and on a venue
    as thin as CDE the decay is the binding constraint rather than the forecast.
    A per-unit edge is not a fundable one.
    """
    return CapacityResult({float(equity): float(run(equity)) for equity in equities})


# ---------------------------------------------------------------------------
# Assembling the verdict
# ---------------------------------------------------------------------------


@dataclass
class SimulationReport:
    """Everything the promotion gates need, in one object."""

    bootstrap: Optional[BootstrapResult] = None
    synthetic: Optional[PathDistribution] = None
    stress: Optional[StressResult] = None
    surface: Optional[SurfaceResult] = None
    capacity: Optional[CapacityResult] = None
    per_period: Optional[PathDistribution] = None
    pbo: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    oos_trades: int = 0
    # Largest share of a bar an exit took. Not a control — the barrier fires
    # where it fires — so it is gated rather than capped: a book whose exits
    # routinely take a fifth of a bar did not trade the market the backtest
    # priced, and its fills are fiction at the size it claims.
    max_exit_participation: Optional[float] = None

    def measurements(self) -> dict[str, Optional[float]]:
        """Flat mapping for `core.metrics.evaluate_gates`.

        Anything not measured comes back None, and a gate with no measurement
        fails rather than passing — "we did not run that test" is not evidence
        of safety.
        """
        return {
            'walk_forward_median_sharpe': self.per_period.median if self.per_period else None,
            'walk_forward_p05_sharpe': self.per_period.p05 if self.per_period else None,
            'pbo': self.pbo,
            'deflated_sharpe': self.deflated_sharpe,
            'bootstrap_positive_fraction': (
                self.bootstrap.probability_positive if self.bootstrap else None
            ),
            'synthetic_positive_fraction': (
                self.synthetic.positive_fraction if self.synthetic else None
            ),
            'stressed_median_sharpe': self.stress.worst if self.stress else None,
            'parameter_plateau': self.surface.retention if self.surface else None,
            'oos_trades': float(self.oos_trades) if self.oos_trades else None,
            'max_exit_participation': self.max_exit_participation,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            'bootstrap': self.bootstrap.as_dict() if self.bootstrap else None,
            'synthetic': self.synthetic.as_dict() if self.synthetic else None,
            'stress': self.stress.as_dict() if self.stress else None,
            'surface': self.surface.as_dict() if self.surface else None,
            'capacity': self.capacity.as_dict() if self.capacity else None,
            'per_period': self.per_period.as_dict() if self.per_period else None,
            'pbo': self.pbo,
            'deflated_sharpe': self.deflated_sharpe,
            'oos_trades': self.oos_trades,
            'max_exit_participation': self.max_exit_participation,
        }
