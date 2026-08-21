"""Campaign search over a declarative space, with an append-only ledger.

This replaces five scripts — `comprehensive_search`, `gap_search`,
`weekly_search`, `reverify_profiles` and `coin_backtests`, about 1,900 lines —
which all did the identical thing: import a config, call a backtest, and apply a
pass/fail rule. They differed only in which parameter combinations they
enumerated and what gate they applied, and both of those are data.

Their docstrings were the giveaway. `comprehensive_search` opened with "NEW vs
overnight_exit_search.py", a file that no longer existed. `gap_search` recorded
"what was already tested in comprehensive_search (NOT repeated here)" as a prose
comment — search state stored in a docstring. `weekly_search` existed because the
first two had a screen/verify gap, and `reverify_profiles` because `gap_search`
only covered a subset. Each was a patch on the last one's flaw, and all five were
still in the tree.

The ledger is the part that matters beyond tidiness. A deflated Sharpe ratio
needs the true number of configurations evaluated, and when that count is spread
across five scripts it is unknowable, so it never reached the calculation. Here
every trial is appended to one Parquet ledger with its parameters, seed, data
version and feature-set hash, and `trial_count` is a query.
"""

from __future__ import annotations

import itertools
import math
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config
from core.metrics import (
    HOURS_PER_YEAR,
    Gate,
    deflated_sharpe,
    evaluate_gates,
    probability_of_backtest_overfitting,
    summarise_paths,
)

logger = logging.getLogger(__name__)

DEFAULT_LEDGER = Path(os.getenv('SEARCH_LEDGER', 'data/search/ledger.parquet'))


# ---------------------------------------------------------------------------
# The space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchSpace:
    """Parameters to vary, as data rather than as nested loops.

    `grid` maps a Config field to the values to try. `fixed` pins fields for the
    whole campaign. Writing it this way means a campaign is a file, so what was
    searched is recoverable afterwards — which is what the old docstring-comments
    were failing to do.
    """

    name: str
    grid: dict[str, Sequence[Any]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = (7,)

    def __post_init__(self) -> None:
        known = set(Config.__dataclass_fields__)
        unknown = (set(self.grid) | set(self.fixed)) - known
        if unknown:
            raise ValueError(f'unknown Config fields in search space: {sorted(unknown)}')

    @property
    def size(self) -> int:
        combinations = 1
        for values in self.grid.values():
            combinations *= max(len(values), 1)
        return combinations * max(len(self.seeds), 1)

    def combinations(self) -> Iterator[tuple[dict[str, Any], int]]:
        """Every (parameters, seed) pair, in a deterministic order."""
        keys = sorted(self.grid)
        value_lists = [list(self.grid[k]) for k in keys]
        for values in itertools.product(*value_lists) if keys else [()]:
            parameters = dict(zip(keys, values))
            parameters.update(self.fixed)
            for seed in self.seeds:
                yield parameters, seed

    def configure(self, base: Config, parameters: dict[str, Any]) -> Config:
        """Apply one combination to a base Config.

        The varied fields are recorded in `cli_overrides` so they outrank per-coin
        profile values. A search that silently lost to a profile default would be
        searching nothing.
        """
        return replace(
            base, **parameters,
            cli_overrides=frozenset(set(base.cli_overrides) | set(parameters)),
        )


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One evaluated configuration, and everything needed to reproduce it."""

    campaign: str
    trial_id: str
    parameters: dict[str, Any]
    seed: int
    metrics: dict[str, float]
    fold_scores: list[float] = field(default_factory=list)
    passed: bool = False
    failed_gates: list[str] = field(default_factory=list)
    feature_set_hash: str = ''
    cost_config_version: str = ''
    data_as_of: Optional[str] = None
    evaluated_at: str = ''
    error: Optional[str] = None

    def to_row(self) -> dict[str, Any]:
        return {
            'campaign': self.campaign,
            'trial_id': self.trial_id,
            'seed': self.seed,
            'parameters': json.dumps(self.parameters, sort_keys=True, default=str),
            'metrics': json.dumps(self.metrics, sort_keys=True, default=str),
            'fold_scores': json.dumps(self.fold_scores),
            'passed': self.passed,
            'failed_gates': json.dumps(self.failed_gates),
            'feature_set_hash': self.feature_set_hash,
            'cost_config_version': self.cost_config_version,
            'data_as_of': self.data_as_of,
            'evaluated_at': self.evaluated_at,
            'error': self.error,
            **{f'param_{k}': v for k, v in self.parameters.items()
               if isinstance(v, (int, float, str, bool))},
            **{f'metric_{k}': v for k, v in self.metrics.items()
               if isinstance(v, (int, float))},
        }


class SearchLedger:
    """Append-only record of every configuration ever evaluated.

    Append-only is the point. A ledger that can be rewritten cannot support a
    deflated Sharpe ratio, because the trial count becomes whatever the last
    writer decided it was. The number that matters is `trial_count`, and it has
    to include the failures.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_LEDGER)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trials: Iterable[Trial]) -> int:
        rows = [trial.to_row() for trial in trials]
        if not rows:
            return 0

        frame = pd.DataFrame(rows)
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            frame = pd.concat([existing, frame], ignore_index=True)

        temporary = self.path.with_suffix('.parquet.tmp')
        frame.to_parquet(temporary, index=False, compression='zstd')
        os.replace(temporary, self.path)
        return len(rows)

    def read(self, campaign: Optional[str] = None) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(self.path)
        if campaign is not None:
            frame = frame[frame['campaign'] == campaign]
        return frame

    def trial_count(self, campaign: Optional[str] = None) -> int:
        """Configurations evaluated, including the ones that failed.

        This is the input a deflated Sharpe ratio needs. Counting only survivors
        understates it, sometimes by orders of magnitude, and understating it is
        exactly how a lucky backtest passes a significance test.
        """
        return len(self.read(campaign))

    def score_matrix(self, campaign: Optional[str] = None) -> np.ndarray:
        """(candidates x folds) out-of-sample scores, for the PBO calculation."""
        frame = self.read(campaign)
        if frame.empty or 'fold_scores' not in frame:
            return np.zeros((0, 0))

        rows = [json.loads(value) for value in frame['fold_scores']]
        rows = [row for row in rows if isinstance(row, list) and len(row) >= 2]
        if len(rows) < 2:
            return np.zeros((0, 0))

        width = min(len(row) for row in rows)
        return np.array([row[:width] for row in rows], dtype=float)

    def best(self, metric: str = 'sharpe', campaign: Optional[str] = None,
             passed_only: bool = True) -> Optional[pd.Series]:
        frame = self.read(campaign)
        column = f'metric_{metric}'
        if frame.empty or column not in frame:
            return None
        if passed_only and 'passed' in frame:
            frame = frame[frame['passed']]
        if frame.empty:
            return None
        return frame.loc[frame[column].idxmax()]


# ---------------------------------------------------------------------------
# Running a campaign
# ---------------------------------------------------------------------------


@dataclass
class CampaignResult:
    """What a campaign found, and how hard it had to look to find it."""

    campaign: str
    trials: list[Trial]
    trial_count: int
    pbo: Optional[float] = None
    deflated: Optional[float] = None

    @property
    def survivors(self) -> list[Trial]:
        return [t for t in self.trials if t.passed]

    @property
    def errors(self) -> list[Trial]:
        return [t for t in self.trials if t.error]

    def summary(self) -> dict[str, Any]:
        return {
            'campaign': self.campaign,
            'evaluated': len(self.trials),
            'ledger_trial_count': self.trial_count,
            'survivors': len(self.survivors),
            'errors': len(self.errors),
            'pbo': self.pbo,
            'deflated_sharpe': self.deflated,
        }

    def __str__(self) -> str:
        pbo = f'{self.pbo:.2f}' if self.pbo is not None else 'n/a'
        dsr = f'{self.deflated:+.2f}' if self.deflated is not None else 'n/a'
        return (
            f'{self.campaign}: {len(self.survivors)}/{len(self.trials)} survived '
            f'| PBO {pbo} | DSR {dsr} '
            f'| {self.trial_count} trials on the ledger'
        )


# A trial evaluator returns (metrics, fold_scores). Kept as a callable so the
# same runner drives a real backtest, a synthetic panel, or a stub in tests.
Evaluator = Callable[[Config, int], tuple[dict[str, float], list[float]]]


def run_campaign(
    space: SearchSpace,
    evaluator: Evaluator,
    *,
    base_config: Optional[Config] = None,
    ledger: Optional[SearchLedger] = None,
    thresholds: Optional[dict[str, tuple[float, str]]] = None,
    feature_set_hash: str = '',
    data_as_of: Optional[str] = None,
    observations: int = 0,
    horizon_bars: int = 0,
) -> CampaignResult:
    """Evaluate every combination, gate it, and record all of it.

    Failures are recorded too. A ledger of only the winners cannot support a
    deflated Sharpe ratio, and the deflation is the whole reason the ledger
    exists.
    """
    base = base_config or Config()
    ledger = ledger or SearchLedger()
    trials: list[Trial] = []

    logger.info('campaign %s: %d combinations', space.name, space.size)

    for parameters, seed in space.combinations():
        config = space.configure(base, parameters)
        trial = Trial(
            campaign=space.name,
            trial_id=uuid.uuid4().hex[:12],
            parameters=parameters,
            seed=seed,
            metrics={},
            feature_set_hash=feature_set_hash,
            cost_config_version=config.cost_config_version,
            data_as_of=data_as_of,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            metrics, folds = evaluator(config, seed)
            trial.metrics = {k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float)) and np.isfinite(float(v))}
            trial.fold_scores = [float(f) for f in folds if np.isfinite(float(f))]
            promoted, gates = evaluate_gates(trial.metrics, thresholds=thresholds)
            trial.passed = promoted
            trial.failed_gates = [g.name for g in gates if not g.passed]
        except Exception as exc:                      # a failed trial is data
            logger.warning('trial failed (%s): %s', parameters, exc)
            trial.error = str(exc)[:400]
        trials.append(trial)

    ledger.append(trials)
    total = ledger.trial_count(space.name)

    # PBO over the campaign's own fold scores, and a deflated Sharpe against the
    # ledger's full trial count rather than this campaign's.
    matrix = ledger.score_matrix(space.name)
    pbo_result = probability_of_backtest_overfitting(matrix) if matrix.size else None

    best = max(
        (t.metrics.get('walk_forward_median_sharpe', float('-inf')) for t in trials),
        default=float('-inf'),
    )
    dsr = None
    if np.isfinite(best) and observations > 1 and total > 0 and horizon_bars:
        # `best` is annualised by `HOURS_PER_YEAR`, and `deflated_sharpe` needs a
        # ratio at the same frequency as `observations`, because both terms of the
        # correction scale as 1/sqrt(n).
        #
        # De-annualising divides by `sqrt(HOURS_PER_YEAR / h)` where **h is the
        # holding period in hours** — the span one observation covers. An earlier
        # version of this used `observations` (a count, ~18-232) in h's place,
        # which is a different quantity entirely: at the 8h horizon CLAUDE.md
        # recommends it read 5.4x too confident and flipped the verdict from
        # BLOCK to pass on every case tested.
        per_observation = best / math.sqrt(HOURS_PER_YEAR / max(int(horizon_bars), 1))
        dsr = deflated_sharpe(
            sharpe=per_observation,
            observations=observations,
            trials=total,
        ).statistic

    return CampaignResult(
        campaign=space.name,
        trials=trials,
        trial_count=total,
        pbo=pbo_result.pbo if pbo_result and pbo_result.valid else None,
        deflated=dsr,
    )


# ---------------------------------------------------------------------------
# Campaign definitions
# ---------------------------------------------------------------------------


def default_campaigns() -> dict[str, SearchSpace]:
    """The spaces worth searching, given what the formulation now exposes.

    Deliberately small. At roughly forty independent observations per fold, a
    thousand-trial search does not find a better strategy — it finds a luckier
    one, and the deflated Sharpe then rejects it. These vary the decision
    thresholds and the risk appetite, not the feature set, because the feature
    set is a modelling question rather than a search one.
    """
    return {
        'thresholds': SearchSpace(
            name='thresholds',
            grid={
                'min_edge_over_cost': (0.25, 0.5, 1.0),
                'max_positions': (3, 5),
            },
            seeds=(7, 17),
        ),
        'risk': SearchSpace(
            name='risk',
            grid={
                'leverage': (2, 3, 4),
                'min_vol_24h': (0.004, 0.008),
                'max_vol_24h': (0.06, 0.10),
            },
            seeds=(7,),
        ),
        'horizon': SearchSpace(
            name='horizon',
            grid={
                'max_hold_hours': (24, 48, 96),
                'min_edge_over_cost': (0.5, 1.0),
            },
            seeds=(7,),
        ),
    }
