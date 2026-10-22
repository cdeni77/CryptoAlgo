"""Measure each feature group alone, and the set with the control removed.

Gain share (`control_gain_share`) counts the splits spent on a group, not
whether it forecasts. An ablation tests whether a group forecasts.

Forecast skill only, no money: money numbers from a run this size must not
choose between configurations.

    python -m scripts.ablate
    python -m scripts.ablate --offsets 3
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.cv import assert_no_leakage, effective_observations, folds_for_config
from core.dataset import apply_fold, fit_fold
from core.features import ALL_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS
from core.metrics import weighted_fold_mean
from core.model import fit_model
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset,
    log_trial, print_header, setup_logging,
)

# Derived from `core.features`, never spelled out, so a survey cannot silently
# run without its control.
CONTROL = CONTROL_GROUPS[0]


def trials(groups: tuple[str, ...],
           drop: Sequence[str] = ()) -> dict[str, tuple[str, ...]]:
    """The cells to score.

    Default: the full set, the full set without the control, and every group
    alone. `drop` instead scores the full set against the full set MINUS each
    named group -- a leave-one-out, which is the comparison a keep-or-remove
    decision needs and which costs two cells instead of one per group.

    Alone and leave-one-out answer different questions. Alone asks whether a
    group forecasts at all; leave-one-out asks whether it adds anything the
    others do not already carry. A group can score near zero alone and still
    matter, or forecast well alone and be redundant.
    """
    out: dict[str, tuple[str, ...]] = {'all groups': groups}
    if drop:
        for group in drop:
            if group not in groups:
                raise SystemExit(f'--drop {group}: not in the selected groups '
                                 f'{list(groups)}')
            rest = tuple(g for g in groups if g != group)
            if rest:
                out[f'all minus {group}'] = rest
        return out
    without = tuple(g for g in groups if g != CONTROL)
    if CONTROL in groups and without:
        out[f'all minus {CONTROL}'] = without
    for group in groups:
        label = f'{group} alone' + ('  (the CONTROL)' if group == CONTROL else '')
        out[label] = (group,)
    return out


def _graded_label(test, config):
    """The outcome to score against, honouring `label_source`.

    `test['outcome']` is the Coinbase proxy. Grading on it while the market
    prices on BRTI hands the baseline free skill -- the ledger measured
    `baseline - market` reversing sign between the two labels.
    """
    proxy = test['outcome'].to_numpy(dtype=float)
    if getattr(config, 'label_source', 'proxy') != 'venue':
        return proxy
    if 'venue_outcome' not in test.columns:
        return proxy
    venue = pd.to_numeric(test['venue_outcome'], errors='coerce').to_numpy(dtype=float)
    return np.where(np.isfinite(venue), venue, proxy)


def _null_for(test, config):
    """What the model is scored AGAINST, which `init_score_source` decides.

    **A market-initialised model must be scored against the MARKET.** It
    starts from the market logit and the trees fit a residual, so measuring
    it against `F(x/sigma)` reports

        [log_loss(baseline) - log_loss(market)] + [log_loss(market) - log_loss(model)]

    where the first term is the market's advantage over the barrier null --
    identical in every ablation cell -- and only the second is what the
    features add. The constant swamps the signal and its fold variance sets
    the standard error, so every group reads as contributing nothing.

    Measured 2026-10-22: with the baseline as the null, all eight groups came
    back between -0.00014 and +0.00031 against a standard error of 0.00078,
    and "all groups" scored +0.002237 -- about the size of the market's own
    advantage over the baseline in the campaign ledger. The ablation was
    reporting the init score, eight times.

    `core/backtest.py` already documents the same swap for `log_loss_skill`:
    "a silent swap is how a market-init model comes to be judged as a
    baseline-init one".
    """
    if getattr(config, 'init_score_source', None) == 'market' \
            and 'market_probability' in test.columns:
        return pd.to_numeric(test['market_probability'],
                             errors='coerce').to_numpy(dtype=float)
    return test['baseline_probability'].to_numpy(dtype=float)


# Set in the parent before the pool forks. The workers inherit the built
# dataset through copy-on-write rather than rebuilding or pickling it -- the
# build is 16 minutes and the frame is gigabytes, so neither is affordable.
_SHARED: dict = {}


def _score_one(task):
    """One (cell, fold) pair: fit, predict, and return the fold's skill.

    A worker, so it must stay a module-level function and read its inputs from
    `_SHARED`. `fit_model` is seeded and `deterministic` is set, so the result
    does not depend on which process computes it or in what order.
    """
    label, groups, index = task
    dataset, config, folds = _SHARED['dataset'], _SHARED['config'], _SHARED['folds']
    fold = folds[index]
    fit, train = fit_fold(dataset, fold.train, config, groups=groups)
    test = apply_fold(dataset, fit, fold.test, config, groups=groups)
    model = fit_model(train, fit.baseline, config, groups=groups)
    y = _graded_label(test, config)
    b = _null_for(test, config)
    m = np.asarray(model.predict(test), dtype=float)
    keep = np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
    return (label, index,
            log_loss(y[keep], b[keep]) - log_loss(y[keep], m[keep]),
            model.best_iteration or 0, effective_observations(test))


def _score_cells(cells: dict, folds, dataset, config, jobs: int) -> dict:
    """Every (cell, fold) pair, over a process pool when `jobs` > 1.

    The cells are independent and each fold is independent inside a cell, so
    the whole grid is one flat task list. LightGBM is given no thread count and
    the loop is pandas-bound, so a single run uses one core out of the box --
    the parallelism has to come from processes, not from the library.
    """
    tasks = [(label, groups, i) for label, groups in cells.items()
             for i in range(len(folds))]
    out = {label: ([None] * len(folds), [None] * len(folds), [None] * len(folds))
           for label in cells}
    # Before the branch: the serial path calls the same worker, so it needs
    # the same inputs.
    _SHARED.update(dataset=dataset, config=config, folds=folds)
    if jobs <= 1:
        results = (_score_one(t) for t in tasks)
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        # fork, so the dataset is shared copy-on-write. spawn would rebuild it
        # in every worker.
        ctx = multiprocessing.get_context('fork')
        pool = ProcessPoolExecutor(max_workers=jobs, mp_context=ctx)
        with pool:
            results = list(pool.map(_score_one, tasks, chunksize=1))
    for label, index, skill, trees, size in results:
        out[label][0][index] = skill
        out[label][1][index] = trees
        out[label][2][index] = size
    return out


def _paired_deltas(per_fold: dict) -> pd.DataFrame:
    """`skill(all groups) - skill(all minus G)`, differenced fold by fold.

    The cells score the same folds on the same rows, so the difference is
    paired and its standard error comes from the dispersion of the per-fold
    differences. Folds are weighted by window count, as the gates weight them.
    """
    base = per_fold.get('all groups')
    if base is None:
        return pd.DataFrame()
    out = []
    for label, (v, sizes) in per_fold.items():
        if not label.startswith('all minus '):
            continue
        d = base[0] - v
        agg = weighted_fold_mean(d, sizes)
        out.append({'group': label[len('all minus '):], 'delta': agg.mean,
                    'se': agg.standard_error, 't': agg.t, 'p': agg.p_value,
                    'folds_positive': int((d > 0).sum()), 'folds': len(d)})
    return pd.DataFrame(out).sort_values('t', ascending=False)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--drop', nargs='*', default=(),
                        help='Leave-one-out: score the full set against the '
                             'full set minus each named group. Two cells '
                             'instead of one per group.')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Score (cell, fold) pairs over this many '
                             'processes. The dataset is built once and shared '
                             'by fork. Default 1, the serial path.')
    parser.add_argument('--out', type=str, default=None,
                        help='Write the table to ablation.csv in this directory')
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    print_header('Ablation', config)

    dataset = load_dataset(args, config)
    folds = folds_for_config(dataset.window_index, config)
    for fold in folds:
        assert_no_leakage(fold)

    # **The groups the CONFIG selected, not every group that exists.** This
    # read `g in set(FEATURE_GROUPS)`, which is always true, so `--groups` was
    # ignored and the survey scored families the model does not use. It also
    # selected both book framings at once, which `feature_columns` now refuses
    # outright -- the bug had been masked by dedup.
    # **From the ARGS, not the config.** `Config` carries no group selection;
    # it lives on the parsed arguments, the way every other script reads it.
    # Reading it off the config instead raised at run time while the
    # cell-building tests passed, because those exercise `trials` and never
    # this function.
    available = tuple(groups_from_args(args) or ALL_GROUPS)
    print(f'  {len(folds)} folds, groups: {", ".join(available)}\n')
    print(f"  {'groups':30s} {'skill':>11s} {'se':>10s} {'t':>7s} {'folds+':>8s} {'trees':>7s}")

    rows = []
    # Per-fold skills per cell. A leave-one-out delta is a PAIRED quantity:
    # every cell scores the same folds, so the standard error must come from
    # the dispersion of the per-fold differences, not from differencing two
    # aggregates. Differencing aggregates ignores the pairing and inflates the
    # error, which hides a real group behind the Bonferroni threshold.
    per_fold: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cells = trials(available, args.drop)
    if args.jobs > 1:
        print(f'  scoring {len(cells) * len(folds)} (cell, fold) pairs over '
              f'{args.jobs} processes\n')
    scored = _score_cells(cells, folds, dataset, config, args.jobs)
    for label in cells:
        skills, trees, sizes = scored[label]
        v = np.asarray(skills, dtype=float)
        per_fold[label] = (v, np.asarray(sizes, dtype=float))
        # The window-weighted aggregation the gates read, so an ablation cell
        # and a promotion report agree.
        weighted = weighted_fold_mean(v, sizes)
        plain = weighted_fold_mean(v)
        print(f'  {label:30s} {weighted.mean:+11.6f} {weighted.standard_error:10.6f} '
              f'{weighted.t:+7.2f} {int((v > 0).sum()):>6d}/{len(v)} '
              f'{np.mean(trees):>7.0f}')
        rows.append({'groups': label, 'skill': weighted.mean,
                     'se': weighted.standard_error, 't': weighted.t,
                     'p': weighted.p_value, 'skill_unweighted': plain.mean,
                     'se_unweighted': plain.standard_error,
                     'folds_positive': int((v > 0).sum()), 'folds': len(v)})

    frame = pd.DataFrame(rows)
    deltas = _paired_deltas(per_fold) if args.drop else pd.DataFrame()
    if not deltas.empty:
        print(f"\n  leave-one-out delta = skill(all) - skill(all minus G), "
              f"paired by fold\n")
        print(f"  {'group':24s} {'delta':>11s} {'se':>10s} {'t':>7s} {'folds+':>8s}")
        for row in deltas.to_dict('records'):
            print(f"  {row['group']:24s} {row['delta']:+11.6f} "
                  f"{row['se']:10.6f} {row['t']:+7.2f} "
                  f"{row['folds_positive']:>6d}/{row['folds']}")
    control = frame.loc[frame['groups'].str.startswith(CONTROL)]
    print()
    if not control.empty:
        value = float(control['skill'].iloc[0])
        positive = int(control['folds_positive'].iloc[0])
        if value > 0 and positive >= len(folds) - 1:
            print(f'  WARNING: the control scores {value:+.6f} on {positive} of '
                  f'{len(folds)} folds. Time of day cannot forecast direction, so '
                  f'this is a measurement error and not a finding — check the fold '
                  f'boundaries and the feature timestamps before reading anything '
                  f'else here.')
        else:
            print(f'  the control scores {value:+.6f} on {positive} of {len(folds)} '
                  f'folds, which is what a control should do')
    print('\n  Read every t against the fold correlation, not against a normal '
          'table: consecutive\n  expanding folds share 50-83% of their training '
          'windows, so "6 of 6 positive" is\n  roughly a 22% event under the null '
          'at rho 0.7, not 1.6%.')
    if args.out:
        from pathlib import Path
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path / 'ablation.csv', index=False)
        print(f'\n  wrote {path / "ablation.csv"}')
        if not deltas.empty:
            deltas.to_csv(path / 'ablation_delta.csv', index=False)
            print(f'  wrote {path / "ablation_delta.csv"}')
        folds_frame = pd.DataFrame({k: v[0] for k, v in per_fold.items()})
        folds_frame.to_csv(path / 'ablation_folds.csv', index=False)
        print(f'  wrote {path / "ablation_folds.csv"}')
    # One trial line for the run, listing every cell, because each cell counts
    # as a trial.
    log_trial('ablate', args, config, groups=available, dataset=dataset, metrics={
        'folds': len(folds),
        'cells': frame.to_dict('records'),
        'deltas': deltas.to_dict('records'),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
