"""Fit and report the barrier baseline alone. The Phase 1 gate.

Run this before anything else, and read it before believing any model result.
It answers one question: **how much of a 15-minute up/down market is already
settled by arithmetic?**

The answer is most of it. Displacement is known exactly; only the remaining
volatility has to be forecast; and volatility is forecastable. Expect the
baseline to take log loss from 0.693 — a coin flip — to somewhere near 0.51,
which is a 26% improvement using no features at all.

That number is the trap this whole system is built around. A model measured
against 50% would report a 40-point edge on late-window rows. There is no edge
there; there is a clock. Everything downstream is measured as a *difference*
against what this script prints.

**The gate: the baseline must be calibrated out of sample.** If its reliability
table is off — predicting 0.90 where 0.84 happens — then the "skill" a model
shows against it is partly the baseline's own miscalibration, and the residual
architecture that makes incremental skill readable stops meaning what it says.
Fix the baseline before training anything.

    python -m scripts.baseline
    python -m scripts.baseline --baseline-distribution normal   # the comparison
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from core.baseline import log_loss, reliability
from core.cv import assert_no_leakage, effective_observations, purged_walk_forward
from core.dataset import apply_fold, fit_fold
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--compare-distributions', action='store_true',
                        help='Fit both a Gaussian and a Student-t barrier and report '
                             'both. Which tail calibrates better is a measurement.')
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    print_header('Barrier baseline — the null hypothesis', config)

    dataset = load_dataset(args, config)
    folds = purged_walk_forward(dataset.window_index, n_folds=config.n_folds,
                                embargo_minutes=config.embargo_minutes)
    groups = groups_from_args(args)

    distributions = (['normal', 'student_t'] if args.compare_distributions
                     else [config.baseline_distribution])
    results: dict[str, list[dict]] = {}

    for distribution in distributions:
        variant = config.with_overrides(baseline_distribution=distribution)
        rows = []
        pooled_y, pooled_p = [], []
        for fold in folds:
            assert_no_leakage(fold)
            fit, _ = fit_fold(dataset, fold.train, variant, groups=groups)
            test = apply_fold(dataset, fit, fold.test, variant, groups=groups)
            y = test['outcome'].to_numpy(dtype=float)
            p = test['baseline_probability'].to_numpy(dtype=float)
            pooled_y.append(y)
            pooled_p.append(p)
            rel = reliability(y, p)
            rows.append({
                'fold': fold.index,
                'test_start': fold.test_start, 'test_end': fold.test_end,
                'windows': effective_observations(test),
                'log_loss': log_loss(y, p),
                'coin_flip': log_loss(y, np.full_like(p, float(y.mean()))),
                'ece': rel.expected_calibration_error,
                'max_dev': rel.max_deviation,
                'n_non_finite': rel.n_non_finite,
                'nu': fit.baseline.nu,
                'scale': ' '.join(f'{o}m={s:.3f}' for o, s in sorted(fit.baseline.scale.items())),
            })
            print(f'  fold {fold.index} [{fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d}] '
                  f'{rows[-1]["windows"]:>7,}w  log loss {rows[-1]["log_loss"]:.5f}  '
                  f'ECE {rows[-1]["ece"]:.5f}  max dev {rows[-1]["max_dev"]:.4f}  '
                  f'nu {fit.baseline.nu:.1f}')
        results[distribution] = rows
        y = np.concatenate(pooled_y)
        p = np.concatenate(pooled_p)
        rel = reliability(y, p)
        frame = pd.DataFrame(rows)
        print()
        print(f'  {distribution}: pooled log loss {log_loss(y, p):.5f} against a '
              f'{log_loss(y, np.full_like(p, float(y.mean()))):.5f} coin flip '
              f'({1 - log_loss(y, p) / log_loss(y, np.full_like(p, float(y.mean()))):.1%} '
              f'better, from arithmetic alone)')
        print(f'  worst-fold ECE {frame["ece"].max():.5f}, worst max deviation '
              f'{frame["max_dev"].max():.4f}, base rate {y.mean():.4f}')
        print('\n  out-of-sample reliability:')
        print(rel.table())
        print()

    if len(distributions) > 1:
        print('  distribution comparison (pooled out of sample):')
        for distribution, rows in results.items():
            frame = pd.DataFrame(rows)
            print(f'    {distribution:<10} log loss {frame["log_loss"].mean():.5f}  '
                  f'ECE {frame["ece"].max():.5f}')
        print('\n  Fat tails matter here because the system trades its confident\n'
              '  predictions: a Gaussian barrier assigns 0.999 where 0.99 is right,\n'
              '  and a model that merely knew that would look skilful.')

    print()
    # `pandas.Series.max()` skips NaN, and `nan > 0.02` is False, so this gate
    # used to fail OPEN: with 31 non-finite rows in 99,388 it printed
    # "gate passed: worst-fold calibration error 0.01516 <= 0.02" while five of
    # six folds had reported `log loss nan, ECE nan`. Refuse on unmeasured folds
    # before comparing anything, so a data hole reads as a data hole.
    all_ece = np.concatenate([
        pd.DataFrame(rows)['ece'].to_numpy(dtype=float) for rows in results.values()])
    all_non_finite = int(sum(
        int(pd.DataFrame(rows)['n_non_finite'].sum()) for rows in results.values()
        if 'n_non_finite' in pd.DataFrame(rows).columns))
    unmeasured = int((~np.isfinite(all_ece)).sum())
    if unmeasured:
        print(f'  GATE FAILED: {unmeasured} of {all_ece.size} folds could not be '
              f'measured at all (calibration came back NaN).')
        print('  That is a hole in the data, not a verdict on the baseline, and it '
              'is not the same answer as "no skill".')
        print('  Find the missing bars before reading anything below.')
        return 1
    if all_non_finite:
        print(f'  GATE FAILED: {all_non_finite} row(s) carried a non-finite '
              f'probability or outcome.')
        print('  Those rows used to be counted in the 0.95-1.00 reliability bin, '
              'which is the band this system trades.')
        return 1

    worst_ece = float(np.max(all_ece))
    if worst_ece > 0.02:
        print(f'  GATE FAILED: worst-fold calibration error {worst_ece:.5f} exceeds 0.02.')
        print('  Skill measured against a miscalibrated baseline is partly the '
              'baseline\'s error.')
        print('  Fix this before training anything.')
        return 1
    print(f'  gate passed: worst-fold calibration error {worst_ece:.5f} <= 0.02 '
          f'across {all_ece.size} folds, none unmeasured')
    print('  next: python -m scripts.evaluate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
