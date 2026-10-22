"""Compare two values of one config field, paired by fold.

The comparison a capacity or regularisation question needs: same rows, same
folds, same features, one field moved. The delta is paired fold by fold,
because the arms share most of their training windows and differencing two
aggregates throws that away -- measured fold correlation on this data averages
0.87, which is a 2.8x difference in the standard error.

Forecast skill only, no money.

    python -m scripts.config_arm --field max_depth --values 3 4
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.cv import assert_no_leakage, effective_observations, folds_for_config
from core.dataset import apply_fold, fit_fold
from core.features import ALL_GROUPS
from core.metrics import weighted_fold_mean
from core.model import fit_model
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset,
    log_trial, print_header, setup_logging,
)
from scripts.ablate import _graded_label, _null_for

#: Below this, an effect cannot be told from an information-free feature on
#: this data. NC2 measured a noise group at +0.000152 with a standard error of
#: 0.000104, so a delta inside that band is reported as UNRESOLVED however its
#: t reads. It is a yardstick from one draw, not a constant -- a multi-seed
#: calibration would replace it with a distribution. It was measured on
#: the ablation's sample, so it applies to an arm run on THAT sample;
#: another sample has its own floor and needs its own control.
NOISE_FLOOR = 0.000152


def _coerce(text: str):
    """`--values 3 4` are ints; a rate like 0.03 is a float."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--field', required=True,
                        help='The Config field to move, e.g. max_depth')
    parser.add_argument('--values', nargs=2, required=True,
                        help='Exactly two values: the baseline, then the arm')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    print_header('Config arm', config)

    low, high = (_coerce(v) for v in args.values)
    groups = tuple(groups_from_args(args) or ALL_GROUPS)
    dataset = load_dataset(args, config)
    folds = folds_for_config(dataset.window_index, config)
    for fold in folds:
        assert_no_leakage(fold)
    print(f'  {len(folds)} folds, {args.field}: {low} vs {high}, '
          f'groups: {", ".join(groups)}\n')

    arms = {}
    for value in (low, high):
        armed = config.with_overrides(**{args.field: value})
        skills, sizes, trees = [], [], []
        for fold in folds:
            fit, train = fit_fold(dataset, fold.train, armed, groups=groups)
            test = apply_fold(dataset, fit, fold.test, armed, groups=groups)
            model = fit_model(train, fit.baseline, armed, groups=groups)
            y, b = _graded_label(test, armed), _null_for(test, armed)
            m = np.asarray(model.predict(test), dtype=float)
            keep = np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
            skills.append(log_loss(y[keep], b[keep]) - log_loss(y[keep], m[keep]))
            sizes.append(effective_observations(test))
            trees.append(model.best_iteration or 0)
        arms[value] = (np.asarray(skills, float), np.asarray(sizes, float),
                       float(np.mean(trees)))
        agg = weighted_fold_mean(arms[value][0], arms[value][1])
        print(f'  {args.field}={value!r:>8}  skill {agg.mean:+.6f}  '
              f'se {agg.standard_error:.6f}  t {agg.t:+.2f}  '
              f'{int((arms[value][0] > 0).sum())}/{len(folds)}  '
              f'{arms[value][2]:.0f} trees')

    delta = arms[high][0] - arms[low][0]
    paired = weighted_fold_mean(delta, arms[low][1])
    print(f'\n  paired delta ({args.field} {high} minus {low}): '
          f'{paired.mean:+.6f}  se {paired.standard_error:.6f}  '
          f'  t {paired.t:+.2f}  {int((delta > 0).sum())}/{len(folds)} folds')
    if abs(paired.mean) < NOISE_FLOOR:
        print(f'\n  UNRESOLVED: |delta| is inside the {NOISE_FLOOR:+.6f} noise '
              f'band, where an\n  information-free feature scored just as well. '
              f'The arms cannot be ranked\n  on this evidence, whatever the t '
              f'reads.')
    rows = [{'field': args.field, 'low': low, 'high': high,
             'delta': paired.mean, 'se': paired.standard_error, 't': paired.t,
             'p': paired.p_value, 'folds_positive': int((delta > 0).sum()),
             'folds': len(folds),
             'resolved': bool(abs(paired.mean) >= NOISE_FLOOR)}]
    if args.out:
        from pathlib import Path
        path = Path(args.out)
        path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path / 'config_arm.csv', index=False)
        pd.DataFrame({f'{args.field}={v}': arms[v][0]
                      for v in (low, high)}).to_csv(
            path / 'config_arm_folds.csv', index=False)
        print(f'\n  wrote {path / "config_arm.csv"}')
    log_trial('config_arm', args, config, groups=groups, dataset=dataset,
              metrics={'folds': len(folds), 'arms': rows})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
