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

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.cv import assert_no_leakage, effective_observations, folds_for_config
from core.dataset import apply_fold, fit_fold
from core.features import CONTROL_GROUPS, FEATURE_GROUPS
from core.metrics import weighted_fold_mean
from core.model import fit_model
from scripts._common import (
    add_data_arguments, config_from_args, load_dataset, log_trial, print_header,
    setup_logging,
)

# Derived from `core.features`, never spelled out, so a survey cannot silently
# run without its control.
CONTROL = CONTROL_GROUPS[0]


def trials(groups: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Return every group alone, the full set, and the full set without the control."""
    out: dict[str, tuple[str, ...]] = {'all groups': groups}
    without = tuple(g for g in groups if g != CONTROL)
    if CONTROL in groups and without:
        out[f'all minus {CONTROL}'] = without
    for group in groups:
        label = f'{group} alone' + ('  (the CONTROL)' if group == CONTROL else '')
        out[label] = (group,)
    return out


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
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

    available = tuple(g for g in FEATURE_GROUPS if g in set(FEATURE_GROUPS))
    print(f'  {len(folds)} folds, groups: {", ".join(available)}\n')
    print(f"  {'groups':30s} {'skill':>11s} {'se':>10s} {'t':>7s} {'folds+':>8s} {'trees':>7s}")

    rows = []
    for label, groups in trials(available).items():
        skills, trees, sizes = [], [], []
        for fold in folds:
            fit, train = fit_fold(dataset, fold.train, config, groups=groups)
            test = apply_fold(dataset, fit, fold.test, config, groups=groups)
            model = fit_model(train, fit.baseline, config, groups=groups)
            y = test['outcome'].to_numpy(dtype=float)
            b = test['baseline_probability'].to_numpy(dtype=float)
            m = np.asarray(model.predict(test), dtype=float)
            # Rows with no volatility estimate carry no forecast; exclude them, as
            # core/metrics.py does.
            keep = np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
            skills.append(log_loss(y[keep], b[keep]) - log_loss(y[keep], m[keep]))
            trees.append(model.best_iteration or 0)
            sizes.append(effective_observations(test))
        v = np.asarray(skills, dtype=float)
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
    # One trial line for the run, listing every cell, because each cell counts
    # as a trial.
    log_trial('ablate', args, config, groups=available, dataset=dataset, metrics={
        'folds': len(folds),
        'cells': frame.to_dict('records'),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
