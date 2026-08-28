"""Measure each feature group alone, and the set with the control removed.

**This exists because `control_gain_share` measures the wrong thing.** That gate
reads LightGBM's gain share for the `clock` group and refuses above 0.30, on the
reasoning that hour-of-day cannot forecast direction so a model leaning on it is
broken. The reasoning is right and the measurement does not test it: gain share
says how many splits the booster spent on a feature, not whether the feature
forecasts anything.

Measured on 326 days of real bars, the two disagree completely:

    control_gain_share      0.279   (against a 0.30 gate — nearly "carrying" it)
    clock group ALONE      -0.000008   t=-0.26   2/6 folds positive
    all groups minus clock +0.000315   t=+2.12   6/6   (slightly BETTER than all five)

So the clock was absorbing splits and forecasting nothing, and the gate could
neither confirm that nor rule it out. An ablation can. It is also the measurement
that found what the skill actually is: `cross_asset` alone at +0.000183, t=+3.39,
6/6 folds, against `vol_state` alone at **-0.000101** — which is the opposite of
the sigma-disagreement mechanism this project was designed around.

Forecast only, no book. `AUDIT_REPORT.md` records why the money numbers from a run
this size must not be used to choose between configurations: across six
configurations, skill and return were decoupled, with the *higher*-skill setting
losing half the account.

    python -m scripts.ablate
    python -m scripts.ablate --offsets 3          # where the skill turned out to live
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.cv import assert_no_leakage, purged_walk_forward
from core.dataset import apply_fold, fit_fold
from core.features import CONTROL_GROUPS, FEATURE_GROUPS
from core.model import fit_model
from scripts._common import (
    add_data_arguments, config_from_args, load_dataset, print_header, setup_logging,
)

# Derived, never spelled out. A hardcoded second copy of the control's name is
# a definition that agrees with `core.features` until one of them changes — and
# the failure mode is a survey that quietly runs without a control, which is how
# the previous incarnation of this project came to rank its own control first
# and not notice.
CONTROL = CONTROL_GROUPS[0]


def trials(groups: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Every group alone, the full set, and the full set without the control."""
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
    setup_logging(args)
    config = config_from_args(args)
    print_header('Ablation', config)

    dataset = load_dataset(args, config)
    folds = purged_walk_forward(dataset.window_index, n_folds=config.n_folds,
                                embargo_minutes=config.embargo_minutes)
    for fold in folds:
        assert_no_leakage(fold)

    available = tuple(g for g in FEATURE_GROUPS if g in set(FEATURE_GROUPS))
    print(f'  {len(folds)} folds, groups: {", ".join(available)}\n')
    print(f"  {'groups':30s} {'skill':>11s} {'se':>10s} {'t':>7s} {'folds+':>8s} {'trees':>7s}")

    rows = []
    for label, groups in trials(available).items():
        skills, trees = [], []
        for fold in folds:
            fit, train = fit_fold(dataset, fold.train, config, groups=groups)
            test = apply_fold(dataset, fit, fold.test, config, groups=groups)
            model = fit_model(train, fit.baseline, config, groups=groups)
            y = test['outcome'].to_numpy(dtype=float)
            b = test['baseline_probability'].to_numpy(dtype=float)
            m = np.asarray(model.predict(test), dtype=float)
            # Rows with no volatility estimate carry no forecast; excluding them is
            # the same rule core/metrics.py applies.
            keep = np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
            skills.append(log_loss(y[keep], b[keep]) - log_loss(y[keep], m[keep]))
            trees.append(model.best_iteration or 0)
        v = np.asarray(skills, dtype=float)
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')
        t = v.mean() / se if se and np.isfinite(se) and se > 0 else float('nan')
        print(f'  {label:30s} {v.mean():+11.6f} {se:10.6f} {t:+7.2f} '
              f'{int((v > 0).sum()):>6d}/{len(v)} {np.mean(trees):>7.0f}')
        rows.append({'groups': label, 'skill': v.mean(), 'se': se, 't': t,
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
