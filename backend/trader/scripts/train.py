"""Fit one model on all available history, for inspection. Not for promotion.

Trains on all history except the last fold, scores that fold once, and prints
what the model uses. Use `scripts.evaluate` to judge it and `scripts.promote`
to install anything.

* alpha, the residual scale: how much of the correction survives on held-out
  rows. Near zero means the model found nothing.
* the control's share of gain: hour of day cannot forecast direction, so a
  model the `clock` group carries is suspect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline import log_loss, reliability
from core.cv import folds_for_config, recency_weights
from core.dataset import apply_fold, fit_fold
from core.features import population_report
from core.model import fit_for_config, table_groups
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, log_trial,
    print_header, setup_logging,
)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--save', type=str, default=None,
                        help='Write the fitted model here. This does not promote it — '
                             '`scripts.promote` is the only path to the live artifact.')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    groups = groups_from_args(args)
    print_header('Train one model, for inspection', config)

    dataset = load_dataset(args, config)
    stacked = config.stack_mode != 'off'
    if stacked:
        from core.backtest import stack_folds
        folds = stack_folds(dataset, config)
    else:
        folds = folds_for_config(dataset.window_index, config)
    groups = table_groups(config, groups)
    fold = folds[-1]
    print(f'  {fold.label()}  (the last fold; this is one look, not an evaluation)')

    fit, train = fit_fold(dataset, fold.train, config, groups=groups)
    print('\n' + fit.summary())

    weights = recency_weights(train['window_open'], config.recency_half_life_days)
    model = fit_for_config(train, fit.baseline, config, groups=groups, weights=weights)
    print('\n' + model.summary())

    test = apply_fold(dataset, fit, fold.test, config, groups=groups)
    p = model.predict(test)
    # A stacked model is market-relative and scores NaN without a quote, so
    # its null is the price and only quoted rows can be compared.
    null = 'market' if stacked else 'baseline'
    pb = test[f'{null}_probability'].to_numpy(dtype=float)
    y = test['outcome'].to_numpy(dtype=float)
    keep = np.isfinite(p) & np.isfinite(pb) & np.isfinite(y)
    if not keep.all():
        print(f'\n  {int((~keep).sum()):,} of {len(keep):,} held-out rows carry no '
              f'forecast and are left out')
    y, p, pb = y[keep], p[keep], pb[keep]
    print(f'\n  held-out: log loss {log_loss(y, p):.5f} vs {null} {log_loss(y, pb):.5f} '
          f'(skill {log_loss(y, pb) - log_loss(y, p):+.5f})')
    print(f'  held-out calibration error {reliability(y, p).expected_calibration_error:.5f} '
          f'vs {null} {reliability(y, pb).expected_calibration_error:.5f}')
    print('\n  held-out reliability:')
    print(reliability(y, p).table())

    print(f'\n  top {args.top} features by gain:')
    importance = model.importance().head(args.top)
    for row in importance.itertuples():
        flag = '  <- CONTROL' if row.is_control else ''
        print(f'    {row.feature:<28} {row.share:6.2%}{flag}')
    print(f'\n  control group takes {model.control_importance_share:.1%} of total gain. '
          f'Hour of day cannot\n  forecast direction, so a large share here indicts the '
          f'measurement, not the market.')

    populated = population_report(test, groups)
    thin = populated[populated['share'] < 0.9]
    if not thin.empty:
        print('\n  features under 90% populated (an empty group has the same shape as a '
              'working one):')
        print('    ' + thin.to_string(index=False).replace('\n', '\n    '))

    if args.save:
        path = model.save(Path(args.save))
        print(f'\n  wrote {path} and {path.with_suffix(".provenance.json")}')
        print('  this is NOT promoted — run `python -m scripts.promote` for that')

    # One look at one fold is still a look, so it counts as a trial.
    log_trial('train', args, config, groups=groups, dataset=dataset, metrics={
        'held_out_skill': log_loss(y, pb) - log_loss(y, p),
        'held_out_fold': int(fold.index),
        'residual_scale': model.residual_scale,
        'control_gain_share': model.control_importance_share,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
