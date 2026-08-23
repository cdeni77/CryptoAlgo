"""Fit one model on all available history, for inspection. Not for promotion.

This is the microscope, not the gate. It trains on everything up to the most
recent windows, holds back the last fold's worth for a single out-of-sample
look, and prints what the model paid attention to. Use it to answer "what is it
actually using" and "did the correction survive"; use `scripts.evaluate` to
answer "does it work" and `scripts.promote` to install anything.

Two numbers here are worth more than the rest:

* **alpha**, the residual scale — how much of the model's claimed correction
  survives on held-out rows. Near zero means it found nothing, however good the
  training loss looked.
* **the control's share of gain.** Hour of day cannot forecast direction. If the
  `clock` group carries the model, the measurement is broken rather than the
  market interesting. The previous incarnation of this project ran a 27-cell
  survey whose best cell was its own control, and that was the most useful
  result it produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core.baseline import log_loss, reliability
from core.cv import purged_walk_forward, recency_weights
from core.dataset import apply_fold, fit_fold
from core.features import population_report
from core.model import fit_model
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
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
    folds = purged_walk_forward(dataset.window_index, n_folds=config.n_folds,
                                embargo_minutes=config.embargo_minutes)
    fold = folds[-1]
    print(f'  {fold.label()}  (the last fold; this is one look, not an evaluation)')

    fit, train = fit_fold(dataset, fold.train, config, groups=groups)
    print('\n' + fit.summary())

    weights = recency_weights(train['window_open'], config.recency_half_life_days)
    model = fit_model(train, fit.baseline, config, groups=groups, weights=weights)
    print('\n' + model.summary())

    test = apply_fold(dataset, fit, fold.test, config, groups=groups)
    y = test['outcome'].to_numpy(dtype=float)
    p = model.predict(test)
    pb = test['baseline_probability'].to_numpy(dtype=float)
    print(f'\n  held-out: log loss {log_loss(y, p):.5f} vs baseline {log_loss(y, pb):.5f} '
          f'(skill {log_loss(y, pb) - log_loss(y, p):+.5f})')
    print(f'  held-out calibration error {reliability(y, p).expected_calibration_error:.5f} '
          f'vs baseline {reliability(y, pb).expected_calibration_error:.5f}')
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
