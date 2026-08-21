"""Train a forecast model and save it with its provenance.

    python -m scripts.train
    python -m scripts.train --as-of 2026-06-01 --out models/forecast_june.joblib

The saved artifact records its feature-set hash, the cost config that priced its
targets, and the training window — so a model can always say what it was trained
on, and a stale one is detectable rather than silently scoring wrong inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.model import MODELS_DIR, cross_validate_forecast, train_forecast_model
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--out', default=None, help='Where to write the model')
    parser.add_argument('--cv-folds', type=int, default=6)
    parser.add_argument('--skip-cv', action='store_true',
                        help='Train only; skip the out-of-sample scoring')
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    print(f'\ndataset: {dataset}')
    print(json.dumps(dataset.summary(), indent=2, default=str))

    model = train_forecast_model(
        dataset.features, dataset.targets, config=config, data_as_of=args.as_of,
        horizon_bars=dataset.horizon_bars,
    )
    if model is None:
        print('\nnot enough resolved rows to train')
        return 1

    print('\nprovenance:')
    for key, value in model.provenance().items():
        print(f'  {key}: {value}')

    print('\nper-head validation:')
    for head, metrics in model.metrics.items():
        rendered = '  '.join(
            f'{k}={v:+.4f}' if isinstance(v, float) else f'{k}={v}'
            for k, v in metrics.items()
        )
        print(f'  {head:11s} {rendered}')

    if not args.skip_cv:
        report = cross_validate_forecast(
            dataset.features, dataset.targets, config=config, n_folds=args.cv_folds,
            horizon_bars=dataset.horizon_bars,
        )
        print(f'\ncross-validation: {report}')
        print(json.dumps(report.as_dict(), indent=2, default=str))
        if report.memorisation_suspected:
            print(
                '\nWARNING: price IC is close to the hindsight identity ceiling, '
                'which means the ranking may be reproducing instrument level '
                'rather than timing.'
            )

    out = Path(args.out) if args.out else MODELS_DIR / 'forecast.joblib'
    model.save(out)
    print(f'\nsaved {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
