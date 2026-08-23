"""Evaluate a candidate and install it, gates permitting.

The only path to `models/forecast.joblib`. The live signal writer loads that
file by name and nothing else, so this is the single place a model becomes real.

Every attempt is recorded in `models/promotions/`, passed or blocked. That
ledger is the trial count, and the trial count is what any claim of skill has
to be discounted by — a project that deletes its failures cannot compute its own
multiple-testing correction.

    python -m scripts.promote
    python -m scripts.promote --history
    python -m scripts.promote --force --reason "skill is on the >0.9 tail; the
        average forecast is flat and the gates read averages"
"""

from __future__ import annotations

import argparse

import pandas as pd

from core.backtest import walk_forward
from core.metrics import evaluate_gates, gate_report
from core.promotion import history, load_live, promote, trial_count
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--history', action='store_true',
                        help='What has been tried, and why not. Then exit.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Evaluate and score the gates, install nothing.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--reason', type=str, default=None,
                        help='Required with --force, and stored with the artifact.')
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.history:
        frame = history()
        if frame.empty:
            print('no promotion attempts recorded')
            return 0
        pd.set_option('display.width', 200, 'display.max_colwidth', 60)
        print(frame.to_string(index=False))
        print(f'\n{len(frame)} attempts, {int(frame["installed"].sum())} installed. '
              f'Any claim of skill discounts by the trial count.')
        return 0

    if args.force and not args.reason:
        raise SystemExit(
            '--force needs --reason. The one good argument for overriding these '
            'gates — skill on a high-conviction tail that the average forecast '
            'does not show — is also the argument that kept a losing system '
            'alive, so it has to be written down.')

    config = config_from_args(args)
    groups = groups_from_args(args)
    print_header('Promotion', config)
    print(f'  attempts so far: {trial_count()}')
    live = load_live()
    if live is not None:
        print(f'  currently live: alpha={live.residual_scale:.3f}, '
              f'{len(live.features)} features, trained on '
              f'{live.n_train_windows:,} windows')
    print()

    dataset = load_dataset(args, config)
    result = walk_forward(dataset, config, groups=groups, trade=True)
    print('\n' + result.report.summary())

    if args.dry_run:
        print('\n' + gate_report(evaluate_gates(result.report)))
        print('\ndry run: nothing installed, nothing recorded')
        return 0

    # The last fold's model is the candidate: it is the one trained on the most
    # history, which is what would be deployed. The earlier folds are its
    # evidence, not alternatives to it.
    candidate = result.models[-1]
    attempt = promote(candidate, result.report, force=args.force,
                      force_reason=args.reason, trades=result.trades())
    print('\n' + attempt.summary())
    return 0 if attempt.installed else 1


if __name__ == '__main__':
    raise SystemExit(main())
