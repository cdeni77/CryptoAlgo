"""Walk-forward evaluation: skill, calibration, money, gates, stress.

Fits and scores across purged expanding folds, runs the probabilities through
the `decide()` the live path uses, and reports in the order to read it:

1. Log-loss skill against the baseline, with the standard error from fold
   dispersion. Offsets share labels and symbols correlate, so a breadth
   formula is wrong.
2. Fold agreement. Necessary, never sufficient.
3. Calibration. The system trades only its confident predictions.
4. The rejection funnel, expected to be dominated by `edge_below_gate`.
5. The money, per fold and on one continuous account.
6. Cost stress. The half-spread is an assumption; a strategy that survives
   only at the assumed spread is not demonstrated.
7. The edge curve. Read the shape, not the argmax: choosing `min_edge_pp` from
   it selects on the rows it is scored on.

    python -m scripts.evaluate
    python -m scripts.evaluate --groups clock        # the control, alone
    python -m scripts.evaluate --min-edge-pp 1.5 --half-spread-cents 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core.backtest import cost_stress, edge_curve, walk_forward
from core.metrics import evaluate_gates, gate_report, gate_spec_hash, gates_passed
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, log_trial,
    print_header, setup_logging,
)


def stack_table(models) -> pd.DataFrame:
    """Return one row per fold of a stacked run: a, b, c and the stage-2 sample."""
    rows = []
    for i, model in enumerate(models):
        if not hasattr(model, 'stack_summary'):
            continue
        s = model.stack_summary()
        rows.append({'fold': i, 'a': round(s['a'], 4), 'b': round(s['b'], 4),
                     'c': round(s['c'], 4), 'n_stage2_rows': s['n_rows'],
                     'fitted': s['fitted'],
                     'oof_skill_vs_market': s['inner_market_log_loss']
                     - s['inner_log_loss']})
    return pd.DataFrame(rows)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--no-trade', action='store_true',
                        help='Measure the forecast only. Faster, and the right mode '
                             'when the question is skill rather than money.')
    parser.add_argument('--no-stress', action='store_true')
    parser.add_argument('--out', type=str, default=None,
                        help='Write scored rows, trades and the report to this directory')
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    groups = groups_from_args(args)
    print_header('Walk-forward evaluation', config)

    dataset = load_dataset(args, config)
    result = walk_forward(dataset, config, groups=groups, trade=not args.no_trade)
    report = result.report

    print('\n' + '-' * 78)
    print('FORECAST')
    print('-' * 78)
    print(report.summary())

    if report.folds_positive >= 5 and report.mean_skill > 0:
        print(f'\n  Read the {report.folds_positive}/{report.folds_total} agreement with '
              f'the p-value beside it ({report.sign_agreement_p_value:.3f}): under no '
              f'skill\n  each fold is a coin flip, so this much agreement is not rare.')

    stack = stack_table(getattr(result, 'models', ()) or ())
    if not stack.empty:
        print('\n  stage 2 per fold — b is how much of the spot model\'s disagreement '
              'with the\n  price survives out of fold (0 = the price is right, '
              '1 = all of it):')
        print('    ' + stack.to_string(index=False).replace('\n', '\n    '))

    per_offset = report.per_offset()
    if not per_offset.empty:
        print('\n  skill by decision offset (this is how the offset set gets narrowed):')
        print('    ' + per_offset.to_string(index=False).replace('\n', '\n    '))
        print('    The barrier framing predicts the edge peaks where |x|/sigma is near 1 —')
        print('    mid-window with a moderate displacement — and decays late, because a')
        print('    probability pinned near 1 is insensitive to a sigma error.')

    if not args.no_trade:
        print('\n' + '-' * 78)
        print('THE FUNNEL')
        print('-' * 78)
        funnel = result.rejections[result.rejections > 0]
        total = int(result.rejections.sum())
        for reason, count in funnel.items():
            print(f'  {reason:<26} {count:>10,}  {count / total:6.2%}')
        print(f'  {"total considered":<26} {total:>10,}')

        print('\n' + '-' * 78)
        print('THE MONEY')
        print('-' * 78)
        for fold in report.folds:
            if fold.stats is not None:
                print(f'  fold {fold.index}: ' + fold.stats.summary().replace('\n', '\n  '))
        if report.continuous is not None:
            print(f'\n  continuous $100 account across the whole out-of-sample span:')
            print('    ' + report.continuous.summary().replace('\n', '\n    '))
            print(f'    fees were {report.continuous.fees_share_of_gross:.1%} of gross')

        if not args.no_stress:
            print('\n' + '-' * 78)
            print('COST STRESS')
            print('-' * 78)
            print(cost_stress(result.scored, config).to_string(index=False))
            print('\n  The half-spread is the assumption, not the fee. If the answer only')
            print('  survives at 1c, it has not been demonstrated — one filled limit order')
            print('  on the venue would settle it.')

            print('\n' + '-' * 78)
            print('WHERE THE GATE BELONGS')
            print('-' * 78)
            print(edge_curve(result.scored, config).to_string(index=False))
            print('\n  Monotone improvement as the gate tightens means the forecast is real')
            print('  and concentrated. A peak that then falls means the tail is noise.')

    print('\n' + '-' * 78)
    print('GATES')
    print('-' * 78)
    # Read the market gates exactly as `promote` does, so both report the same
    # verdict.
    from scripts.promote import market_measurement

    # Forward `entry_offsets`: pooling every scored offset measures a policy
    # nobody runs.
    market = market_measurement(result.scored, entry_offsets=config.entry_offsets)
    gates = evaluate_gates(report, extra=market)
    print(gate_report(gates))

    # Every evaluation is a trial, whether or not anything is promoted after it.
    log_trial('evaluate', args, config, groups=groups, dataset=dataset,
              report=report, metrics={
                  **{name: market.get(name) for name in (
                      'model_minus_market', 'model_minus_market_p', 'min_folds',
                      'market_windows')},
                  'gates_passed': gates_passed(gates),
                  'failed_gates': [g.name for g in gates if not g.passed],
                  **({'stack_b_median': float(stack['b'].median()),
                      'stack_a_median': float(stack['a'].median())}
                     if not stack.empty else {})})

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        result.scored.to_parquet(out / 'scored.parquet', index=False)
        trades = result.trades()
        if not trades.empty:
            trades.to_parquet(out / 'trades.parquet', index=False)
        (out / 'report.json').write_text(json.dumps({
            'gate_values': report.gate_values(),
            'gates': [{'name': g.name, 'value': g.value, 'threshold': g.threshold,
                       'passed': g.passed} for g in gates],
            'gate_spec_hash': gate_spec_hash(),
            'fold_geometry': report.fold_geometry,
            'config': report.config_provenance,
            'per_offset': per_offset.to_dict('records'),
            'stack': stack.to_dict('records'),
            'rejections': result.rejections.to_dict(),
        }, indent=2, default=str))
        print(f'\nwrote {out}/scored.parquet, trades.parquet, report.json')

    return 0 if gates_passed(gates) else 1


if __name__ == '__main__':
    raise SystemExit(main())
