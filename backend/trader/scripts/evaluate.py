"""Walk-forward evaluation: skill, calibration, money, gates, stress.

The one script that answers "does this work". It fits and scores across purged
expanding folds, runs the resulting probabilities through the same `decide()`
the live path uses, and reports the answer in the order it should be read:

1. **Log loss skill against the baseline.** Positive or nothing else matters.
   With the standard error from fold dispersion, not from a breadth formula —
   four offsets share a label and the three symbols are ~0.7 correlated, and a
   breadth-derived error bar on that structure is not merely optimistic but
   degenerate.
2. **Fold agreement.** Five of six positive happens 10.9% of the time by
   chance, so agreement is necessary and never sufficient.
3. **Calibration.** The system trades only its confident predictions, so being
   wrong about *how* confident matters more than the average.
4. **The rejection funnel.** Expected to be dominated by `edge_below_gate`.
   That is the system working: the forecast does not cover the fee, so it
   declines. On the perp system the equivalent number was the single most
   informative output it ever produced.
5. **The money**, per fold and then on one continuous $100 account.
6. **Cost stress.** The half-spread is an assumption — no Kalshi order ticket
   has been read against `core/costs.py` — and it is larger than the fee at
   price above 83c (where 0.07*p(1-p) falls below a cent). A strategy that
   survives only at the assumed
   spread has not been demonstrated.
7. **The edge curve.** A diagnostic on concentration — read the shape, not the
   argmax. Choosing `min_edge_pp` from it would be selecting on the same
   out-of-sample rows it is scored on; see `core/backtest.py:edge_curve` for the
   measurement showing why the money on a run this size cannot choose anything.

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
from core.metrics import evaluate_gates, gate_report, gates_passed
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
)


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
    # The market gates read live-recorded quotes, which this run does not have and
    # structurally cannot: a backtest has no book. Reading them here anyway means
    # `evaluate` reports the same verdict `promote` will, rather than passing a
    # candidate that promotion then blocks for a reason `evaluate` never mentioned.
    from scripts.promote import market_measurement

    gates = evaluate_gates(report, extra=market_measurement(result.scored))
    print(gate_report(gates))

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
            'config': report.config_provenance,
            'per_offset': per_offset.to_dict('records'),
            'rejections': result.rejections.to_dict(),
        }, indent=2, default=str))
        print(f'\nwrote {out}/scored.parquet, trades.parquet, report.json')

    return 0 if gates_passed(gates) else 1


if __name__ == '__main__':
    raise SystemExit(main())
