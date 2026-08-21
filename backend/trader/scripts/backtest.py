"""Walk-forward backtest, then the full simulation stack and the gates.

    python -m scripts.backtest
    python -m scripts.backtest --periods 8 --equity 25000 --full

Walk-forward is the only mode offered. Backtesting a model over its own training
window measures memorisation: on driftless random walks it returned a mean price
PnL of +95,000 with a t-statistic of +7, and there is nothing to find in a
driftless random walk.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from core.backtest import walk_forward_backtest
from core.metrics import evaluate_gates, gate_report, summarise_paths
from core.simulation import (
    SimulationReport,
    bootstrap_trades,
    cost_stress,
    synthetic_panel,
)
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--periods', type=int, default=6, help='Walk-forward retrains')
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--spread-bps', type=float, default=4.0)
    parser.add_argument('--full', action='store_true',
                        help='Also run synthetic panels and cost stress (slow)')
    parser.add_argument('--synthetic-paths', type=int, default=20)
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    print(f'\ndataset: {dataset}')
    result, generated = walk_forward_backtest(
        dataset.features, dataset.targets,
        bars_by_symbol=dataset.bars, funding_by_symbol=dataset.funding,
        config=config, profiles=dataset.profiles,
        n_periods=args.periods, initial_equity=args.equity,
        spread_bps=args.spread_bps, horizon_bars=dataset.horizon_bars,
    )

    if not result.trades:
        print('\nno trades. Gate counts show what closed:')
        print(f'  {result.gates}')
        return 0

    print(f'\nwalk-forward: {result}')
    print(f'forecasts: {json.dumps(generated.summary(), default=str)}')
    print(f'gates: {result.gates}')
    print(json.dumps(result.summary(), indent=2, default=str))

    report = SimulationReport(
        oos_trades=result.n_trades,
        max_exit_participation=result.max_exit_participation,
    )
    returns = result.trades_frame()['net_return'].to_numpy()
    report.bootstrap = bootstrap_trades(returns, n_resamples=2_000)
    print(f'\nbootstrap: {report.bootstrap}')

    # Per-period Sharpe stands in for the CPCV path distribution here: each
    # walk-forward period is an independent out-of-sample stretch.
    period_sharpes = []
    for start, end in generated.periods:
        window = result.equity_curve.loc[
            (result.equity_curve.index >= start) & (result.equity_curve.index <= end)
        ]
        if len(window) > 2:
            from core.metrics import sharpe_ratio
            period_sharpes.append(sharpe_ratio(window.pct_change().dropna()))
    if period_sharpes:
        report.per_period = summarise_paths(period_sharpes)
        print(f'per-period Sharpe: {report.per_period.as_dict()}')

    if args.full:
        def run_with(cfg) -> float:
            outcome, _ = walk_forward_backtest(
                dataset.features, dataset.targets,
                bars_by_symbol=dataset.bars, funding_by_symbol=dataset.funding,
                config=cfg, profiles=dataset.profiles,
                n_periods=args.periods, initial_equity=args.equity,
                spread_bps=args.spread_bps, horizon_bars=dataset.horizon_bars,
            )
            return outcome.sharpe

        report.stress = cost_stress(run_with, config)
        print(f'\ncost stress: {json.dumps(report.stress.as_dict(), indent=2)}')

        synthetic_sharpes = []
        for seed in range(args.synthetic_paths):
            bars = synthetic_panel(dataset.bars, seed=seed)
            outcome, _ = walk_forward_backtest(
                dataset.features, dataset.targets,
                bars_by_symbol=bars, funding_by_symbol=dataset.funding,
                config=config, profiles=dataset.profiles,
                n_periods=args.periods, initial_equity=args.equity,
                spread_bps=args.spread_bps, horizon_bars=dataset.horizon_bars,
            )
            synthetic_sharpes.append(outcome.sharpe)
        report.synthetic = summarise_paths(synthetic_sharpes)
        print(f'\nsynthetic panels: {report.synthetic.as_dict()}')
        print(
            '  note: a generator contains only the structure calibrated into it, '
            'so this measures robustness and sizing, never edge'
        )

    promoted, gates = evaluate_gates(report.measurements())
    print(f'\n{gate_report(gates)}')
    return 0 if promoted else 2


if __name__ == '__main__':
    sys.exit(main())
