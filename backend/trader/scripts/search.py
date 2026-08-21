"""Run a search campaign and record every trial on the ledger.

    python -m scripts.search --list
    python -m scripts.search --campaign thresholds
    python -m scripts.search --ledger-report

Replaces comprehensive_search, gap_search, weekly_search, reverify_profiles and
coin_backtests. They all imported a config, called a backtest and applied a
pass/fail rule, differing only in what they enumerated and what they gated — both
of which are data, and are now campaign definitions.

Every trial is recorded, including failures. A deflated Sharpe ratio needs the
true number of configurations evaluated, and counting only the survivors is how
a lucky backtest passes a significance test it should fail.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from core.backtest import walk_forward_backtest
from core.cv import effective_sample_size
from core.metrics import sharpe_ratio, summarise_paths
from core.search import SearchLedger, default_campaigns, run_campaign
from core.simulation import bootstrap_trades
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--campaign', default=None, help='Campaign to run')
    parser.add_argument('--list', action='store_true', help='Show campaigns and exit')
    parser.add_argument('--ledger', default=None, help='Ledger path')
    parser.add_argument('--ledger-report', action='store_true',
                        help='Summarise the ledger and exit')
    parser.add_argument('--periods', type=int, default=4)
    parser.add_argument('--equity', type=float, default=100_000.0)
    args = parser.parse_args()
    configure_logging(args.log_level)

    campaigns = default_campaigns()

    if args.list:
        for name, space in campaigns.items():
            print(f'{name:12s} {space.size:4d} combinations')
            for field, values in sorted(space.grid.items()):
                print(f'  {field:24s} {list(values)}')
            print(f'  {"seeds":24s} {list(space.seeds)}')
        return 0

    ledger = SearchLedger(args.ledger)

    if args.ledger_report:
        frame = ledger.read()
        if frame.empty:
            print('ledger is empty')
            return 0
        print(f'{len(frame)} trials across {frame["campaign"].nunique()} campaigns')
        print(frame.groupby('campaign').agg(
            trials=('trial_id', 'count'), passed=('passed', 'sum')
        ).to_string())
        best = ledger.best('walk_forward_median_sharpe')
        if best is not None:
            print(f'\nbest passing trial: {best["trial_id"]} ({best["campaign"]})')
            print(f'  parameters: {best["parameters"]}')
        return 0

    if args.campaign not in campaigns:
        print(f'unknown campaign. Available: {", ".join(campaigns)}')
        return 1

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    print(f'\ndataset: {dataset}')
    space = campaigns[args.campaign]
    print(f'campaign {space.name}: {space.size} combinations\n')

    observations = int(sum(
        effective_sample_size(
            dataset.features.xs(symbol, level='symbol').index, dataset.horizon_bars
        )
        for symbol in dataset.symbols
    ))
    print(f'effective observations across the panel: {observations}')
    print('  (this is the denominator every significance test uses)\n')

    def evaluate(trial_config, seed):
        result, generated = walk_forward_backtest(
            dataset.features, dataset.targets,
            bars_by_symbol=dataset.bars, funding_by_symbol=dataset.funding,
            config=trial_config, profiles=dataset.profiles,
            n_periods=args.periods, initial_equity=args.equity,
            # Every trial must be measured at the same horizon the targets were
            # built at, or the campaign is comparing configurations across
            # different problems.
            horizon_bars=dataset.horizon_bars,
        )
        if not result.trades:
            return {'oos_trades': 0.0}, []

        per_period = []
        for start, end in generated.periods:
            window = result.equity_curve.loc[
                (result.equity_curve.index >= start) & (result.equity_curve.index <= end)
            ]
            if len(window) > 2:
                per_period.append(sharpe_ratio(window.pct_change().dropna()))

        distribution = summarise_paths(per_period)
        bootstrap = bootstrap_trades(
            result.trades_frame()['net_return'].to_numpy(), n_resamples=500, seed=seed
        )
        return (
            {
                'walk_forward_median_sharpe': distribution.median,
                'walk_forward_p05_sharpe': distribution.p05,
                'bootstrap_positive_fraction': bootstrap.probability_positive,
                'oos_trades': float(result.n_trades),
                'max_drawdown': result.drawdown.max_drawdown,
                'carry_contribution': result.carry_contribution,
                'net_pnl': result.net_pnl,
            },
            per_period,
        )

    outcome = run_campaign(
        space, evaluate, base_config=config, ledger=ledger,
        data_as_of=args.as_of, observations=observations,
        horizon_bars=dataset.horizon_bars,
    )
    print(f'\n{outcome}')
    print(json.dumps(outcome.summary(), indent=2, default=str))

    for trial in outcome.survivors:
        print(f'\nsurvivor {trial.trial_id}: {trial.parameters}')
        print(f'  {json.dumps(trial.metrics, indent=2, default=str)}')
    if not outcome.survivors:
        print('\nno survivors. The gates rejecting everything is the expected '
              'outcome at this sample size, not a malfunction.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
