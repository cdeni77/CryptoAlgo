"""How much of the backtest's result is the forecast, and how much is the gates?

    python -m scripts.gate_sensitivity --exclude HYPE,PEPE,NEAR,ONDO --horizon 1
    python -m scripts.gate_sensitivity --horizon 24 --min-history-days 231

The production gate stack accepts a few dozen bars out of 75,000, so a headline
Sharpe computed from it is a statement about two trades rather than about the
model. This relaxes one gate at a time, then all of them, and finally removes the
cost hurdle itself — while the simulation keeps charging the real fee schedule at
every fill. That separates three different failures that look identical from the
outside:

* the forecast is fine and the gates are too tight,
* the forecast is weak but positive and fees eat it,
* the forecast has no edge at all.

`ignore_cost_hurdle` is the one scenario that changes what `decide()` is told
rather than what it will accept. `expected_net` is replaced by the *gross*
forecast, so the decision stops asking whether the edge covers the round trip —
but `core/simulation.py` still charges it on entry and exit. That is the literal
form of "just let it trade with fees", and it is a diagnostic, never a
configuration: a system that trades on a gross forecast is buying a known cost
against an unknown edge.

Forecasts are generated once and replayed under every scenario, so the walk-
forward training happens a single time. Every row of the output is the same
model on the same out-of-sample bars; only the decision rule differs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from core.backtest import generate_walk_forward_forecasts, run_backtest
from core.config import Config
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data


def _override(config: Config, **fields) -> Config:
    """Set fields *and* mark them as overrides, so a coin profile cannot win.

    `Config.resolve` implements CLI > profile > default, and the gate thresholds
    are resolved against the per-coin profile. Setting the field alone therefore
    changes nothing for any instrument whose profile carries its own value —
    which is most of them.
    """
    marked = frozenset(config.cli_overrides) | set(fields)
    return replace(config, cli_overrides=marked, **fields)


def _scenarios(config: Config, n_symbols: int) -> list[tuple[str, Config, str]]:
    """(name, config, signal mode) in order of increasing permissiveness.

    Signal mode is 'net' (as shipped), 'gross' (cost-blind) or 'inverted'
    (cost-blind and sign-flipped).
    """
    loose_conviction = dict(min_edge_over_cost=0.0)
    loose_risk = dict(min_edge_to_risk=0.0)
    loose_vol = dict(min_vol_24h=0.0, max_vol_24h=10.0)
    loose_portfolio = dict(
        cooldown_hours=0.0,
        max_positions=max(int(n_symbols), 1),
        max_portfolio_correlation=0.0,
    )
    everything = {**loose_conviction, **loose_risk, **loose_vol, **loose_portfolio}

    return [
        ('production',          config,                                'net'),
        ('no_conviction',       _override(config, **loose_conviction), 'net'),
        ('no_edge_to_risk',     _override(config, **loose_risk),       'net'),
        ('any_volatility',      _override(config, **loose_vol),        'net'),
        ('no_portfolio_caps',   _override(config, **loose_portfolio),  'net'),
        ('all_gates_loose',     _override(config, **everything),       'net'),
        ('ignore_cost_hurdle',  _override(config, **everything),       'gross'),
        ('inverted_gross',      _override(config, **everything),   'inverted'),
    ]


def _restate(forecasts: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Decide on the gross forecast, so the cost hurdle stops being a gate.

    `decide()` refuses a row whose `expected_net` is not positive, and
    `expected_net` is net of the round trip — which is the `edge_below_cost`
    rejection. Replacing it with `|price + carry|` and taking that sign makes the
    decision cost-blind. Nothing about the *accounting* changes: the fill still
    pays the fee, the spread and the commission.

    `mode='inverted'` flips the sign as well. That is not a strategy proposal —
    inverting a signal because it lost on one sample is the purest form of
    fitting to it. It is a *turnover* measurement, and the only one that
    separates "the forecast is bad" from "no forecast could pay at this
    frequency": if the sign-flipped book still loses, then sign accuracy is not
    the binding constraint at this horizon and no improvement to the model
    reaches profitability without cutting trade count.
    """
    out = forecasts.copy()
    gross = out['price'] + out['carry']
    if mode == 'inverted':
        gross = -gross
    out['side'] = np.sign(gross)
    out['expected_net'] = gross.abs()
    out['edge_to_risk'] = np.where(
        out['sigma'] > 1e-9, out['expected_net'] / out['sigma'], 0.0
    )
    return out


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--periods', type=int, default=6, help='Walk-forward retrains')
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--json', action='store_true', help='Also emit the rows as JSON')
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    print(f'\ndataset: {dataset}')
    generated = generate_walk_forward_forecasts(
        dataset.features, dataset.targets, config=config,
        profiles=dataset.profiles, n_periods=args.periods,
        horizon_bars=dataset.horizon_bars,
    )
    if generated.forecasts.empty:
        print('no out-of-sample forecasts')
        return 1
    print(f'forecasts: {json.dumps(generated.summary(), default=str)}')

    rows = []
    header = (f"{'scenario':<20} {'trades':>7} {'net $':>10} {'price $':>10} "
              f"{'fees $':>9} {'ret %':>7} {'Sharpe':>7} {'win %':>6} "
              f"{'accepted':>9} {'top gate':>22}")
    print('\n' + header)
    print('-' * len(header))

    for name, scenario, mode in _scenarios(config, len(dataset.bars)):
        forecasts = (generated.forecasts if mode == 'net'
                     else _restate(generated.forecasts, mode))
        result = run_backtest(
            forecasts=forecasts,
            bars_by_symbol=dataset.bars,
            funding_by_symbol=dataset.funding,
            config=scenario,
            profiles=dataset.profiles,
            initial_equity=args.equity,
            horizon_bars=generated.horizon_bars,
        )
        gates = result.gates
        counts = getattr(gates, 'counts', None) or {}
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else ('-', 0)
        summary = result.summary()
        rows.append({'scenario': name, 'signal_mode': mode, **summary})
        print(f"{name:<20} {result.n_trades:>7} {result.net_pnl:>10,.0f} "
              f"{result.price_pnl:>10,.0f} {result.fees:>9,.0f} "
              f"{summary['return_pct']:>7.2f} {result.sharpe:>7.2f} "
              f"{result.win_rate * 100:>6.0f} "
              f"{summary['gates']['accepted']:>9} "
              f"{f'{top[0]}={top[1]:,}':>22}")

    if args.json:
        print('\n' + json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
