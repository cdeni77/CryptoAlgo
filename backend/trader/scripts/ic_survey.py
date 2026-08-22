"""Survey horizon x feature-group for directional skill, and count the trials.

    python -m scripts.ic_survey
    python -m scripts.ic_survey --horizons 1,4,24 --min-history-days 231
    python -m scripts.ic_survey --ledger-report

A wide search over a fixed sample produces a winner whether or not one exists,
so the only useful form of "try everything" is one that reports what trying
everything is expected to turn up by chance. This walks the grid, records every
cell on the append-only ledger — failures included, because the trial count is
what a deflated Sharpe discounts by — and prints the null expectation next to
the observed hit count.

The pre-registered hit rule, stated here rather than chosen after the fact: a
cell is a hit when its median price IC is positive **and** at least five of the
six purged folds share that sign. Under the null each fold's sign is a coin
flip, so

    P(>= 5 of 6 agree) = (C(6,5) + C(6,6)) / 2^6 = 7/64 = 10.9%

which over 27 cells expects 2.9 hits from noise alone. A survey that returns
three hits has found nothing; the count has to clear the null by a margin the
binomial tail makes explicit.

Scored on price IC, never net IC. `expected_net` and `realised_net` share the
`-cost` term, so net IC is positive before any prediction happens — the
cost-only floor is reported alongside so the difference is visible, but the
ranking is on the metric that cannot be inflated that way.

Every cell also carries `required_ic`: the median across the universe of
`round_trip_cost / sigma_h`, which is the IC the cell would need to pay for its
own trading. An IC of +0.01 against a requirement of 0.20 is not a weak edge, it
is a rounding error on a toll.

This measures forecast skill only. It runs no simulation, so it clears no
promotion gate — every trial records the full gate set as `ungated`, and a
survivor here is a candidate for a backtest, never for promotion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.config import Config
from core.dataset import load_dataset
from core.datastore import ResearchStore
from core.features import GROUPS
from core.metrics import DEFAULT_GATES
from core.model import cross_validate_forecast
from core.search import SearchLedger, Trial
from core.targets import round_trip_cost_series
from scripts._common import add_data_arguments, build_config, configure_logging

# The group sets worth spending trials on. `seasonality,cost` is a deliberate
# control: hour-of-day and a fee hurdle cannot forecast direction, so a cell that
# scores there is measuring the survey's own noise floor rather than an edge.
GROUP_SETS: tuple[tuple[str, ...], ...] = (
    ('all',),
    ('cross_venue',),
    ('trend',),
    ('volatility',),
    ('market_factor',),
    ('liquidity',),
    ('cross_venue', 'trend'),
    ('volatility', 'trend', 'market_factor'),
    ('seasonality', 'cost'),
)

MIN_FOLD_AGREEMENT = 5
N_FOLDS = 6
# P(at least MIN_FOLD_AGREEMENT of N_FOLDS signs agree with the mean), under a
# fair coin per fold.
NULL_HIT_RATE = sum(math.comb(N_FOLDS, k) for k in range(MIN_FOLD_AGREEMENT, N_FOLDS + 1)) / 2 ** N_FOLDS


def _required_ic(dataset, config: Config, horizon: int) -> float:
    """Median `cost / sigma_h` across the universe: the IC a cell has to reach.

    Cost is fixed per round trip and dispersion grows as sqrt(h), so this falls
    with the hold. It is the number that decides whether an IC is worth anything,
    and it belongs next to the IC rather than in a doc.
    """
    ratios: list[float] = []
    for symbol, bars in dataset.bars.items():
        if bars.empty or 'open' not in bars or len(bars) < 200:
            continue
        frame = bars.sort_index()
        opens = frame['open']
        forward = (opens.shift(-(1 + horizon)) / opens.shift(-1) - 1.0).std()
        cost = float(round_trip_cost_series(symbol, frame['close'], config).median())
        if forward and np.isfinite(forward) and forward > 0:
            ratios.append(cost / float(forward))
    return float(np.median(ratios)) if ratios else float('nan')


def _survey(args, config: Config, ledger: SearchLedger) -> pd.DataFrame:
    store = ResearchStore(args.store)
    horizons = [int(h) for h in str(args.horizons).split(',') if h.strip()]
    rows: list[dict] = []
    trials: list[Trial] = []
    gate_names = sorted(DEFAULT_GATES)

    for horizon, groups in itertools.product(horizons, GROUP_SETS):
        label = ','.join(groups)
        requested = None if groups == ('all',) else list(groups)
        cell = f'h={horizon}h groups={label}'
        logging.info('surveying %s', cell)

        row = {'horizon': horizon, 'groups': label}
        try:
            dataset = load_dataset(
                store, venue=args.venue, reference_venue=args.reference_venue or None,
                config=config, as_of=args.as_of, min_quality=args.min_quality,
                horizon_bars=horizon, feature_groups=requested,
                min_history_days=args.min_history_days,
            )
            report = cross_validate_forecast(
                dataset.features, dataset.targets, config=config,
                n_folds=N_FOLDS, horizon_bars=horizon,
            )
            fold_ic = [float(f.price_ic) for f in report.folds
                       if np.isfinite(f.price_ic)]
            median = float(report.price_ic.median)
            agree = sum(1 for v in fold_ic if (v > 0) == (median > 0))
            row.update({
                'price_ic': median,
                'agree': agree,
                'folds': len(fold_ic),
                'price_ic_xs': float(report.price_ic_xs.median),
                'net_ic': float(report.net_ic.median),
                'net_ic_floor': float(report.net_ic_cost_only.median),
                'net_ic_skill': float(report.net_ic_skill),
                'identity_ratio': float(report.identity_ratio),
                'effective_obs': float(report.total_effective_observations),
                'required_ic': _required_ic(dataset, config, horizon),
                'n_features': int(dataset.features.shape[1]),
                'error': None,
            })
            row['hit'] = bool(median > 0 and agree >= MIN_FOLD_AGREEMENT)
            row['ic_share_of_required'] = (
                median / row['required_ic'] if row['required_ic'] else float('nan')
            )
            metrics = {k: v for k, v in row.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            trials.append(Trial(
                campaign=args.campaign, trial_id=cell,
                parameters={'horizon': horizon, 'feature_groups': label,
                            'min_history_days': args.min_history_days,
                            'recency_half_life_days': config.recency_half_life_days},
                seed=0, metrics=metrics, fold_scores=fold_ic,
                passed=row['hit'], failed_gates=[],
                feature_set_hash=getattr(dataset, 'feature_set_hash', '') or '',
                cost_config_version=config.cost_config_version,
                ungated=gate_names,
                data_as_of=args.as_of,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as exc:                                  # noqa: BLE001
            logging.warning('%s failed: %s', cell, exc)
            row.update({'price_ic': float('nan'), 'agree': 0, 'hit': False,
                        'error': str(exc)[:200]})
            trials.append(Trial(
                campaign=args.campaign, trial_id=cell,
                parameters={'horizon': horizon, 'feature_groups': label},
                seed=0, metrics={}, passed=False, ungated=gate_names,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc)[:200],
            ))
        rows.append(row)

    # Appended once, after the grid, but every cell is here including failures:
    # a ledger that only records survivors cannot support a deflated Sharpe.
    ledger.append(trials)
    return pd.DataFrame(rows)


def _report(frame: pd.DataFrame) -> None:
    if frame.empty:
        print('no cells surveyed')
        return

    order = ['horizon', 'groups', 'price_ic', 'agree', 'required_ic',
             'ic_share_of_required', 'net_ic', 'net_ic_floor', 'net_ic_skill',
             'identity_ratio', 'effective_obs', 'hit']
    shown = frame[[c for c in order if c in frame.columns]].copy()
    print('\n' + shown.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

    n = int(len(frame))
    hits = int(frame['hit'].sum())
    expected = n * NULL_HIT_RATE
    # P(at least `hits` hits | null), so a survey cannot report a hit count
    # without reporting how ordinary that count is.
    tail = sum(math.comb(n, k) * NULL_HIT_RATE ** k * (1 - NULL_HIT_RATE) ** (n - k)
               for k in range(hits, n + 1))

    print(f'\ncells surveyed        {n}')
    print(f'hit rule              median price IC > 0 and >= {MIN_FOLD_AGREEMENT}/{N_FOLDS} folds agree')
    print(f'hits observed         {hits}')
    print(f'hits expected (null)  {expected:.1f}   (P(>= {MIN_FOLD_AGREEMENT}/{N_FOLDS}) = {NULL_HIT_RATE:.3f})')
    print(f'P(>= {hits} hits | null)   {tail:.3f}')
    if tail > 0.05:
        print('\nVERDICT: the hit count is what noise produces. No cell here is '
              'evidence of directional skill.')
    else:
        print('\nVERDICT: more hits than the null expects. Take the hits to a '
              'walk-forward backtest; this survey clears no promotion gate.')

    usable = frame.dropna(subset=['ic_share_of_required']) if 'ic_share_of_required' in frame else frame
    if not usable.empty:
        best = usable.loc[usable['ic_share_of_required'].idxmax()]
        print(f"\nclosest any cell came to paying for itself: "
              f"h={int(best['horizon'])}h {best['groups']} at "
              f"{best['ic_share_of_required']:.1%} of the IC its own round trip needs")


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--horizons', default='1,4,24',
                        help='Comma-separated forecast horizons in hours')
    parser.add_argument('--campaign', default='ic_survey')
    parser.add_argument('--ledger', default=None, help='Ledger path')
    parser.add_argument('--ledger-report', action='store_true',
                        help='Summarise the ledger for this campaign and exit')
    args = parser.parse_args()
    configure_logging(args.log_level)

    ledger = SearchLedger(args.ledger)
    if args.ledger_report:
        frame = ledger.read(args.campaign)
        if frame.empty:
            print(f'no trials on the ledger for campaign {args.campaign!r}')
            return 0
        print(f'{len(frame)} trials, {int(frame["passed"].sum())} hits')
        print(f'trial count for deflation: {ledger.trial_count(args.campaign)}')
        print(frame[['trial_id', 'passed']].to_string(index=False))
        return 0

    config = build_config(args)
    _report(_survey(args, config, ledger))
    print(f'\nledger: {ledger.path} | trials recorded for this campaign: '
          f'{ledger.trial_count(args.campaign)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
