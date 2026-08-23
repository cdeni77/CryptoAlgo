"""Arguments every script shares, so no two scripts can disagree about the data.

Seven scripts read the research store, and every one of them can change an
answer by loading a different universe, a different span, a different offset
set or a different fee schedule. They take those arguments from here, so a run
cannot silently differ from the run it is being compared against — and the
`Config` that comes back records which fields the command line moved.

The operational scripts (`scrape`, `sync_store`, `paper`, `orchestrator`) take a
different set and roll their own, because they are about moving bytes rather
than about measurement.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from core.config import Config, DEFAULT_CONFIG, find_fee_config
from core.dataset import Dataset, load_minute_bars
from core.datastore import ResearchStore
from core.features import ALL_GROUPS


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-22s %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
    )
    for noisy in ('urllib3', 'matplotlib', 'numba'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Every argument that changes which rows a run sees, or how it treats them."""
    data = parser.add_argument_group('data')
    data.add_argument('--symbols', type=str, default=None,
                      help='Comma-separated spot products. Default: '
                           + ','.join(DEFAULT_CONFIG.symbols))
    data.add_argument('--venue', type=str, default=None,
                      help=f'Research-store venue label (default {DEFAULT_CONFIG.venue})')
    data.add_argument('--start', type=str, default=None, help='ISO date, inclusive')
    data.add_argument('--end', type=str, default=None, help='ISO date, exclusive')
    data.add_argument('--train-window-days', type=float, default=None,
                      help='Hard cut to the last N days of windows. Distinct from '
                           '--recency-half-life-days, which is a soft weighting; '
                           'both are useful and neither substitutes for the other.')
    data.add_argument('--research-store', type=str, default=None,
                      help='Override RESEARCH_STORE')

    market = parser.add_argument_group('market structure')
    market.add_argument('--window-minutes', type=int, default=None,
                        help='Kalshi crypto up/down windows are 15 minutes. Changing '
                             'this describes a different market.')
    market.add_argument('--offsets', type=str, default=None,
                        help='Comma-separated decision offsets in minutes. Default: '
                             + ','.join(str(o) for o in DEFAULT_CONFIG.decision_offsets))

    model = parser.add_argument_group('model and validation')
    model.add_argument('--groups', type=str, default=None,
                       help='Comma-separated feature groups. Available: '
                            + ','.join(ALL_GROUPS) + '. `clock` is the control — '
                            'keep it in any survey so the noise floor is visible.')
    model.add_argument('--baseline-distribution', type=str, default=None,
                       choices=['normal', 'student_t'])
    model.add_argument('--n-folds', type=int, default=None)
    model.add_argument('--embargo-minutes', type=int, default=None,
                       help='Purge on both sides of every test block. Must cover the '
                            'longest feature lookback (1440), not just the label span.')
    model.add_argument('--recency-half-life-days', type=float, default=None,
                       help='Soft exponential decay by age. Off by default: at this '
                            'window size the sample is large enough that decay costs '
                            'more than the non-stationarity it buys.')

    economics = parser.add_argument_group('economics')
    economics.add_argument('--bankroll', type=float, default=None,
                           help=f'Starting account (default ${DEFAULT_CONFIG.starting_bankroll:.0f})')
    economics.add_argument('--kelly-fraction', type=float, default=None)
    economics.add_argument('--min-edge-pp', type=float, default=None,
                           help='Surplus over break-even demanded before trading, in '
                                'probability points. Abstention is the default action.')
    economics.add_argument('--half-spread-cents', type=float, default=None,
                           help='An assumption, not a measurement — larger than the fee '
                                'at every price above 83c, so it is the parameter most '
                                'worth stressing.')
    economics.add_argument('--assume-maker', action='store_true', default=None)
    economics.add_argument('--fee-config', type=str, default=None,
                           help='Venue fee schedule JSON. Unset still prices correctly '
                                '(the defaults match the published schedule) but records '
                                'no version.')

    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def _parse_list(value: Optional[str]) -> Optional[tuple]:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(',') if part.strip())


def config_from_args(args: argparse.Namespace) -> Config:
    """Build the run's Config, recording which fields the command line moved."""
    if getattr(args, 'research_store', None):
        os.environ['RESEARCH_STORE'] = args.research_store

    overrides: dict = {}
    symbols = _parse_list(getattr(args, 'symbols', None))
    if symbols:
        overrides['symbols'] = symbols
    offsets = _parse_list(getattr(args, 'offsets', None))
    if offsets:
        overrides['decision_offsets'] = tuple(int(o) for o in offsets)
    for field, attr in (
        ('venue', 'venue'), ('window_minutes', 'window_minutes'),
        ('baseline_distribution', 'baseline_distribution'), ('n_folds', 'n_folds'),
        ('embargo_minutes', 'embargo_minutes'),
        ('recency_half_life_days', 'recency_half_life_days'),
        ('train_window_days', 'train_window_days'),
        ('starting_bankroll', 'bankroll'), ('kelly_fraction', 'kelly_fraction'),
        ('min_edge_pp', 'min_edge_pp'), ('half_spread_cents', 'half_spread_cents'),
        ('assume_maker', 'assume_maker'),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            overrides[field] = value

    config = DEFAULT_CONFIG.with_overrides(**overrides)
    path = Path(args.fee_config) if getattr(args, 'fee_config', None) else find_fee_config()
    if getattr(args, 'fee_config', None) and not path.exists():
        raise SystemExit(f'fee config not found: {path}')
    return config.with_fee_assumptions(path)


def groups_from_args(args: argparse.Namespace) -> Optional[tuple[str, ...]]:
    return _parse_list(getattr(args, 'groups', None))


def load_dataset(args: argparse.Namespace, config: Config) -> Dataset:
    """Read the store, lay the window grid, report coverage."""
    logger = logging.getLogger('dataset')
    store = ResearchStore()
    bars = load_minute_bars(
        config, store=store,
        start=pd.Timestamp(args.start, tz='UTC') if getattr(args, 'start', None) else None,
        end=pd.Timestamp(args.end, tz='UTC') if getattr(args, 'end', None) else None,
    )
    dataset = Dataset.build(bars, config).trailing(config.train_window_days)
    coverage = dataset.coverage()
    logger.info('coverage:\n%s', coverage.to_string(index=False))
    worst = coverage['boundary_drop_rate'].max() if len(coverage) else 0.0
    if worst > 0.02:
        logger.warning(
            'up to %.2f%% of windows are dropped for a missing boundary minute. '
            'Each missing minute kills two windows — the one settling on it and '
            'the one opening on it — so this is twice the gap rate.', worst * 100)
    logger.info('%s windows, %s rows, %.0f days',
                f'{len(dataset.window_index):,}', f'{len(dataset.windows):,}',
                dataset.span_days)
    return dataset


def print_header(title: str, config: Config) -> None:
    print('=' * 78)
    print(title)
    print('=' * 78)
    print(f'universe          {", ".join(config.symbols)} on {config.venue}')
    print(f'window            {config.window_minutes}min, decisions at '
          f'{", ".join(f"+{o}m" for o in config.decision_offsets)}')
    print(f'baseline          {config.baseline_distribution}'
          + (f' (nu={config.baseline_nu})' if config.baseline_nu else ' (nu fitted)'))
    print(f'fees              {config.fee_rate:.3f} x p(1-p) per contract + '
          f'{config.half_spread_cents:.1f}c half-spread [{config.fee_config_version}]')
    print(f'account           ${config.starting_bankroll:.2f}, '
          f'{config.kelly_fraction:.2f} Kelly, gate {config.min_edge_pp:.2f}pp')
    if config.cli_overrides:
        print(f'cli overrides     {", ".join(sorted(config.cli_overrides))}')
    print()
