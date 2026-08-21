"""Shared CLI plumbing: arguments every script needs, resolved one way."""

from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from core.config import (
    COST_CONFIG_SEARCH_PATHS,
    DEFAULT_COST_CONFIG_NAME,
    Config,
    find_cost_config,
)
from core.dataset import Dataset, load_dataset, report_warnings
from core.datastore import ResearchStore


def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--store', default=None, help='Research store root')
    parser.add_argument('--venue', default='coinbase', help='Venue supplying the traded price')
    parser.add_argument('--reference-venue', default='binance',
                        help='Deeper venue for basis and lead-lag; empty to disable')
    parser.add_argument('--symbols', default=None, help='Comma-separated CDE codes')
    parser.add_argument('--as-of', default=None,
                        help='Bound reads by available_time for a reproducible run')
    parser.add_argument('--min-quality', default='valid',
                        choices=['valid', 'suspicious', 'unvalidated', 'all'])
    parser.add_argument('--horizon', type=int, default=None,
                        help='Forecast horizon in hours (default: the profile hold)')
    parser.add_argument('--cost-config', default=DEFAULT_COST_CONFIG_NAME,
                        help="Venue fee schedule: a path, or a filename looked up "
                             "in configs/exchange. 'none' to use the hardcoded default.")
    parser.add_argument('--train-window-days', type=float, default=None,
                        help='Fit on the most recent N days only (default: all history)')
    parser.add_argument('--recency-half-life-days', type=float, default=None,
                        help='Exponential decay on training weights, in days. '
                             '0 disables it (default: the Config value)')
    parser.add_argument('--log-level', default=os.getenv('LOG_LEVEL', 'INFO'))
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format='%(levelname)s %(message)s')


def build_config(args: argparse.Namespace) -> Config:
    """A Config with the venue's real fee schedule loaded unless refused.

    Loading it by default is deliberate: the hardcoded 10bp/side is wrong for
    every Coinbase contract, and the previous system never loaded the file at all.
    """
    config = Config()
    if not args.cost_config or args.cost_config.lower() == 'none':
        logging.warning(
            'no cost config: pricing every contract at the hardcoded %.1fbp/side '
            'with no per-contract floor, which is wrong for every Coinbase CDE '
            'contract by 0.06x to 2.5x',
            config.fee_pct_per_side * 10_000,
        )
        return config

    path = find_cost_config(args.cost_config)
    if path is None:
        logging.error(
            'cost config not found: %s (searched %s). Falling back to the '
            'hardcoded default, which misprices every contract.',
            args.cost_config,
            ', '.join(str(d) for d in COST_CONFIG_SEARCH_PATHS),
        )
        return _with_recency(config, args)
    return _with_recency(config.with_cost_assumptions(path), args)


def _with_recency(config: Config, args: argparse.Namespace) -> Config:
    """Apply --recency-half-life-days, and say what the weighting will do.

    The half-life decides how much of a long history actually reaches the model:
    weights sum to about `24 * H / ln 2` bar-equivalents however far back the
    store goes, so past roughly 3H the extra history changes the fit very little.
    That is the intended behaviour of a decay, but it has to be visible, because
    it is easy to scrape two more years and wonder why nothing moved.
    """
    half_life = getattr(args, 'recency_half_life_days', None)
    if half_life is not None:
        config = replace(config, recency_half_life_days=float(half_life))

    if config.recency_half_life_days > 0:
        saturation_days = config.recency_half_life_days / math.log(2)
        logging.info(
            'recency half-life %.0fd: the weighted sample saturates near %.0f '
            'days of full-weight data, so history much beyond %.0fd mostly '
            'informs the folds rather than the fit',
            config.recency_half_life_days, saturation_days,
            3 * config.recency_half_life_days,
        )
    else:
        logging.info('recency weighting off: every row weighted by uniqueness alone')
    return config


def load(args: argparse.Namespace, config: Config) -> Dataset:
    store = ResearchStore(args.store) if args.store else ResearchStore()
    symbols = [s.strip().upper() for s in args.symbols.split(',')] if args.symbols else None
    dataset = load_dataset(
        store,
        venue=args.venue,
        reference_venue=args.reference_venue or None,
        symbols=symbols,
        config=config,
        as_of=args.as_of,
        min_quality=None if args.min_quality == 'all' else args.min_quality,
        horizon_bars=args.horizon,
    )
    window = getattr(args, 'train_window_days', None)
    if window:
        dataset = dataset.trailing(window)
    report_warnings(dataset)
    return dataset


def require_data(dataset: Dataset, venue: str) -> bool:
    if dataset.features.empty:
        logging.error(
            'empty dataset. Populate the research store first:\n'
            '  python -m scripts.migrate_to_research_store --venue %s --coverage', venue
        )
        return False
    return True
