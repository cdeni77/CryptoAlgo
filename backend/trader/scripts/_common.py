"""Shared CLI plumbing: arguments every script needs, resolved one way."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from core.config import Config
from core.dataset import Dataset, load_dataset, report_warnings
from core.datastore import ResearchStore

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COST_CONFIG = REPO_ROOT / 'configs/exchange/coinbase_us_perps_cde_v202602.json'


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
    parser.add_argument('--cost-config', default=str(DEFAULT_COST_CONFIG),
                        help="Venue fee schedule. 'none' to use the hardcoded default.")
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
    if args.cost_config and args.cost_config.lower() != 'none':
        path = Path(args.cost_config)
        if path.exists():
            config = config.with_cost_assumptions(path)
        else:
            logging.warning('cost config not found: %s', path)
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
