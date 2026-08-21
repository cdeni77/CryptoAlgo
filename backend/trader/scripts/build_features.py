"""Build the feature panel from the research store.

This is the step that makes `core.features` real: it reads venue-keyed,
point-in-time bars, funding and open interest out of the Parquet store, assembles
the nine mechanism groups per instrument, standardises the relative ones across
the universe, and materialises the result with a content hash.

    python -m scripts.build_features --list
    python -m scripts.build_features --venue coinbase
    python -m scripts.build_features --venue coinbase --as-of 2026-06-01

`--as-of` is what makes a build reproducible: it bounds every read by
`available_time`, so rebuilding a matrix for a past date gives the data as it
stood then, not as it was later revised.

Replaces the feature half of `scripts/compute_features.py`, which wrote per-coin
CSVs from `features/engineering.py`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

import pandas as pd

from core.config import Config
from core.costs import symbols_missing_fee_schedule
from core.datastore import ResearchStore
from core.features import GROUPS, SymbolInputs, build_panel, feature_columns
from core.profiles import COIN_PROFILES

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'),
                    format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# The instrument whose returns define the market factor.
MARKET_SYMBOL = 'BIP'


def build(args, config):
    """Assemble the panel through the shared loader.

    This script previously carried its own copy of the loading logic, which is
    the duplication pattern the rebuild exists to remove. `core.dataset` is the
    one place that turns "which venue, which symbols, as of when" into a panel,
    so training, backtesting and live signals cannot disagree about the data.
    """
    from scripts._common import load
    return load(args, config)


def _report(panel: pd.DataFrame, config: Config, symbols: Sequence[str]) -> None:
    print(f"\npanel: {panel.shape[0]:,} rows x {panel.shape[1]} features")
    timestamps = panel.index.get_level_values('event_time')
    print(f"span:  {timestamps.min()} -> {timestamps.max()}")

    per_symbol = panel.groupby(level='symbol').size().sort_values(ascending=False)
    print(f"\nrows per instrument ({len(per_symbol)} present):")
    for symbol, rows in per_symbol.items():
        print(f"  {symbol:6s} {rows:>7,}")

    coverage = 1.0 - panel.isna().mean()
    thin = coverage[coverage < 0.5].sort_values()
    if not thin.empty:
        print(f"\nfeatures under 50% coverage ({len(thin)}):")
        for name, value in thin.items():
            print(f"  {name:34s} {value:.1%}")
        print("  (a group whose source data is absent yields empty columns)")

    missing = symbols_missing_fee_schedule(symbols, config)
    if missing:
        print(
            f"\nno explicit fee schedule for {', '.join(missing)} — "
            f"falling back to ${config.min_fee_per_contract:.2f}/contract, "
            "which understates profitability rather than overstating it"
        )


def main() -> int:
    from scripts._common import add_data_arguments, build_config, configure_logging, require_data

    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--name', default='panel', help='Feature matrix name')
    parser.add_argument('--list', action='store_true', help='Print the feature set and exit')
    parser.add_argument('--dry-run', action='store_true', help='Build and report without writing')
    args = parser.parse_args()
    configure_logging(args.log_level)

    # `--groups` used to be declared here and never read, so a "restricted" build
    # produced the full panel and recorded itself as complete. It is gone rather
    # than wired: the panel comes from `core.dataset.load_dataset`, the one place
    # that turns "which venue, which symbols, as of when" into features, and a
    # subset panel built here would be a different feature set from the one
    # training and signals expect. `--list` prints the groups; that is the use
    # the flag was reaching for.

    if args.list:
        for group in GROUPS:
            from core.features import _group_column_names
            columns = _group_column_names(group)
            flag = 'cross-sectional' if group.standardize else 'absolute'
            print(f'\n{group.name}  ({len(columns)} features, {flag})')
            for column in columns:
                print(f'  {column}')
        print(f'\ntotal: {len(feature_columns())}')
        return 0

    config = build_config(args)
    dataset = build(args, config)
    if not require_data(dataset, args.venue):
        return 1

    panel = dataset.features
    _report(panel, config, dataset.symbols)

    if args.dry_run:
        print('\n--dry-run: nothing written')
        return 0

    store = ResearchStore(args.store) if args.store else ResearchStore()
    path, digest = store.write_features(
        panel, name=args.name,
        meta={
            'venue': args.venue,
            'reference_venue': args.reference_venue,
            'symbols': dataset.symbols,
            'as_of': args.as_of,
            'min_quality': args.min_quality,
            'groups': [g.name for g in GROUPS],
            'cost_config': config.cost_config_version,
            'horizon_bars': dataset.horizon_bars,
            'warnings': dataset.warnings,
        },
    )
    print(f'\nwrote {path}\nhash {digest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
