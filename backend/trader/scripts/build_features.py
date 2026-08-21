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


def _load_symbol(
    store: ResearchStore,
    symbol: str,
    *,
    venue: str,
    reference_venue: Optional[str],
    market_bars: Optional[pd.DataFrame],
    as_of: Optional[str],
    min_quality: Optional[str],
) -> Optional[SymbolInputs]:
    """Assemble one instrument's inputs, or None if it has no bars."""

    def frame(dataset: str, source_venue: str, index_only: bool = False) -> pd.DataFrame:
        rows = store.read(
            dataset, venue=source_venue, symbols=[symbol],
            as_of=as_of, min_quality=min_quality,
        )
        if rows.empty:
            return pd.DataFrame()
        rows = rows.set_index(pd.to_datetime(rows['event_time'], utc=True)).sort_index()
        return rows.drop(columns=[c for c in ('event_time', 'symbol', 'venue') if c in rows])

    bars = frame('bars', venue)
    if bars.empty:
        logger.warning('%s: no bars on %s', symbol, venue)
        return None

    funding = frame('funding', venue)
    open_interest = frame('open_interest', venue)

    # Open interest has no Coinbase-native source, so it may only exist under a
    # proxy venue. Fall back rather than dropping the positioning group.
    if open_interest.empty and reference_venue:
        open_interest = frame('open_interest', reference_venue)

    reference_bars = frame('bars', reference_venue) if reference_venue else pd.DataFrame()

    return SymbolInputs(
        symbol=symbol,
        bars=bars,
        funding=funding if not funding.empty else None,
        open_interest=open_interest if not open_interest.empty else None,
        reference_bars=reference_bars if not reference_bars.empty else None,
        market_bars=market_bars,
    )


def build(
    store: ResearchStore,
    *,
    venue: str,
    reference_venue: Optional[str],
    symbols: Sequence[str],
    config: Config,
    as_of: Optional[str] = None,
    min_quality: Optional[str] = 'valid',
    groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Feature panel for `symbols`, MultiIndexed by (event_time, symbol)."""
    market = store.read(
        'bars', venue=venue, symbols=[MARKET_SYMBOL], as_of=as_of, min_quality=min_quality,
    )
    if market.empty:
        logger.warning(
            '%s has no bars on %s — the market_factor group will be empty, so no '
            'instrument can express itself relative to the market',
            MARKET_SYMBOL, venue,
        )
        market_bars = None
    else:
        market_bars = market.set_index(
            pd.to_datetime(market['event_time'], utc=True)
        ).sort_index()

    inputs = []
    for symbol in symbols:
        item = _load_symbol(
            store, symbol, venue=venue, reference_venue=reference_venue,
            market_bars=market_bars, as_of=as_of, min_quality=min_quality,
        )
        if item is not None:
            inputs.append(item)

    if not inputs:
        return pd.DataFrame()

    return build_panel(inputs, config=config, groups=groups)


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', default=None, help='Research store root')
    parser.add_argument('--venue', default='coinbase',
                        help='Venue supplying the traded price')
    parser.add_argument('--reference-venue', default='binance',
                        help='Deeper venue for basis and lead-lag. Empty to disable.')
    parser.add_argument('--symbols', default=None,
                        help='Comma-separated CDE codes (default: every profile)')
    parser.add_argument('--as-of', default=None,
                        help='Bound reads by available_time for a reproducible build')
    parser.add_argument('--min-quality', default='valid',
                        choices=['valid', 'suspicious', 'unvalidated', 'all'],
                        help='Lowest quality to include (default: valid only)')
    parser.add_argument('--groups', default=None,
                        help=f"Comma-separated subset of: {', '.join(g.name for g in GROUPS)}")
    parser.add_argument('--cost-config', default=None,
                        help='configs/exchange/*.json for the real fee schedule')
    parser.add_argument('--name', default='panel', help='Feature matrix name')
    parser.add_argument('--list', action='store_true',
                        help='Print the feature set and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build and report without writing')
    args = parser.parse_args()

    if args.list:
        for group in GROUPS:
            from core.features import _group_column_names
            columns = _group_column_names(group)
            flag = 'cross-sectional' if group.standardize else 'absolute'
            print(f"\n{group.name}  ({len(columns)} features, {flag})")
            for column in columns:
                print(f"  {column}")
        print(f"\ntotal: {len(feature_columns())}")
        return 0

    config = Config()
    if args.cost_config:
        config = config.with_cost_assumptions(args.cost_config)
    else:
        logger.warning(
            'no --cost-config: using the %.4f%%/side default rather than the '
            "venue's per-contract schedule, so fee_hurdle_bps will be wrong",
            config.fee_pct_per_side * 100,
        )

    store = ResearchStore(args.store) if args.store else ResearchStore()
    symbols = (
        [s.strip().upper() for s in args.symbols.split(',')]
        if args.symbols
        else [profile.prefixes[0] for profile in COIN_PROFILES.values()]
    )
    groups = [g.strip() for g in args.groups.split(',')] if args.groups else None

    panel = build(
        store,
        venue=args.venue,
        reference_venue=args.reference_venue or None,
        symbols=symbols,
        config=config,
        as_of=args.as_of,
        min_quality=None if args.min_quality == 'all' else args.min_quality,
        groups=groups,
    )

    if panel.empty:
        logger.error(
            'empty panel — is the research store populated? '
            'Run: python -m scripts.migrate_to_research_store --venue %s', args.venue
        )
        return 1

    _report(panel, config, symbols)

    if args.dry_run:
        print('\n--dry-run: nothing written')
        return 0

    path, digest = store.write_features(
        panel,
        name=args.name,
        meta={
            'venue': args.venue,
            'reference_venue': args.reference_venue,
            'symbols': symbols,
            'as_of': args.as_of,
            'min_quality': args.min_quality,
            'groups': groups or [g.name for g in GROUPS],
            'cost_config': config.cost_config_version,
        },
    )
    print(f"\nwrote {path}\nhash {digest}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
