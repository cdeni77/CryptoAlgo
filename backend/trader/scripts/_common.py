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
from core.dataset import Dataset, DatasetError, load_minute_bars
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
    data.add_argument('--complete-cases', action='store_true',
                      help='Keep only rows where the venue book, the other '
                           'venue, a ladder fit and the venue settlement are '
                           'ALL present. Fewer rows, but each is one the live '
                           'path could have traded with everything it claims.')
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
    # The band the backfilled book can actually represent. Kalshi's tick is a
    # TENTH of a cent below 10c and above 90c, and Predexon's snapshot of the
    # same book is whole cents — measured, ~91% of tail quotes are off the cent
    # grid by ~0.24c, about half the measured half-spread. Inside [0.10, 0.90]
    # the backfill is exact. So economics from backfilled quotes are only
    # quantisation-free in that band, and until now it could not be set.
    # Which offsets may OPEN a position, as against which are SCORED. Only
    # `scripts/live.py` had this flag, defaulting to 12, so every backtest ran
    # `entry_offsets=None` — the first-clear policy — while live ran wait_12.
    # Measured over 70 days, per contract: first_clear 0.040c (t=0.10) against
    # 3.304c at +12m alone (t=5.98). The gates were describing the weakest of
    # the three policies and none of them was the deployed one.
    #
    # The default stays None so past ledger entries keep their meaning.
    # Sizing off the running balance instead of the starting bankroll.
    #
    # OFF by default, and the default is the load-bearing part: additive sizing
    # makes the equity curve's slope the per-trade EDGE, while compounding makes
    # it an exponential of the ESTIMATE of that edge — which is dominated by the
    # error in the estimate, and is how an earlier incarnation of this repo
    # turned $100 into $2e17 and reported it as a return.
    #
    # The flag exists so the alternative can be MEASURED. "Never tried" and
    # "tried and rejected" are different claims and only the second is worth
    # holding. Note the money gates change meaning under it: total_return,
    # sharpe and max_drawdown are all computed on the compounded curve, and
    # `sharpe_implausible` will fire on arithmetic rather than on a bug.
    economics.add_argument('--compound', action='store_true', default=None,
                           help='size off the running balance (default: off, '
                                'so the curve stays additive and readable)')
    economics.add_argument('--entry-offsets', type=int, nargs='+', default=None)
    economics.add_argument('--min-traded-price', type=float, default=None)
    economics.add_argument('--max-traded-price', type=float, default=None)
    model.add_argument('--init-score-source', choices=('baseline', 'market'),
                       default=None,
                       help="Which forecaster the correction is fitted on top "
                            "of. 'market' inverts the null in the right "
                            "direction: an untrained model reproduces the PRICE, "
                            "so model_minus_market >= 0 unless the trees hurt. "
                            "Needs a quote on every row — pair with "
                            "--complete-cases.")
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
        ('compound', 'compound'),
        ('entry_offsets', 'entry_offsets'),
        ('min_traded_price', 'min_traded_price'),
        ('max_traded_price', 'max_traded_price'),
        ('init_score_source', 'init_score_source'),
        ('min_edge_pp', 'min_edge_pp'), ('half_spread_cents', 'half_spread_cents'),
        ('assume_maker', 'assume_maker'),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            # argparse `nargs='+'` yields a list; Config's fields are tuples so
            # that a run's provenance hashes and compares equal across processes.
            overrides[field] = tuple(value) if isinstance(value, list) else value

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
    # The venue's own book and the ladder fits, where they exist. Both are
    # OBSERVED rather than fitted, so they attach with the panel; passing them
    # as None simply leaves the columns NaN, which is the correct state for the
    # four and a half years that predate the venue.
    depth = ladder_fits = None
    try:
        depth = store.read('venue_depth')
    except Exception as exc:                                  # noqa: BLE001
        logger.info('no venue_depth (%s); rows will price against the baseline',
                    str(exc)[:60])
    try:
        ladder_fits = store.read('venue_implied_vol')
    except Exception as exc:                                  # noqa: BLE001
        logger.info('no venue_implied_vol (%s)', str(exc)[:60])
    if depth is not None and len(depth):
        logger.info('venue_depth: %s rows', f'{len(depth):,}')
    if ladder_fits is not None and len(ladder_fits):
        logger.info('venue_implied_vol: %s fits', f'{len(ladder_fits):,}')

    # The venue's OWN settlement, where we hold it. The market comparison has to
    # grade both forecasters on the label the market was priced against:
    # measured, `baseline_minus_market` reads +0.00382 on our Coinbase label and
    # -0.00245 on the venue's, because our label and our baseline share a data
    # source the market does not.
    settlements = None
    try:
        settlements = store.read('venue_settlements')
    except Exception:                                         # noqa: BLE001
        settlements = None

    dataset = Dataset.build(
        bars, config, depth=depth, ladder_fits=ladder_fits
    ).trailing(config.train_window_days)

    if settlements is not None and len(settlements) and 'settled_up' in settlements.columns:
        venue = settlements[['symbol', 'window_open', 'settled_up']].dropna()
        venue = venue.drop_duplicates(['symbol', 'window_open'])
        venue['window_open'] = pd.to_datetime(venue['window_open'], utc=True)
        venue = venue.rename(columns={'settled_up': 'venue_outcome'})
        venue['venue_outcome'] = venue['venue_outcome'].astype(float)
        before = len(dataset.windows)
        dataset.windows['window_open'] = pd.to_datetime(
            dataset.windows['window_open'], utc=True)
        dataset.windows = dataset.windows.merge(
            venue, on=['symbol', 'window_open'], how='left')
        assert len(dataset.windows) == before, 'venue label join changed the row count'
        got = int(dataset.windows['venue_outcome'].notna().sum())
        logger.info("venue settlements joined: %s rows carry the venue's own label",
                    f'{got:,}')
    coverage = dataset.coverage()
    logger.info('coverage:\n%s', coverage.to_string(index=False))
    worst = coverage['boundary_drop_rate'].max() if len(coverage) else 0.0
    if worst > 0.02:
        logger.warning(
            'up to %.2f%% of windows are dropped for a missing boundary minute. '
            'Each missing minute kills two windows — the one settling on it and '
            'the one opening on it — so this is twice the gap rate.', worst * 100)
    if getattr(args, 'complete_cases', False):
        from core.dataset import complete_cases
        before = len(dataset.windows)
        dataset.windows = complete_cases(dataset.windows)
        logger.info('complete cases only: %s of %s rows kept (%.1f%%)',
                    f'{len(dataset.windows):,}', f'{before:,}',
                    len(dataset.windows) / max(before, 1) * 100)
        if not len(dataset.windows):
            raise DatasetError(
                'no rows carry every field; the book, the peer venue, a ladder '
                'fit and the venue settlement must all be present')

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
