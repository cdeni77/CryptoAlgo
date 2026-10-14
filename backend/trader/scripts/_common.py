"""Arguments shared by the research scripts, so no two can disagree about the data.

The returned `Config` records which fields the command line moved. The
operational scripts define their own arguments.
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
    """Add every argument that changes which rows a run sees or how it treats them."""
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
    data.add_argument('--quote-sources', type=str, default=None,
                      help='Comma-separated venue_depth observers allowed to '
                           'price a row, e.g. live_touch,live_ws,live. Default: '
                           'every source, ranked by QUOTE_SOURCE_PRIORITY.')
    data.add_argument('--null-quote-age', choices=('fresh', 'stale'),
                      default=None,
                      help="'fresh' (default) keeps a quote with no recorded "
                           "age; 'stale' drops it, after bounding live_touch "
                           "ages by available_time - event_time.")
    data.add_argument('--peer-max-quote-age-seconds', type=float, default=None,
                      metavar='S',
                      help='staleness cap for the cross-venue PEER book only '
                           '(default: follow the Kalshi cap). Setting it also '
                           'applies --null-quote-age to the peer.')
    data.add_argument('--decision-latency-seconds', type=float, default=None,
                      metavar='S',
                      help='seconds after the minute that live reads the book '
                           '(default 3). A book seen later is not used.')
    data.add_argument('--start', type=str, default=None, help='ISO date, inclusive')
    data.add_argument('--end', type=str, default=None, help='ISO date, exclusive')
    data.add_argument('--train-window-days', type=float, default=None,
                      help='Hard cut to the last N days of windows. Distinct from '
                           '--recency-half-life-days, which is a soft weighting; '
                           'both are useful and neither substitutes for the other.')
    data.add_argument('--research-store', type=str, default=None,
                      help='Override RESEARCH_STORE')
    data.add_argument('--holdout-start', type=str, default=None, metavar='DATE',
                      help='Seal every window opening at or after this ISO date. '
                           'No research script sees it until --unseal-holdout.')
    data.add_argument('--unseal-holdout', type=str, default=None, metavar='REASON',
                      help='Read the sealed holdout. Recorded in the trial log, the '
                           'ledger and models/holdout_unsealed.jsonl; refused on a '
                           'second unseal of the same date without --force-unseal.')
    data.add_argument('--force-unseal', action='store_true',
                      help='Allow a repeat unseal of a holdout already read. '
                           'Separate from promote --force, which installs past gates.')

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
    model.add_argument('--fold-scheme', choices=('calendar', 'count'),
                       default=None,
                       help="how the walk-forward timeline is cut: 'calendar' "
                            "equal spans of time (default), 'count' equal "
                            "numbers of windows (what entries before "
                            "2026-09-03 used)")
    model.add_argument('--fold-block-days', type=float, default=None,
                       help='calendar block length in days (default 21). It is '
                            'the test span AND the step, since blocks do not '
                            'overlap, so 7 makes the walk-forward match the '
                            "weekly retrain's cadence")
    model.add_argument('--max-residual-scale', type=float, default=None,
                       metavar='A',
                       help='ceiling on residual_scale (default 2.0). 1.0 lets '
                            'the shrinkage discount a correction but never '
                            'amplify it -- measured, alpha exceeded 1 on 19 of '
                            '28 weekly refits and hit the 2.0 ceiling once.')
    model.add_argument('--shrinkage-focus-quantile', type=float, default=None,
                       metavar='Q',
                       help='fit residual_scale on rows whose |correction| is '
                            'in the top 1-Q (e.g. 0.90 = top decile), the subset '
                            'that becomes trades. 0 (default) fits on every row, '
                            'which optimises the average -- and the average is '
                            'already fine while the traded rows are 6.7pp '
                            'overconfident.')
    model.add_argument('--ensemble-size', type=int, default=None, metavar='K',
                       help='average K boosters (default 1); their spread '
                            'feeds --edge-lower-bound-z and '
                            '--require-member-agreement')
    model.add_argument('--stack-mode', choices=('off', 'spot_then_market'),
                       default=None,
                       help="'spot_then_market': stage 1 learns spot -> "
                            "probability on every bar, stage 2 fits "
                            "a*logit(market) + b*(logit(spot) - logit(market)) "
                            "only on rows with a usable quote")
    model.add_argument('--stack-spot-groups', type=str, default=None,
                       help='comma-separated stage-1 feature groups (default: '
                            + ','.join(DEFAULT_CONFIG.stack_spot_groups) + ')')
    model.add_argument('--stack-book-features', type=str, default=None,
                       help='comma-separated book columns entering stage 2 '
                            'linearly (L2-penalised, standardized)')
    model.add_argument('--stack-l2', type=float, default=None)
    model.add_argument('--stack-crossfit-blocks', type=int, default=None,
                       help='contiguous blocks used to give stage-2 rows an '
                            'out-of-sample stage-1 prediction (default 4)')
    model.add_argument('--stack-intercept', action='store_true', default=None,
                       help='let stage 2 fit an intercept c in [-0.5, 0.5]')
    model.add_argument('--label-source', choices=('proxy', 'venue'),
                       default=None,
                       help="'venue' trains on the venue's own settlement "
                            "where held, the Coinbase proxy elsewhere")
    economics.add_argument('--edge-lower-bound-z', type=float, default=None,
                           metavar='Z',
                           help='also require the edge to clear the gate at the '
                                'probability minus Z ensemble standard deviations')
    economics.add_argument('--require-member-agreement', action='store_true',
                           default=None,
                           help='refuse unless every ensemble member still '
                                'beats the cost on the chosen side')
    model.add_argument('--fold-train-days', type=float, default=None,
                       help='roll the TRAINING window to this many days '
                            '(default: expanding, which is what the Sunday '
                            'retrain deploys). Changing it here without '
                            'matching it in the retrain evaluates one system '
                            'and trades another')
    # Off by default: additive sizing keeps the equity curve's slope equal to
    # the per-trade edge. Under compounding the money gates read the
    # compounded curve.
    economics.add_argument('--compound', action='store_true', default=None,
                           help='size off the running balance (default: off, '
                                'so the curve stays additive and readable)')
    # Which offsets may open a position; every offset is still scored. The
    # default None keeps past ledger entries' meaning.
    economics.add_argument('--entry-offsets', type=int, nargs='+', default=None)
    # The backfilled book is exact only inside [0.10, 0.90]: outside it the
    # venue tick is a tenth of a cent and the backfill rounds to whole cents.
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
    for field, attr in (('quote_sources', 'quote_sources'),
                        ('stack_spot_groups', 'stack_spot_groups'),
                        ('stack_book_features', 'stack_book_features')):
        listed = _parse_list(getattr(args, attr, None))
        if listed:
            overrides[field] = listed
    for field, attr in (
        ('venue', 'venue'), ('window_minutes', 'window_minutes'),
        ('baseline_distribution', 'baseline_distribution'), ('n_folds', 'n_folds'),
        ('embargo_minutes', 'embargo_minutes'),
        ('recency_half_life_days', 'recency_half_life_days'),
        ('train_window_days', 'train_window_days'),
        ('starting_bankroll', 'bankroll'), ('kelly_fraction', 'kelly_fraction'),
        ('fold_scheme', 'fold_scheme'),
        ('fold_block_days', 'fold_block_days'),
        ('fold_train_days', 'fold_train_days'),
        ('shrinkage_focus_quantile', 'shrinkage_focus_quantile'),
        ('max_residual_scale', 'max_residual_scale'),
        ('compound', 'compound'),
        ('entry_offsets', 'entry_offsets'),
        ('min_traded_price', 'min_traded_price'),
        ('max_traded_price', 'max_traded_price'),
        ('init_score_source', 'init_score_source'),
        ('min_edge_pp', 'min_edge_pp'), ('half_spread_cents', 'half_spread_cents'),
        ('assume_maker', 'assume_maker'),
        ('ensemble_size', 'ensemble_size'),
        ('label_source', 'label_source'),
        ('null_quote_age', 'null_quote_age'),
        ('peer_max_quote_age_seconds', 'peer_max_quote_age_seconds'),
        ('decision_latency_seconds', 'decision_latency_seconds'),
        ('stack_mode', 'stack_mode'), ('stack_l2', 'stack_l2'),
        ('stack_crossfit_blocks', 'stack_crossfit_blocks'),
        ('stack_intercept', 'stack_intercept'),
        ('edge_lower_bound_z', 'edge_lower_bound_z'),
        ('require_member_agreement', 'require_member_agreement'),
        ('holdout_start', 'holdout_start'),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            # argparse yields a list; Config fields are tuples so provenance
            # hashes and compares equal across processes.
            overrides[field] = tuple(value) if isinstance(value, list) else value

    config = DEFAULT_CONFIG.with_overrides(**overrides)
    path = Path(args.fee_config) if getattr(args, 'fee_config', None) else find_fee_config()
    if getattr(args, 'fee_config', None) and not path.exists():
        raise SystemExit(f'fee config not found: {path}')
    return config.with_fee_assumptions(path)


def groups_from_args(args: argparse.Namespace) -> Optional[tuple[str, ...]]:
    return _parse_list(getattr(args, 'groups', None))


def apply_complete_cases(args: argparse.Namespace, dataset: Dataset) -> Dataset:
    """Keep only rows carrying every field this config's groups use.

    The sample depends on the group set, so configs with different groups are
    not scored on identical rows. A sweep that ranks configs should request the
    union of groups.
    """
    from core.dataset import complete_cases
    logger = logging.getLogger('dataset')
    before = len(dataset.windows)
    dataset.windows = complete_cases(dataset.windows, groups_from_args(args))
    logger.info('complete cases only: %s of %s rows kept (%.1f%%)',
                f'{len(dataset.windows):,}', f'{before:,}',
                len(dataset.windows) / max(before, 1) * 100)
    if not len(dataset.windows):
        raise DatasetError(
            'no rows carry every field the requested groups need; the book, the '
            'peer venue, a ladder fit and the venue settlement are the four '
            'requirements and at least one is absent everywhere')
    return dataset


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def resolve_holdout(args: argparse.Namespace, config: Config) -> dict:
    """Decide whether this run may read past `config.holdout_start`.

    Records an unseal (once per date, refusing a repeat without
    `--force-unseal`) and returns the state the trial log and ledger carry.
    """
    reason = getattr(args, 'unseal_holdout', None)
    start = getattr(config, 'holdout_start', None)
    if start is None:
        if reason:
            raise SystemExit('--unseal-holdout without --holdout-start: there is '
                             'no seal to lift')
        return {'start': None, 'sealed': False}
    if not reason:
        return {'start': str(start), 'sealed': True}
    from core.promotion import HoldoutError, unseal_holdout

    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else ''
    try:
        record = unseal_holdout(str(start), reason,
                                force=bool(getattr(args, 'force_unseal', False)),
                                script=script)
    except HoldoutError as exc:
        raise SystemExit(str(exc)) from exc
    logging.getLogger('dataset').warning(
        'HOLDOUT UNSEALED from %s: %s', start, reason)
    return {'start': str(start), 'sealed': False, 'unsealed': record}


def load_dataset(args: argparse.Namespace, config: Config) -> Dataset:
    """Read the store, lay the window grid, and report coverage.

    This is the one place the sealed holdout is enforced. Bars are cut at
    `holdout_start`, and windows opening at or after it are dropped.
    """
    logger = logging.getLogger('dataset')
    holdout = resolve_holdout(args, config)
    args.holdout_state = holdout
    cutoff = _utc(holdout['start']) if holdout.get('sealed') else None
    end = _utc(args.end) if getattr(args, 'end', None) else None
    if cutoff is not None:
        end = cutoff if end is None else min(end, cutoff)
        logger.info('holdout sealed: nothing at or after %s is read', cutoff)
    store = ResearchStore()
    bars = load_minute_bars(
        config, store=store,
        start=_utc(args.start) if getattr(args, 'start', None) else None,
        end=end,
    )
    # The venue's book and the ladder fits, where they exist. Both are observed,
    # not fitted, so they attach with the panel. None leaves the columns NaN.
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

    # The venue's own settlement, where we hold it. The market comparison grades
    # both forecasters on the label the market was priced against, because our
    # label and our baseline share a data source the market does not.
    settlements = None
    try:
        settlements = store.read('venue_settlements')
    except Exception:                                         # noqa: BLE001
        settlements = None

    # Read the index proxy only when its group is asked for, because it is large.
    index_proxy = None
    if 'index_proxy' in (groups_from_args(args) or ()):
        try:
            from core.index_features import load_proxy_minutes
            index_proxy = load_proxy_minutes(store.root, end=end)
            logger.info('index_proxy_1s: %s minute summaries',
                        f'{len(index_proxy):,}')
        except Exception as exc:                              # noqa: BLE001
            logger.warning('no index_proxy_1s (%s); the group will be NaN',
                           str(exc)[:80])

    dataset = Dataset.build(
        bars, config, depth=depth, ladder_fits=ladder_fits,
        index_proxy=index_proxy,
    ).trailing(config.train_window_days)
    if cutoff is not None and len(dataset.windows):
        opens = pd.to_datetime(dataset.windows['window_open'], utc=True)
        dataset.windows = dataset.windows.loc[(opens < cutoff).to_numpy()].reset_index(drop=True)

    if settlements is not None and len(settlements) and 'settled_up' in settlements.columns:
        # Kalshi only, newest revision. The table also holds Polymarket rows,
        # which settle on a different oracle.
        venue = settlements
        if 'venue' in venue.columns:
            venue = venue[venue['venue'] == 'kalshi']
        keep = ['symbol', 'window_open', 'settled_up']
        if 'available_time' in venue.columns:
            keep = keep + ['available_time']
        venue = venue[keep].dropna(subset=['symbol', 'window_open', 'settled_up'])
        if 'available_time' in venue.columns:
            venue = venue.sort_values('available_time')
        venue = venue.drop_duplicates(['symbol', 'window_open'], keep='last')
        venue = venue[['symbol', 'window_open', 'settled_up']]
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
    log_quote_coverage(depth, dataset, config)
    coverage = dataset.coverage()
    logger.info('coverage:\n%s', coverage.to_string(index=False))
    worst = coverage['boundary_drop_rate'].max() if len(coverage) else 0.0
    if worst > 0.02:
        logger.warning(
            'up to %.2f%% of windows are dropped for a missing boundary minute. '
            'Each missing minute kills two windows — the one settling on it and '
            'the one opening on it — so this is twice the gap rate.', worst * 100)
    if getattr(args, 'complete_cases', False):
        if getattr(config, 'stack_mode', 'off') != 'off':
            raise SystemExit(
                '--complete-cases contradicts --stack-mode: stage 1 is fitted '
                'on every bar, and stage 2 already admits only rows carrying a '
                'usable quote. Filter the quotes with --quote-sources and '
                '--null-quote-age instead.')
        apply_complete_cases(args, dataset)

    logger.info('%s windows, %s rows, %.0f days',
                f'{len(dataset.window_index):,}', f'{len(dataset.windows):,}',
                dataset.span_days)
    return dataset


def log_quote_coverage(depth, dataset: Dataset, config: Config) -> dict:
    """Log which quote observers the store holds, their ages, and which priced rows."""
    from core.quotes import quote_coverage, source_age_table

    logger = logging.getLogger('dataset')
    if depth is not None and len(depth) and 'venue' in depth.columns:
        raw = source_age_table(depth[depth['venue'] == 'kalshi'])
        if len(raw):
            logger.info('venue_depth (kalshi) rows by source and recorded age:\n%s',
                        raw.to_string(index=False))
    coverage = quote_coverage(dataset.windows)
    logger.info('quotes priced %s of %s rows (%.1f%%) under sources=%s, '
                'max age %.0fs (peer %s), null age %s',
                f"{coverage['quoted']:,}", f"{coverage['rows']:,}",
                100 * coverage['share'],
                ','.join(config.quote_sources) or 'all',
                float(config.max_quote_age_seconds),
                'follows' if config.peer_max_quote_age_seconds is None
                else f'{config.peer_max_quote_age_seconds:.0f}s',
                config.null_quote_age)
    if coverage['by_source_age']:
        logger.info('priced rows by observer and age:\n%s',
                    pd.DataFrame(coverage['by_source_age']).to_string(index=False))
    return coverage


def data_span(dataset: Optional[Dataset], args: Optional[argparse.Namespace] = None) -> dict:
    """Return the span the run read, for the trial log."""
    span: dict = {}
    if args is not None:
        span.update({'start_arg': getattr(args, 'start', None),
                     'end_arg': getattr(args, 'end', None),
                     'complete_cases': bool(getattr(args, 'complete_cases', False))})
    if dataset is not None:
        index = dataset.window_index
        span.update({
            'first_window': str(index.min()) if len(index) else None,
            'last_window': str(index.max()) if len(index) else None,
            'windows': int(len(index)), 'rows': int(len(dataset.windows)),
            'symbols': list(dataset.symbols),
        })
    return span


def log_trial(script: str, args: argparse.Namespace, config: Config, *,
              groups: Optional[Sequence[str]] = None,
              dataset: Optional[Dataset] = None, report=None,
              metrics: Optional[dict] = None, trial_id: Optional[str] = None,
              extra: Optional[dict] = None) -> dict:
    """Append one line for this run to `trials.jsonl`."""
    from core.promotion import record_trial, trial_metrics

    return record_trial(
        script, trial_id=trial_id, config=config, groups=groups,
        data_span=data_span(dataset, args),
        metrics=trial_metrics(report, metrics),
        holdout=getattr(args, 'holdout_state', None), extra=extra)


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
    if config.quote_sources or config.null_quote_age != 'fresh':
        print(f'quotes            sources {",".join(config.quote_sources) or "all"}, '
              f'null age {config.null_quote_age}, max age '
              f'{config.max_quote_age_seconds:.0f}s')
    if config.stack_mode != 'off':
        print(f'stack             {config.stack_mode}: stage 1 on '
              f'{",".join(config.stack_spot_groups)}; stage 2 book terms '
              f'{",".join(config.stack_book_features) or "none"}, '
              f'{config.stack_crossfit_blocks} cross-fit blocks')
    if config.cli_overrides:
        print(f'cli overrides     {", ".join(sorted(config.cli_overrides))}')
    if getattr(config, 'holdout_start', None):
        print(f'holdout           sealed from {config.holdout_start}')
    print()
