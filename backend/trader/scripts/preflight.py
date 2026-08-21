"""Can this actually train? Answers that before you spend a night scraping.

    python -m scripts.preflight
    python -m scripts.preflight --venue coinbase --min-days 120

Runs the chain in order and stops at the first thing that is not ready, because
a downstream check on missing data reports a confusing symptom rather than the
cause. Each check prints what it measured and what it needed, so a failure names
its own fix.

    1. cost schedule loaded       — otherwise every target is priced wrong
    2. research store populated   — per dataset and venue, with spans
    3. panel builds               — features present, coverage not collapsed
    4. targets resolve            — enough rows past the horizon to learn from
    5. effective sample size      — overlapping labels are not independent rows
    6. cross-section wide enough  — relative features need a universe
    7. model trains               — the heads fit, and identity is not the edge

There is no `--fix` flag. Every failure here is either missing data or a missing
config file, and guessing at either is how a run ends up training on assumptions
nobody wrote down.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import symbols_missing_fee_schedule
from core.cv import average_uniqueness, effective_sample_size
from core.datastore import ResearchStore
from core.features import feature_columns
from scripts._common import add_data_arguments, build_config, configure_logging, load

# A pooled panel model needs enough resolved rows that the effective sample is
# not a handful of independent observations. Below this, every metric downstream
# is a coin flip with error bars wider than the effect being measured.
MIN_EFFECTIVE_OBSERVATIONS = 200

# Relative features standardise across the universe. With fewer than this many
# instruments a cross-sectional z-score is describing noise between two names.
MIN_UNIVERSE = 3

# Coverage below this on a feature means its source data is effectively absent.
MIN_FEATURE_COVERAGE = 0.5


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    fix: str = ''
    # Reported, but not a reason to withhold the rest of the report. Funding is
    # the case that forced this: CDE publishes no historical funding endpoint, so
    # a thin funding series cannot be scraped away and blocking on it made
    # preflight permanently unpassable — while suppressing the effective-sample
    # numbers the script exists to produce.
    advisory: bool = False

    def __str__(self) -> str:
        mark = 'NOTE' if self.advisory else ('PASS' if self.passed else 'FAIL')
        return f'[{mark}] {self.name}: {self.detail}'


def _cost_schedule(config: Config, symbols: list[str]) -> Check:
    if not config.cost_config_version or config.cost_config_version == 'default':
        return Check(
            'cost schedule', False,
            'running on the hardcoded 10bp/side default',
            'pass --cost-config configs/exchange/coinbase_us_perps_cde_v202602.json. '
            'The default is wrong for every Coinbase CDE contract by 0.06x-2.5x, '
            'in both directions, which inverts the ranking between coins.',
        )
    missing = symbols_missing_fee_schedule(symbols, config) if symbols else []
    if missing:
        return Check(
            'cost schedule', True,
            f'loaded {config.cost_config_version}; no explicit fee for '
            f'{", ".join(missing)} (falling back to '
            f'${config.min_fee_per_contract:.2f}/contract)',
        )
    return Check('cost schedule', True, f'loaded {config.cost_config_version}')


def _store_populated(store: ResearchStore, venue: str, min_days: int) -> Check:
    lines: list[str] = []
    shortfalls: list[str] = []

    for dataset in ('bars',):
        try:
            coverage = store.coverage(dataset)
        except Exception as exc:  # noqa: BLE001 - a missing dataset is the finding
            shortfalls.append(f'{dataset}: unreadable ({exc})')
            continue
        if coverage.empty:
            shortfalls.append(f'{dataset}: empty')
            continue

        scoped = coverage[coverage['venue'] == venue]
        if scoped.empty:
            shortfalls.append(f'{dataset}: nothing on {venue}')
            continue

        days = float(scoped['days'].max())
        lines.append(
            f'{dataset}: {len(scoped)} series, {int(scoped["rows"].sum()):,} rows, '
            f'up to {days:.0f} days'
        )
        if days < min_days:
            shortfalls.append(f'{dataset}: longest series is {days:.0f}d < {min_days}d')

        # Name the other venues in the store. The count above is scoped to the
        # traded venue, which is right — but reporting only that number makes a
        # collected reference venue invisible, and "18 series" reads like a
        # total. It is the reference venue that decides whether the cross-venue
        # group has anything to work with, so it belongs in the same line.
        others = coverage[coverage['venue'] != venue]
        if not others.empty:
            summary = ', '.join(
                f'{name} {len(group)} series/{float(group["days"].max()):.0f}d'
                for name, group in sorted(others.groupby('venue'))
            )
            lines.append(f'other venues: {summary}')

    detail = '; '.join(lines) or 'nothing in the store'
    if shortfalls:
        return Check(
            'research store', False, f'{detail} | {"; ".join(shortfalls)}',
            'scrape, then sync: python -m scripts.run_pipeline --backfill-only '
            f'--backfill-days {min_days} --timeframes 1h && '
            f'python -m scripts.migrate_to_research_store --venue {venue}',
        )
    return Check('research store', True, detail)


def _carry_and_positioning(store: ResearchStore, venue: str, min_days: int) -> Check:
    """Funding and open-interest coverage, reported rather than gated.

    Neither can be fixed by scraping harder, which is why blocking on them was
    wrong:

    * **Funding.** CDE publishes no historical endpoint — only the current rate
      on the product, with `funding_time` being the *next* settlement. It
      accumulates forward, one observation per hourly settlement, so the series
      is exactly as long as the hourly loop has been running. Telling someone to
      `--backfill-days 90` for it is advice that cannot work.
    * **Open interest.** Also a snapshot, on the same product payload under
      `future_product_details.open_interest`, so it accumulates forward exactly
      like funding. It used to come through CCXT from another exchange because
      this client had no method for it, which put gate's BTC/USDT:USDT book
      (21,579,279 contracts) behind features describing BIP's (268,164).

    Both feed real feature groups (`carry` 9 features, `positioning` 6), so their
    absence is worth stating plainly — the panel keeps its full 76 columns either
    way and the empty ones arrive as all-NaN.
    """
    notes: list[str] = []
    for dataset, groups in (('funding', 'carry'), ('open_interest', 'positioning')):
        try:
            coverage = store.coverage(dataset)
        except Exception:                                  # noqa: BLE001
            coverage = None
        if coverage is None or coverage.empty:
            notes.append(f'{dataset}: empty, so the {groups} features are all-NaN')
            continue
        scoped = coverage[coverage['venue'] == venue]
        if scoped.empty:
            notes.append(f'{dataset}: nothing on {venue}, so {groups} is all-NaN')
            continue
        days = float(scoped['days'].max())
        rows = int(scoped['rows'].sum())
        suffix = '' if days >= min_days else f' — under {min_days}d, so {groups} is thin'
        notes.append(f'{dataset}: {len(scoped)} series, {rows:,} rows, {days:.0f} days{suffix}')

    return Check(
        'carry & positioning', True, '; '.join(notes), advisory=True,
        fix='neither funding nor open interest can be backfilled on CDE — both '
            'are snapshots on the product endpoint. Start '
            '`scripts.live_orchestrator` and they accrue hourly, one observation '
            'per contract per cycle.',
    )


def _panel_builds(dataset) -> Check:
    panel = dataset.features
    if panel.empty:
        return Check('feature panel', False, 'built zero rows',
                     'check the store coverage above and the --venue argument')

    expected = feature_columns()
    missing = sorted(set(expected) - set(panel.columns))
    coverage = 1.0 - panel.isna().mean()
    thin = coverage[coverage < MIN_FEATURE_COVERAGE].sort_index()

    detail = (
        f'{len(panel):,} rows x {panel.shape[1]} features across '
        f'{len(dataset.symbols)} instruments'
    )
    if missing:
        return Check('feature panel', False, f'{detail}; {len(missing)} columns absent',
                     f'missing: {", ".join(missing[:8])}')
    if len(thin) > len(expected) // 4:
        return Check(
            'feature panel', False,
            f'{detail}; {len(thin)} features under {MIN_FEATURE_COVERAGE:.0%} coverage',
            'a whole group with no coverage means its source dataset is absent: '
            f'{", ".join(list(thin.index)[:8])}',
        )
    if len(thin):
        return Check('feature panel', True,
                     f'{detail}; {len(thin)} thin features '
                     f'({", ".join(list(thin.index)[:4])})')
    return Check('feature panel', True, detail)


def _targets_resolve(dataset) -> Check:
    resolved = len(dataset.resolved_index)
    total = len(dataset.features)
    if resolved == 0:
        return Check('targets', False, f'0 of {total:,} rows resolved',
                     f'the series is shorter than the {dataset.horizon_bars}h horizon')
    share = resolved / total if total else 0.0
    detail = f'{resolved:,} of {total:,} rows resolved ({share:.0%}), horizon {dataset.horizon_bars}h'
    if resolved < 1_000:
        return Check('targets', False, detail,
                     'a pooled model on fewer than a thousand resolved rows is '
                     'fitting noise; scrape more history')
    return Check('targets', True, detail)


def _weighted_effective(index: pd.DatetimeIndex, horizon_bars: int,
                        half_life_days: float) -> float:
    """Effective observations after the recency decay training actually applies.

    `effective_sample_size` answers "how many independent labels are in this
    span". Training then multiplies each by `0.5 ** (age / H)`, and the product
    is what the model is fitted on. The two numbers diverge sharply: the weights
    sum to about `24 * H / ln 2` bar-equivalents no matter how far back the store
    goes, so at H=50d and a 96h horizon the answer is ~18 whether you hold one
    year of history or five. Reporting only the unweighted count is how a scrape
    gets recommended that cannot change the fit.
    """
    if half_life_days <= 0:
        return effective_sample_size(index, horizon_bars)
    uniqueness = average_uniqueness(index, horizon_bars)
    age_days = (index.max() - index).total_seconds() / 86_400.0
    decay = np.power(0.5, np.asarray(age_days) / float(half_life_days))
    return float((uniqueness * decay).sum())


def _effective_sample(dataset, config: Config) -> Check:
    """Overlapping labels are not independent observations.

    A million hourly rows with a 24-hour horizon carry nowhere near a million
    observations' worth of information, and using the row count as the sample
    size is how a t-statistic ends up five times too confident.
    """
    index = pd.DatetimeIndex(
        dataset.resolved_index.get_level_values('event_time').unique()
    ).sort_values()
    if len(index) < 2:
        return Check('effective sample', False, 'fewer than two resolved timestamps')

    unweighted = effective_sample_size(index, dataset.horizon_bars)
    half_life = config.recency_half_life_days
    effective = _weighted_effective(index, dataset.horizon_bars, half_life)
    horizon = max(dataset.horizon_bars, 1)

    detail = (
        f'{effective:.0f} effective observations from {len(index):,} timestamps '
        f'(uniqueness {unweighted:.0f}, horizon {horizon}h'
    )
    detail += (f', half-life {half_life:.0f}d)' if half_life > 0
               else ', no recency decay)')

    if effective >= MIN_EFFECTIVE_OBSERVATIONS:
        return Check('effective sample', True, detail)

    # Two ways out, and both are worth quantifying rather than gesturing at:
    # more history at this horizon, or the same history at a shorter one. The
    # relationship is roughly linear in both, because a label spanning h bars
    # overlaps its h-1 neighbours.
    days_needed = MIN_EFFECTIVE_OBSERVATIONS * horizon / 24.0
    have_days = len(index) / 24.0

    # Three levers, not two. Which one binds depends on the half-life: a decay
    # caps the weighted sample at ~24H/ln2 bar-equivalents regardless of span,
    # so when that cap is the binding constraint, scraping more history is
    # wasted effort and has to be named as such.
    saturated = 24.0 * half_life / math.log(2) / horizon if half_life > 0 else float('inf')

    # The shorter-horizon suggestion has to be solved against whichever budget
    # actually applies. Sizing it off the raw timestamp count while a decay is
    # active produced advice to *lengthen* the horizon — 219h at five years of
    # hourly data — which is the opposite of the fix.
    budget = (24.0 * half_life / math.log(2)) if half_life > 0 else float(len(index))
    horizon_needed = max(int(budget / MIN_EFFECTIVE_OBSERVATIONS), 1)
    fix = (
        f'need at least {MIN_EFFECTIVE_OBSERVATIONS}. A label spanning {horizon} '
        f'bars overlaps its {horizon - 1} neighbours, so they are not independent '
        f'observations and more rows over the same span will not help. Ways out:\n'
    )
    if saturated < MIN_EFFECTIVE_OBSERVATIONS:
        half_life_needed = MIN_EFFECTIVE_OBSERVATIONS * horizon * math.log(2) / 24.0
        fix += (
            f'  - the {half_life:.0f}d recency half-life is the binding limit here: '
            f'it caps this horizon at ~{saturated:.0f} effective observations no '
            f'matter how much history you hold, so more history will not help '
            f'until it is raised. Lengthen it to about {half_life_needed:,.0f}d '
            f'(--recency-half-life-days {half_life_needed:.0f}) or disable it '
            f'(--recency-half-life-days 0)\n'
        )
    else:
        fix += (
            f'  - keep the {horizon}h horizon and scrape about '
            f'{days_needed:,.0f} days ({days_needed / 365:.1f} years); '
            f'you have {have_days:,.0f}\n'
        )
    if horizon_needed < horizon:
        fix += (
            f'  - keep this history and shorten the horizon to about '
            f'{horizon_needed}h (--horizon {horizon_needed})\n'
        )
    fix += (
        'Anything trained below the threshold is not wrong, but every statistic '
        'downstream — Sharpe, PBO, the gates — has error bars far wider than it '
        'looks, and the gates are calibrated for that.'
    )

    return Check('effective sample', False, detail, fix)


def _universe_wide_enough(dataset) -> Check:
    counts = dataset.features.groupby(level='event_time').size()
    typical = float(counts.median()) if len(counts) else 0.0
    detail = f'{len(dataset.symbols)} instruments, {typical:.0f} per bar at the median'
    if len(dataset.symbols) < MIN_UNIVERSE:
        return Check(
            'cross-section', False, detail,
            f'relative features standardise across the universe and need at least '
            f'{MIN_UNIVERSE} instruments; below that the z-scores are noise',
        )
    return Check('cross-section', True, detail)


def _model_trains(dataset, config: Config, as_of: Optional[str]) -> Check:
    # Imported here, not at module scope. This is the one check that needs
    # LightGBM, and LightGBM needs a native library — on macOS an absent
    # `libomp.dylib` raises OSError from dlopen at import time. That took down
    # the whole report, including the sample-size numbers this script exists to
    # produce and which need nothing but pandas. "Can I train?" should not be
    # unanswerable because the trainer will not load.
    try:
        from core.model import train_forecast_model
    except (ImportError, OSError) as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return Check(
            'model fit', False, f'LightGBM unavailable: {first}',
            'every other check above still holds. On macOS this is usually a '
            'missing OpenMP runtime: `brew install libomp`',
        )

    model = train_forecast_model(
        dataset.features, dataset.targets, config=config, data_as_of=as_of,
        horizon_bars=dataset.horizon_bars,
    )
    if model is None:
        return Check('model fit', False, 'training returned nothing',
                     'not enough resolved rows; see the targets check above')

    heads = ', '.join(f'{h}={m.get("ic", float("nan")):+.3f}' for h, m in model.metrics.items()
                      if 'ic' in m)
    detail = (
        f'{len(model.heads)} heads on {model.train_rows:,} rows '
        f'({model.effective_observations:.0f} effective)'
        + (f' | in-sample IC {heads}' if heads else '')
    )
    if model.uses_symbol_identity:
        return Check(
            'model fit', True,
            f'{detail} | WARNING: symbol identity is a feature, so the ranking '
            f'may be reproducing instrument level rather than timing',
        )
    return Check('model fit', True, detail)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--min-days', type=int, default=90,
                        help='History required before training is worth attempting')
    parser.add_argument('--skip-fit', action='store_true',
                        help='Stop before training (the slow check)')
    args = parser.parse_args()
    configure_logging(args.log_level)

    checks: list[Check] = []
    config = build_config(args)
    store = ResearchStore(args.store) if args.store else ResearchStore()

    checks.append(_store_populated(store, args.venue, args.min_days))
    checks.append(_carry_and_positioning(store, args.venue, args.min_days))
    if checks[0].passed:
        dataset = load(args, config)
        checks.append(_cost_schedule(config, dataset.symbols))
        checks.append(_panel_builds(dataset))
        if checks[-1].passed:
            checks.append(_targets_resolve(dataset))
            checks.append(_effective_sample(dataset, config))
            checks.append(_universe_wide_enough(dataset))
            if not args.skip_fit and checks[-3].passed:
                checks.append(_model_trains(dataset, config, args.as_of))
    else:
        checks.append(_cost_schedule(config, []))

    print()
    for check in checks:
        print(check)
        if check.fix and (check.advisory or not check.passed):
            for line in check.fix.split('\n'):
                print(f'       {line}')

    failed = [c for c in checks if not c.passed and not c.advisory]
    print()
    if failed:
        print(f'NOT READY: {len(failed)} of {len(checks)} check(s) failed')
        return 1
    print(f'READY: {len(checks)} check(s) passed. Next: python -m scripts.promote')
    return 0


if __name__ == '__main__':
    sys.exit(main())
