"""Assembling a research dataset from the store.

One place that turns "which venue, which symbols, as of when" into features,
targets and the bars a backtest needs. Every CLI goes through here, so a training
run, a backtest and a live signal cycle cannot disagree about what the data is —
the same class of problem as having three copies of the decision.

Point-in-time is threaded through rather than bolted on: `as_of` bounds every
read by `available_time`, so a run reproduced for a past date sees the data as it
stood then, not as it was later revised.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import pandas as pd

from core.config import Config
from core.costs import resolve_base
from core.costs import symbols_missing_fee_schedule
from core.datastore import ResearchStore
from core.features import SymbolInputs, build_panel
from core.profiles import COIN_PROFILES, CoinProfile
from core.targets import build_target_panel, summarise_targets

logger = logging.getLogger(__name__)

MARKET_SYMBOL = 'BIP'


def _since(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows at or after `cutoff`, tolerating the empty frames the loader stores.

    A symbol with no funding on either venue is held as an empty DataFrame,
    which carries a RangeIndex; comparing that to a Timestamp raises.
    """
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame
    return frame[frame.index >= cutoff]


@dataclass
class Dataset:
    """A feature panel, its targets, and the bars they were built from."""

    features: pd.DataFrame
    targets: pd.DataFrame
    bars: dict[str, pd.DataFrame]
    funding: dict[str, pd.DataFrame]
    profiles: dict[str, CoinProfile]
    venue: str
    reference_venue: Optional[str]
    as_of: Optional[str]
    horizon_bars: int
    warnings: list[str] = field(default_factory=list)
    # Symbols whose funding came from the reference venue instead of the traded
    # one. Structured, not just a warning string, because it has to survive into
    # the model artifact and the promotion gates: funding feeds the `carry`
    # component of the net-return target directly, so a proxy series trains the
    # carry head on a cash flow this account will never receive. Coinbase CDE
    # publishes no historical funding, which makes borrowing it the obvious
    # shortcut and this the guard against taking it silently.
    proxy_funding_symbols: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return sorted(self.bars)

    @property
    def resolved_index(self) -> pd.MultiIndex:
        """Rows with a resolved target — the only ones that can train or score."""
        return self.targets.dropna(subset=['price']).index

    def summary(self) -> dict[str, Any]:
        target_summary = summarise_targets(self.targets)
        times = self.features.index.get_level_values('event_time')
        return {
            'venue': self.venue,
            'reference_venue': self.reference_venue,
            'proxy_funding_symbols': list(self.proxy_funding_symbols),
            'as_of': self.as_of,
            'symbols': self.symbols,
            'rows': int(len(self.features)),
            'features': int(self.features.shape[1]),
            'resolved_targets': int(len(self.resolved_index)),
            'horizon_bars': self.horizon_bars,
            'first_bar': str(times.min()) if len(times) else None,
            'last_bar': str(times.max()) if len(times) else None,
            'carry_share': round(target_summary.carry_share, 3),
            'mean_cost_bps': round(target_summary.mean_cost_bps, 2),
            'warnings': self.warnings,
        }

    def trailing(self, days: float) -> "Dataset":
        """The same dataset restricted to the most recent `days` of event time.

        A training window and a recency half-life are different instruments and
        both are useful: the window bounds what is loaded and fitted, the
        half-life shapes what matters inside it. The window is applied after the
        features are built, so a feature that needed 200 bars of history still
        saw them — only the rows offered to the model are cut.
        """
        if days is None or days <= 0 or self.features.empty:
            return self

        times = pd.DatetimeIndex(self.features.index.get_level_values('event_time'))
        cutoff = times.max() - pd.Timedelta(days=float(days))
        if cutoff <= times.min():
            return self

        keep = times >= cutoff
        features = self.features[keep]
        target_times = pd.DatetimeIndex(self.targets.index.get_level_values('event_time'))
        targets = self.targets[target_times >= cutoff]

        return replace(
            self,
            features=features,
            targets=targets,
            bars={s: _since(f, cutoff) for s, f in self.bars.items()},
            funding={s: _since(f, cutoff) for s, f in self.funding.items()},
            warnings=[
                *self.warnings,
                f'training window: kept the last {float(days):,.0f} days '
                f'({len(features):,} of {len(self.features):,} rows)',
            ],
        )

    def __str__(self) -> str:
        return (
            f"{len(self.symbols)} symbols | {len(self.features):,} rows x "
            f"{self.features.shape[1]} features | "
            f"{len(self.resolved_index):,} resolved targets | "
            f"horizon {self.horizon_bars}h"
        )


def _frame_for(
    store: ResearchStore,
    dataset: str,
    symbol: str,
    venue: str,
    *,
    as_of: Optional[str],
    min_quality: Optional[str],
) -> pd.DataFrame:
    rows = store.read(
        dataset, venue=venue, symbols=[symbol], as_of=as_of, min_quality=min_quality
    )
    if rows.empty:
        return pd.DataFrame()
    indexed = rows.set_index(pd.to_datetime(rows['event_time'], utc=True)).sort_index()
    return indexed.drop(
        columns=[c for c in ('event_time', 'symbol', 'venue', 'quality') if c in indexed]
    )


def _resolve_oi_venue(
    store: ResearchStore,
    symbol: str,
    preferred: list[Optional[str]],
    *,
    as_of: Optional[str],
    min_quality: Optional[str],
) -> Optional[str]:
    """Find a venue that actually reports open interest for this symbol.

    Coinbase exposes no open-interest endpoint, so this is always a proxy, and it
    is looked up rather than assumed: it is not necessarily the same venue used
    for the cross-venue basis. Whichever venue supplies it is recorded, because
    Bybit and Binance report materially different open interest.
    """
    for candidate in preferred:
        if candidate and not _frame_for(
            store, 'open_interest', symbol, candidate,
            as_of=as_of, min_quality=min_quality
        ).empty:
            return candidate

    coverage = store.coverage('open_interest')
    if coverage.empty:
        return None
    available = coverage[coverage['symbol'] == symbol.upper()]
    return str(available.iloc[0]['venue']) if not available.empty else None


def resolve_store_symbols(
    store: ResearchStore,
    requested: Sequence[str],
    *,
    venue: str,
) -> tuple[dict[str, str], list[str]]:
    """Map requested symbols onto the spellings the store actually holds.

    The scraper and the readers disagreed about what a symbol is. `run_pipeline`
    stores whatever the venue calls the product — `BTC-PERP`, or
    `AVP-20DEC30-CDE` — while `load_dataset` asked for the bare profile prefix
    (`BIP`, `ETP`). `ResearchStore._prepare` only upper-cases, so the two never
    met: on a store built by the documented scrape command, every lookup missed
    and the panel came back empty with one "no bars" warning per instrument. It
    looked like a data problem and was a naming problem.

    Resolution goes through `costs._resolve_base`, which already maps every
    spelling — CDE code, ticker, or decorated product id — onto an underlying,
    and is the same function that prices the contract. Returns
    `{requested: stored}` plus the requested symbols nothing in the store matches.
    """
    coverage = store.coverage('bars')
    if coverage.empty:
        return {}, list(requested)

    available = [str(s) for s in coverage.loc[coverage['venue'] == venue, 'symbol'].unique()]
    if not available:
        available = [str(s) for s in coverage['symbol'].unique()]

    by_base: dict[str, list[str]] = {}
    for stored in available:
        base = resolve_base(stored)
        if base:
            by_base.setdefault(base, []).append(stored)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for symbol in requested:
        upper = symbol.upper()
        if upper in available:          # already the stored spelling
            resolved[symbol] = upper
            continue
        base = resolve_base(symbol)
        candidates = by_base.get(base or '', [])
        if candidates:
            # Deterministic when a base has several contracts: prefer an exact
            # prefix match, then the longest name, so the choice never depends on
            # scrape order.
            exact = [c for c in candidates if c.split('-')[0] == upper]
            resolved[symbol] = sorted(exact or candidates, key=lambda s: (-len(s), s))[0]
        else:
            missing.append(symbol)
    return resolved, missing


def load_dataset(
    store: ResearchStore,
    *,
    venue: str = 'coinbase',
    reference_venue: Optional[str] = 'binance',
    oi_venue: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    config: Optional[Config] = None,
    as_of: Optional[str] = None,
    min_quality: Optional[str] = 'valid',
    horizon_bars: Optional[int] = None,
) -> Dataset:
    """Build features and targets for a universe.

    Open interest is resolved to whichever venue reports it, because Coinbase
    exposes no open-interest endpoint. Those figures therefore describe a
    different book than the one being traded, and the positioning features carry
    that caveat — which is recorded in `warnings` rather than left implicit.
    """
    config = config or Config()
    warnings: list[str] = []

    asked = list(symbols) if symbols else [
        profile.prefixes[0] for profile in COIN_PROFILES.values()
    ]
    # `excluded_symbols` was parsed from `--exclude` and `EXCLUDE_SYMBOLS` and read
    # by nothing, so an excluded instrument still traded. Applied here, before the
    # store is touched, so it holds for features, targets, the backtest and the
    # live signal writer alike. Matched on the underlying, so `--exclude BTC`
    # covers `BIP`, `BTC-PERP` and `BIP-20DEC30-CDE`.
    excluded = {s.strip().upper() for s in (config.excluded_symbols or []) if s.strip()}
    if excluded:
        excluded_bases = {resolve_base(s) or s for s in excluded}
        kept = [
            s for s in asked
            if s.upper() not in excluded and (resolve_base(s) or s) not in excluded_bases
        ]
        if len(kept) != len(asked):
            warnings.append(
                f'excluded {len(asked) - len(kept)} symbol(s) by configuration: '
                f'{", ".join(sorted(set(asked) - set(kept)))}'
            )
        asked = kept
    # Translate to the spellings the store holds before anything reads it.
    mapping, unresolved = resolve_store_symbols(store, asked, venue=venue)
    for symbol in unresolved:
        warnings.append(f'{symbol}: nothing in the store resolves to it on {venue}')
    requested = [mapping[s] for s in asked if s in mapping]
    for asked_name, stored_name in mapping.items():
        if asked_name.upper() != stored_name:
            warnings.append(f'{asked_name} resolved to {stored_name}')

    # Profiles are keyed by the stored spelling, since that is what every
    # downstream lookup uses.
    # One profile per symbol, chosen deterministically. A dict comprehension over
    # two loops silently keeps whichever match comes last in COIN_PROFILES order,
    # which is not a decision anyone made.
    profiles: dict[str, CoinProfile] = {}
    for symbol in asked:
        stored = mapping.get(symbol)
        if stored is None:
            continue
        exact = [p for p in COIN_PROFILES.values() if symbol in p.prefixes]
        by_base = [p for p in COIN_PROFILES.values() if resolve_base(symbol) == p.name]
        candidates = exact or by_base
        if len(candidates) > 1:
            warnings.append(
                f'{symbol} matches {len(candidates)} profiles '
                f'({", ".join(p.name for p in candidates)}); using the first'
            )
        if candidates:
            profiles[stored] = candidates[0]

    market_symbol = mapping.get(MARKET_SYMBOL) or MARKET_SYMBOL
    market = _frame_for(
        store, 'bars', market_symbol, venue, as_of=as_of, min_quality=min_quality
    )
    if market.empty:
        warnings.append(
            f'{market_symbol} has no bars on {venue}: the market_factor features '
            f'will be empty, so nothing can be expressed relative to the market'
        )

    inputs: list[SymbolInputs] = []
    bars: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    oi_venues_used: set[str] = set()
    proxy_funding_symbols: list[str] = []
    symbols_with_reference: list[str] = []
    symbols_without_reference: list[str] = []

    for symbol in requested:
        symbol_bars = _frame_for(store, 'bars', symbol, venue, as_of=as_of, min_quality=min_quality)
        if symbol_bars.empty:
            warnings.append(f'{symbol}: no bars on {venue}, skipped')
            continue

        symbol_funding = _frame_for(store, 'funding', symbol, venue, as_of=as_of, min_quality=min_quality)
        if symbol_funding.empty and reference_venue:
            symbol_funding = _frame_for(
                store, 'funding', symbol, reference_venue, as_of=as_of, min_quality=min_quality
            )
            if not symbol_funding.empty:
                proxy_funding_symbols.append(symbol)
                warnings.append(
                    f'{symbol}: funding taken from {reference_venue}, not {venue} — '
                    f'the carry features describe a different venue than the trade'
                )

        resolved_oi_venue = _resolve_oi_venue(
            store, symbol, [oi_venue, venue, reference_venue],
            as_of=as_of, min_quality=min_quality,
        )
        open_interest = (
            _frame_for(store, 'open_interest', symbol, resolved_oi_venue,
                       as_of=as_of, min_quality=min_quality)
            if resolved_oi_venue else pd.DataFrame()
        )
        if not open_interest.empty and resolved_oi_venue != venue:
            oi_venues_used.add(resolved_oi_venue)

        reference_bars = (
            _frame_for(store, 'bars', symbol, reference_venue, as_of=as_of, min_quality=min_quality)
            if reference_venue else pd.DataFrame()
        )
        if reference_venue:
            (symbols_with_reference if not reference_bars.empty
             else symbols_without_reference).append(symbol)

        bars[symbol] = symbol_bars
        funding[symbol] = symbol_funding
        inputs.append(SymbolInputs(
            symbol=symbol,
            bars=symbol_bars,
            funding=symbol_funding if not symbol_funding.empty else None,
            open_interest=open_interest if not open_interest.empty else None,
            reference_bars=reference_bars if not reference_bars.empty else None,
            market_bars=market if not market.empty else None,
        ))

    if not inputs:
        return Dataset(
            pd.DataFrame(), pd.DataFrame(), {}, {}, profiles,
            venue, reference_venue, as_of, 0, warnings,
            proxy_funding_symbols=sorted(proxy_funding_symbols),
        )

    features = build_panel(inputs, config=config)
    resolved_horizon = horizon_bars or config.label_horizon_hours()
    targets = build_target_panel(
        bars, profiles=profiles, funding_by_symbol=funding, config=config,
        horizon_bars=resolved_horizon,
        index_by_symbol={
            symbol: features.xs(symbol, level='symbol').index
            for symbol in bars if symbol in features.index.get_level_values('symbol')
        },
    )

    # A reference venue that returns nothing is the quietest degradation in the
    # loader: `cross_venue_features` returns an empty frame rather than NaN
    # columns, so the panel is simply seven features narrower and looks healthy.
    # It is also the likely case for a US operator — Binance, OKX and Bybit all
    # answer 451 to a US IP — so it has to be said out loud.
    if reference_venue and symbols_without_reference:
        detail = (
            'none of them' if not symbols_with_reference
            else f'{len(symbols_without_reference)} of '
                 f'{len(symbols_without_reference) + len(symbols_with_reference)}'
        )
        warnings.append(
            f'no {reference_venue} bars for {detail}: '
            f'{", ".join(sorted(symbols_without_reference))}. The cross-venue '
            f'group (basis, lead-lag) is empty for those symbols, so the panel '
            f'is narrower than it looks. If the venue is geo-blocked, scrape '
            f'through a proxy; if the scraper fell back to another exchange, '
            f'point --reference-venue at whichever one it stored.'
        )

    if oi_venues_used:
        warnings.append(
            f'open interest taken from {", ".join(sorted(oi_venues_used))}, not '
            f'{venue}: Coinbase publishes none, so the positioning features '
            f'describe a different book than the one being traded'
        )

    missing_schedule = symbols_missing_fee_schedule(list(bars), config)
    if missing_schedule:
        warnings.append(
            f'no explicit fee schedule for {", ".join(missing_schedule)}: falling '
            f'back to ${config.min_fee_per_contract:.2f}/contract, which '
            f'understates profitability rather than overstating it'
        )
    if config.cost_config_version == 'legacy_default':
        warnings.append(
            'no exchange cost config loaded: costs are the hardcoded '
            f'{config.fee_pct_per_side * 100:.2f}%/side default, which is wrong '
            'for every Coinbase contract'
        )

    return Dataset(
        features=features,
        targets=targets,
        bars=bars,
        funding=funding,
        profiles=profiles,
        venue=venue,
        reference_venue=reference_venue,
        as_of=as_of,
        horizon_bars=resolved_horizon,
        warnings=warnings,
        proxy_funding_symbols=sorted(proxy_funding_symbols),
    )


def report_warnings(dataset: Dataset) -> None:
    """Print data caveats prominently.

    These are the things that quietly invalidate a result — a missing fee
    schedule, funding from the wrong venue — so they are surfaced every run
    rather than logged once and forgotten.
    """
    for warning in dataset.warnings:
        logger.warning('%s', warning)
