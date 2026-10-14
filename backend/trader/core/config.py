"""Configuration for the 15-minute binary system.

One frozen dataclass, one source of truth. Every field a run can disagree about
is here, and the fields that change an answer are CLI arguments, so each run
records them in its provenance.

Field groups: what is traded, how volatility is estimated, the null (`baseline_*`),
what a trade costs, and how much is staked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

_TRADER_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEE_CONFIG_NAME = 'kalshi_v202608.json'


def find_fee_config(name: str = DEFAULT_FEE_CONFIG_NAME) -> Optional[Path]:
    """Locate a venue fee schedule, or return None.

    Returns None, not an error: the defaults reproduce the published Kalshi
    schedule, so an unconfigured run prices correctly but records no schedule version.
    """
    explicit = os.getenv('FEE_CONFIG')
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.exists() else None
    candidate = _TRADER_ROOT / 'configs' / 'venue' / name
    return candidate if candidate.exists() else None


# The one Kalshi series mapping. Every reader uses it, so a `KALSHI_SERIES_*`
# override moves the trader and the recorders together.
# The 15-minute up/down series have no strike suffix on the ticker; the strike is
# the price at the window's open. A strike suffix (`KXBTCD-...-T86749.99`)
# marks the hourly threshold ladder, which is the wrong series.
SERIES_BY_SYMBOL: dict[str, str] = {
    'BTC-USD': os.getenv('KALSHI_SERIES_BTC', 'KXBTC15M'),
    'ETH-USD': os.getenv('KALSHI_SERIES_ETH', 'KXETH15M'),
    'SOL-USD': os.getenv('KALSHI_SERIES_SOL', 'KXSOL15M'),
}


def series_to_symbol() -> dict[str, str]:
    """The inverse of `SERIES_BY_SYMBOL`, for a recorder keyed on the ticker.

    Derived, so an env override reaches both directions.
    """
    return {series: symbol for symbol, series in SERIES_BY_SYMBOL.items()}


@dataclass(frozen=True)
class Config:
    # ---- universe and market structure -----------------------------------
    symbols: tuple[str, ...] = ('BTC-USD', 'ETH-USD', 'SOL-USD')
    venue: str = 'coinbase_spot'
    timeframe: str = '1m'

    # A property of the venue (quarter-hour windows), not a tunable. A field so
    # a test can build a short window.
    window_minutes: int = 15

    # The venue settles on an average, not a point price: the strike and the
    # settlement value are both one-minute means of CF Benchmarks' BRTI, and a
    # window's strike is the previous window's settlement value.
    # Averaging reduces variance: remaining variance at offset m is
    #     sigma^2 * ((window - delta - m) + delta/3)
    # not sigma^2 * (window - m). See `remaining_variance_minutes`.
    settle_average_minutes: float = 1.0

    # Minutes after the window opens at which a decision is scored. Each is a
    # row with its own displacement and remaining variance, and they share one
    # label, so folds split on the window, never the row.
    decision_offsets: tuple[int, ...] = (3, 6, 9, 12)

    # Which scored offsets may open a position. Every decision offset is still
    # scored, so measurement is unaffected; this narrows only entries.
    # The live loop takes the earliest offset that clears, and the late offsets
    # carry the measured per-contract edge, so live narrows entries to +12m via
    # `scripts.live --entry-offsets`. None (every scored offset may enter) is
    # the research default, so evaluation can measure every cell.
    # (9, 12) is the conservative widening if coverage matters.
    entry_offsets: Optional[tuple[int, ...]] = None

    # ---- volatility ------------------------------------------------------
    # Trailing realised-volatility lookbacks, in minutes, blended HAR-style.
    # Short lookbacks carry the state, long ones the level.
    vol_lookbacks_minutes: tuple[int, ...] = (15, 60, 240, 1440)

    # Intraday seasonality: a smoothed multiplicative minute-of-day factor.
    # A per-minute factor from a finite sample is mostly noise.
    # Set `seasonality_smooth_minutes` to 0 to disable seasonality.
    seasonality_smooth_minutes: int = 31
    seasonality_min_days: int = 60

    # A floor under the per-minute volatility forecast, in basis points. Without
    # it a dead-quiet minute divides by ~0 and the baseline returns 0 or 1.
    min_sigma_bps_per_minute: float = 0.5

    # ---- the baseline (the null hypothesis) ------------------------------
    # 'normal' or 'student_t'. One-minute crypto returns are fat-tailed, so a
    # Gaussian barrier is overconfident at large displacements.
    baseline_distribution: str = 'student_t'
    # Fraction of training rows, most recent first, held out of the baseline's
    # scale fit and used to fit one sigma correction. 0.0 disables it.
    # Env-settable (read at import; each evaluation is its own process).
    baseline_calibration_holdout: float = float(
        os.getenv('BASELINE_CALIBRATION_HOLDOUT', '0.0'))
    baseline_nu: Optional[float] = None          # fitted when None
    baseline_fit_scale_per_offset: bool = True   # one scale factor per offset

    # The baseline has no drift and never fits one: a drift is the alpha under test.

    # ---- model -----------------------------------------------------------
    # The classifier predicts a correction: the init forecaster's logit is the
    # LightGBM init_score, so an untrained model reproduces it exactly.
    # 'baseline' fits on top of `F(x/sigma)`; 'market' on the recorded quote's
    # implied probability. 'market' needs enough recorded quotes;
    # `core/model.py` refuses rather than substituting.
    init_score_source: str = 'baseline'

    learning_rate: float = 0.03
    n_estimators: int = 400
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 500
    subsample: float = 0.7
    colsample_bytree: float = 0.6
    reg_lambda: float = 10.0
    early_stopping_rounds: int = 40

    # ---- cross-validation ------------------------------------------------
    # How the walk-forward timeline is cut: 'count' gives equal numbers of
    # windows, 'calendar' equal anchored spans of time.
    # 'calendar' is the default: its boundaries do not move when data is
    # appended, so gate values compare across runs. 'count' re-cuts every
    # boundary on append. `--fold-scheme count` remains for comparison.
    fold_scheme: str = 'calendar'
    n_folds: int = 6
    # The calendar block length, in days. A stated parameter, so boundaries do
    # not slide as data arrives. Seven matches the weekly retrain: blocks do not
    # overlap, so this is both the test span and the refit step.
    # Smaller blocks raise per-fold ECE (binned ECE is biased up at small
    # samples), so calibration gates are not comparable across block sizes.
    fold_block_days: float = 7.0
    # A block full on time can be thin on data. Below this many windows it is
    # skipped, with a warning.
    min_test_windows: int = 200
    # Where the shrinkage is fitted, as a quantile of |model correction|.
    # 0.0 fits on every validation row (the average). A higher value fits on
    # the rows where the model disagrees most, which is where trades come from.
    # See `_fit_residual_scale`.
    shrinkage_focus_quantile: float = 0.0
    # Ceiling on `residual_scale`. Alpha is how much of the claimed correction
    # survives out of sample; a value above 1 amplifies the correction, which is
    # an overfitting signature, not shrinkage.
    max_residual_scale: float = 1.0
    # Number of boosters averaged. Their spread feeds `edge_lower_bound_z` and
    # `require_member_agreement`; 1 is a single model.
    ensemble_size: int = 1
    # 'venue' trains on the venue's own settlement where it exists, falling back
    # to the Coinbase-built proxy elsewhere.
    label_source: str = 'proxy'
    # Two-stage stacking. 'spot_then_market' fits stage 1 (a baseline-init
    # correction on `stack_spot_groups`) on every training row, and stage 2
    # only on rows with a usable quote:
    #   logit p = a*logit(market) + b*(logit(p_spot) - logit(market))
    #             [+ c] [+ L2 linear terms on `stack_book_features`]
    # `b` is how much of the spot model's disagreement with the price survives.
    # Stage 2 sees stage 1 only out of sample, via `stack_crossfit_blocks` blocked fits.
    stack_mode: str = 'off'
    stack_spot_groups: tuple[str, ...] = ('vol_state', 'microstructure',
                                          'cross_asset', 'geometry', 'offset')
    stack_book_features: tuple[str, ...] = ()
    stack_l2: float = 1.0
    stack_crossfit_blocks: int = 4
    stack_intercept: bool = False
    # The rolling training window, in days. None means expanding.
    # The weekly retrain deploys expanding, so any other value must also be
    # applied in the retrain before it is traded.
    fold_train_days: Optional[float] = None
    # Purge and embargo, in minutes, on both sides of every test block. It must
    # cover the longest feature lookback (1440) as well as the label span,
    # because a training row just after a test block uses test-period bars.
    embargo_minutes: int = 1440
    recency_half_life_days: Optional[float] = None
    train_window_days: Optional[float] = None
    # The sealed holdout, as an ISO date. Research scripts drop windows opening
    # at or after it (`scripts._common.load_dataset`). Reading past it needs
    # `--unseal-holdout "<reason>"`, recorded and allowed once per date.
    holdout_start: Optional[str] = None

    # ---- costs (Kalshi) --------------------------------------------------
    # Taker fee per order = ceil(fee_rate * C * P * (1 - P) * 10_000) / 10_000,
    # charged on the trade, never on settlement. See core/costs.py.
    fee_rate: float = 0.07
    # Maker fee = ceil(M * 0.0175 * C * P * (1 - P)) per order. M defaults to 0:
    # the 15-minute crypto series are not on the non-standard list.
    maker_fee_multiplier: float = 0.0
    assume_maker: bool = False
    # Half the quoted bid/ask, in cents. The live BTC 15-minute book showed a
    # one-cent spread. `scripts/evaluate.py` stresses this value.
    half_spread_cents: float = 0.5

    # ---- sizing and risk (a $100 account) --------------------------------
    starting_bankroll: float = 100.0
    # Fraction of full Kelly. This is also an edge filter: `decide` floors the
    # stake to whole contracts, so a lower fraction pushes marginal trades below
    # one contract and refuses them. `kelly_fraction` and `min_edge_pp` are coupled.
    # `max_stake_fraction` is close to inert at this edge size: Kelly binds first.
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05      # of bankroll, per position

    # How much of the depth measured at the touch a stake may claim. The quote is
    # read seconds before the order is sent, and the touch often shrinks in that
    # time. A judgement between no headroom and the measured lower-quartile decay.
    depth_fraction: float = 0.5
    # A hard dollar cap per position, used when no measured depth is available.
    # It binds once the account grows, and keeps a backtest from trading size
    # no venue could fill.
    max_stake_dollars: Optional[float] = 25.0

    # Compound off the current bankroll instead of the starting one. Off by
    # default: compounding turns an edge estimate into an exponential dominated
    # by the estimate's error. Off, the equity curve's slope is the per-trade edge.
    # Turn it on to project deployment, and label the output as a projection.
    compound: bool = False
    max_window_exposure_fraction: float = 0.08
    # How many symbols may hold a position in one window. The symbols' settle
    # directions are correlated (about 0.6), so each extra leg adds less than
    # proportional risk. At 3 the cap does not bind, so `decide()`'s alphabetical
    # walk does not starve the last symbol of entries.
    # `max_window_exposure_fraction` caps the notional.
    max_positions_per_window: int = 3

    # One entry per (symbol, window): the decision offsets are one bet observed
    # at several moments. `decide()` enforces it (`ALREADY_ENTERED`); it is an
    # invariant, not a config field.
    #
    # Hold to settlement, always. Settlement is free; an exit pays a second fee
    # and crosses the spread again. A binary's loss is capped at the stake, so no
    # stop-loss is needed. There is no exit path in `core/decide.py` or `core/book.py`.
    min_contracts: int = 1                # round down; zero contracts is a skip
    # Surplus over break-even, in probability points, required before a trade.
    # It guards against calibration error; the fee is already in break-even.
    # Realised edge rises with this gate out of sample, so the estimate carries
    # signal. 1.5 is a judgement on per-trade quality, not the argmax of return.
    # Do not choose it from `edge_curve`: that is in-sample selection.
    min_edge_pp: float = 1.5
    # Ensemble gates, both off by default and inert without ensemble columns.
    # With z > 0 the edge must clear the gate at the probability minus z member
    # standard deviations. With member agreement on, the least favourable member
    # must still beat the cost.
    edge_lower_bound_z: float = 0.0
    require_member_agreement: bool = False
    # Traded-price band. It must be symmetric about 0.5 (`__post_init__` checks).
    # The edge is a sigma disagreement, which points both ways: buy the favourite
    # or buy the longshot. A one-sided band discards half the strategy.
    # The ends exclude prices where the payoff is so skewed that a small
    # calibration error dominates the expected value. The tick is finer there,
    # not coarser (`tapered_deci_cent`).
    min_traded_price: float = 0.05
    max_traded_price: float = 0.95

    # An outlier guard, not an economic gate: a very large departure from the
    # quote is a bug or a stale price. Kept wide, because at an extreme quote a
    # real sigma disagreement moves the probability far (0.88 to 0.70 is the
    # strategy working). The price band and the two-sided-quote rule catch misparsed quotes.
    max_disagreement_pp: float = 25.0

    # ---- order envelope --------------------------------------------------
    # The hard cap, in cents, on how far a live order may cross to fill. The
    # actual crossing is bounded by `min_edge_pp`: a fill at the gate is still a
    # trade this system accepts. `taker_at_cross` + `immediate_or_cancel` fills
    # from the touch outward, so the limit bounds the worst fill, not the price.
    # The cap guards against a book so thin the order walks it far.
    max_slippage_cents: float = 3.0

    # ---- freshness -------------------------------------------------------
    # Maximum age of the newest bar at decision time. A delayed or partial fetch
    # would otherwise be scored as current.
    max_bar_age_seconds: int = 150
    # Contiguous bars required. `volume_z_15` is a 1,440-minute z-score of a
    # 15-minute mean, so feature parity needs 1,455 bars. With fewer, some
    # features are wrong while the cycle would still trade.
    min_usable_bars: int = 1_455
    # Refuse a decision whose cross-venue peer book is missing. Off by default.
    # Without the peer, the model loses most of its skill; `--complete-cases`
    # already requires the peer in training, so this makes live agree.
    require_peer_book: bool = False
    # Place orders only while this exact artifact version is loaded. A row
    # scored by any other model is refused `model_not_pinned`, so a failed
    # reload or an unreviewed retrain cannot trade. None disables the check.
    require_model_version: Optional[str] = None
    # The one quote-staleness limit. `core.quotes` prices backtests with it,
    # `core.metrics` filters the market comparison with it, and `scripts.live`
    # refuses an order and a peer book past it. `QUOTE_MAX_AGE_SECONDS` overrides
    # it for every reader at once.
    max_quote_age_seconds: float = float(os.getenv('QUOTE_MAX_AGE_SECONDS', '30'))
    # Research: which `venue_depth` observers may price a row. Empty keeps every
    # source (ranked by QUOTE_SOURCE_PRIORITY). ('live_touch', 'live_ws', 'live')
    # excludes the Predexon backfill.
    quote_sources: tuple[str, ...] = ()
    # What a quote with no recorded age means. 'fresh' keeps it; 'stale' drops
    # it, after bounding `live_touch` ages from `available_time - event_time`
    # (see `core.quotes`).
    null_quote_age: str = 'fresh'
    # The cross-venue peer's staleness cap. None follows `max_quote_age_seconds`.
    # Separate because few live Polymarket rows are very fresh: a tight Kalshi
    # cap on the peer would void `venue_prob_gap` almost everywhere.
    # Once set, `null_quote_age` applies to the peer too; unset, a peer row with
    # no age stays fresh.
    peer_max_quote_age_seconds: Optional[float] = None
    # Seconds from a decision's minute to the instant live reads the book. A
    # backtest row may only use a book seen by then.
    decision_latency_seconds: float = 3.0
    # Refuse to enter when less than this remains in the window. `choose_offset`
    # floors to whole minutes, so a late submission would carry stale sigma.
    min_remaining_seconds: int = 45
    # Stop trading below this fraction of the starting bankroll.
    ruin_floor_fraction: float = 0.50

    # ---- live execution --------------------------------------------------
    # Price from the in-process stream book only while its socket delivered a
    # frame this recently and the book is not gapped; otherwise read REST
    # `/orderbook` once.
    stream_quote_max_age_seconds: float = 2.0
    # After this many consecutive decision cycles with the peer book refused
    # for one symbol (no book, wrong window, too old), live asks
    # `record_pm_stream` to drop that symbol's state and resubscribe.
    peer_resubscribe_after_cycles: int = 3
    # A one-minute bar built from the Coinbase trade stream is final once the
    # clock passes its close by this much: trades arrive tens of ms late.
    spot_bar_grace_seconds: float = 0.75

    # ---- the settlement-index proxy (`core/index_proxy.py`) --------------
    # An exchange's mid counts toward the per-second median only while its
    # connection delivered something this recently. Applied at record time.
    index_proxy_max_staleness_seconds: float = 5.0
    # Fewer fresh exchanges than this gives a NaN proxy.
    index_proxy_min_exchanges: int = 2
    # The `index_proxy` feature group reads the newest proxy row stamped
    # strictly before the decision, and only if it is at most this old.
    index_proxy_max_age_seconds: float = 5.0
    # On a decision cycle, venue reconciliation runs after the orders and sizing
    # uses the last reconciled balance. If that is older than this (or missing),
    # the cycle reconciles first.
    max_reconcile_age_seconds: float = 300.0
    # One re-priced resend after an immediate_or_cancel that filled nothing,
    # only while the kill is this fresh. Never after a partial fill.
    kill_retries: int = 1
    kill_retry_window_seconds: float = 2.0

    # ---- circuit breakers ------------------------------------------------
    # Circuit breakers. They bound the day, the streak and the drawdown, and are
    # recorded on the account so they survive a restart. Clearing a halt is manual.
    max_daily_loss_fraction: float = 0.15
    # Consecutive losses mean different things at different prices: a longshot
    # is meant to lose most of the time. 25 is roughly as rare at longshot prices
    # as 12 is at even money. Interim: this should scale with the price paid.
    max_consecutive_losses: int = 25
    # Peak-to-current drawdown on realised equity, live. The same 0.35 as the
    # promotion gate. It catches an intraday give-back that the daily-loss rule
    # cannot see.
    max_drawdown_fraction: float = 0.35

    # ---- provenance ------------------------------------------------------
    # The fee schedule version, a fact about the fitted numbers.
    fee_config_version: str = 'builtin_kalshi_v202608'
    cli_overrides: frozenset[str] = frozenset()

    # ---- invariants ------------------------------------------------------
    def __post_init__(self) -> None:
        """Refuse a configuration whose numbers contradict each other."""
        stray = tuple(o for o in (self.entry_offsets or ())
                      if o not in tuple(self.decision_offsets))
        if stray:
            raise ValueError(
                f'entry_offsets {stray} are not in decision_offsets '
                f'{tuple(self.decision_offsets)}. An offset that is never scored '
                f'can never produce a decision, so this configuration would '
                f'abstain on every window and look like a dead signal rather '
                f'than a misconfiguration.'
            )
        if self.entry_offsets is not None and not self.entry_offsets:
            raise ValueError(
                'entry_offsets is empty, so no window could ever be entered. Use '
                'None for "every scored offset", or --mode paper / --dry-run to '
                'run measurement-only — each of which says so explicitly rather '
                'than looking like a signal that never fires.'
            )
        if self.init_score_source not in ('baseline', 'market'):
            raise ValueError(
                f"init_score_source={self.init_score_source!r}; expected "
                f"'baseline' or 'market'. A typo here would otherwise surface as "
                f"a KeyError deep inside the fit, after the fold's baseline, "
                f"volatility models and seasonality had already been fitted."
            )
        if abs(self.max_traded_price - (1.0 - self.min_traded_price)) > 1e-9:
            raise ValueError(
                f'the price band must be symmetric about 0.5: '
                f'max_traded_price {self.max_traded_price} != '
                f'1 - min_traded_price {1.0 - self.min_traded_price}. The edge is '
                f'a disagreement about sigma and it points both ways — an '
                f'asymmetric band silently permits only one direction of it.'
            )
        # The embargo must cover what reaches forward from a training row under
        # the expanding scheme: the label and the HAR target, both
        # `window_minutes` long. Feature lookbacks extend backwards, away from
        # the test block. The 1440 default matters only for a rolling scheme.
        # `core/cv.py:assert_no_leakage` cannot check this: it compares against
        # the configured embargo.
        if self.embargo_minutes < self.window_minutes:
            raise ValueError(
                f'embargo_minutes {self.embargo_minutes} is under window_minutes '
                f'{self.window_minutes}. A training window inside that gap '
                f'settles after the test block starts, so its label is built '
                f'from test-period bars, and the HAR target reaches there too. '
                f'The default of 1440 is deliberately much larger.'
            )
        if self.kill_retries < 0:
            raise ValueError(f'kill_retries must not be negative; got '
                             f'{self.kill_retries}.')
        if self.max_slippage_cents < 0.0:
            raise ValueError(
                f'max_slippage_cents must not be negative; got '
                f'{self.max_slippage_cents}.'
            )
        # Typos here would otherwise surface deep inside a fold's fit.
        for name, allowed in (('label_source', ('proxy', 'venue')),
                              ('null_quote_age', ('fresh', 'stale')),
                              ('stack_mode', ('off', 'spot_then_market'))):
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name}={getattr(self, name)!r}; expected one '
                                 f'of {allowed}')
        if int(self.stack_crossfit_blocks) < 2:
            raise ValueError(f'stack_crossfit_blocks must be at least 2; got '
                             f'{self.stack_crossfit_blocks}')
        if self.decision_latency_seconds < 0.0:
            raise ValueError(f'decision_latency_seconds must not be negative; '
                             f'got {self.decision_latency_seconds}')
        if self.peer_resubscribe_after_cycles < 1:
            raise ValueError(f'peer_resubscribe_after_cycles must be at least 1; '
                             f'got {self.peer_resubscribe_after_cycles}')
        if (self.peer_max_quote_age_seconds is not None
                and self.peer_max_quote_age_seconds <= 0.0):
            raise ValueError(f'peer_max_quote_age_seconds must be positive; got '
                             f'{self.peer_max_quote_age_seconds}')
        if self.stack_l2 < 0.0:
            raise ValueError(f'stack_l2 must not be negative; got {self.stack_l2}')
        if int(self.ensemble_size) < 1:
            raise ValueError(f'ensemble_size must be at least 1; got '
                             f'{self.ensemble_size}')
        if self.edge_lower_bound_z < 0.0:
            raise ValueError(f'edge_lower_bound_z must not be negative; got '
                             f'{self.edge_lower_bound_z}')
        if self.holdout_start is not None:
            from datetime import datetime
            try:
                datetime.fromisoformat(str(self.holdout_start))
            except ValueError as exc:
                raise ValueError(
                    f'holdout_start={self.holdout_start!r} is not an ISO date; a '
                    f'seal that cannot be parsed would seal nothing.') from exc

    # ---- derived ---------------------------------------------------------
    def remaining_minutes(self, offset: int) -> int:
        """Wall-clock minutes from the decision to settlement.

        Reported only. The barrier divides by `remaining_variance_minutes`.
        """
        return self.window_minutes - offset

    def remaining_variance_minutes(self, offset: float) -> float:
        """Minutes of variance left between the last observed price and settlement.

        The settlement value is the mean over the final `settle_average_minutes`.
        For a diffusion, its variance about a price observed at offset `m` is

            sigma^2 * ((window - delta - m) + delta / 3)

        The `delta/3` term is the variance of a time-average over an interval of
        length `delta`. Kept analytic so the baseline's fitted scale does not absorb it.
        """
        delta = float(self.settle_average_minutes)
        return max(0.0, (self.window_minutes - delta - float(offset)) + delta / 3.0)

    def with_overrides(self, **values: Any) -> 'Config':
        """Return a copy with `values` applied, recording which fields moved."""
        known = {f.name for f in fields(self)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        applied = {k: v for k, v in values.items() if v is not None}
        if not applied:
            return self
        return replace(
            self, **applied,
            cli_overrides=frozenset(self.cli_overrides | set(applied)),
        )

    def with_fee_assumptions(self, path: Optional[Path]) -> 'Config':
        """Load a venue fee schedule, or return self unchanged when there is none."""
        if path is None:
            return self
        import json
        payload = json.loads(Path(path).read_text())
        fees = payload.get('fees', {})
        return replace(
            self,
            fee_rate=float(fees.get('fee_rate', self.fee_rate)),
            maker_fee_multiplier=float(fees.get('maker_fee_multiplier',
                                                self.maker_fee_multiplier)),
            half_spread_cents=float(fees.get('half_spread_cents', self.half_spread_cents)),
            fee_config_version=str(payload.get('version', 'unversioned')),
        )

    def provenance(self) -> dict[str, Any]:
        """Every field, JSON-ready, for the artifact, the ledger and the trial log.

        Tuples become lists, sets sorted lists, and `None` stays `None`
        (`entry_offsets=None` means every offset may enter).
        """
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    def provenance_hash(self) -> str:
        """Identity of the configuration under test.

        `cli_overrides` is left out: it says how values were reached, not what they are."""
        import hashlib
        import json

        material = {k: v for k, v in self.provenance().items() if k != 'cli_overrides'}
        canonical = json.dumps(material, sort_keys=True, separators=(',', ':'),
                               default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (frozenset, set)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


DEFAULT_CONFIG = Config()
