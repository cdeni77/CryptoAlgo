"""Configuration for the 15-minute binary system.

One frozen dataclass, one source of truth. Everything a run can disagree
about is a field here, and every script exposes the fields that change an
answer as CLI arguments — so a run that used a different offset set, a
different fee assumption or a different bankroll says so in its own
provenance rather than being reconstructed later from a shell history.

The fields group into five decisions, and they are independent:

* **What is traded.** Three Coinbase spot series, 15-minute windows, and a
  handful of decision offsets inside each window.
* **What volatility is.** The remaining-variance forecast is the only input the
  barrier baseline needs, so its lookbacks and its seasonality live here.
* **What the null is.** ``baseline_*`` describes the arithmetic a clock and a
  volatility estimate can already do. Beating it is the entire question.
* **What a trade costs.** Kalshi's fee is a function of price, not a spread in
  basis points, and the half-spread is a separate field because it is an
  assumption rather than a measurement.
* **How much is staked.** A $100 account makes the integer-contract minimum a
  real gate rather than a rounding detail, and it makes correlation across the
  three symbols a sizing problem rather than a footnote.
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

    Deliberately returns None rather than raising: the caller decides whether
    an unversioned run is acceptable. The hardcoded defaults below reproduce
    the published Kalshi schedule, so an unconfigured run prices correctly but
    records no schedule version — which is the remaining reason to load one.
    """
    explicit = os.getenv('FEE_CONFIG')
    if explicit:
        candidate = Path(explicit)
        return candidate if candidate.exists() else None
    candidate = _TRADER_ROOT / 'configs' / 'venue' / name
    return candidate if candidate.exists() else None


@dataclass(frozen=True)
class Config:
    # ---- universe and market structure -----------------------------------
    symbols: tuple[str, ...] = ('BTC-USD', 'ETH-USD', 'SOL-USD')
    venue: str = 'coinbase_spot'
    timeframe: str = '1m'

    # A Kalshi crypto up/down market opens on a quarter-hour boundary and
    # settles on the next one. `window_minutes` is therefore a property of the
    # venue, not a tunable — it is a field so a test can build a 4-minute
    # window without monkeypatching a module constant.
    window_minutes: int = 15

    # The venue settles on an *average*, not a point price. From the market's own
    # rules: "the simple average of the sixty seconds of CF Benchmarks' BRTI
    # before 12:45 AM EDT ... is at least the simple average of the sixty seconds
    # ... before 12:30 AM EDT". So both the strike and the settlement value are
    # one-minute means, and the strike of a window is exactly the settlement
    # value of the window before it — consecutive markets chain.
    #
    # This has a real consequence for the barrier: averaging *reduces* variance.
    # The unresolved quantity at offset m is the mean over the last minute, whose
    # variance about the last observed price is
    #     sigma^2 * ((window - delta - m) + delta/3)
    # rather than sigma^2 * (window - m). At m=12 that is 2.33 minutes of
    # variance, not 3 — a 22% overstatement if ignored. See
    # `remaining_variance_minutes`.
    settle_average_minutes: float = 1.0

    # A tie resolves YES. `strike_type` on the market is `greater_or_equal`, so a
    # dead-flat window pays the up side — the opposite of what a strict `>` would
    # give it, and worth a field because it is a venue fact rather than a choice.
    tie_resolves_up: bool = True

    # Minutes after the window opens at which a decision is scored. Each is a
    # separate row with its own displacement and its own remaining variance,
    # and they share a label — so folds split on the *window*, never the row.
    # Four offsets spread across the window cover the barrier geometry from
    # near-coin-flip to nearly-settled without carrying fourteen near-copies
    # of every observation.
    decision_offsets: tuple[int, ...] = (3, 6, 9, 12)

    # ---- volatility ------------------------------------------------------
    # Trailing realised-volatility lookbacks, in minutes, blended HAR-style.
    # 15 and 60 carry the state, 240 and 1440 carry the level; a single
    # lookback is either too noisy or too slow and there is no setting that is
    # both.
    vol_lookbacks_minutes: tuple[int, ...] = (15, 60, 240, 1440)

    # Intraday seasonality is a multiplicative minute-of-day factor, smoothed,
    # because a per-minute factor from a finite sample is mostly noise. Set
    # `seasonality_smooth_minutes` to 0 to disable seasonality entirely.
    seasonality_smooth_minutes: int = 31
    seasonality_min_days: int = 60

    # A floor under the per-minute volatility forecast, in basis points. A
    # dead-quiet minute otherwise divides by ~0 and the baseline returns 0 or 1
    # with total confidence.
    min_sigma_bps_per_minute: float = 0.5

    # ---- the baseline (the null hypothesis) ------------------------------
    # 'normal' or 'student_t'. One-minute crypto returns are fat-tailed, so a
    # Gaussian barrier overstates confidence at large displacements. Which
    # distribution calibrates better out of sample is measured, not assumed.
    baseline_distribution: str = 'student_t'
    baseline_nu: Optional[float] = None          # fitted when None
    baseline_fit_scale_per_offset: bool = True   # one scale factor per offset

    # The baseline's drift is structurally zero and is not fitted. A non-zero
    # drift *is* the alpha; it belongs to the model under test, and putting it
    # in the null would hide exactly what the null exists to expose.

    # ---- model -----------------------------------------------------------
    # The classifier predicts a correction to the baseline: the baseline's
    # logit enters as an init_score offset, so an untrained model reproduces
    # the baseline exactly and every parameter it fits is incremental skill.
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
    n_folds: int = 6
    # Purge and embargo, in minutes, applied on both sides of every test
    # block. It must cover the longest feature lookback (1440) as well as the
    # label span (15), because a train row immediately after a test block
    # computes its features from test-period bars.
    embargo_minutes: int = 1440
    recency_half_life_days: Optional[float] = None
    train_window_days: Optional[float] = None

    # ---- costs (Kalshi) --------------------------------------------------
    # fee per contract = ceil(fee_rate * price * (1 - price) * 100) / 100,
    # charged on the trade and never on settlement. See core/costs.py.
    fee_rate: float = 0.07
    maker_fee_rate: float = 0.0025
    assume_maker: bool = False
    # Half the quoted bid/ask, in cents of a dollar contract.
    #
    # MEASURED, no longer assumed: the live BTC 15-minute book quoted 0.19/0.20
    # and 0.10/0.11 — a one-cent spread, so half a cent. The previous default of
    # 1.0 was twice too pessimistic, which made every backtested required-edge
    # figure too high rather than too low. Still a single observation on one
    # symbol at one time of day, so `scripts/measure_book.py` samples it properly
    # and `scripts/evaluate.py` stresses it either way.
    half_spread_cents: float = 0.5

    # ---- sizing and risk (a $100 account) --------------------------------
    starting_bankroll: float = 100.0
    # Fraction of full Kelly. Full Kelly on a binary at an extreme price asks
    # for a third of the account on a 5pp edge; a quarter is the largest value
    # that survives being wrong about the edge by a factor of two.
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05      # of bankroll, per position

    # A hard dollar cap per position, standing in for market depth. This is an
    # ASSUMPTION and an unmeasured one: nobody has read the depth of a Kalshi
    # 15-minute crypto book at a given price. It matters because it binds long
    # before a percentage cap does — at $25 it starts constraining as soon as the
    # account passes about $500 — and without it a backtest compounds a $100
    # account into size no venue could fill and reports the result as a return.
    # Measure the book, then set this from the measurement.
    max_stake_dollars: Optional[float] = 25.0

    # Size from the *starting* bankroll rather than the current one.
    #
    # Default off — meaning additive, non-compounding — because compounding turns
    # a per-trade edge estimate into an exponential, and an exponential is
    # dominated by the error in that estimate rather than by the estimate. On a
    # 2.8pp edge over 28,000 trades the compounded figure came out at 2e17
    # percent, which is arithmetic rather than a finding. With this off, the
    # equity curve's slope *is* the per-trade edge and can be read directly.
    #
    # Turn it on to project deployment, and label the output as a projection.
    compound: bool = False
    max_window_exposure_fraction: float = 0.08
    # The three symbols' 15-minute returns are ~0.7 correlated, so three
    # simultaneous same-direction positions are one position at three times
    # the size. Cap the count as well as the notional.
    max_positions_per_window: int = 2

    # One entry per (symbol, window). The four decision offsets are the same
    # bet observed at four moments, not four bets, so letting each fire
    # independently would put four times the intended size on one 15-minute
    # move. The live-honest rule is to walk the offsets in order and take the
    # first that clears every gate; `scripts/evaluate.py` reports edge per
    # offset separately, which is how the offset set gets narrowed on evidence
    # rather than by taking the best one in hindsight.
    max_entries_per_window: int = 1

    # Settlement is free; an exit is not. Selling a contract back pays a second
    # fee and crosses the spread a second time, so an early exit at 85c costs
    # 3.8pp against the 1.9pp of holding to settle. And there is no risk reason
    # to override that: a binary's loss is capped at the stake from the instant
    # of entry, so there is no liquidation to avoid and nothing a stop-loss
    # protects. An exit therefore has to be justified by the forecast flipping
    # far enough to beat a fresh round of costs, which is a high bar — hence
    # off by default, and `exit_edge_pp` on top of it when enabled.
    allow_early_exit: bool = False
    exit_edge_pp: float = 1.0
    min_contracts: int = 1                # round down; zero contracts is a skip
    # Surplus over break-even, in probability points, demanded before a trade
    # is considered at all. Abstention is the default action, and this is the
    # dial that decides how often it is overridden. It guards against
    # calibration error rather than against fees — the fee is already inside
    # break-even — so the right value is whatever the measured calibration
    # error turns out to be, and `scripts/evaluate.py` reports the whole curve
    # rather than assuming one.
    min_edge_pp: float = 0.5
    # Traded-price band, and it must be symmetric. The edge here is a
    # disagreement about sigma_remaining, and that disagreement points both
    # ways: a smaller sigma than the market assumes makes the probability more
    # extreme than the quote, so buy the favourite; a larger sigma makes the
    # favourite overpriced, so buy the longshot. A one-sided band such as
    # [0.55, 0.95] permits only the first and silently discards half the
    # strategy.
    #
    # What the ends exclude is where the *microstructure* becomes the dominant
    # uncertainty, not where the forecast is weak.
    #
    # The original justification for the low end was wrong and is worth
    # recording: "below 10c a one-cent tick is a 10% relative price error". The
    # venue's `price_level_structure` is `tapered_deci_cent` — the tick is a
    # *tenth* of a cent below 10c and above 90c, and a cent only in between. So
    # quantisation is finer in the tails, not coarser, and the real reason for
    # care at a low price is that the payoff is 50:1 and a small calibration
    # error dominates the expected value.
    min_traded_price: float = 0.05
    max_traded_price: float = 0.97

    # An outlier guard, not an economic gate. A sigma disagreement produces
    # modest departures from the quote; a 40-point departure is a bug, a stale
    # price, or a signal that is not the one under test. Rejecting it loudly is
    # better than sizing it.
    max_disagreement_pp: float = 25.0
    # Stop trading below this fraction of the starting bankroll.
    ruin_floor_fraction: float = 0.50

    # ---- provenance ------------------------------------------------------
    fee_config_path: Optional[str] = None
    fee_config_version: str = 'builtin_kalshi_v202608'
    cli_overrides: frozenset[str] = frozenset()

    # ---- derived ---------------------------------------------------------
    @property
    def settle_offset(self) -> int:
        """Minutes from window open to settlement."""
        return self.window_minutes

    def remaining_minutes(self, offset: int) -> int:
        """Wall-clock minutes from the decision to settlement.

        Reported, not used for scaling volatility — `remaining_variance_minutes`
        is what the barrier divides by, and the two differ because the settlement
        value is an average.
        """
        return self.window_minutes - offset

    def remaining_variance_minutes(self, offset: float) -> float:
        """Minutes of *variance* left between the last observed price and settlement.

        The settlement value is the mean over the final `settle_average_minutes`,
        not the price at the boundary. For a diffusion, the variance of that mean
        about a price observed at offset `m` is

            sigma^2 * ((window - delta - m) + delta / 3)

        The first term is the drift to the start of the averaging window; the
        `delta/3` is the variance of a time-average over an interval of length
        `delta`, which is a third of the variance of its endpoint. Ignoring the
        averaging overstates remaining variance — at offset 12 of a 15-minute
        window, by 3 minutes against 2.33, which is 13% on sigma.

        The baseline's fitted per-offset scale would absorb most of this, but
        absorbing a known analytic correction into a fitted nuisance parameter is
        how a fitted parameter stops meaning anything.
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
            maker_fee_rate=float(fees.get('maker_fee_rate', self.maker_fee_rate)),
            half_spread_cents=float(fees.get('half_spread_cents', self.half_spread_cents)),
            fee_config_path=str(path),
            fee_config_version=str(payload.get('version', 'unversioned')),
        )

    def provenance(self) -> dict[str, Any]:
        """The fields that change an answer, for the model artifact."""
        return {
            'symbols': list(self.symbols),
            'venue': self.venue,
            'timeframe': self.timeframe,
            'window_minutes': self.window_minutes,
            'decision_offsets': list(self.decision_offsets),
            'vol_lookbacks_minutes': list(self.vol_lookbacks_minutes),
            'baseline_distribution': self.baseline_distribution,
            'baseline_nu': self.baseline_nu,
            'n_folds': self.n_folds,
            'embargo_minutes': self.embargo_minutes,
            'recency_half_life_days': self.recency_half_life_days,
            'train_window_days': self.train_window_days,
            'fee_rate': self.fee_rate,
            'half_spread_cents': self.half_spread_cents,
            'fee_config_version': self.fee_config_version,
            'kelly_fraction': self.kelly_fraction,
            'min_edge_pp': self.min_edge_pp,
            'cli_overrides': sorted(self.cli_overrides),
        }


DEFAULT_CONFIG = Config()
