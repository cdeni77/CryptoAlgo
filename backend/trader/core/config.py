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

    # Which of those offsets may actually OPEN a position.
    #
    # Every offset above is still scored — that sample is what `market_benchmark`
    # and the retroactive forecast test read, and narrowing it would destroy the
    # measurement. This narrows only entries.
    #
    # Measured over 70 days and 19,339 symbol-windows, per-contract edge after the
    # measured fee and a 0.5c half-spread, one entry per (symbol, window), 1pp
    # gate:
    #
    #     earliest offset that clears (what ran)   0.040c   t=0.10
    #     +9m or +12m                              1.206c   t=2.68
    #     +12m only                                3.304c   t=5.98
    #
    # The loop books one position per symbol-window and takes whichever offset the
    # clock has passed, so the earliest that clears wins and `already_entered`
    # locks out the rest. In production 250 of 277 settled entries — 90% — landed
    # at +3m, the weakest cell, and exactly one landed at +12m. At +3m the plain
    # barrier baseline is genuinely *worse* than the market (-0.00065 log loss),
    # so the trades clearing there are disproportionately false positives, and
    # taking one forfeits the +12m opportunity in the same window.
    #
    # Robust on every split: all three symbols, both calendar halves (4.475c vs
    # 4.282c at a 2pp gate), all three months, 80% of days positive, and
    # strongest at spreads <= 1c (t=5.95) — i.e. in the most tradeable books.
    #
    # **None means every scored offset may enter**, which is what a backtest
    # and every evaluation sweep need — narrowing the research default would
    # make `scripts.evaluate` unable to measure the very cells this comment
    # cites. The narrowing is a *deployment* choice: `scripts.live` defaults
    # `--entry-offsets` to 12, so the trading policy is explicit at the
    # command line and visible in `docker-compose.yml` rather than buried in a
    # library default.
    #
    # (9, 12) is the conservative widening if the coverage loss matters.
    entry_offsets: Optional[tuple[int, ...]] = None

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
    # Which forecaster the LightGBM correction is fitted on top of: 'baseline'
    # for `F(x/sigma)`, 'market' for the recorded quote's implied probability.
    #
    # 'baseline' is the default and the only one currently fittable. 'market' is
    # the better objective and is blocked on data, not on code: over 1,109
    # live-recorded rows the market's log loss was 0.331 against the baseline's
    # 0.428, so a baseline-init model spends itself correcting the forecaster
    # that is already 0.10 nats behind the price. It also inverts the null in the
    # right direction — an untrained market-init model reproduces the price, so
    # nothing trades, where an untrained baseline-init model disagrees with the
    # price by 5.79pp on average and trades on it.
    #
    # 285 symbol-windows of quotes exist against a `windows_evaluated >= 20,000`
    # gate. `core/model.py` refuses clearly rather than substituting.
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
    # Fractional Kelly. **This is also an edge filter, and that is not obvious.**
    #
    # `decide` floors the stake to whole contracts, so a lower Kelly fraction does
    # not merely stake less — it pushes marginal trades below one contract and
    # refuses them as BELOW_MIN_CONTRACTS. Measured on 326 days, dropping 0.25 to
    # 0.10 left `edge_below_gate` *identical* at 242,571 while
    # `below_min_contracts` went 1,813 -> 8,218 and trades fell 3,221 -> 1,941.
    # Realised edge per contract rose from +0.99pp to +3.32pp and the drawdown fell
    # from 58% to 21%, not because the sizing was safer but because the surviving
    # trades were the higher-edge ones.
    #
    # So `kelly_fraction` and `min_edge_pp` are coupled: anyone lowering Kelly to
    # control drawdown is also raising the effective edge threshold, and anyone
    # reading the two as independent knobs will be surprised. `max_stake_fraction`
    # below, by contrast, is close to inert at this edge size — Kelly binds first,
    # and cutting that cap 5x barely moved the drawdown.
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05      # of bankroll, per position

    # A hard dollar cap per position, standing in for market depth. This is an
    # ASSUMPTION and an unmeasured one: nobody has read the depth of a Kalshi
    # 15-minute crypto book at a given price. It matters because it binds long
    # before a percentage cap does — at $25 it starts constraining as soon as the
    # account passes about $500 — and without it a backtest compounds a $100
    # account into size no venue could fill and reports the result as a return.
    # Measure the book, then set this from the measurement.
    # How much of the depth measured at the touch a stake may claim.
    #
    # Sizing to 100% of it is sizing to a number that has already moved: the
    # quote is read ~4s before the order is sent.
    #
    # Measured on 392 windows of real book at +12m where the two samples are
    # genuinely 45s apart: the size resting at the touch retains a median 1.00x
    # but only **0.29x at the 25th percentile** (bid 0.31x). A quarter of the
    # time, over two thirds of the touch is gone within a minute.
    #
    # An earlier reading put that p25 at 0.55x and was contaminated: 30% of
    # collected rows had `after` equal to `at_decision` — the same snapshot twice,
    # when every book update in the queried span fell before the decision instant
    # — each contributing a spurious 1.00x.
    #
    # 0.5 is chosen against the ~4s horizon that actually applies, not the 45s
    # the measurement spans, so it sits between no headroom and the p25 of a
    # window ten times longer. It is a judgement, and the fuller collection now
    # running will let it be set from the 4s decay directly.
    #
    # On the median trade this is free — 241 contracts resting against ~9
    # wanted, so the cap does not bind. It bites only in the thin tail (p10 is
    # 6 contracts), which is exactly where `fill_or_kill_insufficient_resting_volume`
    # was coming from.
    depth_fraction: float = 0.5
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
    # How many of the three symbols may hold a position in one window.
    #
    # This was 2, justified as "the three symbols' 15-minute returns are ~0.7
    # correlated, so three simultaneous same-direction positions are one position
    # at three times the size." **Measured, that overstates it.** On 122 live
    # windows where all three settled, the pairwise correlation of settle
    # direction is BTC-ETH +0.607, BTC-SOL +0.590, ETH-SOL +0.656 — mean +0.618,
    # with all three agreeing 71.3% of the time against 25% under independence.
    #
    # High, but not one bet. The variance of n unit bets at rho = 0.618:
    #
    #   1 leg   sd 1.00x   2 legs  sd 1.80x   3 legs  sd 2.59x
    #
    # so the third leg adds 50% of the stake for 44% more risk. Per unit of risk
    # that is 1.16 against 1.11 for two — slightly *better*, provided the edge is
    # real and roughly even across symbols.
    #
    # The reason it is 3 now is measurement, not that argument. `decide()` walks
    # symbols alphabetically and refuses at the exposure gates *before* computing
    # an edge, so a binding cap dropped whoever came last in the alphabet and
    # recorded no edge for them: SOL was blocked 184 times against BTC's 104 and
    # ETH's 95. Per-symbol performance cannot be compared when one symbol is
    # structurally starved of entries, and the cap was the thing making the
    # comparison unfair. At 3 it stops binding, so the ordering bias mostly
    # disappears with it.
    #
    # `max_window_exposure_fraction` still caps the notional, which is the limit
    # that was actually doing the work.
    max_positions_per_window: int = 3

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
    # Raised from 0.5 on the out-of-sample edge curve, five years and six folds.
    # `realised_edge_pp` is monotone in the gate across all three sweep cells
    # (0.85 -> 2.80, 0.99 -> 2.68, 1.11 -> 2.95), which is only true if the edge
    # estimate carries signal — filtering on a noise estimate would not raise
    # realised edge. Live agreed independently: the two lowest edge buckets lost
    # and the two highest won.
    #
    # 1.5 rather than the argmax of total return, because that argmax is noise —
    # the same curve computed three ways peaks at 0.25, 1.00 and 1.50. What the
    # data supports is "tighter is better per trade"; where to stop is a judgment.
    # At 1.5 total expected edge is flat against 0.5 (-2%) for 40% less exposure,
    # so it is the same money at 26% better return per unit of risk — and 40% less
    # capital at risk is right under both hypotheses, since if the edge turns out
    # to be negative it loses 40% less.
    min_edge_pp: float = 1.5
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
    #
    # Symmetric about 0.5, deliberately. It was [0.05, 0.97], and 1 - 0.97 is
    # 0.03, not 0.05 — so the band admitted 96-97c favourites (the smaller-sigma
    # view) while refusing 3-4c longshots (the larger-sigma view), silently
    # clipping half the thesis. The asymmetry also ran the wrong way on cost: at
    # 96c a 1pp calibration error destroys ~43% of the gross edge, at 4c about
    # 1%. Keep `max_traded_price == 1 - min_traded_price`; the invariant is
    # asserted in `__post_init__`.
    min_traded_price: float = 0.05
    max_traded_price: float = 0.95

    # An outlier guard, not an economic gate. A sigma disagreement produces
    # modest departures from the quote; a 40-point departure is a bug, a stale
    # price, or a signal that is not the one under test. Rejecting it loudly is
    # better than sizing it.
    # Kept at 25.0 deliberately, having tried 8.0 and measured what it costs. A
    # sigma disagreement at an extreme price moves the probability a long way:
    # at a 0.88 quote, believing sigma is much larger takes P(up) to 0.70, an
    # 18-point departure that is the strategy working rather than a bug. Several
    # tests encode exactly that, and tightening this gate silently deletes the
    # buy-the-longshot half of the thesis — the same failure the asymmetric band
    # caused. The misparsed-quote risk this looked like it was guarding is
    # actually handled by the price band (a cents-as-dollars quote reads 20.0 and
    # is out of band; a 100x-low one reads 0.002 and is too) and by refusing to
    # trade live without a real two-sided quote.
    max_disagreement_pp: float = 25.0

    # ---- order envelope --------------------------------------------------
    # How much of the claimed edge may be paid away to get filled, and the hard
    # cap on that in cents. The live path used to send `price + edge` as the
    # limit — literally the break-even price — so under `fill_or_kill` a thin
    # book could walk the order to a zero-EV fill and call it a trade. Measured:
    # 0.7832 sent against a 0.60 ask, 18c of slippage tolerance on a 1c spread.
    slippage_share_of_edge: float = 0.25
    max_slippage_cents: float = 1.0

    # ---- freshness -------------------------------------------------------
    # There was no staleness guard of any kind. The feed's last `event_time` was
    # logged and never asserted against the wall clock or against the decision
    # minute, so a delayed or partial fetch was scored as though it were current:
    # ten missing minutes moved the displacement from +4.93bp to +2.41bp and the
    # cycle traded anyway. A quote is worse — the book moves within the window,
    # and `now` was read once at the top of the cycle and never revalidated
    # before the order went out, with a Coinbase fetch, four authenticated
    # reconcile calls, inference and six 15-second quote calls in between.
    max_bar_age_seconds: int = 150
    max_quote_age_seconds: int = 45
    # Refuse to enter when this little of the window remains. `choose_offset`
    # floors to whole minutes, so a row built as "minute 12" could be submitted
    # at minute 14.9 carrying sigma for 2.33 minutes when 0.1 remained.
    min_remaining_seconds: int = 45
    # Stop trading below this fraction of the starting bankroll.
    ruin_floor_fraction: float = 0.50

    # ---- circuit breakers ------------------------------------------------
    # The ruin floor was the only limit that existed, and it only fires after
    # half the account is gone — 96 windows a day x 3 symbols means a broken
    # model bleeds continuously, and the nominal worst case was $768/day against
    # a $100 bankroll. These bound the day and the streak instead, and unlike the
    # floor they are recorded on the account so they survive a restart. Clearing
    # a halt is deliberately manual: an automatic reset makes a circuit breaker a
    # speed bump.
    max_daily_loss_fraction: float = 0.15
    max_consecutive_losses: int = 12
    # Peak-to-current drawdown on realised equity, live. Same 0.35 the promotion
    # gate applies to the backtest — a threshold worth enforcing on a simulation
    # is worth enforcing on the account.
    #
    # It exists because the daily-loss rule cannot see this shape. Measured on the
    # first two days: equity ran $100 -> $166.86 by 13:00 UTC and gave back $63.92
    # over the next ten hours, all inside one UTC day. Realised for that day was
    # **+$3.81** against a -$15.00 limit, so the daily rule saw a good day while
    # the account was down 38.3% from its high and nothing was watching.
    max_drawdown_fraction: float = 0.35

    # ---- provenance ------------------------------------------------------
    fee_config_path: Optional[str] = None
    fee_config_version: str = 'builtin_kalshi_v202608'
    cli_overrides: frozenset[str] = frozenset()

    # ---- invariants ------------------------------------------------------
    def __post_init__(self) -> None:
        """Refuse a configuration whose numbers contradict each other.

        These were all comments before, and every one of them was violated by the
        shipped defaults or reachable from a CLI flag.
        """
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
        # What the embargo must actually cover, for the *expanding* scheme this
        # repo uses. Features are all trailing, so a training row's lookback
        # extends backwards, away from the test block — the feature lookback
        # length is NOT the binding constraint, and requiring the embargo to
        # exceed it would be superstition. What genuinely reaches forward is:
        #   - the label, which settles `window_minutes` after the window opens;
        #   - `core/vol.py:forward_realised_vol`, the HAR target, which reads to
        #     `window_open + window_minutes`.
        # Both need `window_minutes`. The 1440 default is deliberately far
        # larger, and becomes load-bearing only if the scheme ever goes rolling,
        # where a training block *follows* a test block and `log_rv_1440` would
        # then read across it.
        #
        # This check exists because `core/cv.py:assert_no_leakage` cannot do it:
        # it compares each fold's gap against the *configured* embargo, which the
        # fold was built from, so it passes at every value — measured, it passes
        # at `--embargo-minutes 0`.
        if self.embargo_minutes < self.window_minutes:
            raise ValueError(
                f'embargo_minutes {self.embargo_minutes} is under window_minutes '
                f'{self.window_minutes}. A training window inside that gap '
                f'settles after the test block starts, so its label is built '
                f'from test-period bars, and the HAR target reaches there too. '
                f'The default of 1440 is deliberately much larger.'
            )
        if not 0.0 <= self.slippage_share_of_edge < 1.0:
            raise ValueError(
                f'slippage_share_of_edge must lie in [0, 1]; got '
                f'{self.slippage_share_of_edge}. At 1.0 the limit price is the '
                f'break-even price and a fill is worth nothing.'
            )

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
            # Which forecaster the correction sits on. Without it in the ledger,
            # two attempts with the same `log_loss_skill` could be measuring
            # skill over two different benchmarks.
            'init_score_source': self.init_score_source,
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
