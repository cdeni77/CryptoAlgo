# CLAUDE.md — Quarter

Full architecture and agent guidance lives in `AGENTS.md`. This file adds
Claude Code-specific notes and, more importantly, the reasoning that is easy to
lose.

**Read `AUDIT_REPORT.md` first.** An adversarial audit in August 2026 found that
the research core is sound — no lookahead, and the backtest and live paths compute
identical feature vectors to 16 decimal places — while everything that traded was
broken: the live loop could not score the window it was deciding, the paper
bankroll was never credited a win, and the scraper had been losing one minute in
every 301 for five years. Several claims in *this* file were among the things
disproved, and are corrected in place below. `AUDIT_FIX_PLAN.md` tracks what
remains.

## Project in One Sentence

Barrier-probability trading on Kalshi 15-minute BTC/ETH/SOL up-down markets:
FastAPI + PostgreSQL serving store, React/Vite dashboard, Python research
pipeline over one-minute Coinbase spot bars.

## Repository Layout

```
backend/api/        FastAPI read-only telemetry (Python 3.12, PostgreSQL)
backend/trader/     the pipeline: scrape, features, baseline, model, live
frontend/           React 18 + Vite + TypeScript + Tailwind
docker-compose.yml
AGENTS.md           architecture and conventions
```

## The reframe, and why everything below follows from it

**A 15-minute up/down market is not a direction bet.** It opens on a
quarter-hour boundary, records the price there as its strike, and settles on
whether the price at the next boundary is strictly above it. By the time a
decision is made, part of the window has already happened. So the question is
never "which way will it go" but:

> given that price has already moved `x` from the strike, and `n` minutes of
> movement remain, what is the chance it finishes above?

That is a barrier crossing, and its answer is `F(x / sigma_n)`. **The
displacement `x` is known exactly. The only forecast required is `sigma_n`** —
and volatility is the one thing this project has ever measured as forecastable
(the old dispersion head scored an out-of-sample IC of +0.34 while every
direction head sat near +0.02).

### The benchmark is not 50%. It is `F(x / sigma_n)`

This is the single most important line in the repository, and getting it wrong
would produce the most convincing false positive it has ever generated. A model
fed mid-window state will report 70-90% accuracy. **None of that is alpha.** At
nine minutes in with three left, a 20bp displacement against 8bp of remaining
sigma is already 99% settled — anyone with a clock and a volatility estimate
knows it, and the market prices it.

Measured on synthetic bars, the baseline alone takes log loss from 0.693 to
0.513 — a 26% improvement over a coin flip, from arithmetic, with no features
and no model. Against 50% that reads as a 40-point edge. It is a clock.

So every measurement in this system is **incremental against the baseline**:
log-loss skill, Brier skill, edge in probability points. `scripts/baseline.py`
exists to print the null before anything else runs, and it gates on the
baseline's own out-of-sample calibration — skill measured against a
miscalibrated baseline is partly the baseline's error.

### What the baseline is allowed to fit, and what it must not

It fits a scale factor per decision offset and a tail thickness. Both are
arithmetic: one-minute returns carry bid-ask bounce which inflates realised
variance, the last observed price is up to a minute stale, and `sqrt(n)` is an
approximation at n = 3.

**It has no drift, and never will.** A non-zero drift *is* the alpha under test.
A null allowed to fit one absorbs the finding and reports no skill for the wrong
reason. `tests/test_baseline.py::test_the_baseline_has_no_drift_to_fit` plants a
drift and asserts the baseline leaves it visible.

**Scale and tail are not separately identified.** From binary outcomes only the
composite map `z -> P(up)` is determined, and a thicker tail with a larger scale
mimics a thinner tail with a smaller one. Measured: against a sigma inflated
1.2-1.4x the fit returned `scale ~ 1.001` and `nu = 2.93` — it put the whole
correction in the tail parameter. So do not read `scale` as "the sigma
inflation". Judge the pair the only way it can be judged: by whether the
resulting probabilities are calibrated out of sample.

## The economics, which are why this venue and not the last one

Kalshi charges, per order:

```
fee = ceil(0.07 * contracts * price * (1 - price) * 10_000) / 10_000
```

Settlement is free, so a held-to-expiry binary pays **one** fee, not a round
trip. The `p(1-p)` term is the whole reason the barrier framing and this venue
fit together: **a confident bet is a cheap bet**, which is the opposite of a
perpetual future's fixed toll.

| price | fee/contract | share of stake | required edge (0.5c half-spread) |
|------:|-------------:|---------------:|---------------------------------:|
|   50c |      $0.0175 |          3.50% |                           2.25pp |
|   70c |      $0.0147 |          2.10% |                           1.96pp |
|   85c |      $0.0089 |          1.05% |                           1.37pp |
|   90c |      $0.0063 |          0.70% |                           1.10pp |
|   95c |      $0.0033 |          0.35% |                           0.80pp |

**The half-spread is 0.5c, measured.** The live BTC 15-minute book quoted
0.19/0.20 and 0.10/0.11 — a one-cent spread. The previous default of 1.0c was
twice too pessimistic, so every required-edge figure in this table used to be too
high. It is still one symbol at one time of day; `scripts/measure_book.py`
samples it properly and `scripts/evaluate.py` stresses it either way.

**The book is deeper than the sizing rules need.** That same order book had
thousands of contracts within a few cents of the touch, and 441 resting at the
best level — about $48. `max_stake_dollars` at $25 is therefore conservative
rather than optimistic, which is the direction to be wrong in. `decide()` now
prefers a measured depth over that guess whenever the row carries one.

Three consequences worth holding onto:

* **The ceiling is per order, not per contract, and it barely bites.** This used
  to say a contract at 50c owes $0.0175 and is charged $0.02 — 14% more — and
  called that "the dominant correction rather than a rounding detail". Measured
  against 328 real fills, the granularity is a hundredth of a cent, not a whole
  cent: that order pays exactly $0.0175, and the per-order effect is worth under
  a hundredth of a cent. The old whole-cent assumption over-charged 7% in
  aggregate and ~17% on the smallest orders, which made every net-edge gate too
  strict and refused trades that were profitable. `decide()` still re-checks
  expected value against what will actually be charged, after rounding to whole
  contracts — rounding is per order, so splitting an order can only cost more,
  never less.
* **The half-spread overtakes the fee at 92.26c** at the current 0.5c default,
  where `0.07*p*(1-p)` falls below half a cent. Above that the assumption is the
  larger cost, which is why it is a separate, stressed parameter. This number has
  now been wrong twice: "~60c", then "83c" — which is the crossover against a
  *one-cent* half-spread, the old default, and was left behind when the measured
  spread halved it. At 83c the fee is still $0.0099, about 2x the half-spread.
  `tests/test_costs.py` derives it; the prose is the thing that keeps drifting.
* **The price band must be symmetric.** The edge here is a disagreement about
  `sigma_remaining`, and that points both ways: a smaller sigma than the market
  assumes makes the probability *more* extreme than the quote, so buy the
  favourite; a larger sigma makes the favourite overpriced, so buy the longshot.
  A one-sided band such as [0.55, 0.95] permits only the first and silently
  discards half the strategy. **This was not hypothetical: the shipped default was
  [0.05, 0.97].** 1 - 0.97 is 0.03, so it admitted 96-97c favourites and refused
  3-4c longshots — and the asymmetry ran the wrong way on cost, since at 96c a 1pp
  calibration error destroys ~43% of the gross edge and at 4c about 1%. It is
  [0.05, 0.95] now, and `Config.__post_init__` refuses an asymmetric band rather
  than leaving it to this paragraph.
* **The tick is tapered, and this corrects a stated reason.** The venue's
  `price_level_structure` is `tapered_deci_cent`: a *tenth* of a cent below 10c
  and above 90c, a full cent in between. The band's low end used to be justified
  as "below 10c a one-cent tick is a 10% relative price error", which is wrong by
  a factor of ten — quantisation is finer in the tails, not coarser. The real
  reason for care at a low price is that the payoff is 50:1, so a small
  calibration error dominates the expected value.

### Two policy decisions, both driven by the fee shape

**Hold to settle.** Settlement is free; an exit pays a second fee and crosses the
spread again — 3.8pp against 1.9pp at 85c. And unlike a perp there is no risk
reason to override that: a binary's loss is capped at the stake from the instant
of entry, so there is no liquidation to avoid and nothing a stop-loss protects.
There is no early-exit code path at all — not a flag defaulting off, an
unconditional policy. An earlier `allow_early_exit` config field implied a gated
mechanism existed; it did not, and was removed rather than left describing a
feature nothing built.

**One entry per (symbol, window).** The four decision offsets are the same bet
observed at four moments, not four bets. Letting each fire independently puts 4x
the intended size on one 15-minute move. The live-honest rule is to walk the
offsets in order and take the first that clears every gate — at offset 3 you
cannot know what offset 12 will look like, so best-of-offsets is not a strategy
that can be run. `scripts/evaluate.py` reports edge per offset separately, which
is how the offset set gets narrowed on evidence.

The barrier framing predicted where that edge should live: `P` is most sensitive
to a sigma error when `|x|/sigma` is near 1, so the edge should peak mid-window
with a moderate displacement and decay late. **That prediction has now been tested
and it is wrong.** On 326 days of real bars the skill peaks at the *earliest*
offset and is dead by offset 9:

```
offset      n   mean_skill  folds+
     3  79765     0.000368     6/6
     6  79769     0.000265     5/6
     9  79770    -0.000004     3/6
    12  79770     0.000103     4/6
```

And the mechanism is not sigma at all. `vol_state` alone scores **-0.000101**
(2/6 folds) while `cross_asset` alone scores **+0.000183 at t=+3.39, 6/6 folds** —
the strongest single group, and the only one with independent prior support (the
archive's cross-sectional residual at h=4h: +0.0186, t=4.54, 6/6 folds). An
earliest-offset peak is the wrong shape for a sigma error and the right one for
lead-lag: a BTC move needs time for ETH and SOL to follow, so twelve remaining
minutes express it and three do not.

So the honest state of the thesis: the barrier reframing is still what makes the
question tractable and the null strong, but the *edge* on top of it looks like
cross-asset lead-lag concentrated at offset 3, not a volatility disagreement.
Offsets 9 and 12 should probably go. See the Edge Investigation section of
`AUDIT_REPORT.md` for the full measurement, including why the money numbers from
that run must not be used to choose a configuration.

## Architecture

```
scripts.scrape  ->  SQLite  ->  scripts.sync_store  ->  Parquet + DuckDB
                                                              |
                                    core/dataset.py  <---------+
                                          |
       core/windows.py (the 15-min grid, one row per symbol/window/offset)
                                          |
              core/vol.py (HAR + intraday seasonality)  ->  sigma_remaining
                                          |
              core/baseline.py  F(x/sigma)  ->  THE NULL
                                          |
       core/features.py (4 mechanisms + 1 control)  ->  core/model.py
                    (LightGBM on the baseline's logit as init_score)
                                          |
              core/decide.py  decide()  ->  core/backtest.py walk-forward
                                          |
              core/metrics.py gates  ->  core/promotion.py  ->  models/forecast.joblib
                                          |
                                  scripts.live  ->  Kalshi
```

### The market data pipeline: one schema per concept for the BOOK — `venue_quotes` is still a fifth, separate thing

Collection used to be four incompatible things for the book — Kalshi live
ladders, Polymarket live ladders, a Predexon backfill in a JSONL file outside
the research store, and the three were joined into `venue_depth`. **This
section used to also claim `venue_quotes` was folded in; it was not, and
`scripts/build_depth.py` never reads it.** `venue_quotes` remains a Kalshi-only
Predexon backfill at an irregular seven offsets (2, 3, 4, 6, 9, 12, 14),
written separately by `scripts/backfill_quotes.py`, and roughly a dozen files
still read it directly (`retro_economics.py`, `refit_market_init.py`,
`research/analysis/_book_analysis.py`, `research/analysis/_offset_vs_market.py` among them). Folding it into
`venue_depth` too is real, deliberately deferred work — a different offset
grid, a different producer, a dozen consumers to repoint — not a rename.

```
                     live (recorded)          backfill (Predexon / gamma)
  raw ladder      venue_ladder  (kalshi)      -- (tick series is packed)
                  pm_ladder     (polymarket)
  summarised      venue_depth <-------------- venue_depth      every minute 0..15
  settlement      venue_settlements <-------- venue_settlements  both venues
  spot bars       minute_bars <-------------- minute_bars        five years
  quotes, sparse  --                          venue_quotes      Kalshi only, 7 offsets, NOT unified
```

`scripts/build_depth.py` is the one path into `venue_depth`, from every BOOK
source, at **every minute** — because the offset grid is itself under test and
a table sampled where the model currently scores would foreclose the question.
`source` separates a book somebody recorded from the same book reconstructed
afterwards; `research/validate/_validate_depth.py` compares them where they overlap, which is the
only independent evidence the backfill describes the same object. That check
needs the backfill to run up to *two hours* ago rather than a day, or the two
never share a minute.

`scripts/collect_settlements.py` writes both venues' own results into one table
and stops at history it already holds, so keeping it current is a page or two
rather than 200 requests. **Predexon's market metadata reaches much further back
than its order books** — ~196 days of settlements against ~70 days of book.

Everything Predexon and Polymarket serve refuses a bare `Python-urllib` or
default `aiohttp` User-Agent with a Cloudflare 1010. The header is not
decoration.

### The model predicts a *correction*, not a probability

The baseline's logit enters LightGBM as an `init_score`, so the model fits the
residual. Three things follow, and they are the reason for the choice:

* An untrained model reproduces the baseline **exactly**, so every tree it grows
  is incremental skill by construction rather than by comparison.
* The objective is the quantity being traded. The previous incarnation regressed
  net return and took its sign, which counted every flat bar as a miss and
  optimised magnitude accuracy nobody was paid for.
* Overconfidence is one number. `residual_scale` (alpha) is a single coefficient
  fitted on held-out training rows: how much of the claimed correction survives.
  The last version of this repo discovered its predictions were 34x too
  confident only by regressing realised on predicted after the fact.

### Features, grouped by the mechanism each exploits

A feature earns its place only by naming a way the baseline is *wrong*.

| group | the claim |
|---|---|
| `vol_state` | the baseline's sigma is mis-estimated, and that error is predictable |
| `microstructure` | its drift is zero and reality's is not (bid-ask bounce, flow bursts) |
| `cross_asset` | BTC leads ETH and SOL at the minute scale; the barrier sees one symbol |
| `geometry` | the baseline is Markov and the path is not — a window that spiked 40bp and came back to 2 is not one that drifted to 2 |
| `clock` | **the control.** Time of day cannot forecast direction |

Keep `clock` in any survey. The previous project ran a 27-cell grid whose best
cell was its own control, and that was the most useful result it produced.
`ForecastModel.control_importance_share` is a gate at 0.30 for the same reason.

**But the gain-share gate is measuring the wrong quantity, and an ablation is the
real test.** Measured on 326 days: `control_gain_share` read 0.279 — nearly
carrying the model, on the face of it — while `clock` *alone* scored
**-0.000008 (t=-0.26, 2/6 folds)** and removing it slightly *improved* skill. A
high LightGBM gain share means splits were spent there, not that the feature
forecasts anything. Run the group alone before believing the share either way.

## Invariants — break these and the numbers stop meaning anything

- **Both ends of the target are one-minute averages, and a tie resolves UP.**
  Read off a live market's own `rules_primary`: *"the simple average of the sixty
  seconds of CF Benchmarks' BRTI before 12:45 ... is **at least** the simple
  average of the sixty seconds ... before 12:30"*. So the strike is the mean over
  `[t0 - 1min, t0)`, the settlement value is the mean over `[t1 - 1min, t1)`, and
  the comparison is `>=` because `strike_type` is `greater_or_equal`. An earlier
  version used `open(t0)`, `open(t1)` and a strict `>` — a defensible reading of
  "up/down in the next 15 minutes", and wrong in three places at once. **That fix
  landed in `core/windows.py` and not in `scripts/live.py`**, where `settle_due`
  kept reading `open(t1)` with a strict `>` until the audit; measured, the two
  disagreed on 3.4-8.2% of windows, and its docstring still justified the `open`
  on the grounds that "the strike was read the same way".

  The `>=` is right because it matches `strike_type`, and for no other reason.
  The justification this file used to give — that "flat windows are not rare on a
  minute grid" — is wrong by four orders of magnitude: **1 exact tie in 173,937**
  real BTC windows. Both ends are OHLC means of a liquid asset, so ties are
  essentially measure-zero. Relatedly `windows.base_rate` claimed the base rate
  should sit slightly *below* 0.5; measured it is **0.5009 (BTC) / 0.5031 (ETH)**,
  above, exactly as `>=` implies.
- **A window's strike is the previous window's settlement value.** Both are the
  mean over the same minute. Consecutive markets chain, which is a real
  structural dependence and one more reason the embargo is a day.
- **Averaging reduces variance, so the barrier divides by less than the clock
  says.** The unresolved quantity is a one-minute mean, and the variance of a
  time-average over an interval is a third of its endpoint's — remaining variance
  at offset `m` is `(W - delta - m) + delta/3`, which at `m=12` is 2.33 minutes
  rather than 3. Ignoring it overstates sigma by 13%, and the baseline's fitted
  scale would have quietly absorbed it; a fitted parameter that absorbs a known
  analytic correction stops meaning anything.
- **The settlement index is CF Benchmarks BRTI, not Coinbase spot, and the
  proxy is now MEASURED.** The target is built from Coinbase bars because that is
  the history that exists, and Coinbase is a large BRTI constituent — a close
  proxy, not the same number. This used to say "an **unmeasured risk**", and that
  was true only because nothing here held the venue's own answer.
  `venue_settlements` does now: Predexon serves `result` on every settled Kalshi
  market (56,385 of them, ~196 days) and Polymarket publishes `winning_side`.

  Measured against 56,284 shared windows, **Kalshi's own settlement agrees with
  our label 96.98% of the time**, and the disagreement is exactly the shape a
  benign proxy should have — it lives entirely in the near-ties:

  ```
  move from strike   windows   disagree
       <1bp             2806     34.85%
       1-2bp            2826     13.91%
       2-5bp            7860      3.18%
       5-10bp          11101      0.32%
      10-25bp          18147      0.05%
       >25bp           13472      0.01%
  ```

  A near-tie on Coinbase is a coin flip on BRTI; a real move is never
  mislabelled. Polymarket, settling on **Binance**, agrees on 96.96% of its 493
  windows, and the two venues agree with *each other* on **99.52% of 209 shared
  windows** — one disagreement. Two independent settlement sources converging
  that tightly is the strongest available evidence that `core/windows.py` builds
  the target correctly.

  **So the risk is bounded, not absent: ~3% of every training label is wrong.**
  Against a measured log-loss skill of +0.001 that is not negligible, and it
  belongs in how that skill is read. `research/validate/_validate_label.py` recomputes it.
- **A one-minute OHLC mean stands in for sixty seconds of index prints.** Both
  ends use the same approximation, so most of its bias cancels in the comparison
  — which is the only reason it is tolerable.
- **A decision at offset `m` sees the close of bar `m-1` and nothing after.** A
  one-minute leak in a fifteen-minute window is 7% of the whole question and
  reads exactly like skill.
- **Cross-validation splits on the window, never the row.** Four offsets share
  one settlement, so a row split puts offset 3 in train and offset 12 in test —
  the same fifteen minutes on both sides, one of them nine minutes closer to
  knowing. The embargo is a **day**, not fifteen minutes: what needs the day is
  `log_rv_1440`, because a training row just after a test block computes it from
  test-period bars.
- **Standard errors come from fold dispersion.** Not `N/(1+(N-1)rho)`: offsets
  share labels, the three symbols are ~0.7 correlated within a window, and a
  breadth formula on that structure is degenerate. Six folds give five degrees of
  freedom — honestly few, which beats a precise-looking number from the wrong
  formula.
- **One `decide()`.** The backtest, the live writer and the paper engine all call
  it. The previous repo had per-family strategy classes and the backtest and live
  path disagreed about entry price for months.
- **`price_source` on every row.** A backtest has no quotes and stands the
  calibrated baseline in for the market; a live decision reads the real ask. Those
  are different claims, and a row that cannot distinguish them makes a backtest
  look like a fill.
- **`kelly_fraction` is also an edge filter, and the two are not independent.**
  `decide()` floors the stake to whole contracts, so a smaller Kelly fraction does
  not just stake less — it pushes marginal trades under one contract and refuses
  them. Measured on 326 days, 0.25 -> 0.10 left `edge_below_gate` *identical* at
  242,571 while `below_min_contracts` went 1,813 -> 8,218; realised edge per
  contract rose +0.99pp -> +3.32pp and drawdown fell 58% -> 21%, because the
  survivors were the higher-edge trades rather than because the sizing was safer.
  `max_stake_fraction` is nearly inert at this edge size by comparison: Kelly binds
  first, and cutting that cap fivefold barely moved the drawdown.
- **Sizing is additive by default** (`compound=False`). Compounding turns a
  per-trade edge estimate into an exponential, and the exponential is dominated
  by the *error* in the estimate. The first full run compounded $100 into
  $2 x 10^17 and reported it as a return. With compounding off, the equity
  curve's slope *is* the per-trade edge.
- **`max_stake_dollars` stands in for market depth**, defaulting to $25. It is an
  **assumption and an unmeasured one** — nobody has read the depth of a 15-minute
  Kalshi book. It binds as soon as the account passes ~$500, and without it a
  backtest trades size no venue could fill.
- **Promotion is the gate.** `core/promotion.py` stages, then atomically renames
  into place, only if every gate passed. Rejections stay in `models/promotions/`
  because **the ledger is the trial count**, and a project that deletes its
  failures cannot compute its own multiple-testing correction. `--force` needs a
  written reason and records it.
- **The API serves measurements, never substitutes.** A missing value is null
  with a reason. The old research surface reported `pr_auc` as
  `holdout_auc - 0.06` and a hardcoded table of feature importances; all of it
  rendered identically to real data.
- **`account.mode` reaches every surface showing one of its numbers.** A live
  account that renders identically to a paper one is the worst failure the schema
  could permit.
- **Live P&L is the venue's, and a missing field is not a zero.** Our books
  estimate all three of the fill price, the fee and the settlement value; the venue
  holds them. `venue_settlements.pnl` is null when the venue left a field absent,
  and nothing downstream may read that as break-even — the API counts it as
  `incomplete` and says the total is short. Stored rather than derived on read
  because the API cannot import the trader, and one definition of the arithmetic
  beats two.

## Three things are fitted, and all three live inside the fold

The volatility model, the intraday seasonality factor, and the baseline's
scale/tail. None is a headline number, which is exactly why a leak would go
unnoticed — a seasonality factor fitted on the full sample makes the baseline
stronger and the model look *weaker*, and nobody audits a result in that
direction.

`ScoringBundle` carries all three with the promoted artifact.
`ForecastModel.deployable` reports whether they are there. Before that existed,
an artifact could be evaluated and not deployed, and nothing said so until the
live path tried to score a window.

## Gates

**Eighteen**, ordered so the **market comparison** is read before the
**forecast** before the **money** — a candidate that fails on skill should not
have its Sharpe discussed. This table used to list fourteen; four were added
later (`market_windows`, `model_minus_market`, `calibration_max_deviation`,
`non_finite_share`) and the prose was never updated to match. On the old perp
system every gate read a simulated outcome, so a model 34x short of its cost
hurdle failed all of them without any saying why.

```
market_windows             >= 2,000  enough live-recorded quotes to compare against
model_minus_market         >= 0      beats the PRICE, not just F(x/sigma) — see below
log_loss_skill             >= 0      the model must beat F(x/sigma)
folds_skill_positive       >= 5      (of 6; five agreeing is 10.9% likely by chance)
calibration_error          <= 0.02   the system trades its confident predictions
calibration_max_deviation  <= 0.04   worst adequately-populated bin, not the mean
non_finite_share           <= 0.001  a NaN prediction is a data hole, not a forecast
residual_scale             >= 0.25   how much of the correction survives
control_gain_share         <= 0.30   the clock must not carry the model
windows_evaluated          >= 20,000
trades                     >= 200
coverage                   >= 0.0005 abstaining on everything passes trivially
realised_edge_pp           >= 0
total_return                >= 0
sharpe                      >= 0.5
sharpe_implausible          == 0     a Sharpe above 5 is a bug signature
max_drawdown                 <= 0.35
halted                       == 0
```

`market_windows` and `model_minus_market` cannot be computed from a backtest,
which has no book — both read NaN and fail until the live loop has recorded
enough quotes. That is the honest state of the question rather than an obstacle
to route around; `--force` with a written reason is the documented way past it,
and the ledger records that it was used.

`sharpe_implausible` is the unusual one among the rest, and it earned its place
immediately: the first full run reported +12.6 and every other gate passed it.
Every other gate asks whether the number is good; this one asks whether it is
possible.

## Commands

```bash
# Phase 0 — the data. ~2.6M minutes per symbol, ~8,800 requests each, a few
# hours. Resumable: an interrupted run continues rather than restarting.
cd backend/trader
python -m scripts.scrape --backfill-days 1825
python -m scripts.scrape --fill-gaps        # recover any batch that gave up
python -m scripts.sync_store

# Phase 1 — the null. Read this before believing any model result.
python -m scripts.baseline
python -m scripts.baseline --compare-distributions

# Phase 2-4 — the model, and whether it pays
python -m scripts.train                       # one model, for inspection
python -m scripts.evaluate                    # walk-forward, gates, cost stress
python -m scripts.evaluate --groups clock     # the control, alone
python -m scripts.promote                     # install, gates permitting
python -m scripts.promote --history           # what has been tried, and why not

# Phase 5 — trading
python -m scripts.check_venue                       # prove the key, read-only
python -m scripts.live                              # paper
python -m scripts.live --mode live --dry-run        # real book, no orders
python -m scripts.live --mode live --place-orders   # real orders
python -m scripts.live --loop --cycle-seconds 60
python -m scripts.sync_venue                        # pull the venue's own ledger
python -m scripts.sync_venue --dry-run              # read and total, write nothing

# Tests. pytest.ini sets -n auto: ~650+ tests in ~90s (`-m "not slow"` for the
# fast loop). The exact count is not pinned here on purpose — this line said
# 207, then 230, then 381, and each was wrong within days. Run
# `pytest --collect-only -q` for the true count rather than trusting a number
# in prose.
cd backend/trader && pytest
cd backend/api && pytest

# Frontend
cd frontend && npm ci && npm run dev
cd frontend && npm run typecheck && npm run lint && npm run build
```

## Going live

`--place-orders` is a separate flag from `--mode live` on purpose: one flag
guarding an irreversible action is one typo away from being wrong. The client
also refuses to place unless constructed `live=True`, and `--require-gates` (the
default) refuses to trade an artifact whose promotion was blocked.

**Markets are resolved by asking the venue** which market closes when this window
settles — never by building a ticker from a series prefix and a date. A pattern is
a guess that keeps working until the venue renames a series, and then it finds
nothing or, worse, the wrong contract. Resolution returns None and the cycle
abstains.

Credentials: `KALSHI_KEY_ID` plus `KALSHI_PRIVATE_KEY` (the PEM) or
`KALSHI_PRIVATE_KEY_PATH`. Auth is RSA-PSS over SHA-256 of
`timestamp + METHOD + path`, not an HMAC secret.

**Live, the venue is the account of record.** In paper mode the bankroll is
arithmetic and settlement comes from our own bars. Live, both are estimates of
someone else's ledger, and where they disagree the venue is right:

* Balance from `/portfolio/balance` every cycle. Our running figure is kept
  alongside and the drift is logged rather than silently overwritten — a drift
  that grows is an unrecorded fill, a partial, or a fee we mispriced.
* Settlement from `/portfolio/settlements` where it knows. Ours is an OHLC mean
  of Coinbase standing in for sixty seconds of CF Benchmarks BRTI, which is a
  close proxy and not the same number.
* Fills read back, not assumed. A `fill_or_kill` that killed leaves a ticket and
  no position, and a position the venue does not report is logged as exactly
  that.

`--no-reconcile` turns it off, which is only reasonable for debugging.

**P&L comes from `/portfolio/settlements`, and the endpoint that looks like the
answer is not.** A binary bought once and held pays one fee at entry and settles
once at $1 or $0, so a settlement row is a position's complete economic history:
`yes_total_cost`/`no_total_cost`, `revenue`, `fee_cost`, `market_result`. Realised
P&L is `revenue - cost - fee_cost` per market, summed. Those rows are stored in
`venue_settlements`, the fills that opened them in `venue_fills`, and the balance
sampled every cycle in `venue_balance` — all three written by the reconcile the
live loop already performs, so the ledger costs no extra request.

`/historical/trades` is **the public tape**: every print in a market, by anyone,
with no account attribution at all. It is the natural thing to reach for and it
cannot compute a portfolio — summing it sums the exchange. Its two honest jobs are
marking an open position at a price the market printed and checking that a fill
printed where the venue said it did, joined on `trade_id`; both live in
`KalshiClient.market_trades` and neither is P&L.

Three consequences that were each a plausible wrong answer:

* **Zero is a measurement.** `_price` maps a zero *quote* to None, because a zero
  level means there is nothing there. A losing position settles at revenue exactly
  0, so `_money` keeps it — reusing the quote parser would have turned every loser
  into a null and flattered the curve, the one direction of error an equity curve
  must never make.
* **The curve is built from settlement P&L, never from balance differences.**
  Nothing in the ledger distinguishes a deposit from a profit, so a
  balance-difference curve reports the first deposit as the best day the strategy
  ever had. `balance_check` compares the two and reports an `implied_starting_balance`;
  what matters is whether it *moves* between syncs, which is what a double-counted
  fee or an unrecorded fill looks like.
* **The win rate reads `market_result`, not the sign of the P&L.** This system
  buys favourites: 100 contracts at 97c returns $100 on $97 of cost, and a fee
  above $3 makes the net negative. Classifying that as a loss would put the win
  rate at odds with the venue's own settlement record.

**Since 2026-02-19 the ledger has two tiers.** The live endpoints refuse to look
back past a moving cutoff (~3 months) and everything older answers on
`/historical/fills` and `/historical/settlements` — on a *different host*,
`external-api.kalshi.com`, which is why `KalshiClient` carries a separate
`historical_base_url`. `all_fills` and `all_settlements` query both and
deduplicate on `trade_id`/`ticker`, because the tiers overlap around the cutoff
and a fill counted twice doubles a cost basis. `GET /historical/cutoff` is read
rather than assumed; unreachable, both tiers are queried anyway.

Our own arithmetic is kept beside the venue's rather than replaced: the dashboard
shows both and `pnl_gap` is the disagreement, which is a mispriced fee, a
settlement our Coinbase proxy called differently, or a fill nobody booked.
`scripts/sync_venue.py` prints all of it, and is the tool for a store built before
these tables existed or a gap while the loop was down.

**Prices arrive as dollar-denominated strings.** The venue serves
`yes_bid_dollars: "0.1900"`, and the integer-cent fields the older documentation
describes (`yes_bid: 19`) come back `null`. Reading only the latter parsed every
quote as empty and reported "no two-sided book on any symbol" against a market
quoting 0.19/0.20 with 1,594 contracts on the bid. Both encodings are accepted
now. Sizes are `_fp` fixed-point strings on the same pattern.

`scripts/check_venue.py` proves all of it before any money moves, and answers the
four questions separately so a failure names itself: does the PEM load, does the
venue accept the signature, do the series tickers exist, and can a market be
resolved for the next window. It constructs the client without `live=True`, so it
cannot place an order even if something in it tried. It also prints the real
book, which is how the assumed 1c half-spread gets checked against a measured
one.

**The series are `KXBTC15M` / `KXETH15M` / `KXSOL15M`**, and the `15M` matters.
`KXBTCD` was tried first: it exists, it has hundreds of open markets, and every
window abstained — because it closes on the *hour* and its tickers carry an
explicit strike (`KXBTCD-26AUG2317-T86749.99`), making it a threshold ladder
rather than an up/down market. `KXBTC15M-26AUG230030` is series + date + HHMM
with no strike suffix, which is the tell: the strike is the price at the window's
open, which is what `core/windows.py` builds.

That abstention was the resolution logic working. A ticker built from a pattern
would have found *something* 15 or 30 minutes away and traded it.

**The backfilled book is quantised to whole cents, and the venue's is not.**
Kalshi's `price_level_structure` is `tapered_deci_cent` — a tenth of a cent below
10c and above 90c — and `GET /markets/{ticker}/orderbook` returns those levels:
103 of them on a live BTC market, priced `0.0010, 0.0020, 0.0030 ...`. Predexon's
snapshot of the same book returns 21, priced as integer cents. It is not a
truncated ladder — total resting size agrees at a ratio of 1.000 when the two are
matched to the same instant — it is a **coarser price grid in the tails**.

Two consequences, and the second is the one that bites:

* `levels_bid` / `levels_ask` are **not comparable across sources** (measured
  ratio 0.579, unchanged by any time filter). A feature built on level counts
  must be source-consistent or it is measuring which pipe the row came through.
* **The backfill cannot represent a tail price to better than a cent**, and the
  traded band is [0.05, 0.95]. Measured against 3,680 live top-of-book
  observations, which carry the venue's true deci-cent precision:

  ```
  mid band     n      best bid off the cent grid   mean rounding error
  below 10c    652              90.3%                    0.230c
  10c - 90c   2474               0.4%                    0.001c
  above 90c    554              91.2%                    0.250c
  ```

  **Inside [0.10, 0.90] the backfill is exact** — the tick is a full cent there
  and there is nothing to round. Outside it roughly nine quotes in ten are
  rounded, by ~0.24c on average: about half the measured 0.5c half-spread, and
  15% of the 1.5pp edge gate. Roughly a third of all observations sit in that
  region, and 41% of live contracts are bought under 15c.

  So economics from backfilled quotes should either restrict to [0.10, 0.90] or
  carry an explicit +/-0.25c uncertainty in the tails. The live path reads Kalshi
  directly and is unaffected.

**A Polymarket slug's trailing unix stamp is the window's OPEN, not its close.**
Read as a close it shifts every window fifteen minutes, and *nothing raises*:
every window is a valid window and every book is a real book. It surfaced only
as agreement — Kalshi 96.98% against our label, Polymarket 49.85%, and Kalshi
against Polymarket exactly 50.0%, which places an alignment error in the mapping
rather than a quality problem in either venue. The venue states it three ways
that agree: the slug stamp, the title ("9:30PM-9:45PM ET"), and `end_time`,
which is the stamp plus fifteen minutes. A live recorder building the slug with
`ceil` names the *next* window, which already exists and already trades — so the
request returns a healthy book that is then stamped with the current window's
open. A wrong answer that looks entirely right.

**`no_levels` is NO-denominated on both venues, and that is a conversion, not a
convention.** Kalshi's orderbook is two BID stacks (`yes_dollars`, `no_dollars`),
so the YES ask is `1 - best_no_bid`. Polymarket's CLOB serves `bids`/`asks` on
one token, so its asks are YES-denominated and are converted at write time in
`scripts/record_pm_ladder._no_levels`. Storing them as served put a 0.51 YES ask
in the column holding a 0.51 NO bid on the other venue: same name, opposite
meaning, wrong by the spread with imbalance inverted, and no exception anywhere.

**Expect ~0.4% of minutes to be missing, and 86% of them were our own bug.** A
five-year BTC backfill returned 2,617,876 of 2,628,000 minutes. This file used to
say most of the shortfall was minutes in which nothing traded. That was false, and
it was the stated reason `--fill-gaps` skipped single-minute holes by default —
so the repair tool skipped precisely the holes the scraper was creating.

`get_candles_range` asked for a 300-minute span with `limit=300`, and Coinbase
treats both `start` and `end` as inclusive. The request therefore named 301 candle
starts, the venue returned the newest 300, and the loop advanced past the one it
dropped: **one minute in every 301, for five years.** Five independent
confirmations — 98.9% of the gaps between holes were exactly 301 minutes apart;
the rate was 0.003318 against 1/301 = 0.003322; it was flat across UTC hour,
weekday and year; and 5,121 of BTC's 8,721 isolated holes fell on the identical
minute in ETH against 28.9 expected by chance. Two independent order books do not
go untraded in the same 8,700 minutes.

Fixed, and `--min-gap-minutes` now defaults to 1 so the singles are re-requested.
Genuinely untraded minutes do exist; re-asking is cheap and idempotent, so the
default errs toward asking. **Run `--fill-gaps` on any store built before this.**

The cost of a missing minute is specific now that both ends of the target are
one-minute averages: only a minute at index 14 of a window matters, and each one
kills two windows (that window's settlement and the next one's strike). At 0.4%
missing spread across all fifteen positions, that is roughly 0.8% of windows lost —
`Dataset.coverage()` reports it per symbol and `load_dataset` warns above 2%.

**The honest state of things, corrected from the audit's snapshot.** This used
to say SOL was still scraping and there was no promoted model — both were true
at the time of the audit and neither is now. All three symbols hold ~2.63M
one-minute bars each (2021-08 to present), and a model has been promoted and
trading live on a $100 account since 2026-08-23
(`models/promotions/20260823T144827Z.json`).

That does not retire the caveat underneath it, which is about how to read any
one walk-forward run rather than about whether a model exists. The five-year
BTC run from the audit reported mean log-loss skill +0.000897 +/- 0.000240 over
6/6 positive folds — read `AUDIT_REPORT.md` before believing a number like that
at face value: the backtest's counterfactual price *is* the baseline the model
is fitted to correct, four of six folds were scored with an unfitted shrinkage
constant, the correction peaks at the offset where the null is worst calibrated
rather than mid-window as predicted, and two of the top four features by gain
are the clock control. At rho 0.7 between folds, "6 of 6 positive" is a 22%
event under the null, not 1.6%. The phase gates exist because the edge is a
*hypothesis*, and passing `scripts.evaluate` once is not the same claim as the
edge being established — that is measured continuously, not settled by one
promotion.

## Archive: what has already been rejected, so it is not re-run

The previous system traded Coinbase CDE perpetual futures directionally. **That
was rejected four independent ways**, and none of it should be revisited:

1. **Cost arithmetic.** Required IC is `round_trip_cost / sigma_h`. Best measured
   direction IC was +0.012 at h=1h against a required 0.404 — **34x short**, and
   7x short at h=24h. The round trip was 27-65bp; the edge was 0.8bp.
2. **A 27-cell pre-registered survey** found 3 hits against 3.0 expected by
   chance, and **the highest-scoring cell in the grid was the control**
   (`seasonality,cost`, identity_ratio 1.0147 — it was ranking by instrument
   identity).
3. **A sign-inverted simulation** made gross PnL *worse*, so the loss was not an
   inverted signal.
4. **h=96h measured end to end**: Sharpe -3.87, and the binding gate moved from
   `edge_below_cost` to `participation_limit` at 70% of rejections — short holds
   cannot pay the toll, long holds cannot be filled at size.

Two secondary findings from that work that still apply here: a cross-sectional
residual signal at h=4h was real (+0.0186, t=4.54, 6/6 folds over five years and
four regimes) and worth **2.2bp against a 29.3bp toll**; and the useful output of
the whole line of work was a threshold, not a model — **break-even needs a round
trip at or below ~2.5bp**. Kalshi's `0.07*p(1-p)` reaches 0.33c at 95c, which is
why this venue is worth trying at all.

Also settled and not worth re-deriving: CDE publishes no historical funding
endpoint, so carry could only accumulate forward at ~0.09bp/hour against a 27bp
round trip (zero of eighteen contracts covered a round trip within 48 hours);
Coinbase spot at 1.20%/side is 9x worse than the perps; and the funding and
open-interest archive has been deleted, so nothing in this repository can reason
about carry any more.

## Environment Variables

See `AGENTS.md` for the full table. Minimum for live workflows:
`COINBASE_API_KEY`, `COINBASE_API_SECRET`, `DATABASE_URL`, `POSTGRES_PASSWORD`
(compose refuses to start without it), `KALSHI_KEY_ID` and
`KALSHI_PRIVATE_KEY`/`KALSHI_PRIVATE_KEY_PATH` for `--mode live`, and
`API_TOKEN` + `VITE_API_TOKEN` for the dashboard's job runner. `FEE_CONFIG` is
optional — unset still prices correctly, because the defaults are the published
schedule, but records no version.
