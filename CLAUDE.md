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
fee = ceil(0.07 * contracts * price * (1 - price) * 100) / 100
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

* **The ceiling is per order, not per contract.** One contract at 50c owes
  $0.0175 and is charged $0.02 — 14% more. At a $100 account every order is a
  small order, so this is the dominant correction rather than a rounding detail.
  `decide()` re-checks expected value against what will actually be charged,
  after rounding to whole contracts.
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
`allow_early_exit` exists, defaults off, and is expected to essentially never
fire.

**One entry per (symbol, window).** The four decision offsets are the same bet
observed at four moments, not four bets. Letting each fire independently puts 4x
the intended size on one 15-minute move. The live-honest rule is to walk the
offsets in order and take the first that clears every gate — at offset 3 you
cannot know what offset 12 will look like, so best-of-offsets is not a strategy
that can be run. `scripts/evaluate.py` reports edge per offset separately, which
is how the offset set gets narrowed on evidence.

The barrier framing predicts where that edge should live: `P` is most sensitive
to a sigma error when `|x|/sigma` is near 1, so the edge should peak mid-window
with a moderate displacement and decay late, because a probability pinned near 1
is insensitive to sigma. Untested — but it is a prediction, not a hope.

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
- **The settlement index is CF Benchmarks BRTI, not Coinbase spot.** The target is
  built from Coinbase bars because that is the history that exists, and Coinbase
  is a large BRTI constituent — a close proxy, not the same number. Live, the
  venue publishes `floor_strike` and `scripts/live.py` prefers it, so the basis is
  only taken in the backtest. It is an **unmeasured risk**.
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

Fourteen, ordered so the **forecast** is read before the **money** — a candidate
that fails on skill should not have its Sharpe discussed. On the old perp system
every gate read a simulated outcome, so a model 34x short of its cost hurdle
failed all of them without any saying why.

```
log_loss_skill        >= 0        the model must beat F(x/sigma)
folds_skill_positive  >= 5        (of 6; five agreeing is 10.9% likely by chance)
calibration_error     <= 0.02     the system trades its confident predictions
residual_scale        >= 0.25     how much of the correction survives
control_gain_share    <= 0.30     the clock must not carry the model
windows_evaluated     >= 20,000
trades                >= 200
coverage              >= 0.0005   abstaining on everything passes trivially
realised_edge_pp      >= 0
total_return          >= 0
sharpe                >= 0.5
sharpe_implausible    == 0        a Sharpe above 5 is a bug signature
max_drawdown          <= 0.35
halted                == 0
```

`sharpe_implausible` is the unusual one, and it earned its place immediately: the
first full run reported +12.6 and every other gate passed it. Every other gate
asks whether the number is good; this one asks whether it is possible.

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

# Tests. pytest.ini sets -n auto: 334 tests in ~61s.
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

**The honest state of things.** There are now ~2.6M one-minute bars per symbol
for BTC and ETH (2021-08 to 2026-08); SOL was still scraping at the time of the
audit. There is still no promoted model. A five-year BTC walk-forward run during
the audit reported mean log-loss skill +0.000897 +/- 0.000240 over 6/6 positive
folds — but read `AUDIT_REPORT.md` before believing it: the backtest's
counterfactual price *is* the baseline the model is fitted to correct, four of six
folds were scored with an unfitted shrinkage constant, the correction peaks at the
offset where the null is worst calibrated rather than mid-window as predicted, and
two of the top four features by gain are the clock control. At rho 0.7 between
folds, "6 of 6 positive" is a 22% event under the null, not 1.6%. The phase gates exist because the edge is a
*hypothesis*: `scripts/evaluate.py` failing is the expected outcome until proven
otherwise, and nothing about the live plumbing existing changes that. Trading
before Phase 3 passes is risking real money on an unestablished edge.

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
