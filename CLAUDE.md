# CLAUDE.md — CryptoAlgo

Full architecture and agent guidance lives in `AGENTS.md`. This file adds Claude Code-specific notes.

## Project in One Sentence

Full-stack crypto trading research platform: FastAPI + PostgreSQL backend, React/Vite frontend, Python ML/optimization trader pipeline — all orchestrated via Docker Compose for Coinbase CDE/perpetual workflows.

## Repository Layout

```
backend/api/        FastAPI service (Python 3.12, PostgreSQL)
backend/trader/     ML pipeline: data collection, features, training, optimization
frontend/           React 18 + Vite + TypeScript + Tailwind CSS
docker-compose.yml
AGENTS.md           Detailed architecture and coding conventions
```

## Running Tests

```bash
# All trader tests. `pytest.ini` sets `-n auto`, so this is already parallel:
# ~5m50 across four cores against ~9m50 serially. The tail is one 209s
# promotion evaluation, so more workers buy little.
cd backend/trader && pytest

# The fast loop — everything except the model-training end-to-ends. 33s.
cd backend/trader && pytest -m "not slow"

# Serial, when a traceback or a debugger matters
cd backend/trader && pytest -n 0 tests/test_<name>.py -v

# API tests (serial; they share one SQLite fixture)
cd backend/api && pytest
```

Tests live in `backend/trader/tests/` (16 files) and `backend/api/tests/` (5). No frontend tests exist — `package.json` has no test script and no runner.

## Key Commands

```bash
# Start full stack
docker compose up --build db backend frontend trader paper-engine

# Frontend dev (hot-reload)
cd frontend && npm ci && npm run dev

# API dev (hot-reload)
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload

# Trader scripts (run from backend/trader/). The seven research scripts —
# train, backtest, promote, search, signals, preflight, build_features — share
# scripts/_common.py:add_data_arguments, so they cannot disagree about the
# dataset. The four operational ones below (run_pipeline,
# migrate_to_research_store, paper_engine, live_orchestrator) hand-roll their
# own argparse and take a different set.
python -m scripts.run_pipeline                 # scrape into SQLite
python -m scripts.migrate_to_research_store    # sync SQLite -> Parquet store
python -m scripts.build_features               # assemble the feature panel
python -m scripts.preflight                    # can this train? run before a long scrape
python -m scripts.train                        # fit a model for inspection
python -m scripts.backtest                     # walk-forward + simulation stack + gates
python -m scripts.promote                      # evaluate and install, gates permitting
python -m scripts.promote --history            # what has been tried, and why not
python -m scripts.search                       # one campaign runner, append-only ledger
python -m scripts.signals                      # decide() on the latest bar
python -m scripts.paper_engine                 # act on signals, account honestly
python -m scripts.live_orchestrator            # the hourly loop: scrape -> sync ->
                                               # features -> signals, plus a
                                               # promotion attempt on its own
                                               # cadence. It does not run search,
                                               # train, backtest or the paper
                                               # engine — paper_engine is its own
                                               # compose service.

# Frontend checks
cd frontend && npm run typecheck && npm run lint && npm run build

# API tests
cd backend/api && pytest
```

## The Research Pipeline

`docs/RESEARCH_PIPELINE.md` is the design spec. Read it before changing anything
under `core/`.

The old path is gone: `scripts/train_model.py` (2,857 lines),
`features/engineering.py`, `core/labeling.py`, `core/meta_labeling.py`,
`core/coin_profiles.py` and the five search scripts have all been deleted. There
is one of each thing now, and the data flows one way:

```
run_pipeline  ->  SQLite  ->  migrate_to_research_store  ->  Parquet + DuckDB
                                                                    |
                                    core/dataset.py  <--------------+
                                          |
              core/features.py  ->  panel   +   core/targets.py  ->  net returns
                                          |
                                   core/model.py  (three heads: price, carry, sigma)
                                          |
                    core/signal.py decide()  ->  core/backtest.py walk-forward
                                          |
                          core/simulation.py  ->  core/metrics.py gates
                                          |
                                  core/promotion.py  ->  models/forecast.joblib
                                          |
                          scripts/signals.py  ->  scripts/paper_engine.py
```

**The formulation changed.** The old system classified triple-barrier outcomes
behind a momentum gate. It could not express carry, which is the most plausible
edge on hourly-funding perps — though at the funding rates since measured it is
0.1-0.5bp/hour against a 27-65bp round trip, not the 2bp/hour these docs assumed.
`core/targets.py` now regresses *net return*, decomposed into price, carry and
cost, and `net_long + net_short == -2 * cost` holds exactly.

**Nothing reaches live except through the gates.** `core/promotion.py` trains,
walk-forward backtests, bootstraps, stresses and gates a candidate, then installs
it only if every gate passed. Rejections stay in `models/promotions/` because the
trial count is what the deflated Sharpe discounts by.

## Step one: can the instrument pay for its own trading?

```bash
python -m scripts.instrument_screen                       # every horizon, both gates
python -m scripts.instrument_screen --horizons 24,96,168
```

`required IC = round_trip_cost / sigma_h`. Cost is fixed per round trip and
dispersion grows as `sqrt(h)`, so the bar an instrument sets is a property of the
venue and the contract, settled before a model exists. This screen was built
last. Everything below it was built first, and most of what it cost could have
been avoided by dividing 27bp by 46bp on day one.

**Two gates, and they pull opposite ways.** `required_ic` falls as `1/sqrt(h)`;
the effective sample falls as `1/h` and is capped again at roughly `3 x
half_life` of history. Reporting either alone recommends a horizon the other
forbids. Measured on this store, 18 contracts, half-life 50d:

| horizon | required IC (best / median) | win-rate ceiling | effective obs | cost | sample |
|---|---:|---:|---:|:--:|:--:|
| 1h  | 0.251 / 0.404 | 54% | 27,621 | no | yes |
| 4h  | 0.128 / 0.203 | 75% | 6,899 | no | yes |
| 24h | 0.051 / 0.082 | 91% | 1,143 | no | yes |
| **96h** | **0.024 / 0.045** | **95%** | **279** | **yes** | **yes** |

**h=96h is the only horizon on this venue that clears both.** That contradicts
the section below, which opens by rejecting 96h for having 18 effective
observations — and both numbers are right. The 18 is a *single series*; the panel
is pooled across 18 instruments, which is 279. The pooled figure is what
`cross_validate_forecast` fits on, so it is the relevant one, and the older
number is not.

Two cautions before reading this as a recommendation. Pooled is generous: pairwise
correlation across this book is 0.658, so 18 instruments are nearer 1.5
independent ones, and 279 pooled observations is perhaps 25 independent. And
h=96h is **untested** — clearing the cost gate says it is the only place worth
trying, not that a forecast exists there. The best measured direction IC at any
horizon is +0.018 at h=24h with 3 of 6 folds agreeing, which is a coin flip.

The `--min-effective-obs` and `--max-required-ic` thresholds are arguments, so a
run can disagree with them rather than silently accept whatever it finds.

## Before You Train: the horizon governs the sample size

Overlapping labels are not independent observations. A label spanning `h` bars
overlaps its `h-1` neighbours, so the effective sample is roughly
`timestamps / h`, not `timestamps`. Using the row count as the sample size is how
a t-statistic ends up several times too confident, and it is what the promotion
gates are calibrated against.

Measured on 92 days of hourly data across five instruments:

| horizon | effective observations | verdict |
|---------|-----------------------|---------|
| 96h (the profile default) | 18 from 1,768 timestamps | far too few |
| 8h | 232 from 1,856 timestamps | enough to start |

### The recency half-life is usually the binding constraint, not the history

Uniqueness is only half the calculation. Training then multiplies each row by
`0.5 ** (age_days / H)` where `H` is `Config.recency_half_life_days`, and the
product is what the model is fitted on. Those weights sum to about
`24 x H / ln 2` bar-equivalents **no matter how far back the store goes**, so the
weighted sample saturates at roughly `24 x H / ln 2 / h` and more history stops
helping past about `3H`.

Measured against `core/cv.py`, at the default `H = 50` days:

| history | horizon | uniqueness says | training sees |
|---------|---------|-----------------|---------------|
| 1 year  | 96h | 91  | 17 |
| 2.2 years | 96h | 200 | 18 |
| 5 years | 96h | 456 | **18** |
| 5 years | 24h | 1,825 | 72 |
| 5 years | 8h  | 5,475 | 216 |

So at the profile default horizon, five years of history buys one effective
observation over one year. Scraping more data is the wrong lever there; the
half-life is. Same five years, varying `H` instead:

| horizon | H=50d | H=180d | H=365d | H=730d | off |
|---------|-------|--------|--------|--------|-----|
| 96h | 18 | 64 | 127 | 216 | 456 |
| 24h | 72 | 259 | **510** | 867 | 1,825 |
| 8h  | 216 | 778 | 1,530 | 2,601 | 5,475 |

24h with `H = 365d` is the combination the cost schedule also argues for: hours
of funding carry needed to cover a round trip are 2.8h (XRP), 2.9h (DOGE), 3.1h
(SOL), 9.9h (BTC), 24.1h (ETH), so a 24h hold pays for itself on carry alone on
four of five contracts. At 96h you must set `H` near 730d to clear the gate,
which nearly disables the decay and defeats its purpose.

`scripts/preflight.py` reports both numbers and names whichever lever binds —
including telling you plainly when more history cannot help. **Run it before a
long scrape**, not after.

```bash
python -m scripts.preflight                                        # profile default
python -m scripts.preflight --horizon 24 --recency-half-life-days 365
python -m scripts.preflight --horizon 8                             # a shorter hold
```

Both controls are on every research script (`scripts/_common.py`), and
`live_orchestrator` forwards them to the training step only — a window passed to
the signal writer would truncate the panel it has to score the latest bar from.
`--train-window-days` is a hard cut applied by `Dataset.trailing()`;
`--recency-half-life-days` is the soft weighting. They are different instruments
and both are useful.

## The target must start from a price you can still buy

`price_return` measured `close(t+h) / close(t) - 1`. That is wrong here, and it
invalidated every performance number this repo produced before 2026-08-21.

A bar's `available_time` is the moment it closes, so a decision using bar `t` is
made at `t+1` and the earliest price it can fill at is `open(t+1)`. But `close(t)`
is the *last trade* in bar `t`, and on a thin nano perp that can be twenty minutes
before the bar ends while Coinbase spot keeps moving. The basis computed from that
stale print looks extreme, the next trade corrects it, and a close-anchored target
books the correction as profit.

Measured on 399 days, the 14 contracts with >= 231 days, three walk-forward
quarters:

| measurement | IC |
|---|---:|
| `basis_z_168h` vs the `close(t) -> open(t+1)` gap alone | **-0.50** |
| `basis_z_168h` vs the old close-to-close target | -0.167 |
| `basis_z_168h` vs `open(t+1) -> open(t+1+h)` | -0.0065 |
| cross_venue+trend model, h=1h, close-to-close | **+0.114** |
| the same model, same split, open-to-open | **+0.002** |

**Ninety-eight percent of the apparent edge was unreachable.** That single fact
explains the central puzzle: reported IC of 0.05-0.17 alongside backtests that
lost money on every configuration. `core/simulation.py` always entered at the next
open — correctly — and the metric it was scored against never did.

Everything that looked like a finding dissolves into it. `cross_venue` "beating all
61 features"; basis mean reversion at IC -0.167 stable across three quarters;
shorter horizons scoring better (the gap is a larger share of a shorter horizon);
the per-instrument table favouring the thinnest contracts (the thinnest have the
stalest closes). All one artifact.

`entry='next_open'` is the default. `entry='close'` exists only to reproduce an old
artifact — nothing should train on it, and
`tests/test_targets.py::test_the_target_cannot_be_filled_at_a_price_that_is_already_gone`
fails if `build_targets` reaches for it.

### What the data actually supports, on the honest target

**Direction: nothing.** Fifteen group-by-horizon combinations, walk-forward across
three quarters, tradeable target. Per-quarter IC, and how many of the three shared
the mean's sign:

| group | h=1h | h=2h | h=4h |
|-------|------|------|------|
| cross_venue | +0.006 +0.011 +0.018 (**3/3**) | -0.003 -0.006 +0.015 (1/3) | -0.004 +0.007 -0.004 (1/3) |
| cross_venue+trend | +0.002 +0.005 +0.007 (3/3) | +0.004 -0.014 +0.011 (2/3) | -0.002 -0.006 +0.018 (1/3) |
| trend | +0.003 -0.000 -0.001 (1/3) | -0.001 -0.016 +0.003 (1/3) | +0.004 -0.009 +0.013 (2/3) |
| liquidity | -0.014 -0.008 -0.007 (0/3) | +0.002 -0.008 +0.002 (2/3) | +0.001 -0.019 -0.005 (1/3) |
| all populated | +0.005 +0.005 -0.009 (2/3) | +0.010 +0.015 -0.005 (2/3) | +0.018 +0.002 -0.037 (2/3) |

The largest is `cross_venue` at h=1h: +0.012, positive in all three quarters, and
an expected edge of **0.8bp against a 27bp cheapest round trip**. It also flips
sign at h=2h and h=4h. Three-of-three sign agreement happens by chance in a
quarter of cells, and 15 were tested, so three or four such cells are expected from
noise. Treat this table as a negative result, and don't mine it.

`cross_venue` remains the most interesting group because it has a *mechanism* — a
thin nano perp lagging the deep Coinbase spot book its own index is built from —
rather than a fitted pattern. But a mechanism that produces 0.8bp against a 27bp
toll is not a strategy.

**How far off, in the units that decide it.** Required IC is `cost / sigma_h`,
and cost is fixed per round trip while `sigma` grows as `sqrt(h)`, so the ratio
improves only with the hold. Measured on this store, median across the 18
contracts:

| horizon | required IC (median) | range | measured |
|---------|---------------------:|-------|---------:|
| 1h  | 0.404 | 0.25 (NER) - 0.82 (SHP) | +0.012 |
| 2h  | 0.287 | 0.18 - 0.59 | flips sign |
| 4h  | 0.203 | 0.13 - 0.42 | flips sign |
| 24h | 0.082 | 0.05 - 0.18 | untested |

The best measured IC is 34x short of what h=1h needs and 7x short of h=24h. To
close it on horizon alone — required IC scaling as `1/sqrt(h)` — the hold would
have to reach roughly **48 days**, at which point the effective sample is a few
dozen observations. There is no horizon at which this forecast pays for the
round trip.

### The cross-sectional residual is the target the features were built for

Two measurements, taken together, changed the best available configuration.

**Breadth.** Removing the cross-sectional mean takes effective breadth from 1.38
names to 13.82, because the residuals are near-uncorrelated by construction:

| horizon | raw rho | breadth | residual rho | breadth | SE gain |
|---|---:|---:|---:|---:|---:|
| 1h  | +0.651 | 1.48 | -0.086 | 13.82 | 3.06x |
| 4h  | +0.704 | 1.38 | -0.086 | 13.82 | 3.17x |
| 24h | +0.717 | 1.36 | -0.085 | 13.82 | 3.19x |
| 96h | +0.753 | 1.29 | -0.116 | 10.89 | 2.91x |

**Capacity.** `scripts.model_capacity --demeaned-target` runs the ladder against
that residual. Every rung at h=4h turns positive, where against the raw target
almost all were negative, and the fold agreement is unlike anything else measured
here — `ridge_all`, `stump_depth2`, `lgbm_tiny` and `lgbm_production` all at
**6 of 6**. Five of the 24 residual cells reach 6/6, against 0.75 expected by
chance.

So the signal is real and replicated across model classes. Then the economics:

| horizon | sigma_resid | required IC | measured | x required | eff obs | SE | sigmas | folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4h  |  78.8bp | 0.3718 | +0.0279 | **0.08x** | 3,702 | 0.016 | 1.70 | 6/6 |
| 24h | 189.3bp | 0.1548 | +0.0136 | 0.09x | 617 | 0.040 | 0.34 | 5/6 |
| 96h | 398.3bp | 0.0736 | **+0.0784** | **1.07x** | 122 | 0.091 | 0.86 | 6/6 |

**h=96h is the first cell in this entire project whose measured IC exceeds its own
break-even requirement** — 1.07x, on 6 of 6 folds, with the smallest
generalisation gap on the ladder (0.049 against the production head's 0.686).

And it is still not established. At 122 effective observations per fold the IC
standard error is 0.091, so +0.0784 is 0.86 standard errors from zero: the point
estimate clears the hurdle and the interval contains zero comfortably. 6/6
agreement is 3.1% per cell by chance, which is suggestive and not proof across 24
cells.

**The vise, one notch better.** h=4h is measurable (1.70 sigma) and 12x short of
its requirement. h=96h clears its requirement and cannot be measured. Same shape
as before, but for the first time one end of it is on the right side of the cost
line.

**What would settle it is exactly the backfill already on the table.** To put the
h=96h estimate 4 standard errors from zero needs 2,603 effective observations
against the ~729 the store holds — a factor of 3.6, or about **1,424 days of
history**. The 1,825-day spot backfill covers it with room to spare. That is the
first time a data request in this project has had a specific number behind it
rather than a hope.

Two caveats that have to travel with this result. It is a *market-neutral*
formulation — long the top residuals, short the bottom — and
`core/simulation.py` and `decide()` are built for directional single positions, so
none of the gate stack has been run against it. And nothing here is promoted:
`ic_covers_cost` reads the price head against the raw target, so a residual model
would need its own measurement wired in before it could pass a gate.

### h=96h was the last untested cell, and it fails on every gate

The cost screen said h=96h was the only horizon clearing both the economics and
the sample gate — required IC 0.024-0.045 against a 95% win-rate ceiling, 279
effective observations. It had never been run. Run against the 14 contracts with
>= 231 days, half-life 365d, honest target:

    213 trades | net -21,509 (price -11,549, funding +0, fees 9,960)
    Sharpe -3.87 | maxDD 21.8% | win rate 39.9% | liquidations 0
    gates: 213 of 70,077 accepted (0.30%)
    bootstrap: Sharpe median -3.48 [p05 -5.25, p95 -1.67] | P(positive) 0% | ruin 98.7%
    cost stress: baseline -3.87, fees 2x -5.57, spread 3x -4.24, both -6.34 | survives: no
    per-period: 6 walk-forward paths, all negative, positive fraction 0.00

**The economics gate did exactly what the screen predicted, and it did not help.**
Acceptance rose from 0.003% at h=1h to 0.30% here — a hundredfold, because the
forecast now clears the round trip far more often. The win rate rose from 24% to
39.9%. Both moved the way the arithmetic said they would. The Sharpe got *worse*,
from -2.36 to -3.87.

**And the binding constraint changed, which is the useful part.** The dominant
gate is no longer `edge_below_cost` (6,157) but **`participation_limit` at
49,387 — 70% of all rejections.** At a four-day hold the sized position is large
relative to a bar's volume, so the book cannot absorb it. Below h=24h this system
was constrained by the fee schedule; at h=96h it is constrained by the fact that
these are nano contracts on a thin venue. There is no horizon where neither
binds: short holds cannot pay the toll, long holds cannot be filled at size.

Fees are 9,960 of the 21,509 loss — 46%. Price PnL is -11,549. Both halves are
still broken, and cost stress kills it in every scenario.

This closes the last configuration that had an argument behind it rather than a
hope. Direction on this venue has now been rejected four independent ways: the
cost arithmetic (34x short at h=1h, 7x at h=24h), a 27-cell pre-registered survey
(3 hits, 3.0 expected, and the winner was the control), a sign-inverted simulation
(inverting made gross PnL *worse*), and h=96h measured end to end.

### The feature panel is built for a strategy this venue cannot afford

`cross_sectional_standardize` converts 8 of the 9 feature groups to z-scores
across the universe at each bar — so the **common component is removed from the
features**. The target keeps it. Measured on this store at h=4h:

    share of target variance that is the common (market) component:  69.9%
    sd(raw price target)       143.6bp
    sd(demeaned price target)   78.8bp

And the two halves of the panel behave in opposite directions, which is the
mismatch stated as a measurement:

| | mean abs IC vs **raw** target | vs **demeaned** target | improve when demeaned |
|---|---:|---:|---:|
| the 52 standardized features | 0.0067 | **0.0128** | 45 / 52 |
| the 6 absolute features (seasonality, cost) | **0.0163** | 0.0068 | 0 / 6 |

A clean double dissociation. Standardized features predict *relative* returns;
absolute features predict *absolute* ones. The pipeline pairs standardized
features with a raw directional target, which is the wrong half for 52 of 58.

**This also explains the survey control.** `seasonality,cost` scored highest of
27 cells partly because it is the only group left un-demeaned — the only feature
set actually aligned with what the target measures. The instrument-identity
reading (`identity_ratio` 1.0147) is the other half of it; both hold.

**But the obvious fix is worse, and it was tested rather than assumed.**
`--no-cross-sectional-standardize` exists now, and A/B against measured price IC:

| horizon | standardized | absolute | identity_ratio |
|---|---:|---:|---|
| 4h  | -0.0132 | -0.0148 | 0.61 -> **0.68** |
| 24h | **+0.0175** | **-0.0273** | 0.33 -> **0.52** |

Removing the demeaning trades a target mismatch for a memorisation problem: the
trees split on absolute level and recover instrument identity, and the identity
ratio rises in both. So the panel is not simply misconfigured.

**Aligning the other way is what the features want, and the economics forbid it.**
The matching strategy is cross-sectional market-neutral — predict relative return,
long the top and short the bottom. That pays *two* round trips to trade a
lower-variance quantity, so at h=4h the requirement becomes
`2 x 29.3bp / (78.8bp x sqrt 2) = 0.53` against the directional 0.203. Two and a
half times harder.

So the feature layer is correctly built for cross-sectional trading, and this
venue cannot afford cross-sectional trading. That is not a bug to fix here; it is
the same conclusion the cost screen reaches, arriving from the feature side. It
is also precisely the pairing that works at 1-2bp round trips and fails at 27bp.

`Dataset.cross_sectional_standardized` and the model provenance now record which
panel a model was fitted on, because `feature_set_hash` hashes column *names* and
cannot tell a demeaned panel from an absolute one.

### The alignment canary

`tests/test_model.py::test_a_perfect_feature_is_recovered_end_to_end` injects a
feature equal to the realised outcome and asserts the pipeline recovers price IC
above 0.9. Nothing in the suite checked this, and it is the first thing any "why
is the IC zero" investigation has to rule out: a one-bar shift between panel and
target, in either direction, destroys signal silently and reads exactly like no
edge. Lookahead tests assert the model cannot see the *future*; this asserts it
can see the *present*, which is the opposite failure and had no guard.

It passes, and a mutation test alongside it shifts the target by one bar and
asserts the canary fails — so the pipeline is aligned, and the near-zero ICs
above are the data rather than a defect.

### The 27-cell survey, and why its best cell is the control

`scripts.ic_survey` walks three horizons against nine feature-group sets, six
purged folds each, every cell appended to `data/search/ledger.parquet`. The hit
rule is pre-registered in the module: median price IC positive and at least five
of six folds sharing that sign. Under the null each fold is a coin flip, so
`P(>= 5/6) = 7/64 = 10.9%` and 27 cells expect 2.9 hits from noise.

    cells surveyed        27
    hits observed          3
    hits expected (null) 3.0
    P(>= 3 hits | null)  0.579

**Three hits, three expected.** The survey found exactly what noise produces.

The detail that settles it: **the highest-scoring cell in the whole grid is the
control.** `seasonality,cost` at h=24h — hour-of-day and day-of-week sin/cos,
`is_weekend`, and four fee-hurdle columns — scored price IC **+0.0538** with 5 of
6 folds agreeing, reaching **60.7%** of the IC its own round trip needs. That is
the best any cell came to paying for itself, and hour-of-day cannot forecast
direction. Its `identity_ratio` is **1.0147**, above the hindsight ceiling: the
cost columns are near-constant per instrument, so the model is ranking by
instrument identity with a continuous variable, which is exactly the failure
`contract_notional_usd` was removed from the feature set for.

The control was put in the grid to measure the survey's own noise floor. It fired,
and it beat every real feature set. Nothing else in the table needs interpreting.

The other two hits are both h=1h `cross_venue` (+0.0047, 6/6) and
`cross_venue,trend` (+0.0120, 5/6) — reproducing the cell these docs previously
called "the largest", at **1.1% and 2.9%** of required IC respectively.

Best real feature set anywhere in the grid: `all` at h=24h, price IC +0.0175,
19.7% of required IC, and 3 of 6 folds — a coin flip.

`net_ic_skill` is negative in 20 of 27 cells. At h=1h the cost-only floor is
+0.139 against net ICs of +0.11 to +0.14, so essentially all of net IC there was
the fee schedule appearing on both sides of the correlation.

**Volatility: a lot.** The dispersion head scores **+0.34** on the holdout, and it
went *up* when the target was fixed, because it never depended on the artifact.
Returns are near-unpredictable and volatility is very predictable, which is the
textbook result and not a disappointment — but on perps, with no options, a
volatility forecast improves sizing and gates entries. It is not itself tradeable.

**Carry: still untested.** 18 funding rows. It is the one part of the original
thesis that has never been evaluated, and it accumulates forward only.

### The gate stack, once both defects were fixed

Run against the 14 contracts with >= 231 days, h=1h, honest target, union-span
simulation:

    27 trades | net -3,455 (price -3,269, fees 187) | Sharpe -2.70 | win rate 33%
    gates: 27 of 75,545 accepted (0.04%) | edge_below_cost rejected 61,750
    bootstrap: Sharpe median -2.36 [p05 -4.84, p95 -0.56] | P(positive) 1% | ruin 0.0%
    cost stress: baseline -2.70, fees 2x -2.83, spread 3x -2.96, both -3.09

`edge_below_cost` rejecting 82 percent of candidates is the system working: the
forecast does not cover the fee, so it declines. The 0.04 percent that got through
still lost, and the loss is now almost entirely the prediction rather than costs.

**These numbers predate the fee correction below and are not re-measured.** The
schedule they were produced under priced the cheap contracts at 6-13bp against
the 27-33bp the venue charges, so `edge_below_cost` will reject strictly more and
the accepted trades were charged too little. Re-run before quoting them; the
qualitative reading — the forecast does not cover the toll — is the one thing the
correction can only strengthen.

Both fixes moved the risk numbers a long way, and the earlier figures were the
wrong ones:

| | before | after |
|---|---:|---:|
| simulated span | 1,776 bars | **9,430** |
| ruin probability | 99.9% | **0.0%** |
| maxDD p95 | 89.0% | 28.4% |
| bootstrap Sharpe median | -4.64 | -2.36 |
| cost stress (both) | -8.19 | -3.09 |

A 99.9 percent ruin probability on a strategy that loses 3.5 percent over eight
months was always implausible. It came from measuring a levered book over 74 days
of a single directional quarter. The strategy loses steadily; it does not blow up.

### The fee schedule is measured, not assumed

Everything above is a comparison between a forecast and a round trip, so the
round trip is the denominator of every conclusion here. It was wrong twice, in
different ways, and both were found by reading order tickets off the venue's own
app rather than by reasoning about a schedule document.

**Wrong shape.** The model was `max(pct_fee, per_contract / notional)` — a floor
under the percentage fee. The venue charges both, added. A floor understates every
leg by the smaller component's share, which is 1.5bp a side on a $782 BIP
contract and 5bp on a $242 ETP one.

**Wrong numbers.** Coinbase publishes a member commission table (group A
$0.75/contract on BTC and ETH, group B $0.10 on the rest) and the schedule
encoded it with `taker_bps: 0` — per-contract only. Its retail app does not
charge that. Three tickets:

| contract | notional | app fee | old model | new model |
|---|---:|---:|---:|---:|
| BIP | $782.05 | $0.90 | $0.78 | **$0.9021** |
| XPP | $740.50 | $0.86 | $0.74 | **$0.8605** |
| ETP | $242.50 | $0.36 | $0.24 | **$0.3625** |

`0.10% of notional + $0.12 per contract` reproduces all three to under half a
cent. No single percentage does: the implied rate is 0.115% at $782 and 0.149% at
$242. **The three notionals spanning 3.2x is what makes them decisive** — the
first two, at $782 and $740, are consistent with a flat percentage *and* with a
flat dollar amount, and for most of a day they were read as the former.
`tests/test_costs.py` pins all three, and asserts that a flat percentage still
fails to fit, so the per-contract term cannot be quietly dropped as redundant.

The consequence is that **the fee schedule no longer distinguishes contracts.**
It used to imply a 6.1bp-to-65.4bp spread across the book, and that spread was
load-bearing: it is what made XPP and SLP look like the only affordable
contracts, what made BIP's clean prices look like they came at 3x XPP's fee, and
what put six contracts' fill uncertainty above their own cost. Every contract now
pays the same percentage and the same commission, so the round trip is 27-65bp
and what varies is notional per contract — a fact about contract sizes, not about
fees.

What follows from that correction is recorded where it applies: the fill
uncertainty table below, the required-IC table above, the carry hours-to-cover
table, and the horizon the carry thesis has to be tested at. The direction of
every change is the same — costs went **up** on the contracts that looked cheap —
so no conclusion here became more optimistic.

### Fill uncertainty is a second cost, and it is not in the fee schedule

`close(t)` is a bar's last *trade*, so the move between it and `open(t+1)` — the
first price a decision can fill at — is pure fill uncertainty. It is absent from
`core/costs.py`, symmetric, and no model removes it.
`scripts/preflight.py:_prices_are_fresh` reports it per instrument.

Measured on 399 days. Median close-to-next-open gap against each contract's own
round trip, priced from the fee schedule the venue's app actually charges (0.10%
of notional plus $0.12/contract — see **The fee schedule is measured, not
assumed** below):

| | median gap | round trip | gap / cost |
|---|---:|---:|---:|
| **BIP** | 1.7bp | 27.0bp | **0.06** |
| **ETP** | 2.9bp | 34.0bp | **0.08** |
| **XPP** | 4.4bp | 27.3bp | **0.16** |
| **SLP** | 4.6bp | 29.3bp | **0.16** |
| SHP | 17.1bp | 64.7bp | 0.26 |
| AVP, POP | 15-16bp | 43-50bp | 0.31-0.34 |
| HYP, ADP, LCP | 10-13bp | 28-34bp | 0.36-0.41 |
| DOP, LNP, SUP, BCP | 13-16bp | 28-29bp | 0.44-0.54 |
| OND, XLP, NER | 17-18bp | 27-31bp | 0.55-0.65 |
| PEP | 28.3bp | 31.3bp | **0.90** |

**No contract has median fill uncertainty above its own round trip.** That is a
correction, not a finding: this table read "six of fourteen" while the schedule
billed 6.1-65.4bp across the book, and the venue does not price that way — every
contract pays the same 0.10% and the same $0.12, so the round trip is 27-65bp
and the spread that made cheap contracts look cheap was a modelling error. PEP at
0.90 is the only one close to the line.

**The trap dissolved with it.** The old reading was structural and pessimistic:
cheap contracts are cheap because they are thin, thin means stale, so the only
affordable contracts were the unusable ones and XPP and SLP were the sole
survivors. With one fee schedule across the book, freshness alone selects, and it
selects the four contracts with the most history: **BIP (0.06), ETP (0.08), XPP
and SLP (0.16)** — an order of magnitude cleaner than the rest, and BIP is at the
*low* end of the cost range rather than three times XPP's.

Fill uncertainty is still a real cost absent from `core/costs.py`, and it is
still worth ranking contracts by. It is no longer a reason to exclude any of
them.

This is also why the per-instrument IC table from the close-anchored target ranked
the thinnest contracts highest: staleness was the signal.

### The sample is one bear market and one reversal

Do not read a backtest here as a general result. Equal-weight across all 18:

| quarter | return | Sharpe |
|---|---:|---:|
| 2025 Q3 | +12.5% | +1.23 |
| 2025 Q4 | -34.1% | -2.65 |
| 2026 Q1 | -25.8% | -1.92 |
| 2026 Q2 | -18.8% | -1.42 |
| 2026 Q3 | +28.6% | +4.14 |

Fourteen of eighteen contracts are down 14-55% with 48-76% drawdowns. An 80/20
chronological split therefore trains on three down quarters and tests on a
melt-up, which is why a model fit that way calls shorts into a rally and realises
a 31% win rate. Use walk-forward quarters, not one split.

### Contract history is ragged, and it truncates the simulation

`core/simulation.py` runs on the shortest common span. With all 18 that is **54
days** of 398, because HYP listed 2026-06-05. Only four contracts (BIP, ETP, SLP,
XPP) have a year; ten have ~240 days; four have under 170.

`--exclude HYPE,PEPE,NEAR,ONDO` leaves the 14 with >= 231 days — every one
spanning three down quarters plus the rally — and takes the simulated span to 190
days. Prefer that to `--symbols`, which also shrinks the training universe.

Note that the four excluded are the four that *rose*, because they only exist
during the rally. Selecting instruments on measured performance here selects for
listing date.

## Critical Architecture Notes

- **`core/profiles.py` is the single source of truth** for per-coin feature sets, thresholds, and ML hyperparameters. Coins are described by a feature *archetype* (`mean_reversion`, `momentum_breakout`, `meme`, `trend_persistence`, `compression_breakout`) plus tuned deltas. Changes cascade into training, search, and signal generation.
- **`core/costs.py` is the single source of truth for the money *inputs*** — contract specs, exchange fee assumptions, and the per-contract commission. Trade PnL and sizing are deliberately elsewhere and the module's closing comment says where: round-trip cost in `core/targets.py`, entry fee and Kelly/risk-budget sizing in `core/execution.py`. Load a venue's real schedule with `Config.with_cost_assumptions(find_cost_config())` — `configs/` lives under `backend/trader/` so the Docker build context includes it; the hardcoded defaults are 0.10%/side plus $0.12/contract, which is what the venue's app was measured charging — so an unconfigured run now prices correctly but records no schedule version, which is the remaining reason to load one.
- **Duplicated ORM models**: `backend/trader/core/pg_writer.py` duplicates the API ORM models for container isolation. `backend/trader/tests/test_orm_parity.py` fails when they diverge in columns, types, nullability, defaults, or migration lists — a note in a doc was not enough; `wallet.balance` had already drifted 10,000 against 100,000, so whichever container created the row decided the paper account's starting balance.
- **No react-router**: Frontend routing is manual via `window.history.pushState` in `App.tsx`. Add a page by adding an entry to `ROUTES` (path → label) **and** to `PAGES` (path → component). `RoutePath` derives from `ROUTES`; `PAGES` is a `Record<RoutePath, ComponentType>`, which is what makes a route with no component a `tsc` error. The render used to be a chain of `route === '/x' && <XPage />`, and this file used to claim that was exhaustive — it was not.
- **Frontend HTTP goes through `src/api/client.ts`**: one base URL, one error type, one place the `X-API-Token` header is set. Poll with `usePolling`, which pauses on hidden tabs and surfaces failures. Five copies of `fetchWithError` had already drifted in how they reported errors, and every `.catch(() => {})` made a dead backend look like a quiet market.
- **The API serves measurements, never substitutes.** A missing value is null with a reason. The research surface used to report `pr_auc` as `holdout_auc - 0.06`, `precision_at_threshold` as `holdout_auc - 0.04`, and — when the artifact it wanted was absent, which was always — a hardcoded table of six feature importances. All of it rendered identically to real data.
- **`ic_covers_cost` is the gate that was missing**: measured out-of-sample price IC divided by `core/targets.py:required_information_coefficient` — the IC the traded universe's own round trip requires. Threshold 1.0. Every other gate reads a *simulated outcome*, so a model 34x short of its cost hurdle failed all of them without any saying why, and a Sharpe needs far more data to estimate than an IC does. Both halves are recorded separately, because a weak forecast and an expensive venue are the same ratio and opposite fixes. A candidate can in principle clear costs on a high-conviction tail while its average forecast cannot — that is a real argument and also the one that kept a losing system alive, so it needs `--force` with a written reason.
- **Promotion is the gate**: `core/promotion.py` stages into `models/.staging/{version}/`, then atomically renames into place — but only after every gate in `core/metrics.py:DEFAULT_GATES` passed. `--force` needs a reason and records it. `live_orchestrator.py` decides *when* to ask; it never decides the answer.
- **One `decide()`**: `core/signal.py` is the only place a trade is chosen. The backtest and the live signal writer both call it, which is why they cannot drift. The old per-family strategy classes under `core/strategies/` are deleted — they were orphaned by the reformulation.
- **All trader scripts** share `scripts/_common.py` for data arguments, and accept CLI args that override env vars.

## Coding Conventions

- **Python**: Type hints throughout, dataclasses for config, `logging` module, `os.getenv()` with defaults. Scripts run as `python -m scripts.<name>`.
- **TypeScript/React**: Functional components + hooks only, no class components. Fetch-based API layer (no axios). `recharts` for charts.
- **Tests**: `pytest` for trader. The suite's job is to catch the failure modes that produced fake edge before: lookahead (`test_backtest.py`), symbol-identity memorisation (`test_model.py`), leaked fold statistics (`test_cv_and_metrics.py`), the cost identity (`test_targets.py`), and blocked candidates reaching live (`test_promotion.py`). Mark anything over ~10s `@pytest.mark.slow`.

## Carry: CDE publishes no history, and the rate is far smaller than assumed

Two things established by probing a live US account
(`python -m scripts.probe_funding`), both of which change the plan:

**1. There is no historical funding endpoint.** The current rate lives at
`/api/v3/brokerage/products/{id}` under `future_product_details`:

```
"funding_interval": "3600s",
"funding_rate": "0.000009",
"funding_time": "2026-08-21T16:00:00Z",
```

`funding_time` is the *next* settlement, and there are no range parameters and
no cursor. `/api/v3/brokerage/intx/funding-rates` — which the scraper used to
call — is Coinbase *International* Exchange; `GET /api/v3/brokerage/portfolios`
on a US account returns a single `DEFAULT` portfolio and no INTX one, so every
`intx/` path is a 404 by design. (The old INTX implementation is kept as
`_unreachable_intx_funding_history` for an account that has that venue.)

So carry cannot be backfilled. It accumulates forward, one observation per
hourly settlement, which `run_pipeline` now takes on every run. **The carry head
can only ever be validated on history collected since collection started**, so
start the hourly loop before you need it. Bars, by contrast, backfill fine.

**2. The observed rate is ~22x smaller than the 2bp/hour these docs assumed.**
`0.000009` per hour is **0.09 bp/hour**, or 2.16 bp/day. Against the round trip
the venue actually charges, the hold needed for carry alone to cover a round trip
is measured from the one funding snapshot per contract that exists (18 rows,
2026-08-21) and each contract's median round trip:

| contract | rate | round trip | hours to cover |
|----------|-----:|-----------:|---------------:|
| NER (NEAR) | 0.5 bp/h | 26.5 bp |  **52 h** |
| HYP (HYPE) | 0.3 bp/h | 27.2 bp |  80 h |
| ADP, BCP, PEP | 0.3-0.4 bp/h | 30-35 bp | 92-98 h |
| LNP (LINK) | 0.3 bp/h | 28.1 bp | 108 h |
| XPP (XRP)  | 0.2 bp/h | 27.5 bp | 153 h |
| DOP (DOGE) | 0.2 bp/h | 29.5 bp | 164 h |
| SLP (SOL)  | 0.1 bp/h | 29.2 bp | 209 h |
| BIP (BTC)  | 0.1 bp/h | 27.1 bp | 226 h |
| ETP (ETH)  | 0.1 bp/h | 33.8 bp | 226 h |
| SHP (SHIB) | 0.004 bp/h | 68.1 bp | 6,812 h |

**Zero of eighteen cover a round trip within 48 hours**, and the four contracts
with clean prices need 150-226 hours — six to nine days. The earlier table said
2.9h for XPP because it priced the round trip at 5.7bp; the venue charges 27.5bp,
so that number was five times too kind, and the fee correction changed the answer
rather than the precision.

That matters because it moves the horizon the carry thesis has to be tested at,
and the horizon governs the sample size. A one-week hold at `h = 168` saturates
at `24 x 730 / ln 2 / 168 = 150` effective observations even with the recency
half-life pushed to two years, and reaching 200 with the decay off needs about
**1,400 days of funding history** — which cannot be backfilled at all. So "carry
alone pays for the round trip" is not testable on any timeline worth planning
around at these rates.

What keeps the thesis alive is that carry does not have to cover the round trip
by itself. It has to make a hold that is *otherwise* marginal profitable, and it
has to be measured on its distribution rather than on this one print: funding is
volatile, sometimes negative, and one snapshot taken while BTC was +6.7% on the
day is not a rate. Collect the distribution. `scripts/preflight.py` reports
sample size; it cannot tell you the edge is real.

### Use Coinbase spot as the reference venue, not Binance

The cross-venue group (basis, lead-lag) needs a deeper venue quoting the same
underlying. Binance, OKX and Bybit are all 451 from a US IP — but Coinbase's own
spot book is deeper than the nano perp, reachable, and is the market the perp's
index is built from, which makes its basis the thing that actually drives
funding. It is the better reference for this account, not a fallback.

Two things make it work:

1. **Scrape spot under its own venue label.** The perp and its spot index both
   resolve to the same base (`BIP-20DEC30-CDE` and `BTC-USD` both -> `BTC`), so
   storing them under one label makes the basis a comparison between an
   instrument and itself — identically zero, and a column full of a plausible
   number that measures nothing. `run_pipeline --venue-label coinbase_spot`.
2. **Reference symbols resolve separately.** `load_dataset` used to look up the
   *trade* spelling on the reference venue, which only worked by accident — the
   old CCXT path stored Binance bars under the Coinbase product id. Coinbase spot
   calls it `BTC-USD`, so the direct lookup found nothing and the whole group
   came back empty. It now resolves against the reference venue's own spellings.

```bash
# Perps
python -m scripts.run_pipeline --backfill-only --backfill-days 400 --timeframes 1h

# Spot, under its own label, for the basis. Same 400 days, and --spot-universe
# resolves all 18 products rather than naming nine by hand.
python -m scripts.run_pipeline --backfill-only --backfill-days 400 --timeframes 1h \
  --spot-universe

python -m scripts.migrate_to_research_store --venue coinbase
python -m scripts.preflight --horizon 24 --recency-half-life-days 365
```

**Both legs are 400 days, not 1100 on spot.** This used to ask for 1100, and it
was wasted: `cross_venue_features` reindexes the reference series onto the *perp*
index, so a spot bar older than the oldest perp bar is dropped by that reindex —
and the reference venue feeds nothing else, since `load_dataset`'s funding and
open-interest fallbacks both consult it and Coinbase spot has neither. Measured,
not reasoned: one spot path truncated to the perp span versus left at full depth
produces bit-identical cross-venue columns, which
`tests/test_reference_venue.py::test_reference_history_deeper_than_the_trade_venue_is_unused`
now pins. 400 covers all of CDE anyway — BIP, the oldest contract, was listed
2025-07-18.

**`--backfill-days` in `docker-compose.yml` is the FIRST cycle only.** Every cycle
after it uses `INCREMENTAL_BACKFILL_HOURS`, so on a populated store it is a cheap
gap-fill and on an empty one it *defines* the dataset. It was 30, which at a 24h
horizon is roughly 30 effective observations against the ~200 the gates need — so
a fresh start could never promote anything, and nothing in the log said why. It is
400 now, and the orchestrator runs `scripts.preflight` once after the first cycle
so the verdict is in the log rather than inferred from repeated gate failures.

Note the panel needs at least three instruments for the relative groups to
standardise (`min_universe=3`); below that they are legitimately NaN.

### Historical funding needs credentials this account does not have

`GET /rest/funding-rate` on `https://api.exchange.fairx.net` does serve
historical CDE funding, keyed on the Perp Style Futures symbol (`BIPZ30`, see
`core.costs.psf_symbol`) with an optional `trading_session_date`. It is not
reachable with a CDP key:

| | the CDP key this repo uses | what that endpoint needs |
|---|---|---|
| host | `api.coinbase.com` | `api.exchange.fairx.net` |
| auth | JWT / ES256 | HMAC-SHA256 |
| credentials | key name + EC private key PEM | access key + secret + **passphrase** |
| issued by | CDP portal | Derivatives Command Center (`dcc.coinbase.com`) |

Confirmed not public — an unauthenticated request returns
`{"error":"missing request header: CB-ACCESS-KEY"}`. The DCC is the exchange's
member portal, so eligibility is the blocker rather than configuration. If
credentials do turn up, `probe_funding --funding-path /rest/funding-rate` tests
it and the host would need adding to the connector.

Until then carry accumulates forward only, and the basis above is the material
from which it could later be *reconstructed* — validated against the forward
actuals, labelled, and gated, never assumed.

### Do not backfill carry from another venue

The tempting shortcut, given that CDE has no funding history, is to take
Binance's. Don't — and the gates now refuse it.

Funding feeds the `carry` component of the net-return target directly
(`core/targets.py`), so a proxy series trains the carry head on a cash flow this
account will never receive. It is not a small approximation: each venue sets
funding from its own basis and its own formula, on its own interval (CDE hourly,
most offshore perps 8h), in books of very different depth — BIP turns ~1.5M
contracts a day against Binance's BTC perp. And the failure is invisible: the
backtest would credit carry from one venue while the paper engine accrues it from
another, and both numbers look plausible.

`load_dataset` already warned. The warning did not reach the artifact or the
gates, so a proxy-funded candidate could clear all ten and install
indistinguishably from a clean one. Now:

- `Dataset.proxy_funding_symbols` records it structurally, not just as a string.
- `ForecastModel.provenance()` carries it, so a forced install keeps the evidence.
- `proxy_funding_symbols` is a promotion gate with a threshold of zero, and
  "not measured" fails like every other gate here.

Research on proxy funding is fine and sometimes useful — measuring whether
funding mean-reverts at all, say. Promotion is what is blocked. `--force` with a
reason remains the escape hatch, and records itself.

The legitimate use of another venue's funding is as a *cross-venue feature*
(basis, lead-lag), which is what the `cross_venue` group already is.

## Contract sizes: settled against the venue

`configs/exchange/coinbase_us_perps_cde_v202602.json` disagreed with
`core/costs.py:CONTRACT_UNITS` on three instruments. The venue answered it:
`future_product_details.contract_size` on the product endpoint reports **10, 50
and 5** for AVAX, LINK and LTC — agreeing with `CONTRACT_UNITS` across all
sixteen contracts. So the code was right and the schedule was wrong; the schedule
is corrected.

The edit changed no computed cost, because the schedule's `contract_sizes` are
read by nothing — `get_contract_spec` always uses `CONTRACT_UNITS`. That is
precisely why the discrepancy survived: dead data cannot be wrong in a way
anything notices. Two mechanisms now watch it, both mutation-verified —
`load_exchange_cost_assumptions` warns on any disagreement at load time, and
`test_orm_parity.py::test_the_venue_schedule_agrees_with_the_contract_units_table`
fails on it.

Reproduce the venue's own table with:

```bash
python -m scripts.probe_funding --sizes-only
```

## Every source is Coinbase now, and no proxy is needed

CCXT is gone — the connector, the dependency, all three call sites. It existed
for perp bars, reference bars and open interest, and each turned out to be
native:

| group | features | source | needs a proxy |
|-------|---------:|--------|---------------|
| `cross_venue` (basis, lead-lag) | 7 | Coinbase spot, `--spot-universe` | no |
| `positioning` (OI) | 6 | `future_product_details.open_interest` | no |
| `carry` | 9 | `future_product_details.funding_rate` | no |
| `market_factor` | 9 | Coinbase BTC bars | no |

**Open interest was the last one found, and the way it was missed is the pattern
worth remembering.** A comment read "Open interest has no Coinbase-native source:
the REST client implements candles, tickers and /intx/funding-rates but no
open-interest endpoint" — an accurate statement about *this client*, recorded as a
fact about the *venue*, exactly like the INTX-vs-CDE confusion. It is on the
product payload, on the contract actually traded. Compare the two books:

| | contracts |
|---|----------:|
| `BIP-20DEC30-CDE` (Coinbase, what we trade) | 268,164 |
| `BTC/USDT:USDT` (gate, what the six features described) | 21,579,279 |

Funding and open interest ride **one** request now
(`get_contract_snapshot`), so they cannot straddle a settlement. Both are
snapshots: current value, no range parameters, no cursor. So **open interest
cannot be backfilled either** — it accumulates forward one observation per
contract per cycle, exactly like carry. Run the loop before you need it.

The CCXT paths that are gone, and why each was a mistake rather than resilience:

- **Funding fallback** wrote `binance_proxy` rates. Funding feeds the target's
  carry component, so another venue's rate trains the carry head on money this
  account never receives — and `proxy_funding_symbols` is a promotion gate with a
  threshold of zero, so those rows could only ever *block* a candidate.
- **OHLCV pre-history fill** filled the span before a contract was listed. BIP
  began 2025-07-18, so a 400-day request legitimately misses 265 days; nothing
  was missing, and the substitute was another exchange's contract stored under
  this symbol's name.
- **`-PERP` placeholder symbols** were appended for any modelled asset with no
  CDE listing, putting instruments this account cannot trade into the universe.
  Unlisted assets are now named and excluded.

**This degrades quietly by construction.** `build_panel` reindexes to the
canonical 76-column list so a saved model always scores against the same matrix,
which means a group that produced nothing arrives as an *all-NaN column*, not an
absent one — the panel keeps its shape and looks healthy. And
`feature_set_hash` hashes column *names*, so a model fit behind a geo-block and
one fit through a proxy have the identical hash.

Two things now say so out loud: `load_dataset` warns when the reference venue
yields no bars, naming the symbols and the consequence, and the model's
provenance carries `empty_features` and `n_features_populated` alongside
`n_features`. `tests/test_reference_venue.py` covers both.

`--reference-venue` defaults to `coinbase_spot`. It used to default to
`binance`, which is 451 from a US IP, so it resolved to no bars and the seven
cross-venue columns arrived all-NaN — with the identical `feature_set_hash` a
populated panel would have.

## What to do next, and when

**Direction is answered: no.** 0.8bp of edge against a 27bp cheapest round trip,
and more history does not move it — the recency half-life saturates, so 399 days
gives 1,678 effective observations at h=4h and 2,000 days gives 3,088. The
constraint is the economics by a factor of **34 at h=1h and 7 at h=24h**, not the
sample size. Do not tune the directional model; 15 group-by-horizon cells have
already been tested on this data and further searching spends statistical budget
on a question the fee schedule has answered.

**Carry is the open question, and the fee correction moved its horizon.** Funding
accrues at 18 rows an hour and cannot be backfilled:

| | funding rows | what becomes possible |
|---|---:|---|
| +90d | 38,880 | monthly folds — a first honest look |
| +106d | 45,800 | h=48h clears the 200-observation gate at H=730 |
| +180d | 77,760 | walk-forward quarters — a real evaluation |

h=48h remains the horizon to test at, but not for the reason this section used to
give. At the measured funding rates 48 hours of carry covers **no** contract's
round trip outright — it covers 92% of NER's, 44-60% on HYP/ADP/BCP/PEP/LNP, and
21-31% on the four with clean prices. So the question to ask is not "does carry
pay for the round trip" but **"does carry plus a weak directional forecast clear
it"**, which is a harder question and the only one the data will support: a hold
long enough for carry to cover the toll alone is ~168h, and that saturates at 150
effective observations however long the loop runs.

**Whether a maker rate even exists is unverified, and it is the cheapest thing to
find out.** This section used to call maker-on-both-sides the one lever available
sooner, on the reasoning that it would move the round trip toward ~5bp. That was
downstream of the wrong schedule. What is actually known: the three observed
tickets are market orders, the schedule assumes `maker_bps == taker_bps == 10`
because no maker ticket has been seen, and the spread component that a resting
order could avoid is ~4bp of a 27bp round trip — a 15% saving, not an 80% one.
If the 0.10% has a maker rate the picture changes materially; if it does not,
execution is not a lever at all. **One limit order on the app answers it**, and
until it does, do not build the fill-probability model `core/simulation.py`
lacks.

**Spot is not the answer.** Coinbase Advanced Trade charges 1.20% per side at this
account's tier: 240bp round trip against BIP's 27bp, 9x worse. Spot prices are
genuinely cleaner (median close-to-open gap 1.1bp against the perps' 14.2bp, and
no flat-OHLC bars at all), but clearing 240bp at h=4h needs an IC of ~2.6 against
BTC's 92bp of four-hour dispersion — above perfect foresight.

**Funding and open interest are the only irreplaceable data.** They are 296K each
against 15M of bars and 49M of features, and unlike those two they cannot be
re-fetched at all. `.gitignore` un-ignores them specifically for that reason;
everything else in the research store regenerates.

## Environment Variables

See `AGENTS.md` for the full table. Minimum required for live workflows: `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `DATABASE_URL`, `POSTGRES_PASSWORD` (compose refuses to start without it), `COST_CONFIG` (unset still prices correctly — the defaults match the measured schedule — but records no fee version), and `API_TOKEN` + `VITE_API_TOKEN` if you want the dashboard's script runner.
