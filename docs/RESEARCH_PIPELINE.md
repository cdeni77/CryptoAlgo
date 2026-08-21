# Research Pipeline Rebuild

Design spec for the feature + training + validation rebuild. Scope: everything
between the scraper and the signal writer. The scraper (`data_collection/`),
the API, and the frontend keep their current contracts.

Instrument context: **Coinbase US perpetual-style futures (CDE)**, hourly
funding (`funding.method = coinbase_us_perps_hourly`), per-contract commission
rather than a percentage fee.

---

## 1. Why the current numbers can't be trusted

Six defects, all verified against the repo. The first two alone invalidate any
cross-coin comparison the search has produced.

### 1.1 The cost model is never loaded — errors of 0.06x to 2.5x

`Config.cost_config_path` was declared but never read, so every run used the
hardcoded default (10 bps per side, no per-contract floor) instead of
`configs/exchange/coinbase_us_perps_cde_v202602.json` (0 bps + $0.75/contract
for BIP/ETP, $0.10 for the rest). The 200 lines of assumption-loading in
`core/costs.py` were unreachable.

Round-trip cost, modelled vs actual:

| Contract | Notional/ct | Modelled RT | Actual RT | Error |
|----------|------------:|------------:|----------:|------:|
| BIP (BTC)  | $600   | 20.0 bp | 25.0 bp | 1.25x |
| ETP (ETH)  | $300   | 20.0 bp | 50.0 bp | 2.50x |
| SLP (SOL)  | $750   | 20.0 bp |  2.7 bp | 0.13x |
| XPP (XRP)  | $1,100 | 20.0 bp |  1.8 bp | 0.09x |
| DOP (DOGE) | $1,750 | 20.0 bp |  1.1 bp | 0.06x |
| AVP (AVAX) | $350   | 20.0 bp |  5.7 bp | 0.29x |
| LCP (LTC)  | $475   | 20.0 bp |  4.2 bp | 0.21x |

The error is not a constant bias — it runs in **both directions**. ETH was
backtested at 40% of its real cost; DOGE at 17x. Fee-aware labels
(`TripleBarrierSpec.fee_pct_per_side`) inherit the same error, so the labels
themselves are wrong. Any ranking of coins or strategies built on this is
uninterpretable.

### 1.2 Contract sizes contradict between code and config

`core/trading_costs.py` and the exchange config disagree for three instruments:

| Contract | Config | Code table | Factor |
|----------|-------:|-----------:|-------:|
| AVP/AVAX | 5.0    | 10         | 2x |
| LNP/LINK | 10.0   | 50         | 5x |
| LCP/LTC  | 1.0    | 5          | 5x |

Contract size scales notional linearly, so PnL for AVAX, LINK and LTC is off by
2x-5x depending on which table the call path happened to read.

### 1.3 Venues are silently blended

Corrected from an earlier draft of this document, which claimed training ran
wholly on Binance data. It does not, and the real defect is subtler.

**Bars** are Coinbase-first: `run_pipeline.py:178` fetches native Coinbase
candles, and only falls back to `CCXTConnector` when Coinbase returns nothing or
when there is a pre-history gap of more than 12 hours (`:188`, `:204`). So a
symbol's history is a *blend* — recent bars from the venue we trade, older bars
from Binance/Bybit via a symbol map (`BIP -> BTC/USDT:USDT`).

**Funding** is also Coinbase-first, falling back to a `binance_proxy` source,
and it already records which in `funding_source`.

**Open interest** is CCXT-only. There is no Coinbase-native alternative: the REST
client implements candles, tickers and `/intx/funding-rates`, but no
open-interest endpoint.

The problem is that the `ohlcv` table had **no venue column**, so the boundary
between the instrument and its proxy was unrecoverable. A model trained across
that seam is trained on two different books' microstructure — different tick
size, different liquidity, different funding mechanics — with no way to tell
where one ends.

*Fixed:* bars now carry `venue`, recorded per row from the fetch path that
produced them, and the research store keys on it.

### 1.4 Three drifted copies of the signal decision

`run_backtest` (676 lines), `run_signals` (329) and `run_inference` (300) each
implement threshold, calibration, regime and momentum logic independently.
`run_signals` and `run_inference` are a copy-paste pair that has drifted. The
backtest and the live path are therefore different programs, which is the
structural cause of "training didn't align to paper trading."

### 1.5 The statistics don't support the conclusions

- 120-day windows at 1h bars = 2,880 rows. Computed with
  `core.cv.effective_sample_size`, the 72-hour barrier horizon leaves **40.0
  independent observations** — not an estimate, the concurrency-weighted count.
  At a 108-hour horizon it is 26.7. Against 77 features.
- A single walk-forward path yields **one** Sharpe estimate with no confidence
  interval.
- ~200 Optuna trials x 15 coins of selection pressure sits on top of that, with
  the true trial count scattered across five scripts and never reaching the
  deflated-Sharpe calculation.

Per-coin models cannot be rescued at this sample size. The fix is structural
(see 3.4), not a better estimator.

### 1.6 Two of three datasets skipped validation

`DataValidator` implements `validate_ohlcv`, `validate_funding_rate` and
`validate_open_interest`. Only the first was reachable: OHLCV went through the
async `DataPipeline`, which validates, while `run_pipeline`'s
`backfill_funding_rates` and `backfill_open_interest` inserted directly.

`validate_open_interest` had **zero callers**, and open-interest records were
constructed with `quality=DataQuality.VALID` written in by hand — the column
asserted the data had been checked when nothing had checked it.

Compounding it, `quality` defaulted to `VALID` on every record type, so an
unvalidated row was indistinguishable from a verified one.

*Fixed:* `data_collection/ingest.py` is now the only path into storage and the
only place that sets `quality`; `UNVALIDATED` is the default, so a bypass is
visible in the data (`SELECT COUNT(*) ... WHERE quality = 'unvalidated'`). The
flag survives into the research store, whose reads exclude flagged rows unless
asked — a 50%-per-hour funding rate is stored as suspicious rather than dropped,
and it has no business reaching the carry features.

### 1.7 XLM trains on none of its intended features

`COIN_FEATURE_MAP` maps XLM to `XRPFlowMicrostructureFeatures`, which emits
`xrp_*` column names, while the XLM profile asks for `xlm_*`. All six archetype
features resolve to nothing and are silently dropped by the column intersection.
XLM has been training on 7 of its 13 intended features.

---

## 2. Design principles

1. **One implementation per decision.** Anything that decides whether to trade
   exists exactly once and is called by backtest, simulation and live alike.
2. **Costs are inputs, not constants.** The venue's fee schedule is data, loaded
   once, used by labels, backtest, sizing and live.
3. **Point-in-time or it doesn't exist.** Every feature read is bounded by
   `available_time <= t`. The scraper schema already carries the pair; enforce it.
4. **A result is a distribution, not a number.** No decision is made on a point
   estimate.
5. **The panel is the unit of study.** Coins are observations of one process, not
   fifteen separate problems.

---

## 3. Architecture

Seven layers, each one module with one job.

```
data      →  features  →  labels  →  model  →  simulation  →  decision  →  execution
(venue-    (mechanism-   (triple-  (pooled  (CPCV, MC,      (one         (fills,
 keyed,     grouped)      barrier,  panel)   bootstrap)      decide())     funding,
 PIT)                     fee-aware)                                       margin)
```

### 3.1 Data

**Research store: Parquet + DuckDB.** Columnar, immutable, no server, and fast
enough that a full-history feature build is a coffee break rather than an
afternoon. Partitioned `dataset/venue=…/symbol=…/month=…`.

**Serving store: PostgreSQL, unchanged.** `signals`, `trades`, `paper_*`,
`wallet`, `model_runs` keep their schemas so the API and frontend are untouched.

Changes to what's collected:

- **Bars keyed by `(venue, symbol)`**, so Coinbase and Binance series coexist
  explicitly instead of one masquerading as the other.
- **Coinbase-native funding and OI** for the traded instrument. Binance funding
  stays, as a *cross-venue* feature rather than a substitute.
- **Book snapshots** from the existing WebSocket handler, persisted for slippage
  calibration. Without depth data, every slippage number is a guess.
- **Feature matrices materialised as Parquet with a content hash**, so a model
  artifact names exactly the feature set it was trained on.

### 3.2 Features — grouped by mechanism, not by coin

The fee table in 1.1 drives the design. Round-trip cost ranges from ~1 bp
(DOGE) to ~50 bp (ETH) across the same universe, so **the tradable horizon and
the minimum edge differ per instrument by an order of magnitude**. A single
feature set with a single horizon cannot be right for both. Cost enters as a
first-class feature, and the label horizon is derived from the instrument's
hurdle rather than hand-tuned per coin.

Groups, each a function of the panel at time `t`:

**Carry** — the perp-specific edge, and the one aligned to hourly funding:
funding level, funding z-score, funding term structure (1h vs 8h vs 24h mean),
funding minus realised vol, cumulative carry over the intended hold, hours to
next settlement.

**Cross-venue** — Coinbase-vs-Binance basis, basis z-score, and lead-lag
correlation. A thinner venue that lags a deeper one is a real, mechanical retail
edge, and it is currently thrown away by treating Binance as the price.

**Volatility** — Parkinson, Garman-Klass and Rogers-Satchell estimators,
bipower variation with its jump component separated, vol-of-vol, and the
short/long vol term structure.

**Liquidity / microstructure** — Amihud illiquidity, Corwin-Schultz spread
proxy, Roll measure, a Kyle-lambda proxy, signed volume imbalance. These are
what make the cost model dynamic instead of a constant.

**Positioning** — OI change, OI-price divergence, liquidation cascade proxy.

**Trend / reversal** — multi-horizon returns and their cross-sectional ranks.

**Market factor** — BTC beta and BTC-residual momentum, so the model can express
"this coin relative to the market" instead of relearning the market in every coin.

**Seasonality** — hour-of-day and day-of-week (crypto session effects are real),
plus proximity to funding settlement.

Every feature is **cross-sectionally standardised at each timestamp** (rank or
z-score across the universe). This is what makes the pooled model in 3.4
coherent, and it removes the per-coin scaling that the current archetypes
hand-encode.

### 3.3 Labels

Keep what is already correct: triple-barrier with the round-trip fee added to
the take-profit barrier, neutral-direction rows excluded, average-uniqueness
sample weights, and meta-labelling as a second stage. The one change is that the
fee input becomes the real per-contract schedule, which moves the barriers
materially (1.1).

### 3.4 Model

**One pooled panel model** over all instruments, with instrument identity as a
feature (embedding or dummies) rather than fifteen separate fits. At 30-50
independent events per coin per fold this is not a preference, it is the
difference between an estimable model and noise: pooling takes the effective
sample from ~40 to ~600 per fold.

- Primary: LightGBM on the cross-sectionally standardised panel.
- Second member: a small GRU or TCN (~50-200k params) over raw normalised
  windows. Not a transformer — the sample size doesn't support one — but a
  genuinely decorrelated view, since it sees window *shape* the reductions
  discard.
- Purged, embargoed CV with the embargo never shorter than the label horizon.
- Per-coin profiles survive for thresholds, exits and sizing, where they are
  genuinely instrument-specific.

---

## 4. The simulation layer

This is the part that replaces "the backtest looked good." Seven techniques that
compose into one verdict; each rules out a specific failure mode.

### 4.1 Combinatorial purged cross-validation (CPCV)

*Implemented in `core/cv.py`.* Split the timeline into N=12 contiguous groups,
take k=2 as test: C(12,2) = 66 train/test splits, which recombine into **11
distinct backtest paths** instead of one. Every path is purged and embargoed, and
`assert_no_leakage` fails a fold whose purge is shorter than the label horizon.

*Output:* a distribution of Sharpe, drawdown and hit rate.
*Rules out:* a strategy that only works on the one train/test cut you chose.
*Also yields:* PBO — the fraction of splits where the in-sample-best
configuration lands below the out-of-sample median.

**What it does not buy.** The 11 paths reuse the same history, so they are 11
correlated views of one sample rather than 11 samples. The spread across them
measures sensitivity to how the data was cut — worth knowing, and the thing a
single walk-forward hides — but it adds no evidence. Measured on a 120-day hourly
window: the whole sample carries **40.0 independent observations** at a 72-hour
horizon, and an individual CPCV test block carries about **7**. Widening the
sample means more instruments, not more folds.

### 4.2 Deflated Sharpe Ratio

Adjust the observed Sharpe for the number of configurations tried, their
variance, and the sample's skew and kurtosis. Requires the **true** trial count,
which means a single search ledger (4.8) rather than five scripts each losing
their own count.

*Rules out:* the best of 3,000 tries looking like a discovery.

### 4.3 Stationary bootstrap of the trade sequence

Politis-Romano stationary bootstrap with expected block length near the mean
holding period, so autocorrelation and volatility clustering survive resampling.

*Output:* Sharpe confidence interval, the distribution of maximum drawdown,
time-to-recovery, and risk of ruin at the intended leverage.
*Rules out:* sizing decisions based on the one drawdown that happened to occur.

### 4.4 Synthetic panel generation

Two generators, used together:

- **Block bootstrap of the whole panel** — resample time blocks across all
  instruments jointly, preserving the cross-sectional correlation structure.
- **Parametric** — a 2-3 state regime-switching model (vol/trend states) with
  GARCH or HAR dynamics and Student-t innovations within each state, plus a jump
  component, calibrated per instrument and coupled through the empirical
  correlation matrix.

Run the full stack — features, model, decide, execute — on 1,000 synthetic
panels.

*Rules out:* a strategy fitted to the single path history took.
*Honest limit:* see section 7. A generator contains only the structure you put
into it, so this measures **robustness and sizing**, never edge.

### 4.5 Execution simulation

The static 2 bps assumption is replaced with a model calibrated to the venue:

- Participation-rate slippage — order size against bar volume, with the spread
  crossing taken from persisted book snapshots rather than assumed.
- The real fee schedule, per-contract, per instrument group (1.1).
- Latency: signal at bar close, fill at next bar open plus a delay.
- **Hourly funding accrued at actual settlement timestamps** — at 4x leverage
  and hourly settlement, carry is not a rounding error.
- Per-bar margin and liquidation check. A backtest that ignores intrabar
  liquidation overstates returns at leverage.

*Rules out:* the backtest-to-paper gap that costs are hiding in.

### 4.6 Parameter sensitivity surface

Perturb each chosen parameter one grid step in both directions and re-run. A
real edge sits on a plateau; an overfit sits on a spike.

*Rules out:* a configuration that only works at exactly the values the optimiser
landed on.

### 4.7 Capacity curve

Sharpe as a function of capital deployed, using 4.5's slippage model. On CDE
liquidity this is a hard constraint, and it tells you the strategy's actual
ceiling rather than its per-unit ceiling.

### 4.8 Search ledger

One append-only ledger of every configuration ever evaluated: parameters, seed,
data version, feature-set hash, and every metric. This replaces the current
practice of recording "what was already tested" in a script docstring, and it is
what makes 4.2 computable.

---

## 5. Promotion gates

A configuration reaches paper trading only by clearing all of these. They are
hard gates, not a score to be traded off.

| Gate | Threshold | From |
|------|-----------|------|
| CPCV median Sharpe | >= 0.5 | 4.1 |
| CPCV 5th percentile Sharpe | > 0 | 4.1 |
| Probability of backtest overfitting | <= 0.30 | 4.1 |
| Deflated Sharpe Ratio | > 0 at true trial count | 4.2 |
| Bootstrap P(Sharpe > 0) | >= 0.90 | 4.3 |
| Synthetic panels with positive Sharpe | >= 60% | 4.4 |
| Cost stress: 2x fees, 3x slippage | median Sharpe still > 0 | 4.5 |
| Parameter plateau | >= 60% of +/-1-step neighbours keep >= 70% of Sharpe | 4.6 |
| OOS trade count | >= 100 | — |
| Paper/CPCV agreement | paper Sharpe inside the 90% band after 30 days | — |

The last gate is the one that closes the loop. If paper falls outside the band
the model is not "unlucky" — the simulation is wrong, and it gets fixed before
anything else ships.

---

## 6. Build order

Each phase leaves the system working and is independently verifiable.

1. **Costs and contract specs.** Load the exchange config; reconcile the
   contract-size conflict against the venue's published schedule; re-run every
   existing backtest to establish a corrected baseline. Nothing else is
   meaningful until this is right.
2. **One `decide()`.** Extract the single decision function; assert
   backtest/live parity on identical inputs as a test.
3. **Venue-correct data.** Coinbase-native bars, hourly funding and OI; keep
   Binance as a cross-venue feature source; start persisting book snapshots.
4. **Research store.** Parquet + DuckDB, point-in-time enforced, feature
   matrices content-hashed.
5. **Feature layer.** The mechanism groups in 3.2, cross-sectionally
   standardised.
6. **Pooled model.** Panel LightGBM with instrument identity; purged CV;
   uniqueness weights; meta-labelling.
7. **Simulation layer.** CPCV first (it changes every number you look at), then
   the bootstrap, then execution simulation, then synthetic panels.
8. **Search collapse.** One campaign runner over the ledger, replacing the five
   scripts.
9. **Gates wired into promotion.** No path to paper except through section 5.

---

## 7. What simulation cannot tell you

Worth stating plainly, because the failure mode here is believing the machinery.

- **Synthetic data cannot prove edge.** A generator only contains the structure
  you calibrated into it. Fit one with momentum and momentum strategies will
  work; fit one without and nothing will. Synthetic panels test whether a
  strategy survives paths that didn't happen — they are a robustness and sizing
  instrument, and they are not evidence of alpha. Only genuinely unseen data is.
- **CPCV reduces selection bias; it does not eliminate it.** Every configuration
  you evaluate against it spends some of its power. That is why 4.8 exists.
- **The gates in section 5 will reject most things.** That is the point. At
  30-50 independent events per fold, the honest prior is that most apparent
  edges are noise, and a framework that keeps confirming your strategies works
  is a framework that isn't measuring anything.
- **A corrected cost model may erase the current results entirely.** Given 1.1,
  the ETH and BTC strategies were backtested at a fraction of their real cost.
  Some of what looks like a working strategy today is a fee error.

---

## 8. Deferred: the app layer

Out of scope for the pipeline rebuild, recorded here so it is not rediscovered.
Audited but not changed. The frontend is 1,945 lines, well organised, and
`tsc --noEmit` passes clean — this is a fix list, not a rewrite.

### 8.1 Security

Severity depends on exposure. `docker-compose.yml` publishes with Docker's
default binding, which is all interfaces, so these are LAN-reachable as
configured.

- **Postgres on `0.0.0.0:5432` with the password `yourpassword`** hardcoded in
  compose. Bind to localhost, move the credential to an env file.
- **No authentication on any endpoint.** There is not one auth dependency in the
  API.
- **CORS admits every origin with credentials.** `allow_origins` lists three
  localhost entries *and* `"*"`; the wildcard defeats the list. Verified against
  Starlette: an arbitrary origin is echoed back into
  `Access-Control-Allow-Origin` alongside `Access-Control-Allow-Credentials:
  true`. With no auth, any page the operator visits can read trades and wallet
  balances and start research jobs.
- **`POST /launch/{job}` passes `args` through** with only whitespace filtering.
  Not remote code execution — the job name is checked against a discovered
  allowlist and `Popen` receives a list, so there is no shell — but arbitrary
  flags reach the launched script.

### 8.2 Duplicated ORM models

`backend/api/models/` and `core/pg_writer.py` each define the same 9 tables:
96 duplicated column definitions. One has already drifted:

    wallet.balance   api = Column(Float, default=100000.0)
                     pg  = Column(Float, default=10000.0)

A 10x difference in starting balance, resolved by whichever process calls
`create_all` first. The container-isolation reason for duplicating them is real,
but a shared schema module installed into both images is a better answer than a
comment asking people to remember.

Both sides also run their own ad-hoc migrations — `app.py` executes
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` at import time, and
`pg_writer._run_pg_migrations` does the same job separately. One migration path,
whichever it is.

### 8.3 Correctness and quality

- **Back button is broken.** `App.tsx` calls `pushState` with no `popstate`
  listener, so the URL changes without the view.
- **No error or loading state in any page.** When the API is unreachable the
  dashboard shows stale numbers with no indication. On a trading dashboard that
  is the wrong failure mode.
- **`npm run lint` has never run.** No ESLint config exists, despite the script
  and three eslint plugins in `devDependencies`. It is documented in CLAUDE.md as
  a check to run.
- **Polling ignores tab visibility.** The dashboard holds 5s, 30s and 60s
  intervals — roughly 17k requests a day per open tab, whether or not anyone is
  looking.
- **Business logic in routing layers.** `endpoints/wallet.py` is 692 lines and
  includes a third-party Ethplorer integration; `controllers/research.py` is 631.

### 8.4 What the app should gain from the rebuild

The pipeline work creates capabilities the current UI has no way to reach, and
this is the more interesting half of the deferred work:

- **A model's provenance.** Every artifact records its feature-set hash, cost
  config version, and data as-of timestamp. The UI should show which model is
  live and exactly what it was trained on.
- **The promotion gates as a screen.** Section 5 is a pass/fail table per
  candidate. That is a UI, and it is the screen that decides what goes live.
- **CPCV path distributions, not point estimates.** A Sharpe is a distribution
  across 11 paths; showing the median and the 5th percentile is the difference
  between a dashboard that informs and one that flatters.
- **One promote action.** Train, gate, paper, and live are currently a sequence
  of scripts launched by PID. With the pipeline unified behind one `decide()` and
  one search ledger, promotion can be a single reviewable transition with an
  audit trail — which is also what makes going from paper to real defensible
  rather than nervous.
