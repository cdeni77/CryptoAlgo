# Research Pipeline Rebuild

Design spec for the feature + training + validation rebuild. Scope: everything
between the scraper and the signal writer.

> **Read this as the design intent, not as the current state.** It was written
> before the rebuild and the build diverged from it in five places that matter.
> Where they disagree, the code wins, and `AGENTS.md` describes what landed.
>
> | This spec says | What landed |
> |---|---|
> | §3.3 triple-barrier labels + meta-labelling | `core/targets.py` regresses **net return**, decomposed into price/carry/cost. No labeller, no `TripleBarrierSpec`, no meta-labelling stage. Only average-uniqueness weighting survived. |
> | §3.4 instrument identity as a feature | `core/model.py` sets `USE_SYMBOL_IDENTITY = False` — identity alone scored IC +0.54 on random walks. `identity_ceiling_ic` reports that ceiling instead. |
> | §3.4 a GRU/TCN second ensemble member | Three LightGBM heads (`price`, `carry`, `dispersion`). No neural member; no torch dependency. |
> | §4.5 spread taken from persisted book snapshots | Still an assumption (`config.spread_bps`, default 4.0). `OrderBookSnapshot` exists and the WebSocket builds them, but nothing persists one, so §3.1's "book snapshots persisted for slippage calibration" is also unlanded. |
> | §5 `CPCV median/p05 Sharpe` gates, and a "paper/CPCV agreement" gate | The gates are `walk_forward_median_sharpe` / `walk_forward_p05_sharpe`, measured across walk-forward sub-periods. CPCV is implemented and tested but off the promotion path by choice: 66 fits against this loop's 6. Paper/CPCV agreement is monitoring, not a gate. `README.md` lists the ten real gates. |
>
> The scraper (`data_collection/`) did *not* keep its contract: venue-scoped
> watermarks, gap detection and a single naive-UTC time helper were added. The
> API and frontend did not either — see `AGENTS.md`.

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

**Correction, 2026-08-22: the "actual" column above was itself wrong.** It came
from the published member commission table, which the venue's retail app does not
charge, and it applied the per-contract amount as a floor under the percentage fee
rather than in addition to it. Three order tickets read off the app give
`0.10% of notional + $0.12/contract`, which puts every contract in this table at
27-34bp rather than 1.1-50bp. The defect §1.1 describes was real — the schedule
was unreachable — but the magnitudes are superseded; see
"The fee schedule is measured, not assumed" in `CLAUDE.md` and
`backend/trader/tests/test_costs.py`. The remainder of §1.1 is kept as the record
of what was found when.

The error is not a constant bias — it runs in **both directions**. ETH was
backtested at 40% of its real cost; DOGE at 17x. The old system's fee-aware
labels (`TripleBarrierSpec.fee_pct_per_side`, since deleted along with the
labeller) inherited the same error, so the labels themselves were wrong. Any ranking of coins or strategies built on this is
uninterpretable.

### 1.2 Contract sizes contradict between code and config

`core/trading_costs.py` (since deleted) and the exchange config disagreed for three instruments:

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
  At a 108-hour horizon it is 26.7. Against 76 features.
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
data      →  features  →  targets  →  model  →  simulation  →  decision  →  execution
(venue-    (mechanism-   (net      (pooled  (bootstrap,     (one         (fills,
 keyed,     grouped)      return:   panel,   regime sim,      decide())     funding,
 PIT)                     price +   3 heads) cost stress)                   margin)
                          carry -
                          cost)
```

### 3.1 Data

**Research store: Parquet + DuckDB.** Columnar, immutable, no server, and fast
enough that a full-history feature build is a coffee break rather than an
afternoon. Partitioned `dataset/venue=…/symbol=…/month=…`.

**Serving store: PostgreSQL.** `signals`, `paper_*`, `wallet`, `model_runs`.
(The `trades` table named here was deleted — nothing ever wrote a row to it. See
`AGENTS.md`.)

Changes to what's collected:

- **Bars keyed by `(venue, symbol)`**, so Coinbase and Binance series coexist
  explicitly instead of one masquerading as the other.
- **Coinbase-native funding and OI** for the traded instrument. Binance funding
  stays, as a *cross-venue* feature rather than a substitute.
- **Book snapshots** from the existing WebSocket handler, persisted for slippage
  calibration. Without depth data, every slippage number is a guess. **Not
  landed:** the snapshots are built and dropped — nothing writes them — so the
  spread is still the assumption §4.5 describes.
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

### 3.3 Labels — **superseded, see the header table**

There is no labeller. `core/targets.py` regresses net return, decomposed into
`price`, `carry` and `cost`, because triple-barrier classification could not
express carry — the most plausible edge on hourly-funding perps. (The rate was
assumed at 2bp/hour; the rates since measured are 0.1-0.5bp/hour against a
27-65bp round trip, so a 48h hold covers 21-92% of a round trip on carry alone.) Of the paragraph below, only the
average-uniqueness weighting survived (`core/cv.py:average_uniqueness`); there
is no meta-labelling stage, and `Config.meta_probability_threshold` is orphaned.
The original text:

Keep what is already correct: triple-barrier with the round-trip fee added to
the take-profit barrier, neutral-direction rows excluded, average-uniqueness
sample weights, and meta-labelling as a second stage. The one change is that the
fee input becomes the real per-contract schedule, which moves the barriers
materially (1.1).

### 3.4 Model — **two departures, see the header table**

Pooling landed. The two things that did not: instrument identity is *excluded*
(`USE_SYMBOL_IDENTITY = False`, because identity alone scored IC +0.54 on random
walks — `identity_ceiling_ic` reports that ceiling instead), and there is no
neural second member. The ensemble is three LightGBM heads: `price`, `carry`,
`dispersion`. The original text:

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

- Participation-rate slippage — order size against bar volume. **The spread
  crossing is still an assumption** (`config.spread_bps`, default 4.0), not a
  measurement: `OrderBookSnapshot` exists and the WebSocket handler builds them,
  but `storage.py` has no insert for one and `ingest.py` never persists one, so
  the calibration input does not exist yet. `execution.py:book_depth_bps` is the
  hook that will replace the assumption when it does.
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

The gate names below are the *design* names. What `core/metrics.py:DEFAULT_GATES`
actually contains is the ten-row table in `README.md`; two differences matter.

The `cpcv_*` gates were renamed `walk_forward_*` because they were measured
across walk-forward sub-periods, not CPCV paths — the old names promised a
stronger guarantee than the loop delivered. CPCV is implemented and tested
(`core/cv.py`), and is off the promotion path by choice: 66 model fits against
this loop's 6.

Paper/CPCV agreement is **not** a gate. It cannot be: it needs 30 days of paper
trading that only happens after promotion. It is the orchestrator's monitoring
job (`live_orchestrator:_monitoring_thresholds`), and a breach quarantines the
live model rather than blocking a candidate. `max_exit_participation <= 0.20`
*is* a gate and is missing from the table below.

| Gate (design name) | Threshold | From |
|------|-----------|------|
| CPCV median Sharpe → `walk_forward_median_sharpe` | >= 0.5 | 4.1 |
| CPCV 5th percentile Sharpe → `walk_forward_p05_sharpe` | > 0 | 4.1 |
| Probability of backtest overfitting | <= 0.30 | 4.1 |
| Deflated Sharpe Ratio | > 0 at true trial count | 4.2 |
| Bootstrap P(Sharpe > 0) | >= 0.90 | 4.3 |
| Synthetic panels with positive Sharpe | >= 60% | 4.4 |
| Cost stress: 2x fees, 3x spread | median Sharpe still > 0 | 4.5 |
| Parameter plateau | >= 60% of +/-1-step neighbours keep >= 70% of Sharpe | 4.6 |
| OOS trade count | >= 100 | — |
| Paper/CPCV agreement | *monitoring, not a gate* — see above | — |

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
6. **Pooled model.** Panel LightGBM, purged CV, uniqueness weights. Landed
   without instrument identity and without meta-labelling — see §3.3 and §3.4.
7. **Simulation layer.** CPCV, the bootstrap, execution simulation, synthetic
   panels. CPCV landed but sits off the promotion path (§5); the walk-forward
   distribution is what the gates read.
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

## 8. The app layer

Landed. Recorded here with what each fix was, because several of the defects were
of the same kind as the pipeline's: numbers displayed as measurements that nobody
had measured.

### 8.1 Security

All four closed.

- **Postgres and the API are bound to loopback**, and the compose password is a
  required `.env` variable — `${POSTGRES_PASSWORD:?...}`, so `docker compose up`
  refuses to start without it rather than falling back to a working default in
  version control. `.env.example` is the template.
- **`POST /research/launch` is authenticated and fails closed.** With no
  `API_TOKEN` configured it returns 503, not 200: a deployment that forgot to set
  the secret refuses to launch rather than launching for anyone. Comparison is
  `hmac.compare_digest`, so a wrong token does not leak its prefix through
  timing. Read-only routes stay open — they serve a local dashboard, expose no
  credentials, and gating them buys nothing the origin policy does not already
  provide.
- **CORS origins come from the environment with `*` filtered out.** The wildcard
  had made every other entry in the list decorative.
- **`args` are validated, and rejected rather than sanitised.** Long lowercase
  flags and a bounded value charset; no spaces, no shell metacharacters, no
  filesystem paths. Rejecting matters: a silently stripped argument means the job
  ran with settings the requester does not believe it ran with, and that result
  looks legitimate.

`backend/api/tests/test_security.py` holds the fail-closed property and the
argument grammar.

### 8.2 Duplicated ORM models

`wallet.balance` is now 100,000 on both sides, and the divergence that produced
it is guarded by `backend/trader/tests/test_orm_parity.py`, which compares every
shared table column by column — name, type, nullability, primary key, default,
server default — and compares the two migration lists as well.

This is a different answer than the shared-schema module suggested above, and
deliberately so. A shared module means one image's package installed into the
other, which reintroduces the coupling the duplication exists to avoid; the
duplication is a legitimate isolation choice, and what it was missing was not
deduplication but *enforcement*. "Keep both in sync" in a doc is a hope. A test
that fails on the next divergence is a mechanism.

### 8.3 Correctness and quality

- **Back button works.** `App.tsx` has a `popstate` listener, and `RoutePath`
  derives from the `ROUTES` table so a route without a render case is a type
  error.
- **Every page has loading, error and empty states.** `src/components/StateBlock.tsx`
  and `src/hooks/usePolling.ts`. An error banner sits *above* existing data
  rather than replacing it — a stale price is worth more than a blank panel, as
  long as it is visibly stale. `loading` is distinct from `empty`, so a
  successful fetch returning nothing no longer renders as a spinner that never
  resolves.
- **`npm run lint` runs.** `.eslintrc.cjs` exists; typecheck, lint and build are
  all clean. `no-empty` with `allowEmptyCatch: false` is on, because
  `.catch(() => {})` in seven places is how the frontend came to hide every
  backend failure behind stale data.
- **Polling pauses on hidden tabs** and refreshes immediately on return, and each
  source polls at the rate its data actually moves. The wallet — which calls
  Coinbase, an Ethereum RPC node, Ethplorer and the Solana RPC — was being
  refetched every minute forever.
- **One HTTP client.** `src/api/client.ts`: one base URL, one `ApiError` carrying
  the server's `detail`, one place the token header is set, and a request timeout
  so a hung fetch does not leave a screen frozen on its last value. Five copies
  of `fetchWithError` had already drifted in what they reported.
- **Logic out of the routing layers.** `endpoints/wallet.py` is 21 lines;
  `controllers/wallet.py` holds the integrations.

### 8.3.1 Fabricated measurements — not in the original audit

Found while rewriting, and the most serious item in this section. Three separate
places presented invented numbers in a form indistinguishable from real ones:

- **`pr_auc` was `holdout_auc - 0.06`** and `precision_at_threshold` was
  `holdout_auc - 0.04`. One number, displayed three times, with two constants
  subtracted. And `holdout_auc` itself came from `signals.model_auc`, which the
  new signal writer leaves null — AUC is undefined for a regression on net
  return.
- **`drift_delta` was `realised_win_rate - holdout_auc * 100`**, subtracting an
  AUC from a percentage.
- **Feature importance fell back to a hardcoded table** — `momentum_24h: 0.26`,
  `trend_strength: 0.22`, four more — whenever `pruned_features_<coin>.json` was
  absent, which was always, because it belonged to the deleted pipeline. An
  explainability panel showing six plausible feature names with plausible weights
  is the worst available failure: it renders exactly like the real thing.
- **`get_research_runs` invented three runs per signal** — a train, an optimize
  and a validate, with start times derived by subtracting twelve minutes from the
  signal timestamp, durations hardcoded to 12, 20 and 8 minutes, and a status of
  "success" for all of them. None of it had happened.
- **The API served its own contract table with `fee_pct: 0.001`** for every
  contract — a third copy of the cost model, carrying the 10bp/side figure the
  whole rebuild exists to correct.

All replaced. The API now serves measurements or nulls with a reason, importances
come from the promoted model's booster, run history comes from `model_runs` joined
to the promotion ledger, and `/coins/cde-specs` reads the real fee schedule from
the same file the research pipeline prices its targets with.
`backend/api/tests/test_model_surface.py` asserts that a missing measurement
arrives as null.

The metric that replaces the AUC family is the one a net-return model can be held
to: the edge `decide()` claimed in basis points before each trade, against what
the trades earned. A model whose realised net runs consistently below its
forecast is mispriced, and it over-sizes every position that clears the
conviction floor.

### 8.4 What the app gained from the rebuild

All four landed, on the new `/model` route.

- **Provenance** — feature-set hash, cost config version, horizon, train window,
  row count *and* effective observation count, with a warning when the effective
  count is under 200 or when symbol identity is a feature.
- **The promotion gates as a screen** — failures first, each with its measured
  value beside its threshold and a plain-language note on what it protects
  against. A gate that missed by 0.01 and one that missed by an order of
  magnitude need different responses, and a red badge cannot tell them apart.
- **Distributions, not point estimates** — bootstrap, per-period and synthetic
  Sharpe each shown as median with p05 and p95.
- **One promote action** — `core/promotion.py` is the only route to live. The
  button launches `scripts.promote` through the authenticated endpoint; it does
  not promote anything itself, because the only thing allowed to install a model
  is the thing that ran the gates. Rejections stay in the ledger: the trial count
  is what the deflated Sharpe discounts by. A forced promotion needs a reason and
  stays visibly forced for as long as it is live.

---

## 9. What the end-to-end rehearsal found

The chain was run once on synthetic data seeded through the real storage layer —
scrape schema, migration, features, targets, training, walk-forward, simulation,
gates — rather than only through pytest's in-process calls. Three defects only a
full run could surface.

### 9.1 Venue was not in the scraper's unique keys

`ohlcv` carried a `venue` column whose unique key stayed
`(symbol, timeframe, event_time)`. Against `INSERT OR REPLACE`, that means a
Binance bar silently replaces the Coinbase bar for the same instrument and hour:
only one venue survives per bar, and the cross-venue features that need both at
the same timestamp — basis, lead-lag — would have produced no rows at all while
the column sat there looking correct.

`funding_rates` and `open_interest` had no venue column, and their inserts wrote
literals: funding recorded `source = "coinbase"` for every row including Binance
proxy rates, and open interest recorded `"ccxt"`, which names the client library
rather than an exchange.

This is §1.3 one level down. The venue column was added; the key was not.

Fixed, with a rebuild path for existing databases (SQLite cannot alter a UNIQUE
constraint) and four tests. Rows the old key already collapsed are unrecoverable —
re-running the backfill is the only way to get the second venue back.

### 9.2 The horizon came from the config, not the data

`train_forecast_model` read the horizon from `config.label_horizon_hours(profile)`
regardless of what the targets were built at. With `--horizon 8`, the targets
resolved at 8h while the model purged its validation split at the profile's 96h
and recorded 96h in its provenance.

Too wide is merely wasteful. The same bug with a horizon *longer* than the
profile's purges less than one label span, which leaks. And the recorded value is
not cosmetic — it drives `effective_observations`, the denominator under every
significance claim. Measured on the rehearsal dataset: 72 effective observations
reported, against 922 actual. A 12.8x understatement.

`horizon_bars` is now a parameter on `train_forecast_model`,
`cross_validate_forecast`, `generate_walk_forward_forecasts` and
`walk_forward_backtest`, and every caller passes `dataset.horizon_bars`.

### 9.3 The horizon is what governs whether there is enough data at all

Not a defect — a constraint, and the most important practical output of the
rehearsal. Overlapping labels are not independent observations: a label spanning
`h` bars overlaps its `h-1` neighbours, so the effective sample is roughly
`timestamps / h`.

On 92 days of hourly data across five instruments:

| horizon | effective observations | verdict |
|---------|-----------------------|---------|
| 96h (the profile default) | 18 from 1,768 timestamps | far too few |
| 8h | 232 from 1,856 timestamps | enough to start |

There are **three** levers, not two, and `scripts/preflight.py` reports which
one binds:

1. **Raise the recency half-life.** Weights sum to about `24 x H / ln 2`
   bar-equivalents however far back the store goes, so the weighted effective
   sample saturates at `24H/ln2/h` regardless of history. When that is already
   below 200, preflight says so and suppresses the scrape advice — more history
   genuinely cannot help until the half-life is raised. This is usually the
   binding constraint.
2. **Shorten the horizon.** Sized off the weighted budget when a decay is
   active, not the raw timestamp count — computing it the naive way produced
   advice to *lengthen* the horizon to 219h at five years of hourly data, which
   is the opposite of the fix.
3. **Scrape more**, about `200 x horizon / 24` days — which only helps once
   lever 1 is not the binding one.

This is much cheaper to learn before an overnight scrape than after.

### 9.4 What the rehearsal confirmed works

On a driftless-price panel with AR(1) funding, the suite reported what it should:
the carry head found the funding (in-sample IC +0.49) while the price head found
nothing (−0.04), and the gates blocked promotion with ten failures. `--quick`
correctly leaves the skipped simulation gates as failures, so a fast development
run can never promote. The API reads the resulting ledger entry — blocked, with
its ten named gates, correct horizon and effective-observation count — and reports
"no promoted model" for feature importances rather than substituting anything.

---

## 10. What taking screenshots found

Running the dashboard against a real evaluation, rather than against tests,
surfaced a defect that no test had caught — because the tests only checked that
bad candidates get *blocked*, which they did, for the wrong reason.

### 10.1 Three gates were structurally unmeasurable

`pbo`, `deflated_sharpe` and `parameter_plateau` were in `DEFAULT_GATES`, but
`evaluate_candidate` never computed any of them. A gate with no measurement fails
by design — that rule is correct, and §5 relies on it — so every candidate came
back with three gates reading "not measured".

The consequence: **`--force` was the only route to live.** A system whose entire
premise is that nothing reaches live except through the gates had gates that
nothing could pass. The tests passed throughout, because a blocked candidate is
what they assert, and eight of ten failing gates looks the same as eleven.

All three are now measured:

- **`deflated_sharpe`** from the observed Sharpe against the expected maximum of
  `trials` worthless strategies. `trials` is the ledger count, passed in by the
  caller — which is what finally makes "rejections are kept" load-bearing rather
  than merely tidy.
- **`parameter_plateau`** from perturbing the three thresholds `decide()` trades
  on — `min_edge_over_cost`, `vol_mult_tp`, `vol_mult_sl` — one step either way
  and re-running the walk-forward. Retention reports zero when the centre itself
  loses money, because "neighbours at least 70% as good as a negative Sharpe"
  would reward being uniformly bad.
- **`pbo`** from the per-period Sharpes of those same runs, which form exactly the
  (candidates x splits) matrix the estimate needs. Two gates out of one set of
  runs.

`test_a_full_evaluation_measures_every_gate` now fails if any measurement comes
back null, and asserts the report and the gate table describe the same set. A
gate added without a measurement breaks the suite instead of silently blocking
everything.

Measured on the demo dataset, all ten now carry numbers: `pbo` 0.50,
`deflated_sharpe` −11.93, `parameter_plateau` 0.00, against eight failures and
two passes. Which is the correct verdict — but it is now a verdict rather than an
absence.

### 10.2 The dashboard was reading the wrong LightGBM API

Feature importance called `Booster.feature_importance(importance_type='gain')` on
heads that are `LGBMRegressor` instances, whose attributes are
`feature_importances_` and `feature_name_`. Every request returned
"no attribute 'feature_importance'". Both APIs are handled now.

### 10.3 Three display defects only visible on screen

- `ModelStatusPanel` still had an AUC column, null on every row the current
  signal writer creates, and its fixed-width flex row let the gate reason and the
  timestamp overprint each other. It shows expected net edge against the round
  trip that edge has to clear.
- The positions and signals tables packed eight and nine columns into a
  third-width card and clipped the last one — the unrealised PnL and the gate
  result, which are the two numbers those panels exist to show.
- `PaperPositionsTable` carried its own contract-size table with a comment
  reading "Must match trading_costs.py", a file deleted in this rebuild. Contract
  size multiplies into unrealised PnL, so a stale entry there misreports a
  position silently. It comes from the API now — and the same table survived one
  more round in `DashboardPage.tsx` as a "fallback until the specs load", where
  three of its nine entries had already drifted against the real schedule (AVAX
  10 against 5, LINK 50 against 10, LTC 5 against 1). There is no fallback now:
  a position whose contract size is unknown shows the stored mark.

### 10.4 `frontend/node_modules` was committed

4,791 files, 55.8 MB, tracked since 072eb02. `package-lock.json` is tracked too,
so `npm ci` reproduces it exactly — which is the point of a lockfile. Untracked.
