# CryptoAlgo Multi-Coin Pipeline Recovery Plan (Research-Driven)

## 1) End-to-end pipeline map (what must work before deployment)

1. **Market + microstructure ingestion** via `scripts.run_pipeline`:
   - Pulls OHLCV (Coinbase first, CCXT fallback), funding, optional OI.
   - Writes to SQLite (`TRADER_DB_PATH`, default `./data/trading.db`).
2. **Feature generation** via `scripts.compute_features`:
   - Builds shared + coin-specific features from OHLCV/funding/OI.
   - Exports per-symbol feature CSVs + ML datasets under `./data/features`.
3. **Model training/backtest** via `scripts.train_model`:
   - Uses per-coin `CoinProfile` (thresholds, exits, filters, sizing).
   - Simulates walk-forward trading with fees/funding and prints edge decomposition + exit-reason diagnostics.
4. **Parameter optimization** via `scripts.optimize` (usually launched by `parallel_launch`):
   - Uses fold-based objective, holdout gates, and can block deployment when holdout fails.
5. **Robustness validation** via `scripts.validate_robustness`:
   - Computes Monte Carlo, CV consistency, DSR/PSR, holdout checks and assigns readiness tier.
6. **Orchestration** via `scripts.parallel_launch` / `scripts.live_orchestrator`:
   - `parallel_launch` runs optimize + paper_screen/full validation.
   - `live_orchestrator` runs recurring pipeline -> features -> signals, and atomic model promotion from staging.

---

## 2) What the latest results say (portfolio diagnostic interpretation)

From your recent run and follow-up context:
- **All tracked coins are currently failing readiness** (no deployable candidates).
- **Cost drag is material** (fees/funding absorb too much raw edge in weak setups).
- **Exit distributions are unhealthy** in failing runs (high stop-loss concentration, low TP conversion).
- **Holdout/readiness gates are working correctly** by blocking weak candidates from deployment.

Interpretation:
- This is not a single-coin problem; it is a **pipeline-level quality and robustness problem**.
- We need a systematic, coin-by-coin and cross-coin recovery process, not one-off BTC tuning.

---

## 3) Research hypotheses to test first (all-coin scope)

### H1 — Cost-adjusted edge is insufficient at current trade frequency
- If raw edge per trade is near zero/negative, Coinbase fee model dominates net returns.
- Action: increase selectivity and reduce churn per coin before broadening parameter search.

### H2 — Label horizon/barrier mismatch differs by coin
- Coins likely require different `label_forward_hours` and `label_vol_target` ranges.
- Action: run constrained per-coin grids and compare strict OOS + holdout performance.

### H3 — Exit geometry mismatch by volatility regime
- Stop-loss dominance suggests SL/TP/momentum entry geometry is misaligned with realized path behavior.
- Action: jointly tune (`vol_mult_sl`, `vol_mult_tp`, `max_hold_hours`) with objective penalties for stop-loss concentration and fee-heavy churn.

### H4 — Regime filters are not selective enough in weak conditions
- Overtrading chop periods kills expectancy.
- Action: stratify all coins by volatility/trend buckets and retune `min_vol_24h`, `max_vol_24h`, and `min_momentum_magnitude`.

### H5 — Portfolio selection is under-constrained
- Even if one coin looks decent in isolation, combined exposure may still be fragile.
- Action: only promote coins that pass individual gates **and** portfolio-level correlation/drawdown constraints.

---

## 4) 14-day execution plan (all tracked coins)

## Phase A (Days 1-2): Data + feature integrity audit (all symbols)

**Goal:** ensure no coin is being optimized on bad inputs.

1. Run data backfill sanity for BTC, ETH, SOL, XRP, DOGE with OI.
2. Run feature generation and verify per-symbol row counts, null coverage, and target balance.
3. Add per-symbol QA artifact (CSV/JSON) with:
   - available rows,
   - missingness by top-20 used features,
   - target positive rate,
   - feature drift snapshot for recent 30d vs prior 90d.

**Exit criteria:** No symbol with severe data gaps or broken targets.

## Phase B (Days 3-6): Per-coin robust search (all coins, independent)

**Goal:** produce at least 2 coins with positive holdout economics.

1. Run optimization for **all 5 coins** under `--preset robust180`.
2. Increase trial budgets by coin priority:
   - Tier 1 (ETH, SOL): higher budget first,
   - Tier 2 (BTC, XRP): medium,
   - Tier 3 (DOGE): medium with stricter filters.
3. Keep `require_holdout_pass` ON and evaluate additional practical checks:
   - holdout return > 0,
   - holdout Sharpe > 0,
   - holdout trades above minimum,
   - stop-loss share < 70%,
   - fee/edge ratio improving vs baseline.
4. Rank candidates by **cost-adjusted net edge + robustness**, not Sharpe alone.

**Exit criteria:** at least 2 coins reach PILOT-or-better with positive holdout.

## Phase C (Days 7-10): Cross-coin portfolio robustness

**Goal:** ensure selected coins survive together.

1. Build portfolio candidates from passing coins only.
2. Evaluate pairwise and basket-level behavior:
   - correlated drawdown windows,
   - simultaneous regime failures,
   - exposure overlap.
3. Drop coins that materially degrade combined robustness.

**Exit criteria:** portfolio basket has acceptable combined drawdown and stable readiness.

## Phase D (Days 11-14): Paper deployment readiness (staged rollout)

**Goal:** controlled launch with hard kill-switches.

1. Deploy only readiness-qualified coins; start with reduced position scale.
2. Enforce kill-switches:
   - max daily drawdown,
   - min rolling win-rate,
   - calibration drift,
   - abnormal fee/edge expansion,
   - coin-level disable if readiness degrades.
3. Retrain cadence remains active only while paper metrics stay aligned with validation expectations.

**Exit criteria:** 2-week stable paper performance with no hard-gate violations.

---

## 5) Immediate command runbook (all coins)

### A. Rebuild clean feature artifacts
```bash
cd backend/trader
python -m scripts.run_pipeline --backfill-only --backfill-days 365 --include-oi --db-path ./data/trading.db
python -m scripts.compute_features
```

### B. Portfolio-wide optimization (all tracked coins)
```bash
cd backend/trader
python -m scripts.parallel_launch \
  --coins BTC,ETH,SOL,XRP,DOGE \
  --trials 900 \
  --preset robust180 \
  --n-cv-folds 5 \
  --holdout-days 180 \
  --require-holdout-pass \
  --screen-threshold 55
```

### C. Validate robustness summary (all)
```bash
cd backend/trader
python -m scripts.validate_robustness --all --mode full --show
```

---

## 6) Pipeline-level improvements to implement next (code tasks)

1. **Add automated diagnostics export from `train_model` results**
   - Persist per-coin stop-loss share, fee/edge ratio, avg holding hours, regime bucket outcomes.
2. **Add objective shaping in `optimize`**
   - Penalize candidates with extreme stop-loss concentration and fee-heavy edge decay.
3. **Add stricter pre-trade quality filters**
   - Increase signal quality gating using calibrated probability spread and volatility/trend state checks.
4. **Add deterministic experiment tracking**
   - Save run config + git commit + artifact hash per coin and run cohort.
5. **Add deployment checklist automation**
   - Single command producing PASS/FAIL by coin and for combined portfolio readiness.

---

## 7) Definition of “ready to deploy” (coin + portfolio)

A **coin** is pilot-ready only if all are true:
- Holdout return > 0, holdout Sharpe > 0, and sufficient holdout trades.
- Robustness tier is `PILOT` or `FULL`.
- Cost-adjusted edge is positive with acceptable fee/edge ratio.
- Drawdown and ruin probabilities are within guardrails.
- CV and parameter stability checks pass.

A **portfolio** is deployable only if:
- At least 2 coins pass coin-level readiness.
- Combined basket drawdown/correlation checks remain within limits.
- No single coin dominates risk budget.

If no coins pass, keep deployment blocked and continue the research loop. Blocking bad models is a feature, not a bug.
