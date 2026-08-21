# CryptoAlgo

A research platform for trading Coinbase CDE perpetual futures. Three services:

- **trader** — data collection, features, model, simulation, promotion gates, paper trading
- **api** — FastAPI over PostgreSQL, serving what the trader wrote
- **frontend** — React/Vite dashboard

The organising idea is that **nothing reaches live except through the gates**. A
model is trained, walk-forward backtested, resampled, cost-stressed, run against
synthetic panels, and measured against a fixed set of promotion criteria. It is
installed only if every one of them passes. Rejections are kept, because the
count of attempts is what the deflated Sharpe ratio has to discount by.

---

## Repository structure

```text
backend/
  api/                FastAPI service, PostgreSQL models, endpoints, controllers
  trader/
    core/             The library: costs, features, targets, model, signal,
                      backtest, simulation, promotion
    data_collection/  Scraper: exchange clients, validation, SQLite storage
    scripts/          One entrypoint per verb
    configs/exchange/ Venue fee schedules
frontend/             React 18 + Vite + TypeScript + Tailwind
docs/RESEARCH_PIPELINE.md   The design spec. Read it before changing core/.
docker-compose.yml
```

---

## The pipeline

```text
Coinbase (+ Binance as a reference venue)
        │  run_pipeline
        ▼
SQLite — the scraper's landing zone, venue-keyed
        │  migrate_to_research_store
        ▼
Parquet + DuckDB — bitemporal, point-in-time readable
        │  build_features
        ▼
Feature panel — 76 features in nine mechanism groups
        │  core/targets.py
        ▼
Targets — net return, decomposed into price, carry and cost
        │  train
        ▼
Model — three heads: price, carry, dispersion
        │  core/signal.py decide()
        ▼
Decisions — one implementation, shared by the backtest and the live writer
        │  backtest → simulation → gates
        ▼
promote — installs only if every gate passes
        │  signals
        ▼
paper_engine — accounted exactly as the backtest accounts
```

### What the model predicts

Net return over the horizon, split into its parts:

    net_long  = price_return + carry_return - cost
    net_short = -price_return - carry_return - cost

so `net_long + net_short == -2 * cost` exactly. The decomposition is the point.
On hourly-funding perps, carry is the most plausible edge available — 2bp/hour is
48bp/day against a round trip of 5bp (DOGE) to 54bp (ETH) — and a model that
cannot express carry separately from price cannot find it.

---

## Running it

### Full stack

```bash
cp .env.example .env      # POSTGRES_PASSWORD is required
docker compose up --build db backend frontend trader paper-engine
```

Postgres and the API bind to loopback. `POST /research/launch` needs `API_TOKEN`
set and fails closed without it.

### Development

```bash
cd frontend && npm ci && npm run dev
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload
```

### The research loop

Every trader script shares the same data arguments (see `scripts/_common.py`), so
none of them can disagree about which dataset they are looking at.

```bash
cd backend/trader

# 1. Collect, and move it into the research store
python -m scripts.run_pipeline --backfill-only --backfill-days 365
python -m scripts.migrate_to_research_store --venue coinbase

# 2. Check it can actually train, BEFORE the long scrape
python -m scripts.preflight

# 3. Build the panel
python -m scripts.build_features

# 4. Train for inspection
python -m scripts.train

# 5. Walk-forward, simulation stack, gates — installs nothing
python -m scripts.backtest --full

# 6. Evaluate a candidate and install it if the gates pass
python -m scripts.promote
python -m scripts.promote --history      # what has been tried, and why not

# 7. Search, decide, trade
python -m scripts.search
python -m scripts.signals
python -m scripts.paper_engine
```

`scripts.live_orchestrator` runs steps 1, 3 and 7 on a cycle, with retraining on
its own cadence through `promote`. It decides *when* to evaluate a candidate; it
never decides whether one is good.

---

## Run preflight before you scrape

Overlapping labels are not independent observations. A label spanning `h` bars
overlaps its `h-1` neighbours, so the effective sample is roughly
`timestamps / h`. Treating the row count as the sample size is how a t-statistic
ends up several times too confident.

Measured on 92 days of hourly data across five instruments:

| horizon | effective observations | verdict |
|---------|-----------------------|---------|
| 96h | 18 from 1,768 timestamps | far too few |
| 8h  | 232 from 1,856 timestamps | enough to start |

Uniqueness is only half of it. Training also weights each row by
`0.5 ** (age_days / H)`, where `H` is `Config.recency_half_life_days`, and those
weights sum to about `24 x H / ln 2` bar-equivalents **however far back the store
goes**. So the weighted sample saturates near `24 x H / ln 2 / h`, and at the
default `H = 50` days a 96h horizon tops out around 18 effective observations
whether you hold one year of history or five — while uniqueness alone reports 456
at five years.

That makes the half-life, not the history, usually the binding constraint:

| horizon | H=50d | H=180d | H=365d | H=730d | off |
|---------|-------|--------|--------|--------|-----|
| 96h | 18 | 64 | 127 | 216 | 456 |
| 24h | 72 | 259 | **510** | 867 | 1,825 |
| 8h  | 216 | 778 | 1,530 | 2,601 | 5,475 |

`scripts/preflight.py` reports both numbers and names whichever lever binds,
including saying plainly when more history cannot help. There are three ways out,
not two: raise the half-life, shorten the horizon, or scrape more — and only the
last one costs a night.

```bash
python -m scripts.preflight --horizon 24 --recency-half-life-days 365
```

It is much cheaper to learn this before an overnight scrape than after.

---

## The promotion gates

Defined in `core/metrics.py:DEFAULT_GATES`, and every one of them has to pass. A
gate with no measurement **fails** — "we did not run that test" is not evidence
of safety.

| gate | threshold | what it protects against |
|------|-----------|--------------------------|
| `walk_forward_median_sharpe` | ≥ 0.5 | picking the best path instead of the middle |
| `walk_forward_p05_sharpe` | ≥ 0.0 | a strategy whose bad paths lose money |
| `pbo` | ≤ 0.30 | the in-sample winner losing out-of-sample |
| `deflated_sharpe` | ≥ 0.0 | the best of fifty random strategies looking good |
| `bootstrap_positive_fraction` | ≥ 0.90 | one lucky ordering of trades |
| `synthetic_positive_fraction` | ≥ 0.60 | fragility to paths that did not happen |
| `stressed_median_sharpe` | ≥ 0.0 | costs being worse than assumed |
| `parameter_plateau` | ≥ 0.60 | a spike fitted to noise rather than a mechanism |
| `oos_trades` | ≥ 100 | statistics on too few trades |
| `max_exit_participation` | ≤ 0.20 | fills that are fiction at the claimed size |

`--force` can override them, but it requires a reason and records it, so a forced
model stays visibly forced for as long as it is live. The `/model` route in the
dashboard shows all of this, including the rejections.

---

## What the platform will not tell you

Worth stating, because the machinery is easy to believe:

- **Synthetic panels are not evidence of edge.** A generator contains only the
  structure calibrated into it. Fit one with momentum and momentum strategies
  will work on it; fit one without and nothing will. They test robustness and
  sizing.
- **A backtest cannot price liquidity it never saw.** Entries are sized against a
  trailing lower-quartile of volume so the position stays exitable, but exits
  land where the barrier fires. Exit participation is reported and gated, not
  capped.
- **Open interest comes from a proxy venue.** Coinbase publishes none, so the
  positioning features describe a different book than the one being traded. The
  loader says so in its warnings rather than leaving it implicit.
- **A missing measurement is served as null.** The dashboard says "not measured"
  instead of showing a plausible number. This is deliberate and hard-won.

---

## Testing

```bash
cd backend/trader && pytest              # the library and the pipeline
cd backend/api    && pytest              # auth, argument validation, response shapes
cd frontend       && npm run typecheck && npm run lint && npm run build
```

The trader suite is organised around the failure modes that produced fake edge
before, and each test names the one it guards:

- `test_backtest.py` — lookahead. Trading in-sample forecasts returned a mean
  price PnL of +95,000 at t = +7 on driftless random walks.
- `test_model.py` — symbol-identity memorisation. Identity alone scored an
  information coefficient of +0.54 on random walks.
- `test_cv_and_metrics.py` — leaked fold statistics, and the gate arithmetic.
- `test_targets.py` — the cost identity.
- `test_ingest.py` — venue keying. Without venue in the unique key, each venue's
  bars silently overwrote the other's.
- `test_promotion.py` — a blocked candidate reaching the live path.
- `test_orm_parity.py` — the two duplicated ORM model sets diverging.

---

## Documentation

- `docs/RESEARCH_PIPELINE.md` — the design spec: the defects that motivated the
  rebuild, the architecture, the simulation layer, the gates, and what
  simulation cannot tell you.
- `AGENTS.md` — module-by-module architecture and conventions.
- `CLAUDE.md` — the short version, plus the commands.
