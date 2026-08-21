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
edge on hourly-funding perps: 2bp/hour is 48bp/day against a 5-54bp round trip.
`core/targets.py` now regresses *net return*, decomposed into price, carry and
cost, and `net_long + net_short == -2 * cost` holds exactly.

**Nothing reaches live except through the gates.** `core/promotion.py` trains,
walk-forward backtests, bootstraps, stresses and gates a candidate, then installs
it only if every gate passed. Rejections stay in `models/promotions/` because the
trial count is what the deflated Sharpe discounts by.

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

## Critical Architecture Notes

- **`core/profiles.py` is the single source of truth** for per-coin feature sets, thresholds, and ML hyperparameters. Coins are described by a feature *archetype* (`mean_reversion`, `momentum_breakout`, `meme`, `trend_persistence`, `compression_breakout`) plus tuned deltas. Changes cascade into training, search, and signal generation.
- **`core/costs.py` is the single source of truth for the money *inputs*** — contract specs, exchange fee assumptions, and the per-contract fee floor. Trade PnL and sizing are deliberately elsewhere and the module's closing comment says where: round-trip cost in `core/targets.py`, entry fee and Kelly/risk-budget sizing in `core/execution.py`. Load a venue's real schedule with `Config.with_cost_assumptions(find_cost_config())` — `configs/` lives under `backend/trader/` so the Docker build context includes it; the hardcoded defaults are 10bps/side, which is wrong for Coinbase CDE by 0.06x-2.5x depending on the contract.
- **Duplicated ORM models**: `backend/trader/core/pg_writer.py` duplicates the API ORM models for container isolation. `backend/trader/tests/test_orm_parity.py` fails when they diverge in columns, types, nullability, defaults, or migration lists — a note in a doc was not enough; `wallet.balance` had already drifted 10,000 against 100,000, so whichever container created the row decided the paper account's starting balance.
- **No react-router**: Frontend routing is manual via `window.history.pushState` in `App.tsx`. Add a page by adding an entry to `ROUTES` (path → label) **and** to `PAGES` (path → component). `RoutePath` derives from `ROUTES`; `PAGES` is a `Record<RoutePath, ComponentType>`, which is what makes a route with no component a `tsc` error. The render used to be a chain of `route === '/x' && <XPage />`, and this file used to claim that was exhaustive — it was not.
- **Frontend HTTP goes through `src/api/client.ts`**: one base URL, one error type, one place the `X-API-Token` header is set. Poll with `usePolling`, which pauses on hidden tabs and surfaces failures. Five copies of `fetchWithError` had already drifted in how they reported errors, and every `.catch(() => {})` made a dead backend look like a quiet market.
- **The API serves measurements, never substitutes.** A missing value is null with a reason. The research surface used to report `pr_auc` as `holdout_auc - 0.06`, `precision_at_threshold` as `holdout_auc - 0.04`, and — when the artifact it wanted was absent, which was always — a hardcoded table of six feature importances. All of it rendered identically to real data.
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
`0.000009` per hour is **0.09 bp/hour**, or 2.16 bp/day. Hours of carry needed to
cover a round trip, measured against the CDE fee schedule:

| contract | round trip | at 0.09 bp/h (observed) | at 2 bp/h (assumed) |
|----------|-----------:|------------------------:|--------------------:|
| XPP (XRP)  |  5.7 bp |  63 h |  2.9 h |
| DOP (DOGE) |  7.3 bp |  81 h |  3.6 h |
| SLP (SOL)  |  8.7 bp |  97 h |  4.3 h |
| BIP (BTC)  | 25.0 bp | 278 h | 12.5 h |
| ETP (ETH)  | 48.1 bp | 534 h | 24.1 h |

This is **one snapshot**, taken while BTC was +6.7% on the day, and funding is
volatile and sometimes negative — so it is not a verdict on the carry thesis.
But it does mean the "24h hold pays for itself on carry alone on four of five
contracts" claim rests on an assumed number, not a measured one, and the first
real observation is 22x lower. Collect the distribution before sizing anything on
it. `scripts/preflight.py` reports sample size; it cannot tell you the edge is
real.

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

## The reference venue needs a proxy from a US IP

Coinbase data — the instrument actually traded — is authenticated and US-legal,
so it scrapes fine. The *reference* venue does not: Binance, OKX and Bybit all
answer HTTP 451 to a US IP, and open interest has no Coinbase-native source at
all (`run_pipeline.py:412`), so both come through CCXT.

Without `HTTPS_PROXY` set you lose two feature groups:

| group | features | source | blocked from a US IP |
|-------|---------:|--------|----------------------|
| `cross_venue` (basis, lead-lag) | 7 | reference venue via CCXT | yes |
| `positioning` (OI) | 6 | CCXT, no Coinbase endpoint | yes (and `--include-oi` is opt-in) |
| `carry` | 9 | Coinbase native funding | no |
| `market_factor` | 9 | Coinbase BTC bars | no |

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

If the scraper's CCXT fallback served a different exchange, bars are stamped with
*that* venue's name — point `--reference-venue` at whichever one it stored, or
the reader asks for `binance` and matches nothing.

## Environment Variables

See `AGENTS.md` for the full table. Minimum required for live workflows: `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `DATABASE_URL`, `POSTGRES_PASSWORD` (compose refuses to start without it), `COST_CONFIG` (unset misprices every contract by 0.06x–2.5x), and `API_TOKEN` + `VITE_API_TOKEN` if you want the dashboard's script runner.
