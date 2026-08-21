# agents.md — CryptoAlgo Codex Agent Configuration

## Project Overview

CryptoAlgo is a full-stack crypto trading research and monitoring workspace built for Coinbase CDE/perpetual workflows. It includes live orchestration, offline backtesting/optimization, and a paper-trading pipeline.

## Repository Structure

```
backend/
  api/              FastAPI service (Python 3.12, PostgreSQL)
  trader/           ML pipeline: data collection, features, training, optimization
frontend/           React 18 + Vite + TypeScript + Tailwind CSS dashboard
docker-compose.yml  Orchestrates all services
```

### Backend — API (`backend/api/`)

- **Framework**: FastAPI 0.115, SQLAlchemy 2.0, Pydantic 2.9
- **Database**: PostgreSQL 16 via `psycopg2-binary`
- **Entry point**: `app.py` creates the FastAPI app, registers CORS, includes all routers
- **Models**: `models/base.py` (declarative base), `models/trade.py` (paper orders, fills, positions, equity curve, `ModelRun`, `PaperEngineConfig` — no live-trade table; see below), `models/signals.py`, `models/wallet.py`, `models/model.py`, `models/research.py`
- **Endpoints** (each in `endpoints/`):
  - `coins.py` — `/coins/prices`, `/coins/cde-prices`, `/coins/cde-specs`, `/coins/history/{symbol}`
  - `signals.py` — `/signals/`
  - `wallet.py` — `/wallet/` (Coinbase spot + perps + Ledger addresses)
  - `paper.py` — `/paper/summary`, `/paper/fills`, `/paper/positions`, `/paper/equity`, `/paper/config`, `/paper/model-status`
  - `research.py` — `/research/summary`, `/research/runs`, `/research/features/{coin}`, `/research/scripts`, `/research/jobs`, `/research/jobs/{pid}/logs`, `POST /research/launch/{job}`
  - `model.py` — `/model/` (live provenance, gates, kill switch), `/model/promotions` (the ledger, rejections included), `/model/features` (real booster gains)
- **Security** (`security.py`): `require_token` gates every mutating route on `API_TOKEN` and **fails closed** — no token configured means 503, not open. `validate_job_args` restricts what a launched script can be handed, rejecting rather than sanitising. `allowed_origins` filters `*` out of the CORS list. Before this, `POST /research/launch` had no authentication and the origin list ended with `*`, so any page the browser had open could start a trader script with arbitrary arguments in a container holding the exchange keys.
- **No live-trade ledger.** There was a `trades` table, a `/trades` router and a `/trades/stats`, and nothing ever wrote a row: the only writers (`PgWriter.open_trade`/`.close_trade`) had no callers, so the endpoint served a win rate of `0.0` computed over an empty table, which renders identically to a measured 0%. All of it is deleted, and `tests/test_orm_parity.py` fails if the table comes back. The paper engine's ledger is `paper_positions`, which carries funding, exit reason and TP/SL; live execution should extend those tables rather than add a parallel schema.
- **Schema bootstrap runs at startup, not at import.** The API runs under `uvicorn --workers 4`; `create_all` used to run at module scope in each worker, and its check-then-create is not atomic, so a loser died with DuplicateTable before FastAPI existed to log it. `app.bootstrap_schema()` now runs from the lifespan hook under a Postgres transaction-scoped advisory lock.
- **Controllers**: Business logic in `controllers/`. Five controllers for six endpoint modules — `endpoints/coins.py` holds its product tables and the Coinbase client inline.
- **External deps**: `coinbase-advanced-py` for Coinbase API integration

### Backend — Trader (`backend/trader/`)

- **Language**: Python 3.12
- **Core modules** (`core/`):
  - `profiles.py` — Per-coin trading profiles (`CoinProfile`, frozen). 16 coins described by a feature *archetype* plus tuned deltas; archetypes are `mean_reversion`, `momentum_breakout`, `meme`, `trend_persistence`, `compression_breakout`. The last two are templates parameterized by coin (`{coin}_trend_spread_12h`), so a profile can never ask for a column the builder doesn't emit.
  - `costs.py` — Single source of truth for the *money inputs*: contract specs (`ContractSpec`, `get_contract_spec`), exchange fee assumptions (`configs/exchange/*.json`), the per-contract fee floor, and `symbols_missing_fee_schedule`. Absorbed the former `trading_costs.py` and `execution_sim.py`. Trade PnL and position sizing are deliberately **not** here — the module's own closing comment lists where each went: round-trip cost to `targets.py:round_trip_cost`, entry fee to `execution.py:entry_cost`, sizing to `execution.py:fractional_kelly` + `size_from_forecast`.
  - `config.py` — One `Config` with `resolve()` implementing CLI > profile > default. There is deliberately **no** declarative CLI/env layer: `CliParam`, `CLI_PARAMS`, `ENV_PARAMS`, `add_cli_args`, `from_env`, `from_args` and `build_parser` were deleted because nothing called them, and they advertised 22 flags and 5 environment variables that parsed, stored and reached nothing. Arguments live in `scripts/_common.py`. `with_cost_assumptions()` loads a venue's real fee schedule; `find_cost_config()` locates it by name across the repo checkout and the container image. `configs/` lives under `backend/trader/` on purpose — the trader's Docker build context is that directory, so a schedule above it is never copied into the image.
  - `features.py` — 76 features in nine mechanism groups (carry, cross_venue, volatility, liquidity, positioning, trend, market_factor, seasonality, cost). Relative groups are cross-sectionally standardised per timestamp so one pooled model can span the universe; absolute ones (fee hurdle, hour-of-day) are left on their own scale. Zero-lookahead is asserted by test.
  - `cv.py` — Purged walk-forward and combinatorial purged CV (12 groups, 2 held out, 66 splits, 11 paths). `assert_no_leakage` fails a fold whose purge is shorter than the label horizon. `effective_sample_size` reports the concurrency-weighted observation count: a 120-day hourly window at a 72h horizon carries 40, not 2,880. Also holds `FoldPreprocessor` for leak-free per-fold scaling.
  - `metrics.py` — Sharpe, drawdown, PSR, deflated Sharpe, PBO, CPCV path distributions, and the promotion gates. A gate with no measurement fails rather than passing.
  - `targets.py` — What the model predicts: net return over the horizon, decomposed into `price`, `carry` and `cost`. Replaces the triple-barrier classification, which could not express carry — the most plausible edge on hourly-funding perps. The identity `net_long + net_short == -2 * cost` holds exactly, which is the test that the decomposition is real.
  - `model.py` — `ForecastModel`: three LightGBM heads (price, carry, dispersion) plus the provenance to trust them. The dispersion head fits on *walk-forward* residuals — fitted in-sample it understated risk 2.35x, which would have over-levered by the same factor. `USE_SYMBOL_IDENTITY = False`: identity alone scored IC +0.54 on random walks, so it is excluded and `identity_ceiling_ic` reports the memorisation ceiling instead.
  - `signal.py` — One `decide()`. Every gate and threshold lives here (sizing is called out to `execution.size_from_forecast`), and both the backtest and the live signal writer call it. The previous system had three implementations, which is why its backtest and paper trading disagreed.
  - `execution.py` — Fills, slippage (square-root participation), funding accrual, barriers, liquidation, and sizing. Entries size against `liquidity_floor` — a trailing lower-quartile of volume — because the exit bar is not the entry bar; capping only on the deciding bar let a 10% entry dump 47% of the bar it exited through.
  - `backtest.py` — The event loop over `decide()`. `walk_forward_backtest` retrains per period and purges one horizon; the in-sample variant raises by default. Backtesting a model over its own training window returned +95,000 mean price PnL at t = +7 on driftless random walks.
  - `simulation.py` — Stationary bootstrap (Politis-Romano, Politis-White block length), regime-switching synthetic panels, cost stress, parameter surface, capacity curve. Turns one backtest number into a distribution.
  - `search.py` — One campaign runner with an append-only Parquet ledger. Replaces five search scripts that each had their own idea of a trial.
  - `dataset.py` — The single loader. Every script and the live path go through it, so they cannot disagree about the data.
  - `promotion.py` — The only route to live: train, walk-forward, bootstrap, stress, gate, install. Rejections stay in `models/promotions/` because the trial count is what the deflated Sharpe discounts by. `--force` requires a reason and records it.
  - `datastore.py` — Bitemporal research store: Parquet partitioned by dataset/venue/symbol/month, queried via DuckDB. Venue is part of the key; every read is point-in-time via `as_of` on `available_time`.
  - `pg_writer.py` — Postgres writer for signals, model runs and paper-trading persistence (`Signal`, `Wallet`, `ModelRun`, `PaperOrder`, `PaperFill`, `PaperPosition`, `PaperEquityCurve`, `PaperEngineConfig`). Duplicates ORM models for container isolation; `tests/test_orm_parity.py` compares columns, defaults, indexes and both migration lists against the API's copy.
- **Data collection** (`data_collection/`):
  - `storage.py` — Abstract `DatabaseBase` + `SQLiteDatabase` implementation with bi-temporal schema
  - `models.py` — Data models: `OHLCVBar`, `FundingRate`, `OpenInterest`
- **Scripts** (`scripts/`) — the seven *research* scripts (`train`, `backtest`, `promote`, `search`, `signals`, `preflight`, `build_features`) share `add_data_arguments` from `_common.py`, so no two can disagree about the dataset. The four *operational* ones (`run_pipeline`, `migrate_to_research_store`, `paper_engine`, `live_orchestrator`) hand-roll their own argparse and accept a different set:
  - `run_pipeline.py` — OHLCV, funding rate and OI collection, routed through `data_collection/ingest.py` so nothing reaches storage unvalidated
  - `migrate_to_research_store.py` — SQLite → Parquet, preserving venue and revision history
  - `build_features.py` — Assemble and materialise the feature panel with a content hash
  - `preflight.py` — Can this train? Cost schedule, store coverage, panel, targets, effective sample size, cross-section width, model fit. Run it before a long scrape.
  - `train.py` — Fit a model for inspection, with CPCV scoring
  - `backtest.py` — Walk-forward, then the simulation stack, then the gates
  - `promote.py` — Evaluate a candidate and install it only if the gates pass. `--history` shows what has been tried.
  - `search.py` — Run a campaign against the ledger
  - `signals.py` — `decide()` on the latest bar, written through `pg_writer`
  - `paper_engine.py` — Act on signals and account for them exactly as the backtest does: one cash movement per side, funding accrued hourly, liquidation modelled
  - `live_orchestrator.py` — The loop: scrape → sync → features → signals, with retraining on its own cadence through `promote`. Decides *when* to ask, never what the answer is.
- **ML stack**: LightGBM, scikit-learn
- **Data**: SQLite for the scraper, Parquet + DuckDB for research, joblib for model artifacts, Postgres for serving

### Frontend (`frontend/`)

- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS 3.4 with CSS custom properties (dark theme with glass-card effects)
- **Pages** (in `src/pages/`):
  - `DashboardPage.tsx` (`/`) — Portfolio value, equity curve marked to live prices, open positions, price grid, recent signals and fills, model status. Contract sizes come from `/coins/cde-specs` or not at all — there is no local fallback table. There was one, carried over from a deleted cost module, and three of its nine entries had drifted against the real schedule (AVAX 10 against 5, LINK 50 against 10, LTC 5 against 1). Contract size multiplies straight into unrealised PnL, so each of those misreported a position by that factor until the API answered. A position whose contract size is unknown now shows the stored mark.
  - `TradingPage.tsx` (`/trading`) — Per-instrument chart with fill markers, range selector, external wallet holdings, per-instrument signals and fills.
  - `ResearchPage.tsx` (`/research`) — Edge calibration (expected net edge against realised, the comparison the model can be held to), per-instrument health with the reason attached, retrain history from `model_runs` joined to the promotion ledger, and the script runner.
  - `ModelPage.tsx` (`/model`) — What is live and why it was allowed to be: the promotion gates with measured values beside thresholds, provenance, the simulation distributions, the candidate ledger including rejections, and the kill switch. Forced promotions stay visibly forced.
- **Components** (in `src/components/`): `Sidebar`, `PriceCard`, `PriceChart`, `EquityChart`, `SignalsTable`, `PaperPositionsTable`, `PaperFillsTable`, `ModelStatusPanel`, `GateTable`, `StateBlock`. `SignalsTable` shows the forecast decomposition — net, price, carry, cost, edge-to-risk — because the classifier columns it used to show (`Mom`, `Trend`, `ML`, `AUC`) are all null now.
- **API layer** (`src/api/`): `client.ts` is the only place that talks HTTP — one base URL, one `ApiError` with a real message, one place the `X-API-Token` header is set, and a request timeout. The rest (`coinsApi`, `signalsApi`, `walletApi`, `paperApi`, `researchApi`, `modelApi`) are thin wrappers over it. Five separate copies of `fetchWithError` had already drifted in what they reported.
- **Polling** (`src/hooks/usePolling.ts`): pauses when the tab is hidden, reports failures instead of swallowing them, and distinguishes "loading" from "empty". Every page previously ran its own `setInterval` forever with `.catch(() => {})`, so a dead backend was indistinguishable from a quiet market.
- **State components** (`src/components/StateBlock.tsx`): `Panel`, `Spinner`, `Empty`, `ErrorBlock`, `Freshness`. An error banner sits *above* existing data rather than replacing it — a stale price is worth more than a blank panel, as long as it is visibly stale.
- **Types**: `src/types.ts` — shared TypeScript interfaces
- **Routing**: Custom history-based routing in `App.tsx` (no react-router). `ROUTES` maps path to label; `PAGES: Record<RoutePath, ComponentType>` maps path to component. Add a page by adding an entry to *both* — `RoutePath` is derived (`keyof typeof ROUTES`), so it is not something you edit, and the `Record` is what makes a route with no component a compile error. The render used to be a chain of `route === '/x' && <XPage />`, exhaustive only by inspection.
- **Config**: `VITE_API_BASE_URL` env var (defaults to `http://localhost:8000`)

## Docker Compose Services

| Service        | Image/Build       | Port | Depends On            |
|----------------|-------------------|------|-----------------------|
| `db`           | `postgres:16`     | 5432 | —                     |
| `backend`      | `backend/api/`    | 8000 | db                    |
| `frontend`     | `frontend/`       | 3000 | backend               |
| `trader`       | `backend/trader/` | —    | db (healthy)          |
| `paper-engine` | `backend/trader/` | —    | db (healthy), trader  |

`paper-engine` runs `scripts.paper_engine` and is the thing that acts on signals
and keeps the paper account. Compose commands here name services explicitly, so
leaving it off the list leaves the engine down and the account frozen.
Health checks exist on `db` and `trader` only; `frontend/Dockerfile` is the only
multi-stage build.

Persistent volumes: `postgres_data`, `trader_data`, `trader_models`, `trader_logs`.

## Key Environment Variables

| Variable                    | Default / Required      | Used By   |
|-----------------------------|-------------------------|-----------|
| `COINBASE_API_KEY`          | Required for full flows | backend, trader |
| `COINBASE_API_SECRET`       | Required for full flows | backend, trader |
| `DATABASE_URL`              | postgres connection URI | backend, trader |
| `POSTGRES_PASSWORD`         | **Required** — compose refuses to start without it | db, backend, trader |
| `API_TOKEN`                 | unset ⇒ `POST /research/launch` returns 503 | backend |
| `VITE_API_TOKEN`            | unset ⇒ the promote button cannot authenticate | frontend |
| `CORS_ALLOW_ORIGINS`        | comma-separated; `*` is filtered out | backend |
| `LEDGER_WALLETS_JSON`       | `[]`                    | backend   |
| `VITE_API_BASE_URL`         | `http://localhost:8000` | frontend  |
| `TRADER_DB_PATH`            | `/app/data/trading.db` in compose; `backend/trader/data/trading.db` when a script is run on a host | trader |
| `RESEARCH_STORE`            | `/app/data/research`    | trader, paper-engine |
| `TRADER_DATA_MOUNT`         | `trader_data` (a named volume). Read by compose, not by code: set it to `./backend/trader/data` to run the containers against a store scraped on the host, which the named volume otherwise masks | compose |
| `COST_CONFIG`               | `coinbase_us_perps_cde_v202602.json` — unset misprices every contract by 0.06x–2.5x | trader, paper-engine |
| `TRADE_VENUE`               | `coinbase`              | trader    |
| `REFERENCE_VENUE`           | `coinbase_spot` — Binance/OKX/Bybit are 451 from a US IP, so the old `binance` default resolved to seven all-NaN columns | trader |
| `HTTPS_PROXY` / `HTTP_PROXY` | unset — **required from a US IP** for open-interest data and for any offshore reference venue (Binance, OKX and Bybit all answer 451). Not needed for the `coinbase_spot` default | trader |
| `SYMBOLS`                   | unset (the whole profile universe) | trader |
| `EQUITY`                    | `100000`                | trader    |
| `LEVERAGE`                  | `4`                     | trader    |
| `CYCLE_INTERVAL_SECONDS`    | `3600`                  | trader    |
| `CYCLE_ALIGN_MINUTE`        | `3`                     | trader    |
| `INCREMENTAL_BACKFILL_HOURS`| `6`                     | trader    |
| `INITIAL_BACKFILL_DAYS`     | see `live_orchestrator`  | trader    |
| `TRAIN_WINDOW_DAYS`         | `0` (all history)       | trader    |
| `RECENCY_HALF_LIFE_DAYS`    | unset (`Config`: `50`)  | trader    |
| `RETRAIN_EVERY_DAYS`        | `7`                     | trader    |
| `WALK_FORWARD_PERIODS`      | `6`                     | trader    |
| `MODELS_DIR`                | `models/`               | trader, backend |
| `SEARCH_LEDGER`             | see `scripts/search.py` | trader    |
| `ORCHESTRATOR_STATE_FILE`   | see `live_orchestrator` | trader    |
| `PAPER_MONITOR_*`           | six thresholds, see `live_orchestrator:_monitoring_thresholds` | trader |
| `TRADER_DIR`                | `../trader`             | backend   |
| `LOG_LEVEL`                 | `INFO`                  | all       |

Every variable compose sets is read by something —
`tests/test_deployment.py::test_every_environment_variable_compose_sets_is_read_by_something`
enforces it. Four rows used to sit in this table naming variables no code read:
`SIGNAL_THRESHOLD`, `MIN_AUC`, `EXCLUDE_SYMBOLS` and `FEATURE_LOOKBACK_DAYS`.
`LEVERAGE` was the dangerous one — it was in compose, documented here, and read
nowhere, so an operator lowering it watched the book keep sizing at 4x. It is
wired now (`--leverage`, threaded from the orchestrator).

## Tracked Coins

Two universes, deliberately different sizes:

- **Served by the API and the frontend (9):** BTC, ETH, SOL, XRP, DOGE, AVAX, ADA, LINK, LTC — each with a spot product (`{COIN}-USD`) and a CDE perpetual contract (e.g. `BIP-20DEC30-CDE`, `ETP-20DEC30-CDE`, `SLP-20DEC30-CDE`, `XPP-20DEC30-CDE`, `DOP-20DEC30-CDE`). `endpoints/coins.py` and `frontend/src/types.ts:ALL_COINS` agree on this list.
- **Modelled by the trader (16):** the nine above plus BCH, DOT, NEAR, PEPE, SHIB, SUI, XLM — `core/profiles.py:COIN_PROFILES`.

## Coding Conventions

- **Python**: Type hints throughout, dataclasses for config, logging via `logging` module. Scripts are runnable as modules (`python -m scripts.<name>`). Use `os.getenv()` with sensible defaults for all config.
- **TypeScript/React**: Functional components with hooks, no class components. Custom CSS variables for theming. `recharts` for charting. Fetch-based API layer (no axios).
- **SQL**: SQLAlchemy ORM for Postgres tables, raw SQLite for trader pipeline data. Bi-temporal schema in trader storage.
- **Docker**: volume mounts for development hot-reload. Multi-stage build on the frontend only; health checks on `db` and `trader` only.

## Common Commands

```bash
# Start everything. Name paper-engine too — services are explicit here, so
# omitting it leaves the engine down and the paper account frozen.
docker compose up --build db backend frontend trader paper-engine

# Is the data ready to train on?
docker compose run --rm trader python -m scripts.preflight

# One-off walk-forward backtest with the full simulation stack and the gates
docker compose run --rm trader python -m scripts.backtest --full

# Evaluate a candidate; installs it only if every gate passes
docker compose run --rm trader python -m scripts.promote

# What has been tried, and why it did not promote
docker compose run --rm trader python -m scripts.promote --history

# One retrain cycle through the gates
docker compose run --rm trader python -m scripts.live_orchestrator --retrain-only

# Parameter search against the append-only ledger
docker compose run --rm trader python -m scripts.search

# Frontend dev
cd frontend && npm ci && npm run dev

# API dev
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload
```

## Agent Instructions

- When modifying trader logic, be aware that `core/profiles.py` is the single source of truth for per-coin feature sets, thresholds, and ML hyperparameters. Changes there cascade into training, search, and signal generation.
- `docs/RESEARCH_PIPELINE.md` is the design spec. Read it before changing anything under `core/`.
- **Nothing reaches live except through `core/promotion.py`.** It trains, walk-forward backtests, bootstraps, stresses and gates a candidate, and installs it only if every gate in `core/metrics.py:DEFAULT_GATES` passed. Rejections are kept: the trial count is what the deflated Sharpe ratio discounts by. If you add a promotion criterion, add it to `DEFAULT_GATES` and to `SimulationReport.measurements()` — a gate with no measurement fails, which is the intended direction.
- **The horizon comes from the data, not the config.** `train_forecast_model`, `cross_validate_forecast`, `generate_walk_forward_forecasts` and `walk_forward_backtest` all take `horizon_bars`, and callers pass `dataset.horizon_bars`. It sets the purge width between train and test and is recorded in the model's provenance, where it drives `effective_observations`. Reading it from the config instead meant a run with `--horizon 8` built targets at 8h while the model purged at the profile's 96h and reported a twelvefold-understated effective sample — and the same bug with a *longer* horizon purges less than one label span, which leaks.
- **One `decide()`.** `core/signal.py` holds every gate and threshold, and calls `execution.size_from_forecast` for the size. Both the backtest and the live signal writer call it; do not add a second path. Three separate implementations is why the previous system's backtest and paper trading disagreed.
- The trader and API run in separate containers with duplicated ORM models (in `pg_writer.py`). Keep them in sync when changing database schema — `tests/test_orm_parity.py` compares columns, types, nullability, defaults, indexes, and both the column and index migration lists, and fails on divergence. `create_all` only creates missing *tables*, so a new column **or index** on an existing model needs a matching `ALTER`/`CREATE INDEX IF NOT EXISTS` in both places.
- The frontend has no router library — routing is handled manually via `window.history.pushState` in `App.tsx`. Add a page by adding an entry to `ROUTES` (path → label) *and* to `PAGES` (path → component). Do not try to edit `RoutePath`: it is `keyof typeof ROUTES`. `PAGES` is typed `Record<RoutePath, ComponentType>`, so a route with no component fails `tsc`.
- The research scripts take CLI args only (`scripts/_common.py:add_data_arguments`); `live_orchestrator.py` and `paper_engine.py` are the two that default their flags from environment variables, which is how compose configures a deployment. Check the `argparse` block at the bottom of each script for what it actually accepts.
- `core/promotion.py` stages into `models/.staging/{version}/` and atomically renames into place. `live_orchestrator.py` decides *when* to evaluate a candidate; it never decides whether one is good.
- Feature engineering is declarative and universe-wide, not per-coin scripts. `core/features.py` emits nine mechanism groups; `core/profiles.py` selects which columns a coin's archetype uses. A profile can never ask for a column the builder does not emit — the templates are parameterised by coin.
- The research endpoints read from model artifacts, the promotion ledger, `signals` and `paper_positions`. There is no live-trade history and no separate research database. Edge calibration compares the mean forecast over a signal window against the mean realised return of paper positions **opened inside that same window** — the two used to come from different periods, so a retrained model was graded on returns earned by its predecessor.
- Tests under `backend/trader/tests/` are organised around the failure modes that produced fake edge before: lookahead (`test_backtest.py`), symbol-identity memorisation (`test_model.py`), leaked fold statistics (`test_cv_and_metrics.py`), the cost identity (`test_targets.py`), and blocked candidates reaching live (`test_promotion.py`). Mark anything over ~10s `@pytest.mark.slow`.
- **Deleted in the rebuild**, so do not reference them: `core/labels.py`, `backend/api/endpoints/trade.py`, `backend/api/controllers/trade.py`, `frontend/src/api/tradesApi.ts`, `scripts/train_model.py`, `scripts/compute_features.py`, the five search scripts, `scripts/validate_robustness.py`, `scripts/prune_features.py`, `scripts/preflight_check.py`, `features/` entirely, `core/coin_profiles.py`, `core/labeling.py`, `core/meta_labeling.py`, `core/paper_profile_overrides.py`, `core/strategies/`, `core/cv_splitters.py`, `core/metrics_significance.py`, `core/study_significance.py`, `core/overfit_diagnostics.py`, `core/preprocessing_cv.py`, `core/reason_codes.py`, `core/run_manifest.py`, `core/trading_costs.py`, `core/execution_sim.py`.