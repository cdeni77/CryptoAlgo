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
- **Models**: `models/base.py` (declarative base), `models/trade.py`, `models/signals.py`, `models/wallet.py`, `models/research.py`
- **Endpoints** (each in `endpoints/`):
  - `coins.py` — `/coins/prices`, `/coins/cde-specs`, `/coins/history/{symbol}`
  - `trade.py` — `/trades/`, `/trades/recent`, `/trades/open`, `/trades/closed`, `/trades/coin/{coin}`, `/trades/{trade_id}`
  - `signals.py` — `/signals/`, `/signals/coin/{coin}`, `/signals/{signal_id}`
  - `wallet.py` — `/wallet/` (Coinbase spot + perps + Ledger addresses)
  - `paper.py` — `/paper/orders`, `/paper/fills`, `/paper/positions`, `/paper/equity`
  - `research.py` — `/research/summary`, `/research/coins/{coin}`, `/research/runs`, `/research/features/{coin}`, `/research/scripts`, `/research/jobs`, `POST /research/launch/{job}`
  - `model.py` — `/model/` (live provenance, gates, kill switch), `/model/promotions` (the ledger, rejections included), `/model/features` (real booster gains)
- **Security** (`security.py`): `require_token` gates every mutating route on `API_TOKEN` and **fails closed** — no token configured means 503, not open. `validate_job_args` restricts what a launched script can be handed, rejecting rather than sanitising. `allowed_origins` filters `*` out of the CORS list. Before this, `POST /research/launch` had no authentication and the origin list ended with `*`, so any page the browser had open could start a trader script with arbitrary arguments in a container holding the exchange keys.
- **Controllers**: Business logic in `controllers/` matching each endpoint module
- **External deps**: `coinbase-advanced-py` for Coinbase API integration

### Backend — Trader (`backend/trader/`)

- **Language**: Python 3.12
- **Core modules** (`core/`):
  - `profiles.py` — Per-coin trading profiles (`CoinProfile`, frozen). 16 coins described by a feature *archetype* plus tuned deltas; archetypes are `mean_reversion`, `momentum_breakout`, `meme`, `trend_persistence`, `compression_breakout`. The last two are templates parameterized by coin (`{coin}_trend_spread_12h`), so a profile can never ask for a column the builder doesn't emit.
  - `costs.py` — Single source of truth for money: contract specs, exchange fee assumptions (`configs/exchange/*.json`), round-trip cost breakdown, trade PnL, position sizing. Absorbed the former `trading_costs.py` and `execution_sim.py`.
  - `config.py` — One `Config` with `resolve()` implementing CLI > profile > default. CLI flags and env vars generated from declarative tables. `with_cost_assumptions()` loads a venue's real fee schedule; `find_cost_config()` locates it by name across the repo checkout and the container image. `configs/` lives under `backend/trader/` on purpose — the trader's Docker build context is that directory, so a schedule above it is never copied into the image.
  - `features.py` — 76 features in nine mechanism groups (carry, cross_venue, volatility, liquidity, positioning, trend, market_factor, seasonality, cost). Relative groups are cross-sectionally standardised per timestamp so one pooled model can span the universe; absolute ones (fee hurdle, hour-of-day) are left on their own scale. Zero-lookahead is asserted by test.
  - `labels.py` — Triple-barrier labelling. One implementation, not two. The take-profit barrier includes the real per-contract round-trip cost, so a move that clears the barrier but not the fees is labelled a loss. Same-bar ties and timeouts resolve against the trade.
  - `cv.py` — Purged walk-forward and combinatorial purged CV (12 groups, 2 held out, 66 splits, 11 paths). `assert_no_leakage` fails a fold whose purge is shorter than the label horizon. `effective_sample_size` reports the concurrency-weighted observation count: a 120-day hourly window at a 72h horizon carries 40, not 2,880. Also holds `FoldPreprocessor` for leak-free per-fold scaling.
  - `metrics.py` — Sharpe, drawdown, PSR, deflated Sharpe, PBO, CPCV path distributions, and the promotion gates. A gate with no measurement fails rather than passing.
  - `targets.py` — What the model predicts: net return over the horizon, decomposed into `price`, `carry` and `cost`. Replaces the triple-barrier classification, which could not express carry — the most plausible edge on hourly-funding perps. The identity `net_long + net_short == -2 * cost` holds exactly, which is the test that the decomposition is real.
  - `model.py` — `ForecastModel`: three LightGBM heads (price, carry, dispersion) plus the provenance to trust them. The dispersion head fits on *walk-forward* residuals — fitted in-sample it understated risk 2.35x, which would have over-levered by the same factor. `USE_SYMBOL_IDENTITY = False`: identity alone scored IC +0.54 on random walks, so it is excluded and `identity_ceiling_ic` reports the memorisation ceiling instead.
  - `signal.py` — One `decide()`. Every gate, threshold and sizing rule lives here, and both the backtest and the live signal writer call it. The previous system had three implementations, which is why its backtest and paper trading disagreed.
  - `execution.py` — Fills, slippage (square-root participation), funding accrual, barriers, liquidation, and sizing. Entries size against `liquidity_floor` — a trailing lower-quartile of volume — because the exit bar is not the entry bar; capping only on the deciding bar let a 10% entry dump 47% of the bar it exited through.
  - `backtest.py` — The event loop over `decide()`. `walk_forward_backtest` retrains per period and purges one horizon; the in-sample variant raises by default. Backtesting a model over its own training window returned +95,000 mean price PnL at t = +7 on driftless random walks.
  - `simulation.py` — Stationary bootstrap (Politis-Romano, Politis-White block length), regime-switching synthetic panels, cost stress, parameter surface, capacity curve. Turns one backtest number into a distribution.
  - `search.py` — One campaign runner with an append-only Parquet ledger. Replaces five search scripts that each had their own idea of a trial.
  - `dataset.py` — The single loader. Every script and the live path go through it, so they cannot disagree about the data.
  - `promotion.py` — The only route to live: train, walk-forward, bootstrap, stress, gate, install. Rejections stay in `models/promotions/` because the trial count is what the deflated Sharpe discounts by. `--force` requires a reason and records it.
  - `datastore.py` — Bitemporal research store: Parquet partitioned by dataset/venue/symbol/month, queried via DuckDB. Venue is part of the key; every read is point-in-time via `as_of` on `available_time`.
  - `pg_writer.py` — Postgres writer for trades, signals, paper-trading persistence. Duplicates ORM models for container isolation.
- **Data collection** (`data_collection/`):
  - `storage.py` — Abstract `DatabaseBase` + `SQLiteDatabase` implementation with bi-temporal schema
  - `models.py` — Data models: `OHLCVBar`, `FundingRate`, `OpenInterest`
- **Scripts** (`scripts/`) — all share `_common.py` for data arguments, so no two can disagree about the dataset:
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
  - `DashboardPage.tsx` (`/`) — Portfolio value, equity curve marked to live prices, open positions, price grid, recent signals and fills, model status. Contract sizes come from `/coins/cde-specs`, not a local table.
  - `TradingPage.tsx` (`/trading`) — Per-instrument chart with fill markers, range selector, external wallet holdings, per-instrument signals and fills.
  - `ResearchPage.tsx` (`/research`) — Edge calibration (expected net edge against realised, the comparison the model can be held to), per-instrument health with the reason attached, retrain history from `model_runs` joined to the promotion ledger, and the script runner.
  - `ModelPage.tsx` (`/model`) — What is live and why it was allowed to be: the promotion gates with measured values beside thresholds, provenance, the simulation distributions, the candidate ledger including rejections, and the kill switch. Forced promotions stay visibly forced.
- **Components** (in `src/components/`): `Sidebar`, `PriceCard`, `PriceChart`, `EquityChart`, `SignalsTable`, `PaperPositionsTable`, `PaperFillsTable`, `ModelStatusPanel`, `GateTable`, `StateBlock`. `SignalsTable` shows the forecast decomposition — net, price, carry, cost, edge-to-risk — because the classifier columns it used to show (`Mom`, `Trend`, `ML`, `AUC`) are all null now.
- **API layer** (`src/api/`): `client.ts` is the only place that talks HTTP — one base URL, one `ApiError` with a real message, one place the `X-API-Token` header is set, and a request timeout. The rest (`coinsApi`, `tradesApi`, `signalsApi`, `walletApi`, `paperApi`, `researchApi`, `modelApi`) are thin wrappers over it. Five separate copies of `fetchWithError` had already drifted in what they reported.
- **Polling** (`src/hooks/usePolling.ts`): pauses when the tab is hidden, reports failures instead of swallowing them, and distinguishes "loading" from "empty". Every page previously ran its own `setInterval` forever with `.catch(() => {})`, so a dead backend was indistinguishable from a quiet market.
- **State components** (`src/components/StateBlock.tsx`): `Panel`, `Spinner`, `Empty`, `ErrorBlock`, `Freshness`. An error banner sits *above* existing data rather than replacing it — a stale price is worth more than a blank panel, as long as it is visibly stale.
- **Types**: `src/types.ts` — shared TypeScript interfaces
- **Routing**: Custom history-based routing in `App.tsx` (no react-router)
- **Config**: `VITE_API_BASE_URL` env var (defaults to `http://localhost:8000`)

## Docker Compose Services

| Service    | Image/Build           | Port  | Depends On |
|------------|-----------------------|-------|------------|
| `db`       | `postgres:16`         | 5432  | —          |
| `backend`  | `backend/api/`        | 8000  | db         |
| `frontend` | `frontend/`           | 3000  | backend    |
| `trader`   | `backend/trader/`     | —     | db         |

Persistent volumes: `postgres_data`, `trader_data`, `trader_models`, `trader_logs`.

## Key Environment Variables

| Variable                    | Default / Required      | Used By   |
|-----------------------------|-------------------------|-----------|
| `COINBASE_API_KEY`          | Required for full flows | backend, trader |
| `COINBASE_API_SECRET`       | Required for full flows | backend, trader |
| `DATABASE_URL`              | postgres connection URI | backend, trader |
| `TRADER_DB_PATH`            | `/app/data/trading.db`  | trader    |
| `SIGNAL_THRESHOLD`          | `0.74`                  | trader    |
| `MIN_AUC`                   | `0.54`                  | trader    |
| `LEVERAGE`                  | `4`                     | trader    |
| `EXCLUDE_SYMBOLS`           | `BIP,DOP`               | trader    |
| `CYCLE_INTERVAL_SECONDS`    | `3600`                  | trader    |
| `INCREMENTAL_BACKFILL_HOURS`| `6`                     | trader    |
| `TRAIN_WINDOW_DAYS`         | `0` (all history)       | trader    |
| `RECENCY_HALF_LIFE_DAYS`    | unset (`Config`: `50`)  | trader    |
| `RETRAIN_EVERY_DAYS`        | `7`                     | trader    |
| `FEATURE_LOOKBACK_DAYS`     | `2190`                  | trader    |
| `VITE_API_BASE_URL`         | `http://localhost:8000` | frontend  |
| `LEDGER_WALLETS_JSON`       | `[]`                    | backend   |

## Tracked Coins

BTC, ETH, SOL, XRP, DOGE — each with spot products (`{COIN}-USD`) and CDE perpetual contracts (e.g., `BIP-20DEC30-CDE`, `ETP-20DEC30-CDE`, `SLP-20DEC30-CDE`, `XPP-20DEC30-CDE`, `DOP-20DEC30-CDE`).

## Coding Conventions

- **Python**: Type hints throughout, dataclasses for config, logging via `logging` module. Scripts are runnable as modules (`python -m scripts.<name>`). Use `os.getenv()` with sensible defaults for all config.
- **TypeScript/React**: Functional components with hooks, no class components. Custom CSS variables for theming. `recharts` for charting. Fetch-based API layer (no axios).
- **SQL**: SQLAlchemy ORM for Postgres tables, raw SQLite for trader pipeline data. Bi-temporal schema in trader storage.
- **Docker**: Multi-stage builds, volume mounts for development hot-reload, health checks on all services.

## Common Commands

```bash
# Start everything
docker compose up --build db backend frontend trader

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
- **One `decide()`.** `core/signal.py` holds every gate, threshold and sizing rule. Both the backtest and the live signal writer call it; do not add a second path. Three separate implementations is why the previous system's backtest and paper trading disagreed.
- The trader and API run in separate containers with duplicated ORM models (in `pg_writer.py`). Keep them in sync when changing database schema.
- The frontend has no router library — routing is handled manually via `window.history.pushState` in `App.tsx`. Add new pages by extending the `RoutePath` type and adding a case.
- All trader scripts support CLI args that override environment variables. Check `argparse` blocks at the bottom of each script for available options.
- `core/promotion.py` stages into `models/.staging/{version}/` and atomically renames into place. `live_orchestrator.py` decides *when* to evaluate a candidate; it never decides whether one is good.
- Feature engineering is declarative and universe-wide, not per-coin scripts. `core/features.py` emits nine mechanism groups; `core/profiles.py` selects which columns a coin's archetype uses. A profile can never ask for a column the builder does not emit — the templates are parameterised by coin.
- The research endpoints read from model artifacts, the promotion ledger and trade history. There is no separate research database.
- Tests under `backend/trader/tests/` are organised around the failure modes that produced fake edge before: lookahead (`test_backtest.py`), symbol-identity memorisation (`test_model.py`), leaked fold statistics (`test_cv_and_metrics.py`), the cost identity (`test_targets.py`), and blocked candidates reaching live (`test_promotion.py`). Mark anything over ~10s `@pytest.mark.slow`.
- **Deleted in the rebuild**, so do not reference them: `scripts/train_model.py`, `scripts/compute_features.py`, the five search scripts, `scripts/validate_robustness.py`, `scripts/prune_features.py`, `scripts/preflight_check.py`, `features/` entirely, `core/coin_profiles.py`, `core/labeling.py`, `core/meta_labeling.py`, `core/paper_profile_overrides.py`, `core/strategies/`, `core/cv_splitters.py`, `core/metrics_significance.py`, `core/study_significance.py`, `core/overfit_diagnostics.py`, `core/preprocessing_cv.py`, `core/reason_codes.py`, `core/run_manifest.py`, `core/trading_costs.py`, `core/execution_sim.py`.