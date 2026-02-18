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
  - `research.py` — `/research/summary`, `/research/coins/{coin}`, `/research/runs`, `/research/features/{coin}`, `POST /research/launch/{job}`
- **Controllers**: Business logic in `controllers/` matching each endpoint module
- **External deps**: `coinbase-advanced-py` for Coinbase API integration

### Backend — Trader (`backend/trader/`)

- **Language**: Python 3.12
- **Core modules** (`core/`):
  - `coin_profiles.py` — Per-coin trading profiles (`CoinProfile` dataclass) with feature lists, thresholds, exit params, ML hyperparams. Profiles defined for BTC, ETH, SOL, XRP, DOGE. Each has base features + coin-specific extras (e.g., `BTC_EXTRA_FEATURES`, `SOL_EXTRA_FEATURES`, `DOGE_EXTRA_FEATURES`).
  - `pg_writer.py` — Postgres writer for trades, signals, paper-trading persistence. Duplicates ORM models for container isolation.
- **Data collection** (`data_collection/`):
  - `storage.py` — Abstract `DatabaseBase` + `SQLiteDatabase` implementation with bi-temporal schema
  - `models.py` — Data models: `OHLCVBar`, `FundingRate`, `OpenInterest`
- **Feature engineering** (`features/engineering.py`):
  - `SOLMomentumFeatures` — momentum acceleration, efficiency ratio, breakout strength, vol term structure, return autocorrelation, range expansion
  - `DOGESentimentFeatures` — FOMO/panic scores, pump-dump signal, extreme move frequency, vol asymmetry, VWAP distance, hype cycle
- **Scripts** (`scripts/`):
  - `run_pipeline.py` — OHLCV, funding rate, OI collection and backfill
  - `compute_features.py` — Feature generation into CSV artifacts
  - `train_model.py` — Backtesting + signal generation with ensemble training (3 lookback offsets), walk-forward validation
  - `live_orchestrator.py` — Scheduled cycle runner: pipeline → features → signals. Handles retrain scheduling, model staging/promotion, graceful shutdown.
  - `optimize.py` — Per-coin Optuna optimization with true holdout evaluation, deflated Sharpe tracking, TPE sampler
  - `parallel_launch.py` — Process-level multi-coin optimization launcher with integrated robustness validation
  - `validate_robustness.py` — Post-optimization robustness validation producing paper-trade readiness scores
- **ML stack**: LightGBM, scikit-learn, Optuna
- **Data**: SQLite for pipeline DB, CSV for feature artifacts, joblib for model artifacts

### Frontend (`frontend/`)

- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS 3.4 with CSS custom properties (dark theme with glass-card effects)
- **Pages** (in `src/pages/`):
  - `TradingTerminalPage.tsx` (`/`) — Spot/CDE price cards, price charting, trades table, signals table, wallet summary. Auto-refreshes prices (3s), trades (10s), signals (15s).
  - `StrategyLabPage.tsx` (`/strategy`) — Model health KPIs, coin strategy scoreboard, experiment timeline, explainability-lite (feature importance + signal distribution), paper trading tabs (positions, equity, performance, fills).
- **Components** (in `src/components/`): `PriceCard`, `PriceChart`, `TradesTable`, `SignalsTable`, `WalletInfo`, `PaperPositionsTable`, `PaperEquityTable`, `PaperPerformancePanel`, `PaperFillsTable`
- **API layer** (`src/api/`): `coinsApi.ts`, `tradesApi.ts`, `signalsApi.ts`, `paperApi.ts`, `researchApi.ts`
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
| `TRAIN_WINDOW_DAYS`         | `90`                    | trader    |
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

# One-off backtest
docker compose run --rm trader python -m scripts.train_model --backtest --threshold 0.80 --min-auc 0.54 --leverage 4

# Retrain models once
docker compose run --rm trader python -m scripts.live_orchestrator --retrain-only --run-once --train-window-days 90

# Parallel optimization
docker compose run --rm trader python -m scripts.parallel_launch --trials 200 --jobs 16 --coins BTC,ETH,SOL,XRP,DOGE

# Single-coin optimization
docker compose run --rm trader python -m scripts.optimize --coin BTC --trials 100 --jobs 1

# Frontend dev
cd frontend && npm ci && npm run dev

# API dev
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload
```

## Agent Instructions

- When modifying trader logic, be aware that `coin_profiles.py` is the single source of truth for per-coin feature lists, thresholds, and ML hyperparameters. Changes there cascade into training, optimization, and signal generation.
- The trader and API run in separate containers with duplicated ORM models (in `pg_writer.py`). Keep them in sync when changing database schema.
- The frontend has no router library — routing is handled manually via `window.history.pushState` in `App.tsx`. Add new pages by extending the `RoutePath` type and adding a case.
- All trader scripts support CLI args that override environment variables. Check `argparse` blocks at the bottom of each script for available options.
- The orchestrator (`live_orchestrator.py`) manages model versioning with a staging directory pattern — new models are trained into `.staging/{version}/` then atomically promoted to the models directory.
- Feature engineering is coin-specific. BTC uses mean-reversion extras (z-scores, RSI extremes), SOL uses momentum acceleration features, DOGE uses sentiment-proxy features. The base feature set is shared.
- The research endpoints read from model artifacts and trade history to produce health KPIs, not from a separate research database.
- Tests are minimal — `pytest` is listed in requirements but test coverage is sparse. New features should include tests where practical.