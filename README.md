# CryptoAlgo

CryptoAlgo is a full-stack crypto trading research and monitoring workspace with:

- a **FastAPI backend** for prices, trades, signals, wallet, and paper-trading telemetry,
- a **React + Vite frontend** dashboard,
- a **trader service** for data collection, feature engineering, signal generation, retraining, and optimization.

It is designed for Coinbase CDE/perpetual workflows and includes both live orchestration and offline backtesting/optimization paths.

---

## Repository Structure

```text
backend/
  api/        FastAPI service + PostgreSQL models/endpoints
  trader/     Data pipeline, feature engineering, training, optimization, orchestrator
frontend/     React/Vite dashboard
docker-compose.yml
```

Core trader scripts:

- `run_pipeline.py` — OHLCV/funding/OI collection and backfill
- `compute_features.py` — feature generation into CSV artifacts
- `train_model.py` — backtesting + signal generation
- `live_orchestrator.py` — scheduled cycle runner (pipeline → features → signals)
- `optimize.py` — single-coin Optuna optimization with true holdout evaluation
- `parallel_launch.py` — process-level multi-coin optimization launcher

---

## High-Level Architecture

```text
Exchanges / Coinbase
        │
        ▼
backend/trader/run_pipeline.py
        │
        ▼
backend/trader/compute_features.py
        │
        ├────────────► backend/trader/train_model.py --signals
        │                  │
        │                  ├─ model artifacts (joblib)
        │                  └─ writes trades/signals/paper telemetry
        │
        └────────────► backend/trader/optimize.py (offline tuning)

backend/api (FastAPI + Postgres) ◄──────── frontend (React dashboard)
```

---

## Services (Docker Compose)

`docker-compose.yml` defines:

- **db**: PostgreSQL 16
- **backend**: FastAPI (`/coins`, `/trades`, `/signals`, `/wallet`, `/paper`)
- **frontend**: Nginx-served Vite build
- **trader**: orchestrated ML/data pipeline service

Default local ports:

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Postgres: `localhost:5432`

---

## Quick Start

### 1) Prerequisites

- Docker + Docker Compose
- Coinbase API credentials (optional for some public reads, required for full workflows)

### 2) Configure environment

At minimum, export or place in shell env:

```bash
export COINBASE_API_KEY="..."
export COINBASE_API_SECRET="..."
```

### 3) Start the stack

```bash
docker compose up --build db backend frontend trader
```

This starts:

- API + frontend,
- trader orchestrator running periodic cycles,
- persisted volumes for trader DB/models/logs and Postgres.

---

## Common Workflows

### Live/trader cycle (default service behavior)

The orchestrator runs:

1. `run_pipeline.py` (backfill/incremental market data)
2. `compute_features.py`
3. `train_model.py --signals`

with scheduled repetition controlled by env/CLI options in `live_orchestrator.py`.

### One-off backtest

```bash
docker compose run --rm trader \
  python train_model.py --backtest --threshold 0.80 --min-auc 0.54 --leverage 4
```

### Retrain-only run once

```bash
docker compose run --rm trader \
  live_orchestrator.py --retrain-only --run-once --train-window-days 90
```

### Parallel Optuna launch

```bash
docker compose run --rm trader \
  python parallel_launch.py --trials 200 --jobs 16 --coins BTC,ETH,SOL,XRP,DOGE
```

Current optimization defaults are tuned for stronger robustness:

- `holdout-days=120`
- `plateau-patience=100`
- `plateau-min-delta=0.02`
- `plateau-warmup=60`

### Direct single-coin optimization

```bash
docker compose run --rm trader \
  python optimize.py --coin BTC --trials 100 --jobs 1
```

---

## API Surface

Base URL: `http://localhost:8000`

- `GET /` — API health message
- `GET /coins/prices` — current tracked prices
- `GET /coins/cde-specs` — contract metadata for tracked symbols
- `GET /coins/history/{symbol}` — historical spot OHLCV
- `GET /trades/*` — trade history endpoints
- `GET /signals/*` — signal endpoints
- `GET /wallet/` — wallet summary exposing paper trading wallet + Coinbase spot/perps totals and coin/position breakdowns
- `GET /paper/*` — paper orders/fills/positions/equity telemetry

Note: legacy `/ops` endpoints were intentionally removed; operational control is CLI/orchestrator-driven.

---

## Data & Artifacts

Trader service stores persistent artifacts via Docker volumes:

- `/app/data` — SQLite pipeline DB + exported features
- `/app/models` — trained model artifacts
- `/app/logs` — orchestrator/runtime logs

Postgres service stores API-facing trade/signal/wallet/paper tables.

---

## Key Configuration Knobs

### Trader / orchestration

- `CYCLE_INTERVAL_SECONDS` (default 3600)
- `INCREMENTAL_BACKFILL_HOURS` (default 6)
- `TRAIN_WINDOW_DAYS`
- `RETRAIN_EVERY_DAYS`
- `SIGNAL_THRESHOLD`
- `MIN_AUC`
- `LEVERAGE`
- `EXCLUDE_SYMBOLS`

### Feature generation

- `FEATURE_LOOKBACK_DAYS` (default `2190`) to cap historical span used by `compute_features.py`.

---

## Frontend Notes

The dashboard shows:

- spot/CDE price cards,
- charting and market history,
- trades/signals tables,
- paper positions/equity/performance views,
- wallet summary.

Operations controls were removed from the frontend to match backend endpoint removal.

---

## Local Development Without Docker

You can run each layer directly, but Docker Compose is the canonical path.

- Frontend: `npm ci && npm run dev` in `frontend/`
- API: install `backend/api/requirements.txt`, run `uvicorn app:app`
- Trader: install `backend/trader/requirements.txt`, run scripts in `backend/trader/`

Make sure DB paths and `DATABASE_URL` are aligned with your local setup.

---

## Troubleshooting

### Not enough data / missing symbols in training

- ensure `run_pipeline.py` has completed backfill,
- verify feature CSVs exist under trader data dir,
- check symbol mappings and available exchange history windows.

### API up but frontend missing data

- verify backend reachable from frontend (`VITE_API_BASE_URL`),
- confirm Postgres health and API DB connection,
- inspect API logs for Coinbase request errors.

### Optimization appears to start from different years per coin

This is expected when historical availability differs by symbol; holdout slicing is relative to each run’s global end, while earlier start dates depend on available historical bars.

---

## Disclaimer

This repository is for research and engineering workflows, not financial advice. Crypto derivatives trading carries substantial risk.
