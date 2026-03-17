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
# All trader tests
cd backend/trader && pytest

# Single file
cd backend/trader && pytest tests/test_<name>.py -v
```

Tests live in `backend/trader/tests/` (28 files). No frontend tests exist.

## Key Commands

```bash
# Start full stack
docker compose up --build db backend frontend trader

# Frontend dev (hot-reload)
cd frontend && npm ci && npm run dev

# API dev (hot-reload)
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload

# Trader scripts (run from backend/trader/)
python -m scripts.run_pipeline
python -m scripts.compute_features
python -m scripts.train_model
python -m scripts.optimize --coin BTC --trials 100 --jobs 1
python -m scripts.parallel_launch --trials 200 --jobs 16 --coins BTC,ETH,SOL,XRP,DOGE

# Frontend checks
cd frontend && npm run typecheck && npm run lint
```

## Critical Architecture Notes

- **`coin_profiles.py` is the single source of truth** for per-coin feature lists, thresholds, and ML hyperparameters. Changes there cascade into training, optimization, and signal generation.
- **Duplicated ORM models**: `backend/trader/core/pg_writer.py` duplicates the API ORM models for container isolation. Keep both in sync when changing DB schema.
- **No react-router**: Frontend routing is manual via `window.history.pushState` in `App.tsx`. Add pages by extending `RoutePath` type and adding a case.
- **Model staging pattern**: `live_orchestrator.py` trains into `.staging/{version}/` then atomically promotes to the models directory.
- **All trader scripts** accept CLI args that override env vars — check `argparse` blocks at the bottom of each script.

## Coding Conventions

- **Python**: Type hints throughout, dataclasses for config, `logging` module, `os.getenv()` with defaults. Scripts run as `python -m scripts.<name>`.
- **TypeScript/React**: Functional components + hooks only, no class components. Fetch-based API layer (no axios). `recharts` for charts.
- **Tests**: `pytest` for trader. New features should include tests where practical.

## Environment Variables

See `AGENTS.md` for the full table. Minimum required for live workflows: `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `DATABASE_URL`.
