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

Tests live in `backend/trader/tests/` (9 files). No frontend tests exist.

## Key Commands

```bash
# Start full stack
docker compose up --build db backend frontend trader

# Frontend dev (hot-reload)
cd frontend && npm ci && npm run dev

# API dev (hot-reload)
cd backend/api && pip install -r requirements.txt && uvicorn app:app --reload

# Trader scripts (run from backend/trader/). Every script takes the same data
# arguments — see scripts/_common.py — so they cannot disagree about the dataset.
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
python -m scripts.live_orchestrator            # the loop that runs all of the above

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

So the horizon and the history length trade off directly. `scripts/preflight.py`
computes both numbers and states the two ways out — scrape roughly
`200 x horizon / 24` days, or shorten the horizon to about
`timestamps / 200` hours. **Run it before a long scrape**, not after.

```bash
python -m scripts.preflight                 # profile default horizon
python -m scripts.preflight --horizon 8     # what a shorter hold buys
```

## Critical Architecture Notes

- **`core/profiles.py` is the single source of truth** for per-coin feature sets, thresholds, and ML hyperparameters. Coins are described by a feature *archetype* (`mean_reversion`, `momentum_breakout`, `meme`, `trend_persistence`, `compression_breakout`) plus tuned deltas. Changes cascade into training, search, and signal generation.
- **`core/costs.py` is the single source of truth for money** — contract specs, exchange fee assumptions, round-trip costs, trade PnL, and position sizing. Load a venue's real schedule with `Config.with_cost_assumptions(find_cost_config())` — `configs/` lives under `backend/trader/` so the Docker build context includes it; the hardcoded defaults are 10bps/side, which is wrong for Coinbase CDE by 0.06x-2.5x depending on the contract.
- **Duplicated ORM models**: `backend/trader/core/pg_writer.py` duplicates the API ORM models for container isolation. `backend/trader/tests/test_orm_parity.py` fails when they diverge in columns, types, nullability, defaults, or migration lists — a note in a doc was not enough; `wallet.balance` had already drifted 10,000 against 100,000, so whichever container created the row decided the paper account's starting balance.
- **No react-router**: Frontend routing is manual via `window.history.pushState` in `App.tsx`. Add a page by adding an entry to `ROUTES` and a case in the render — the `RoutePath` type derives from `ROUTES`, so a missing case is a type error.
- **Frontend HTTP goes through `src/api/client.ts`**: one base URL, one error type, one place the `X-API-Token` header is set. Poll with `usePolling`, which pauses on hidden tabs and surfaces failures. Five copies of `fetchWithError` had already drifted in how they reported errors, and every `.catch(() => {})` made a dead backend look like a quiet market.
- **The API serves measurements, never substitutes.** A missing value is null with a reason. The research surface used to report `pr_auc` as `holdout_auc - 0.06`, `precision_at_threshold` as `holdout_auc - 0.04`, and — when the artifact it wanted was absent, which was always — a hardcoded table of six feature importances. All of it rendered identically to real data.
- **Promotion is the gate**: `core/promotion.py` stages into `models/.staging/{version}/`, then atomically renames into place — but only after every gate in `core/metrics.py:DEFAULT_GATES` passed. `--force` needs a reason and records it. `live_orchestrator.py` decides *when* to ask; it never decides the answer.
- **One `decide()`**: `core/signal.py` is the only place a trade is chosen. The backtest and the live signal writer both call it, which is why they cannot drift. The old per-family strategy classes under `core/strategies/` are deleted — they were orphaned by the reformulation.
- **All trader scripts** share `scripts/_common.py` for data arguments, and accept CLI args that override env vars.

## Coding Conventions

- **Python**: Type hints throughout, dataclasses for config, `logging` module, `os.getenv()` with defaults. Scripts run as `python -m scripts.<name>`.
- **TypeScript/React**: Functional components + hooks only, no class components. Fetch-based API layer (no axios). `recharts` for charts.
- **Tests**: `pytest` for trader. The suite's job is to catch the failure modes that produced fake edge before: lookahead (`test_backtest.py`), symbol-identity memorisation (`test_model.py`), leaked fold statistics (`test_cv_and_metrics.py`), the cost identity (`test_targets.py`), and blocked candidates reaching live (`test_promotion.py`). Mark anything over ~10s `@pytest.mark.slow`.

## Environment Variables

See `AGENTS.md` for the full table. Minimum required for live workflows: `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `DATABASE_URL`.
