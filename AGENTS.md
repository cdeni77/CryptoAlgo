# AGENTS.md — Quarter

Architecture and conventions. `CLAUDE.md` carries the reasoning — the reframe,
the economics, the invariants and what has already been rejected. Read that
first; this file says where things are.

## Project Overview

Barrier-probability trading on Kalshi 15-minute BTC/ETH/SOL up-down markets. A
window opens on a quarter-hour boundary, records the price there as its strike,
and settles on whether the price at the next boundary is strictly above it. The
displacement from the strike is known exactly; only the volatility over the
minutes that remain has to be forecast.

**The benchmark is `F(x / sigma_n)`, never 50%.** Everything is measured
incrementally against that baseline. See `CLAUDE.md`.

## Repository Structure

```
backend/trader/
  core/
    config.py      one frozen dataclass; every field that changes an answer
    windows.py     the 15-min grid as a table: symbol x window x offset
    vol.py         HAR on log realised vol + circular-smoothed minute-of-day
    baseline.py    the null: F(x/sigma), scale and tail fitted, drift never
    features.py    4 mechanism groups + 1 control, 42 columns
    dataset.py     Dataset (unfitted) / FoldFit / ScoringBundle / score_live
    cv.py          window-level purged expanding folds, 1-day embargo
    model.py       LightGBM on the baseline logit as init_score
    costs.py       Kalshi fees: ceil(0.07*C*p*(1-p)) per order, free settlement
    decide.py      the ONLY place a trade is chosen
    book.py        the account: enter, settle, equity at cost
    backtest.py    walk-forward, cost stress, the edge curve
    metrics.py     log-loss skill, calibration, and the 14 gates
    promotion.py   stage, gate, atomic rename, append to the ledger
    datastore.py   Parquet + DuckDB research store, point-in-time reads
    pg_writer.py   the serving store (mirrored in backend/api/models/serving.py)
  data_collection/
    coinbase_connector.py  REST + websocket, one-minute candles
    kalshi_client.py       RSA-PSS auth, quotes, balance, orders, fills
    pipeline.py storage.py queue.py validator.py models.py ingest.py timeutil.py
  scripts/
    _common.py     shared data arguments; no two scripts disagree about data
    scrape.py      one-minute Coinbase spot bars -> SQLite
    sync_store.py  SQLite -> Parquet
    baseline.py    fit and report the null (the Phase 1 gate)
    train.py       one model, for inspection. Not for promotion.
    evaluate.py    walk-forward: skill, funnel, money, stress, gates
    promote.py     the only path to models/forecast.joblib
    live.py        score, price, decide, act. paper | live --dry-run | live
  tests/           207 tests, 26s in parallel
  configs/venue/   fee schedules (optional; defaults match the published one)

backend/api/
  app.py           lifespan bootstrap, advisory-locked on Postgres
  models/serving.py    GENERATED from core/pg_writer.py — do not hand-edit
  controllers/serving.py
  endpoints/serving.py (read-only)  endpoints/jobs.py (the one mutating route)
  security.py      require_token (fails closed) + validate_job_args

frontend/src/
  App.tsx          manual routing; ROUTES + PAGES as a Record, so a route with
                   no component is a tsc error rather than a blank screen
  api/client.ts    one base URL, one error type, one place the token is set
  api/serving.ts   every request the app makes
  components/      Logo, Rail, QuarterTrack, ProbabilityScale, WindowChart,
                   Charts, Primitives
  pages/           Live, Decisions, Calibration, Model, Account
  lib/format.ts    formatters (not components — the fast-refresh rule is right)
```

## The serving store

Six tables plus two for the live path. A binary needs almost none of what a
perpetual future needed — no mark price, no unrealised PnL, no funding accrual,
no stop levels, no leverage.

| table | what |
|---|---|
| `predictions` | one row per (symbol, window, offset), **traded or refused** |
| `positions` | one contract purchase, held to settlement |
| `account` | bankroll, and `mode` — 'paper' or 'live' |
| `equity_curve` | equity at each settlement, open stake at **cost** |
| `model_runs` | every promotion attempt, blocked ones included |
| `calibration` | the reliability table, model and baseline |
| `minute_prices` | a rolling window for the chart (research store stays authoritative) |
| `order_tickets` | a live decision, and what came back |

Storing the refusals is the point: a dashboard showing only trades cannot show
that the system declined 99% of windows because the forecast did not cover the
fee, which is the most informative thing it has to say.

**`backend/api/models/serving.py` is generated from `core/pg_writer.py`.** Change
the trader's module, copy the shared block across, then run
`backend/trader/tests/test_orm_parity.py`. That test compares every column,
type, nullability and default, plus the migration lists — because "keep both in
sync" in a doc is a hope, and it had already failed once by a factor of ten.

## Docker Compose Services

| service | what it does |
|---|---|
| `db` | PostgreSQL. `POSTGRES_PASSWORD` is required; compose refuses without it. |
| `backend` | FastAPI, `uvicorn --workers 4`. Schema bootstrap is in the lifespan hook behind an advisory lock, not at import — four workers racing `create_all` killed the losers before FastAPI existed to log it. |
| `frontend` | Vite dev server or the built bundle. |
| `trader` | `scripts.live --loop`. Defaults to paper mode. |

`--backfill-days` in compose is the **first cycle only**. On an empty store it
*defines* the dataset; every cycle after uses the incremental window.

## Key Environment Variables

| variable | used by | notes |
|---|---|---|
| `COINBASE_API_KEY` / `_SECRET` | scrape, live | one-minute candles |
| `KALSHI_KEY_ID` | live | required for `--mode live` |
| `KALSHI_PRIVATE_KEY` | live | the PEM itself |
| `KALSHI_PRIVATE_KEY_PATH` | live | or a file holding it |
| `KALSHI_BASE_URL` | live | defaults to production; a demo host exists |
| `KALSHI_SERIES_BTC/ETH/SOL` | live | series tickers, resolved to markets by close time |
| `DATABASE_URL` | api, trader | PostgreSQL |
| `POSTGRES_PASSWORD` | db | compose refuses to start without it |
| `RESEARCH_STORE` | all research | Parquet root; absolute, not cwd-relative |
| `MODELS_ROOT` | promote, live | where `forecast.joblib` lives |
| `FEE_CONFIG` | all research | optional; defaults are the published schedule |
| `API_TOKEN` | api | mutating routes fail **closed** without it |
| `VITE_API_TOKEN` | frontend | must match `API_TOKEN` |
| `CORS_ALLOW_ORIGINS` | api | `*` is filtered out |
| `TRADER_DIR` | api | where `endpoints/jobs.py` launches scripts |

## Traded Universe

`BTC-USD`, `ETH-USD`, `SOL-USD` — Coinbase spot product ids, used verbatim from
the scrape command through the research store to the serving database. **There is
no translation step**, and `tests/test_symbols.py` pins that absence: the
previous system stored `BTC-PERP` while the reader asked for `BIP`, and every
lookup missed.

`BTC-USD` is the `cross_asset` reference. Dropping it from the universe weakens
that group to a peer mean rather than erroring.

## Coding Conventions

- **Python**: type hints throughout, frozen dataclasses for config, `logging`,
  `os.getenv` with defaults. Scripts run as `python -m scripts.<name>`.
- **TypeScript/React**: function components and hooks only. Fetch-based API layer,
  no axios. `recharts` for charts. Formatters live outside component modules.
- **Design**: an instrument panel, not a trading terminal. Two poles (`above`
  teal / `below` clay) are the *only* colours that carry direction; `accent` is
  neither, so structure cannot read as a signal; gate semantics are a third set.
  No gradients, no glow, no shadow depth, no hover lift, no emoji, 2px corners,
  tabular numerals everywhere. Both themes complete at token level.
- **Tests**: `pytest`. The suite's job is to catch a pipeline that *manufactures*
  edge, so every capability test is paired with a null. Mark anything over ~10s
  `@pytest.mark.slow`.

## Agent Instructions

1. **Read `CLAUDE.md` before changing anything under `core/`.** The invariants
   there are each the fix to a specific bug that cost this project real time.
2. **Never measure against 50%.** Skill is the difference against the baseline.
3. **Never split cross-validation on the row.** Four offsets share a settlement.
4. **Never mark an open position to the model's own forecast.** That books belief
   as profit.
5. **Never let a blocked candidate reach `models/forecast.joblib`.** Promotion is
   the only path, and `--force` needs a written reason.
6. **When a claim in these docs is measurable, measure it.** Three numbers in the
   previous version of this file were wrong (the fee crossover, the scale
   interpretation, a forward-looking target that looked backwards) and each was
   found by a test rather than by review.
