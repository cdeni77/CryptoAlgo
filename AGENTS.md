# AGENTS.md — Quarter

Where things are and how to work here. `CLAUDE.md` holds the invariants and the
established results. Read it before changing anything under `core/` or
`livetrade/`.

## Repository layout

```
backend/trader/
  core/
    config.py        one frozen dataclass; every field that changes an answer
    windows.py       the 15-minute grid: symbol x window x offset
    vol.py           HAR on log realised vol, minute-of-day seasonality
    baseline.py      the null F(x/sigma): scale and tail fitted, no drift
    features.py      feature groups, the clock control, opt-in book groups
    dataset.py       Dataset, FoldFit, ScoringBundle, score_live
    quotes.py        point-in-time quote join (decision_age)
    book_features.py book, cross-venue and implied-vol features
    book_flow.py     book dynamics (opt-in group)
    index_proxy.py   multi-exchange index proxy (pure)
    index_features.py index_proxy feature group
    cv.py            window-level purged expanding folds, 1-day embargo
    model.py         LightGBM correction on an init logit; ensemble; stack mode
    costs.py         Kalshi taker and maker fee schedules
    decide.py        the only place a trade is chosen
    book.py          the simulated account
    backtest.py      walk-forward, cost stress
    metrics.py       skill, calibration, the gate spec
    promotion.py     staging, gates, ledger, trial log, go-live check
    datastore.py     Parquet + DuckDB research store
    pg_writer.py     the serving store (mirrored in backend/api/models/serving.py)
    stream_book.py   Kalshi WebSocket book cache
    pm_stream_book.py Polymarket book cache
    trade_bars.py    one-minute bars from trades
    spool.py         frame spool, compaction, retention, disk guard
  livetrade/         the live loop's parts: artifact, bars, quotes, features,
                     risk, execution, ledger, state
  data_collection/   Coinbase and Kalshi clients, minute-bar storage
  scripts/
    live.py          CLI, scheduling, run_cycle
    run_live.py      the trader process: live loop plus recorders, supervised
    record_*.py      Kalshi stream, Polymarket stream and ladder, Kalshi ladder,
                     implied vol, Coinbase trades, index proxy
    scrape.py sync_store.py build_depth.py collect_settlements.py correct_outcomes.py
    baseline.py train.py evaluate.py promote.py ablate.py
    check_venue.py sync_venue.py reset_dashboard.py healthcheck.py
  research/
    collect/         Predexon backfill (catalog, collect, consolidate, load_store)
    validate/        checks of recorded data against independent sources
    analysis/        lead_lag.py
  tests/

backend/api/         FastAPI, read-only serving endpoints
frontend/src/        React + Vite dashboard
ops/                 weekly_retrain.sh
```

## Data

Research store (`RESEARCH_STORE`, Parquet + DuckDB), main tables:

| table | contents |
|---|---|
| `minute_bars` | Coinbase one-minute bars, BTC/ETH/SOL |
| `venue_depth` | Kalshi and Polymarket book per minute; `source` is `backfill`, `live`, `live_ws` or `live_touch` |
| `venue_book_events` | raw Kalshi WebSocket frames (retention `RAW_RETENTION_DAYS`) |
| `venue_implied_vol` | strike-ladder volatility fits |
| `venue_settlements` | the venues' own results |
| `spot_trades` | raw Coinbase trades |
| `index_proxy_1s` | per-second multi-exchange index proxy |

Serving store (PostgreSQL): `predictions` (every scored row, traded or
refused), `positions`, `order_tickets`, `account`, `equity_curve`,
`model_runs`, `calibration`, `minute_prices`, and the venue ledger tables.
`backend/api/models/serving.py` mirrors `core/pg_writer.py`;
`tests/test_orm_parity.py` keeps them in step.

## Docker Compose services

| service | what it does |
|---|---|
| `db` | PostgreSQL. Needs `POSTGRES_PASSWORD`. |
| `backend` | FastAPI. |
| `frontend` | the dashboard. |
| `trader` | `scripts.run_live`: live loop plus recorders. Production flags go in `docker-compose.override.yml`. |
| `scrape`, `evaluate`, `retrain` (profile `tools`) | one-off jobs. `retrain` runs `scripts.promote`. |

## Environment variables

| variable | used by |
|---|---|
| `DATABASE_URL`, `POSTGRES_PASSWORD` | api, trader, db |
| `COINBASE_API_KEY`, `COINBASE_API_SECRET` | scrape, live |
| `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY` or `KALSHI_PRIVATE_KEY_PATH` | live |
| `KALSHI_BASE_URL`, `KALSHI_HISTORICAL_BASE_URL`, `KALSHI_SERIES_BTC/ETH/SOL` | live |
| `RESEARCH_STORE`, `MODELS_ROOT`, `FEE_CONFIG` | research, promote, live |
| `QUOTE_MAX_AGE_SECONDS` | quote staleness cap (default 30) |
| `RAW_RETENTION_DAYS`, `INDEX_PROXY_RETENTION_DAYS`, `DISK_GUARD_MIN_FREE_FRACTION`, `DISK_GUARD_MIN_FREE_GB` | trader store sync |
| `INDEX_PROXY_EXCHANGES`, `LIVE_COMPONENTS` | trader |
| `PREDEXON_API_KEY`, `COLLECT_DATA` | research/collect |
| `CORS_ALLOW_ORIGINS` | api |

## Conventions

- Python: type hints, frozen dataclasses for config, `logging`. Scripts run
  as `python -m scripts.<name>`.
- TypeScript/React: function components and hooks, fetch-based API layer,
  `recharts`.
- Tests: `pytest`, parallel by default. Pair every capability test with a
  null. Mark tests over ~10s `@pytest.mark.slow`.
- Comments and docstrings: Simplified Technical English, no history (see
  `CLAUDE.md`).

## Agent instructions

1. Read `CLAUDE.md` before changing `core/` or `livetrade/`.
2. Measure skill against the market and against `F(x/sigma)`, never 50%.
3. Keep every input point-in-time. Use `core/quotes.decision_age`.
4. Split cross-validation on the window, never the row.
5. Do not change a gate without `--accept-gate-change` and a reason.
6. Live, trust the venue's ledger over our own bookkeeping.
7. When a claim in these docs can be measured, measure it before relying on it.
