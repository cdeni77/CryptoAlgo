# CLAUDE.md — Quarter

Trading on Kalshi 15-minute BTC/ETH/SOL up/down markets. FastAPI + PostgreSQL
serving store, React/Vite dashboard, Python research pipeline over one-minute
Coinbase bars and recorded venue books. `AGENTS.md` says where things are. This
file says what must stay true.

**Two companions are binding, not optional.**
`research/RESEARCH_PROTOCOL.md` is the anti-overfitting contract: the effective
sample is windows, not rows; all inference resamples windows; failures are
recorded alongside successes; negative controls are mandatory; and the ranking
criterion is generalization, not return. Read it before designing an
experiment. `DATA.md` is the source of truth for the data itself — what every
dataset is, its real coverage, its defects, what cannot be recovered, and the
traps that fail silently rather than raising.

## Established results

These are measured. Do not re-run them unless the data or the venue changes.

- **No forecast edge over the market price.** On point-in-time data the model
  beats `F(x/sigma)` but not the Kalshi price (`model_minus_market` ≈ 0).
- **Favourite-longshot bias exists.** `logit(p_true) = a·logit(mid)` fits
  `a ≈ 1.08` (stable across offsets, symbols and eras). It is smaller than
  taker costs (fee + half-spread ≈ 1.3–2.9c per contract).
- **Passive (maker) orders are adversely selected.** Simulated fills on the
  favourite side lose about 1.2c per contract at a zero maker fee.
- **Kalshi does not lag spot.** Cross-correlation with Kraken trades peaks at
  lag 0 for all three symbols.
- **Earlier backtest edges were lookahead.** See "Point in time" below.

## Clocks that are running

Data that is not captured before these expire is gone for good.

- `venue_book_events` — **60 days** (`core/spool.DEFAULT_RAW_RETENTION_DAYS`;
  compose passes `RAW_RETENTION_DAYS=60`). This file said 14 for months and the
  code never agreed — corrected 2026-10-17.
- `venue_ladder` is kept forever, but **stopped recording on 2026-09-16 and was
  dead for a month before anyone noticed**, while the raw frames it is built
  from stayed current. Recoverable by replaying the stream, until the 60 days
  above run out.
- The venue serves **no historical order book**: sizes and depth before
  2026-08-25 do not exist at any price. Prices, strikes and settlements DO go
  back — see `DATA.md`, "Enumeration was the wall, not the data".

## Invariants

### The target
- Strike = mean of the minute before the window opens. Settlement = mean of
  the minute before it closes. Up wins on `>=` (`strike_type` is
  `greater_or_equal`). A window's strike is the previous window's settlement.
- Kalshi settles on CF Benchmarks BRTI. Coinbase is a proxy that disagrees
  only near ties. Train and grade on the venue's own result
  (`--label-source venue`) wherever it exists.

### Point in time
- A decision at offset `m` uses bars up to the close of bar `m-1`.
- A book or fit is usable only if it was seen at or before
  `minute + decision_latency_seconds` (`core/quotes.decision_age`).
- `quote_age_seconds` means different things by source. `backfill` counts back
  from the minute. `live_touch` counts back from the decision instant. The
  minute-grid recorders (`live`, `live_ws`, Polymarket `live`) sample about
  29s after the minute, so their rows are never decision-time data.
- Implied-vol fits join on `available_time` (capture time), not `event_time`.

### Measurement
- Always measure against a benchmark: the market (`model_minus_market`) and
  `F(x/sigma)`. Never against 50%.
- Cross-validation splits on the window, never the row. Embargo one day.
  Standard errors come from fold dispersion, folds weighted by window count.
- Gates are frozen in `core/gate_spec.lock.json`. Change them only with
  `promote --accept-gate-change "<reason>"`. Every evaluate, promote, ablate
  and train run is a trial in `trials.jsonl`.
- A sealed holdout (`--holdout-start`) is read once (`--unseal-holdout`).

### Trading
- One `decide()` for backtest and live. `price_source` is on every row.
- Taker fee per order: `ceil(0.07·C·P·(1−P))` to a hundredth of a cent. Maker
  fee: `ceil(M·0.0175·C·P·(1−P))`, `M = 0` for these series. Settlement is free.
  Hold to settlement; there is no exit path.
- One entry per (symbol, window). The price band is symmetric, `[0.05, 0.95]`.
- Live, the venue is the account of record: balance, fills and settlement P&L
  come from Kalshi. A missing field is null, never zero.
- Markets are resolved by asking the venue, never by building a ticker.
- `--place-orders` is separate from `--mode live`. Orders need a client built
  with `live=True`.

## Operating the live system

- Production runs `scripts.run_live` in the `trader` service. Its trading flags
  live in `docker-compose.override.yml` on the production machine.
- Pin the traded model: `--require-model-version <VERSION>`. Any other model
  is refused `model_not_pinned`.
- A force-installed (gate-blocked) model trades only with
  `--no-require-gates --force --reason "<why>"`.
- Restart the trader only between +1 and +2 minutes into a window (decisions
  run at +3/+6/+9/+12). Use `docker compose up -d --force-recreate trader`.
- New models load at the next window boundary without a restart.
- The production research store is root-owned. Write to it through a one-off
  container (`docker compose run --rm ...`).
- `promote --go-live-check` compares a model's live dry-run record with its
  backtest. Read-only.

## Commands

```bash
cd backend/trader
python -m scripts.scrape --backfill-days 1825     # bars, resumable
python -m scripts.scrape --fill-gaps
python -m scripts.sync_store
python -m scripts.baseline                         # the null
python -m scripts.evaluate --help                  # walk-forward + gates
python -m scripts.promote --help                   # the only path to a live model
python -m scripts.promote --go-live-check --days 14
python -m scripts.live --mode live --dry-run       # real book, no orders
python -m scripts.check_venue                      # prove the Kalshi key, read-only
python -m research.collect.load_store --what depth --month YYYY-MM

pytest                       # backend/trader and backend/api; -n auto
cd frontend && npm run typecheck && npm run lint && npm run build
```

## Writing comments and docstrings

Use Simplified Technical English. Keep only what the code cannot say: an
invariant, a unit, a venue or API quirk, the reason a constant has its value,
a safety rule. Short sentences, active voice, present tense, one idea each.
Docstrings are one summary line, plus arguments and returns only when not
obvious. Never write history: no dates, no incident stories, no "used to", no
measurements that are not the reason for a constant. History lives in git.
