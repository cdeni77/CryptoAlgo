# Quarter

Barrier-probability trading on Kalshi 15-minute BTC/ETH/SOL up-down markets.

A window opens on a quarter-hour boundary, records the price there as its
strike, and settles on whether the price at the next boundary is above it. By the
time you are deciding, part of the window has already happened — so the question
is not which way the market will go, but:

> given that price has already moved `x` from the strike, and `n` minutes of
> movement remain, what is the chance it finishes above?

That is a barrier crossing. The displacement is known exactly; the only forecast
needed is the volatility over the minutes that remain.

**The benchmark is `F(x/sigma)`, not 50%.** The barrier arithmetic alone takes log
loss 26% below a coin flip, with no features and no model — so a model measured
against 50% would report a large edge that is entirely a clock. Everything here
is measured as a difference against that baseline.

## Layout

```
backend/trader/     the pipeline: scrape, features, baseline, model, live
backend/api/        read-only telemetry over the serving store
frontend/           the dashboard
```

## Quick start

```bash
cp .env.example .env      # set POSTGRES_PASSWORD at minimum
docker compose up --build db backend frontend

cd backend/trader
python -m scripts.scrape --backfill-days 1825   # a few hours; resumable
python -m scripts.sync_store
python -m scripts.baseline                      # read the null first
python -m scripts.evaluate                      # skill, gates, cost stress
```

## Documentation

- **`CLAUDE.md`** — the reframe, the economics, the invariants, and what has
  already been rejected so it is not re-run.
- **`AGENTS.md`** — architecture, environment variables, conventions.

## Status

This section described an early phase of the project — no scraped data, no
trained model — and was never updated as the project moved past it. Rather than
pin another number here that will drift (the test count alone has been wrong
three times across the other docs), the honest state now lives in one place:
**`CLAUDE.md`'s "The honest state of things"** section, updated as it changes.
As of this fix: five years of bars for all three symbols, a model promoted and
trading live since 2026-08-23, and the edge still treated as a hypothesis under
continuous measurement rather than something one promotion settled — the phase
gates exist for exactly that reason and `scripts/evaluate.py` failing remains
the expected outcome until proven otherwise on the current artifact.
