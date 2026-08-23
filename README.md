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

The plumbing is complete and tested (207 tests). The **edge is unestablished**:
there is no scraped data, no trained model, and no measured skill yet. The phase
gates exist because that is a hypothesis, and `scripts/evaluate.py` failing is
the expected outcome until proven otherwise.
