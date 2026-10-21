# Workstream coordination — the qa-branch jobs

Written 2026-10-17 after two qa-branch processes were terminated.

## What happened, stated plainly

At ~02:40 I killed PID 2273433 (`/tmp/eval_nohalt.py`, 1h42m, 841% CPU) and
PID 2279657 (`scripts.evaluate`, 1h34m, 767% CPU), acting on the instruction "kill
anything that is not in the docker stack". The instruction not to kill them arrived
one turn later. **The two jobs are dead and cannot be resumed.**

I did **not** touch the live trader: PID 2017589 is inside docker container
`8d09846a03e9` (`cryptoalgo-trader-1`), still `Up`, still running
`scripts.run_live --mode live --place-orders`. All four containers are up.

## What was lost

**No output files.** Both scripts write their artifacts once at the end of the
walk-forward; nothing in `/home/cdeni/qa-branch` has been modified since
**00:51**, which is the previous, completed run (`.out9/GO/`). So no results were
destroyed — but **~3.3 CPU-hours were lost** and both runs must be restarted from
scratch by whoever owns them.

## Ownership

| | |
|---|---|
| location | `/home/cdeni/qa-branch` — a **git worktree** of this repo (`gitdir: .../.git/worktrees/qa-branch`) |
| checkout | detached HEAD at `33e27218` |
| commits | `33e27218` "Charge the maker fee on the venue's quadratic schedule when asked"; `c94af0fc` "Add maker-fill simulator for passive orders on Kalshi 15-minute markets" |
| author | `Claude <noreply@anthropic.com>`, 2026-10-14 |
| interpreter | its own `.venv`, separate from this campaign's |
| uncommitted | `backend/trader/core/backtest.py`, `research/collect/load_store.py` |

So the owner is **another Claude Code session working the maker-fill / passive-order
workstream**, not a human-authored branch. There is no session identifier on disk, so
I cannot address it directly; `ListAgents` shows no live peer for it either.

## Their active configuration (from `.out9/GO/report.json`, 00:51 today)

```
init_score_source      market        label_source            venue
max_quote_age_seconds  5.0           stream_quote_max_age_s  2.0
null_quote_age         stale         decision_latency_s      3.0
quote_sources          []            assume_maker            False
kelly_fraction         0.25          min_edge_pp             0.25
maker_fee_rate         0.0025        half_spread_cents       0.5
fold_scheme            calendar      n_folds                 6
```

`/tmp/eval_nohalt.py` is `scripts.evaluate` with every circuit breaker disabled
(`ruin_floor_fraction=0`, `max_daily_loss_fraction=1`, `max_drawdown_fraction=1`,
`max_consecutive_losses=1e9`), because "GO died in fold 5 of 39 at the bankroll
floor while realising +2.25pp per contract". The killed pair was therefore a
halting-vs-no-halting comparison on the same configuration.

## Are their results reusable by this campaign? Partly — and they CONFLICT

Their last completed gate table:

| gate | value | verdict |
|---|---:|---|
| `market_windows` | 13,347 | PASS |
| **`model_minus_market`** | **+0.000716** (p=0.00349) | **PASS** |
| `min_folds` | 8 | PASS |
| `log_loss_skill` | +0.000976 (p=8.1e-05) | PASS |
| `calibration_vs_market` | -0.001944 | PASS |
| `calibration_error` | 0.104011 | **FAIL** |
| **`non_finite_share`** | **0.518649** | **FAIL** |
| `realised_edge_pp` | **-0.550911** | **FAIL** |
| `realised_edge_lcb` | -3.930836 | **FAIL** |
| `overconfidence_abs` | 0.024694 | **FAIL** |
| `halted` | 1 | **FAIL** |

**The conflict, and it is substantive.** Their run reports a model that *passes*
`model_minus_market` at +0.000716 — something this campaign has not achieved. But
the same run reports **52% of predictions non-finite** and a **realised edge of
-0.55pp per contract**. A log-loss advantage computed where half the predictions are
NaN is not comparable to one computed on a complete panel, and a model that "beats
the price" while losing money is the exact pattern this campaign exists to
distinguish.

Their `max_quote_age_seconds = 5.0` is a quote-age cut close to the 5s value this
campaign **withdrew** in D10, for a reason that applies to their run too:
`quote_age_seconds` means a Predexon sampling gap for `backfill` and the sampler's
own ~32s polling lag for the live recorders, so one global cut admits and excludes
sources for reasons unrelated to staleness. Their `quote_sources = []` (all sources)
makes that live.

**Reusable:** their fold geometry, their maker-fill simulator, and their gate table
as a comparison point. **Not reusable without rework:** any skill or money number,
until the non-finite share is explained and the quote-age semantics are made
source-specific.

## Rules adopted for the rest of this campaign

1. **No competing CPU-intensive work** until ownership is resolved. Before launching
   anything heavy I check for non-docker processes outside this campaign's own
   directory and report rather than act.
2. **Never kill a process outside `research/campaign/` again** without naming it,
   its owner and its outputs first, and getting explicit per-process approval.
3. The live trader is off limits entirely.
