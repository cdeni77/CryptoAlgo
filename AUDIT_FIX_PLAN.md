# AUDIT_FIX_PLAN

Ordered, actionable tasks from `AUDIT_REPORT.md`. Each carries severity, files,
expected behaviour, the change, required tests, and dependencies.

## Status, 2026-08-24 (after 24 hours of real orders)

**20 commits, 422 tests, tree clean. Live on a $100 account since 2026-08-23
15:18 UTC.** The Addendum in `AUDIT_REPORT.md` has the detail; this is what it
means for the queue.

Four defects that neither static review nor the backtest found, all four in the
code that *verifies* rather than the code that trades, all four silent:

1. **[CRITICAL] V1 order endpoint retired** (`410`). V2 quotes one book from the
   YES side, so a DOWN order converts to `1 - price` rather than relabelling.
   Fixed; three mutants die where the V1 side-flip left 230 tests green.
2. **[CRITICAL] `position_fp`, not `position`.** `venue_open` was the empty set on
   every cycle, so the position cross-check was dead in *both* directions —
   including the reverse one this plan and the report both claimed had been added.
   Fixed.
3. **[HIGH] Every settlement credited twice for one cycle.** The venue credits on
   settle, and we adopted its balance before crediting our own. Bankroll
   self-healed; Kelly sized off the inflated figure and the drift alarm cried wolf
   on every win. Fixed by ordering.
4. **[HIGH] `client_order_id` enforced one *attempt* per window.** A killed
   fill-or-kill consumes the id, so the first thin-volume kill locked the window.
   57 of 69 refusals were our own key, at a *higher* claimed edge (8.37pp) than
   the fills (5.77pp). Fixed by keying on the offset too.

**Closed from P1: the market benchmark is a gate.** `market_windows` and
`model_minus_market` are read before `log_loss_skill`, because the arithmetic null
is not the counterparty. This mattered immediately: over 1,109 rows the market's
log loss is **0.331** against the model's **0.430**, on every symbol and every
offset, and the model is indistinguishable from its own baseline. A candidate like
that passed all fourteen previous gates.

**The money does not settle it either way.** $100 → $165.23, +$65.67 realised,
venue-confirmed. A bootstrap under "the market's de-spread mid is the true
probability", using each trade's real price, size and fee, puts expected P&L at
**−$23** and makes +$63 a **2–4%** event (ρ 0 → 0.9). So this is mild evidence
*for* an edge, and it is one day and 88 windows.

### New queue, in priority order

1. **[P1] `init_score` should be the market's logit, not the baseline's.** The
   model is fitted to correct the forecaster that loses by 0.10 nats. Blocked on
   enough recorded quote history to fit against; unblocks itself as (2) runs.
2. **[P1] Accumulate 2,000 windows of quotes, then re-read `model_minus_market`.**
   ~a week at 96 windows/day across three symbols. If it is still negative the
   strategy is falsified on its own terms and the honest move is back to research.
   **Keep placing real orders while this runs** — `--dry-run` records the quotes
   but assumes every intended order fills, and 30% do not, so the fill selection
   would go unmeasured.
3. **[P2] Audit every remaining `_fp` / `_dollars` field.** The same trap has now
   been found in quotes, fills and positions. `settlements.revenue` is safe only
   because it is used as a sign test and never as a magnitude — safety that stops
   holding the moment someone reads the field for its value.
4. **[P2] A `killed` ticket blocks the window and probably should not.** A
   confirmed zero fill bought nothing, so the window could reopen the way
   `skipped` does. Conservative as it stands; worth measuring how often it costs
   an entry before changing it.

---

## Status, 2026-08-23 (updated after running the pipeline)

**14 commits on `audit/critical-fixes`, 397 tests (385 fast + 12 slow), tree clean.**

The pipeline has now been run end to end on a 326-day subset, which found four more
bugs that static review could not: `--fill-gaps` fetched nothing at all, and three
of my *own* fixes were over-corrections that a real venue outage exposed (a
`non_finite_rows == 0` gate, a shrinkage guard that raised on any NaN and killed a
whole evaluation, and log-loss computed over NaNs in two places). All corrected to
rates with the rows excluded and counted.

### What the run says

* **Phase 1 passes.** Pooled out-of-sample log loss 0.44948 against 0.69315 —
  35.2% better from arithmetic alone, where the docs claim 26%.
* **`evaluate` passes 15 of 16 gates**, failing only `max_drawdown` (58.4%). Do not
  read that as encouraging: three of the passes are within a hair of their
  thresholds, `residual_scale` is 1.046 (above 1, so it wants to *amplify* the
  correction), and predicted edge +2.00pp realised +0.99pp.
* **The skill is real and it is not what the design says.** It is not the model
  recalibrating the baseline (a free in-fold recalibration of the null gains
  nothing out of sample), not the clock (`clock` alone scores -0.000002 on 1 of 6
  folds — the 28% gain share was a false alarm and the gate measures the wrong
  quantity), and not volatility (`vol_state` alone is negative). It is
  **cross-asset lead-lag concentrated at offset 3**, which contradicts the
  mid-window prediction `CLAUDE.md` and `evaluate.py` both stated. Both corrected.
* **The money numbers cannot choose a configuration.** At essentially constant
  skill, return spanned -50% to +303% across six configurations, and a
  *higher*-skill setting lost half the account.

### Still open, in the order it matters

1. **Rotate the leaked credentials.** Only you can do this. `b70c78c` and
   `6097ed1` are on `origin/main`.
2. **Finish the store repair, then re-run on five years.** `--fill-gaps` is running:
   BTC recovered +8,672 minutes, ETH is nearly done, SOL is queued. Everything
   above is 326 days of one regime with a flat seasonality factor, and thirteen
   exploratory trials with no multiple-testing correction. The five-year run is the
   test.
3. **Collect Kalshi quote history.** `scripts/market_benchmark.py` now exists and
   prints "No settled window has both a recorded market quote and an outcome",
   which is the honest state. Run `scripts.live --mode live --dry-run --loop` for
   weeks; it reads the real book, places nothing, and records a row per window.
   **Until that has thousands of windows, no number in this repository speaks to
   profitability.**
4. **Decide the drawdown question.** `max_drawdown <= 0.35` and
   `max_stake_fraction = 0.05` are jointly inconsistent for a binary: a loss is
   100% of stake, so twelve in a row is 60% of the account, and at ~14 trades a day
   that streak is unremarkable. Either number can move; it is a risk-appetite call,
   not a bug, and changing one to pass the other would be gaming a gate.
5. **Do not `promote` yet.** It would install a model fitted on the unrepaired
   store, and `max_drawdown` blocks it anyway.
6. The remaining P2/P3 items below.

Two things remain deliberately **not** fixed, and are decisions rather than
defects: the backtest's counterparty is the baseline the model corrects (fixing it
means collecting quotes, not inventing a worse price), and `min_edge_pp = 0.5pp` is
below what any calibration measurement on this sample can resolve. Both are argued
out in `AUDIT_REPORT.md`.

## Original status, 2026-08-23

Eight commits on `audit/critical-fixes`. **All of P0 except credential rotation is
done, and most of P1 and P2.** The suite went from 230 tests to 348 (plus 28 in
`backend/api`), and every fix was checked by reintroducing the bug and confirming a
test fails — three mutants that survived all 230 original tests are now killed.

What is still open, in the order it matters:

1. **Rotate the leaked credentials.** Only you can do this. `b70c78c` and
   `6097ed1` are on `origin/main`.
2. **Re-fill the store and retrain.** Every model fitted before the pagination fix
   was fitted on a grid missing one minute in 301, and the loss phase is fixed in
   training but moves in serving. Run `scripts.scrape --fill-gaps` (now defaulting
   to `--min-gap-minutes 1`), then `sync_store`, then re-evaluate.
3. **Collect Kalshi quote history.** `--mode live --dry-run` writes
   `market_probability` on every window now. Until there are months of it, nothing
   here speaks to profitability — see the Final Adversarial Review in
   `AUDIT_REPORT.md`.
4. **Paper-trade for a week and read the funnel.** Nothing in this system has ever
   completed a cycle. Every fix is verified by a test and none by a live run.
5. The remaining P2/P3 items below, chiefly `core/backtest.py` at 0% coverage,
   `run_cycle`/`act_on` untested, six more `inspect.getsource` assertions, and the
   absence of any CI.

Two things are deliberately **not** fixed and are decisions rather than defects:
the backtest's counterparty is the baseline the model corrects (fixing it means
collecting quotes, not inventing a worse price), and `min_edge_pp = 0.5pp` is below
what any calibration measurement on this sample can resolve (500 rows at p=0.9
carry a 1.3pp standard error). Both are argued out in `AUDIT_REPORT.md`.

---

**Read this first (historical — P0-7 has since landed).** The live path was dead —
`score_live` raised on every cycle. That means every other live defect is *latent*. Fixing P0-1
alone would activate duplicate orders (P0-2), debit-only accounting (P0-3), the
wrong settlement rule (P0-4) and phantom positions (P0-6) all at once. **P0-1
must be the last of the P0 live fixes to land, or the whole P0 block must land
together.** The plan below is ordered so that is what happens.

---

## P0 — Must fix before any live trading

### P0-0 · Rotate the leaked credentials  [CRITICAL] [operator action]
**Files:** none in repo — external consoles. Then `.gitignore`, `frontend/.env`.
**Expected:** no credential in git history is still valid.
**Change:** rotate the Coinbase API key/secret, the Postgres password, and
`API_TOKEN`/`VITE_API_TOKEN`. Then `git rm --cached frontend/.env`, broaden
`.gitignore` from `.env` to `.env*`, add `*.pem`/`*.key`. Optionally
`git filter-repo` afterwards — rotation is the control that works, history
rewriting cannot reach existing clones.
**Tests:** a pre-commit/CI check that fails on any staged `.env*`, `*.pem`,
`*.key`, or a diff containing `BEGIN * PRIVATE KEY`.
**Depends on:** nothing. Do this first; it is the only item with an active
external exposure.

### P0-1 · Make the account authoritative and credit settlements  [CRITICAL]
**Files:** `core/pg_writer.py`, `scripts/live.py`
**Expected:** a settled winning position increases the bankroll by its payout;
`realized_pnl` reflects actual settled PnL; the bankroll is never updated by an
unlocked read-then-write.
**Change:** move the bankroll mutation *into* `settle_position` so the position
row and the account move in one transaction; credit `payout`, accumulate
`realized_pnl`. Replace `update_account(bankroll=account.bankroll - stake)` in
`act_on` with an atomic relative update (`UPDATE ... SET bankroll = bankroll - :x`).
Add a unique constraint so exactly one `Account` row can exist.
**Tests:** open a position, settle it up, assert bankroll == start - outlay +
payout and `realized_pnl == pnl`; settle it down and assert bankroll == start -
outlay. Concurrency test: two interleaved debits must both apply.
**Depends on:** nothing.

### P0-2 · Settle on the trained rule, and prefer the venue  [CRITICAL]
**Files:** `scripts/live.py`
**Expected:** `settle_due`'s outcome equals `build_windows`' `outcome` for the
same window; the venue's settlement wins where it exists.
**Change:** replace `settle_price = row['open']` / `settle_price > strike` with
`bar_mean` of the bar at `settle_time - 1min` and `>=`, calling
`core.windows.bar_mean` rather than re-deriving. Apply the `/portfolio/settlements`
rows as authoritative before falling back to bars; persist any disagreement.
Delete the dead `price_reference` branch and the stale docstring that argues for
open-to-open.
**Tests:** property test — for synthetic windows including exact ties,
`settle_due` outcome == `build_windows` outcome. A venue settlement disagreeing
with bars must produce the venue's PnL and record the disagreement.
**Depends on:** P0-1 (the credit path must exist before settlements matter).

### P0-3 · One entry per (symbol, window), enforced durably  [CRITICAL]
**Files:** `scripts/live.py`, `core/pg_writer.py`
**Expected:** across any number of cycles, restarts, or offsets, at most one
entry and one order per (symbol, window). `max_positions_per_window` and
`max_window_exposure_fraction` bind live exactly as in the backtest.
**Change:** seed `WindowExposure` at the top of each cycle from
`writer.open_positions()` filtered to the current window, plus any `order_tickets`
row for it. Make `open_position` an idempotent get-or-create on
`(symbol, window_open)`. Refuse in `act_on` when a ticket for the window already
has status `placed`/`filled`. Add a Postgres advisory lock so only one trader
process runs (mirroring `backend/api/app.py:60`).
**Tests:** two `run_cycle` calls at elapsed 3m and 4m → one position, one ticket,
one order. A third at 6m (different offset) → still one. Restart recovery:
pre-existing position in DB → `ALREADY_ENTERED`. Two concurrent processes → the
second refuses to start.
**Depends on:** P0-1.

### P0-4 · Never book a position without a confirmed fill  [CRITICAL]
**Files:** `scripts/live.py`, `data_collection/kalshi_client.py`, `core/decide.py`
**Expected:** a position exists only for contracts the venue confirms filled, at
the price it filled. No order, no position. No market, no trade.
**Change:** read `status`, `remaining_count`, `taker_fill_count` — treat anything
but a confirmed fill as zero contracts; write the position from the fill, not the
decision. Add `Reason.NO_MARKET` and refuse in `decide()` when running live
without a `market_ticker`, instead of silently pricing off the baseline. Restructure
`act_on` so `open_position` is unreachable unless an order was acknowledged. Stop
`resolve_ticket` nulling `filled_contracts`/`filled_price` on a later call.
**Tests:** mocked filled / killed / partial / empty-body / malformed-JSON, each
asserting position count, price and bankroll delta. Unresolved market → zero
positions, zero bankroll movement, `reason == 'no_market'`.
**Depends on:** P0-1, P0-3.

### P0-5 · Fix the order envelope: limit price, ticks, and slippage cap  [HIGH]
**Files:** `data_collection/kalshi_client.py`, `scripts/live.py`, `core/costs.py`
**Expected:** the limit sent is never below the intended price; the price walked
to is bounded by something that preserves edge, not by break-even.
**Change:** round the buy limit **up** (`math.ceil`) to the venue's tick, then
re-verify EV against the rounded price and abstain if it no longer clears.
Replace the `price + edge` limit (a zero-EV cap — measured 0.7832 sent against a
0.60 ask) with `price + min(edge * slippage_share, max_slippage)`, both config
fields. Record the rounded price on the ticket so the ledger matches the wire.
**Tests:** for every price on the tick ladder assert cents sent >= intended; a
price whose EV goes negative after rounding abstains; the limit never exceeds
`price + max_slippage`.
**Depends on:** P0-4.

### P0-6 · Make the safety flags real  [HIGH]
**Files:** `scripts/live.py`, `core/pg_writer.py`, `core/promotion.py`, `backend/api/endpoints/jobs.py`
**Expected:** every documented guard actually guards.
**Change:** (a) make `--dry-run` and `--place-orders` mutually exclusive at the
parser and gate placement on `args.place_orders and not args.dry_run`;
(b) `ensure_account` must update `mode` on an existing row and refuse to inherit a
paper bankroll into a live run; (c) `_refuse_if_blocked` must test `passed`, not
`installed`, so a `--force` install cannot be traded silently; (d) drop
`scripts.promote` from the API `JOBS` allow-list, or reject `--force` there.
**Tests:** parser rejects `--mode live --dry-run --place-orders`; `--dry-run`
sends zero POSTs; paper→live flips `account.mode`; a forced, gate-failing artifact
refuses to trade; `POST /jobs/scripts.promote --force` returns 4xx.
**Depends on:** nothing.

### P0-7 · Score the window being decided  [CRITICAL] [land last in P0]
**Files:** `core/windows.py`, `core/dataset.py`, `scripts/live.py`
**Expected:** `score_live` returns one row per symbol at every configured offset,
with the same feature values the backtest computes, and abstains on a stale feed.
**Change:** add `include_unsettled: bool = False` to `build_windows`. When set,
pad the minute grid with NaN rows to the end of the window containing the last
bar so the trailing partial window survives the `// window` reshape; require
`strike` present but allow `settle_price`/`settle_return`/`outcome` NaN for that
row; count `minutes_missing` over `[0, offset)` only. `score_live` passes
`include_unsettled=True`. **Do not write a second window builder** — the bit-exact
parity measured in this audit follows from both paths sharing this arithmetic.
Add the freshness precondition in the same change: require a non-NaN bar at
`window_open + offset - 1` and abstain otherwise. Catch `DatasetError` in the
`--loop` body so one bad cycle logs and continues instead of exiting.
**Tests:** for each offset, bars ending at `window_open + offset` → one row per
symbol, finite `displacement`/`sigma_remaining`/`baseline_probability`, NaN
`outcome`, `last_price == close(window_open + offset - 1)` exactly. Bars one
minute short → abstain, not a row. Rewrite
`tests/test_features_and_model.py:203,216`, which feed a settled window and so
cannot fail for the live case.
**Depends on:** P0-1 … P0-6. **This is the switch that turns the live path on.**

---

## P1 — Must fix before increasing capital

### P1-1 · Fix the Coinbase pagination off-by-one and recover the data  [CRITICAL]
**Files:** `data_collection/coinbase_connector.py`, `scripts/scrape.py`, `tests/test_backfill_windows.py`, `CLAUDE.md`
**Expected:** a `limit`-N request spans exactly N candle starts; no minute is lost.
**Change:** request `batch_end = current_start + (limit - 1) * tf_seconds` (or
subtract one timeframe from `end`). Change `--min-gap-minutes` default from 2 to
1, fix its docstring, correct `CLAUDE.md`'s "minutes in which nothing traded"
claim, fix the test that locks in the false premise, then run `--fill-gaps` to
recover ~10,100 minutes per symbol.
**Tests:** mocked API returning candles whose `start` ∈ `[start, end]`; a
3,000-minute range returns 3,000 consecutive minutes with no holes; assert the
requested pair never spans more than `limit` starts.
**Depends on:** nothing. Do before any retraining — training data is affected.

### P1-2 · Exclude the in-progress candle  [HIGH]
**Files:** `data_collection/coinbase_connector.py`, `scripts/scrape.py`, `data_collection/pipeline.py`, `scripts/live.py`
**Expected:** no partial candle is ever stored or scored.
**Change:** clamp every `end` to the last *closed* minute
(`floor(now, 1min) - 1min`) at all four call sites. Make the ingest path able to
overwrite a previously-stored final minute so an existing partial is repaired.
**Tests:** with a clock mid-minute, the newest returned/stored bar is the previous
minute. A stored partial is replaced on re-scrape.
**Depends on:** P1-1 (same module).

### P1-3 · Make NaN fail loudly and consistently  [CRITICAL]
**Files:** `core/baseline.py`, `core/metrics.py`, `scripts/baseline.py`
**Expected:** a non-finite prediction can never pass a gate or be pooled into a
reliability bin.
**Change:** drop or explicitly bucket non-finite predictions in `reliability`
instead of letting `np.digitize` pool them into `[0.95, 1.00]`. Replace
`pandas.max()` / builtin `max()` in gate aggregation with an order-independent
NaN-propagating helper. Add an explicit `non_finite_rows == 0` gate so a data
hole reports as a data hole rather than as "no skill".
**Tests:** one injected NaN → every gate reader fails, `scripts/baseline.py`
exits non-zero, the NaN appears in no bin.
**Depends on:** nothing.

### P1-4 · Make the calibration gate protect the traded band  [HIGH]
**Files:** `core/metrics.py`, `core/baseline.py`, `core/config.py`
**Expected:** a model miscalibrated where it trades cannot pass.
**Change:** gate `max_deviation` over populated bins, not only mean ECE. Add a
gated calibration metric computed **on the traded subset**. Narrow bins in the
tails (2pp, not 10pp). Reconcile `min_edge_pp` and `calibration_error` in one
place so the permitted error cannot exceed the required edge.
**Tests:** the three constructions from the report — traded-band-only error,
sign-cancelling within-bin error, and uniform within-bin error — each must FAIL.
**Depends on:** P1-3.

### P1-5 · Purge and embargo the inner validation split  [MEDIUM]
**Files:** `core/model.py`, `core/cv.py`
**Expected:** `residual_scale` and `best_iteration` are estimated on rows that do
not share feature lookbacks with inner-train; the embargo is validated against
the feature lookbacks rather than against itself.
**Change:** apply the configured embargo to the inner split too. Separate the rows
used for early stopping from those used for `residual_scale`. Assert
`embargo_minutes >= max(vol_lookbacks_minutes) + window_minutes` at config
construction — `assert_no_leakage` currently passes at `--embargo-minutes 0`.
**Tests:** `--embargo-minutes 0` must raise; the inner gap must be >= embargo;
`residual_scale` on pure noise must be < 0.25 (it currently reads 0.902).
**Depends on:** nothing.

### P1-6 · Abstain on stale, incomplete or one-sided input  [HIGH]
**Files:** `scripts/live.py`, `core/decide.py`, `core/config.py`
**Expected:** the system does not trade when its inputs are uncertain.
**Change:** add a max bar age and a max quote age, both config; refuse when the
feed does not reach `decision_time - 1min`. Refuse when any symbol's bars are
missing rather than silently redefining the other symbols' `cross_asset` features.
Refuse a one-sided or crossed book instead of reverting to baseline pricing and
still ordering. Add a finiteness gate over the feature vector before `predict`.
Re-validate the wall clock against `decision_time` immediately before sending.
**Tests:** each condition asserted to abstain with a named reason.
**Depends on:** P0-7.

### P1-7 · Validate the artifact against the running config  [HIGH]
**Files:** `core/model.py`, `core/promotion.py`, `scripts/live.py`
**Expected:** a model cannot be scored under a config it was not fitted for, and
the artifact on disk is the one the ledger gated.
**Change:** record a hash of the artifact in the promotion ledger and verify it
before `joblib.load`. Compare `config_provenance` against the running `Config`
and refuse on any field that changes an answer (offsets, `window_minutes`,
`vol_lookbacks_minutes`, fee params). Assert
`booster.feature_name() == model.features`. Give `ForecastModel` a real `version`
so `Prediction.model_version` stops being NULL.
**Tests:** a mismatched config refuses; a tampered artifact refuses; a reordered
feature list raises.
**Depends on:** nothing.

### P1-8 · Add the risk controls that do not exist  [HIGH]
**Files:** `core/config.py`, `core/decide.py`, `scripts/live.py`, `core/pg_writer.py`
**Expected:** bounded loss per day and a kill switch that survives a restart.
**Change:** add max daily loss, max consecutive losses, a drawdown kill switch on
the *live* account, total and per-symbol exposure caps, and an orders-per-hour
budget. Write `Account.halted`/`halted_reason` from the live path — they are
currently never written by anything, so the dashboard's safety chip cannot turn
on. Range-check the probability in `decide()` (q=1.20 currently yields a
7.1x-Kelly stake). Treat a NaN bankroll as halted.
**Tests:** each limit trips and persists across a restart; a NaN bankroll refuses.
**Depends on:** P0-1, P0-3.

---

## P2 — Important reliability improvements

- **P2-1 [MEDIUM]** `core/decide.py`: stop adding the half-spread to `outlay` when
  the price came from a real ask (`decide.py:379` vs `:213-216`) — measured $4.98
  recorded against $4.94 actually paid. Manufactures the exact reconciliation
  drift CLAUDE.md tells the operator to read as an unrecorded fill. Apply
  `min_edge_pp`, not just `EV > 0`, in the post-rounding re-check.
- **P2-2 [MEDIUM]** Make the price band symmetric. `[0.05, 0.97]` contradicts both
  `config.py`'s own comment and CLAUDE.md, and the 2pp asymmetry runs the wrong
  way: it admits 96-97c favourites (where a 1pp calibration error destroys 43% of
  the gross) and refuses 3-4c longshots (where it destroys 1%).
- **P2-3 [MEDIUM]** Side-adjust `Decision.baseline_probability` as
  `model_probability` already is, or side-adjust neither. Today a DOWN trade
  stores P(down) beside P(up) and the difference is meaningless. Same for
  `market_probability`, which is always `ask_up`. Rename the API fields so
  `Prediction.model_probability` (P(up)) and `Position.model_probability`
  (P(side taken)) stop sharing a JSON key.
- **P2-4 [MEDIUM]** Tighten `max_disagreement_pp` from 25.0. Against a real quote
  it permits trading on a belief that the market is 25 points wrong — which is
  the only guard against a misparsed quote or a broken sigma.
- **P2-5 [MEDIUM]** Fix `n_features_populated` double-subtraction
  (`model.py:133`) — reports 28 for 35 features, and it goes into every
  provenance record the trial count relies on.
- **P2-6 [MEDIUM]** Reclassify `seasonal_ramp` (a deterministic function of
  minute-of-day and offset) out of `vol_state`, and decide explicitly whether
  offset-dependent features belong in the `clock` control. As written the
  `control_gain_share <= 0.30` gate both penalises legitimate per-offset
  recalibration and cannot catch clock-driven models.
- **P2-7 [MEDIUM]** `core/pg_writer.py:353` `_run_migrations` swallows every
  exception at `logger.debug`. Narrow it to the documented SQLite case and let
  real Postgres migration failures surface at startup.
- **P2-8 [MEDIUM]** `KalshiClient.balance()` returns `0.0` on a missing field and
  reads integer cents only, while every sibling parser prefers `*_dollars` and
  returns `None` on failure. A parse failure overwrites a correct bankroll with
  zero. Log every swallowed parse error in `_price`/`_quantity`/`_cents`.
- **P2-9 [MEDIUM]** Wire up `record_model_run` and `write_calibration` — the only
  writers of the `model_runs` and `calibration` tables, currently with zero
  callers, while the API and the Model/Calibration dashboard pages read from
  them. Also mount `trader_models` on the `backend` service.
- **P2-10 [MEDIUM]** Either populate `trade_count` in the connector or delete
  `trade_count_z_15`. It is permanently all-NaN on real data (0 of 2,617,876 rows)
  and only looks alive because `tests/conftest.py:84` fabricates the column.
- **P2-11 [MEDIUM]** Replace the 8 `inspect.getsource` substring assertions with
  behavioural tests. That pattern is what let P0-1 ship green.
- **P2-12 [LOW-MED]** Record Kalshi quotes and outcomes for every window, traded
  or refused, so the model can eventually be measured against the market's own
  price — the only economically meaningful benchmark, and currently absent from
  the schema, the collection, every metric and every gate. `market_probability`
  and `outcome` columns already exist; a `--mode live --dry-run` loop populates
  the unselected sample this needs.
- **P2-13 [LOW-MED]** Add a loop-liveness healthcheck. The trader healthcheck
  tests a Postgres connection, so a crash loop is invisible to compose.
- **P2-14 [LOW]** Fix `--start`/`--end` (`core/datastore.py:258` crashes on pandas
  3.0.0), so runs can be date-limited at all.
- **P2-15 [LOW]** `us_equity_hours` covers EDT only; `np.errstate(invalid=...)` at
  `windows.py:246` does not suppress the All-NaN-slice warning it targets.

---

## P3 — Cleanup / maintainability

- **P3-1** Pin every Python dependency and add a lock file. `lightgbm`,
  `scikit-learn`, `scipy`, `joblib`, `cryptography` and `coinbase-advanced-py` are
  unpinned — the packages that deserialize the model and sign live orders.
- **P3-2** Correct the documentation the audit contradicted: the "83c" fee/spread
  crossover is 92.26c at the current 0.5c default and is stale in six places;
  "ties are not rare" is 1 in 173,937; "0.5% of minutes, mostly untraded" is 86%
  a client-side bug; "207 tests in 26s" is 230 in ~170s; the `settle_due`
  open-to-open docstring; and CLAUDE.md's "no scraped data" (there are 2.6M bars
  per symbol).
- **P3-3** Delete or clearly quarantine the dormant perp-era code: `RedisQueue`,
  funding/OI models, storage tables and validators, the `--live` scrape path,
  `write_features`/`read_features`, and the unused `data_collection/__init__`
  exports.
- **P3-4** Remove dead code: `Config.tie_resolves_up` (never read),
  `live.py:236-237`, the `revenue` local, the `baseline.py:221-222` broken
  weights conditional, `elapsed_fraction`/`remaining_minutes` (correlate exactly
  -1.0).
- **P3-5** Add `Mapped[]` typing to `core/pg_writer.py` so mypy can check the
  bankroll arithmetic — 43 of its 106 errors are in the file that had P0-1.
- **P3-6** Rename one of the two `serving.py` modules so mypy can resolve imports.
- **P3-7** Add a `ruff` config and fix the 545 findings (448 auto-fixable); run
  ruff/mypy/pytest and the frontend checks in CI. No CI config exists today, so
  `test_orm_parity.py` is not actually enforced.
