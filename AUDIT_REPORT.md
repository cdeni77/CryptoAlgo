# Quarter — Adversarial Audit Report

**Date:** 2026-08-23
**Scope:** whole repository — data ingestion, window/label construction, features,
volatility model, barrier baseline, LightGBM correction, cross-validation,
backtest, promotion gates, live/paper trading, order execution, risk controls,
serving store, API, frontend, secrets, dependencies, tests.
**Method:** static review plus executed reproductions. Every CRITICAL and HIGH
finding below was reproduced or confirmed against the code by the coordinator,
not merely reported. Comments, docstrings, `CLAUDE.md`, `AGENTS.md` and test
names were treated as hypotheses throughout; several are contradicted by the
implementation and are called out as such.

---

# Executive Summary

| question | answer |
|---|---|
| Is this system currently safe to trade real money? | **NO** |
| Is the backtest trustworthy? | **UNCERTAIN** — internally sound and leak-free, but it measures skill against an assumed counterparty, not profitability |
| Is there evidence of lookahead / data leakage? | **NO** in training and backtesting (proven experimentally). **YES**, minor, in two inner-fit steps |
| Is train/live inference parity verified? | **YES for `core/`** (bit-exact, 0 of 52 columns differ). **NO for `scripts/live.py`** |
| Is order execution safe / idempotent? | **NO** |
| Are reported profitability metrics believable? | **NO** — no profitability metric has ever been produced; there is no trained model and no measured edge |

> **Addendum, 2026-08-24.** This system has now traded real money for 24 hours.
> Four defects were found that this audit did not, all four in the code that
> *verifies* rather than the code that trades, and the market benchmark was
> measured for the first time. Point 3 of the Final Adversarial Review predicted
> this in these words: *"no fill ever read back, no settlement ever reconciled,
> no drift ever observed, no idea what the venue actually does with a duplicate
> `client_order_id`. ... The first week of paper trading will find things this
> audit did not."* All four items on that list were the four defects. See
> **Addendum — the first 24 hours live**, at the end of this document.

## The one-paragraph version

The research core of this project is unusually well built. The barrier reframing is
correct, the label reproduces the venue's published settlement rule exactly, the
feature pipeline is provably free of lookahead, the three fitted objects are
contained inside their folds, and the backtest and live code paths compute
identical feature vectors to 16 decimal places. That is a genuinely strong
foundation and most of this report's severity does not touch it.

Everything that trades is broken. `scripts/live.py` cannot score the window it is
asked to decide — `build_windows` structurally refuses to emit an unsettled
window, so every live cycle raises `DatasetError` and the process exits. That
single defect has masked at least eight others behind it: the live path re-enters
the same market up to twelve times per window, settles positions against a
different rule than it trains on, discards the venue's own settlement data, books
positions for orders it never placed, records killed orders as filled fills,
sends limit prices below the intended price in exactly the band it wants to
trade, and treats `--dry-run` as a no-op. Because the loop crashes first, none of
this has ever executed — current real-money exposure is zero — but all of it
activates the moment the crash is fixed, which is why the fixes must ship
together.

Separately and independently: live Coinbase credentials and an API token are in
pushed git history and need rotating regardless of anything else here.

## Top 10 highest-risk findings

| # | Sev | Finding | Location |
|---|---|---|---|
| 1 | CRITICAL | Live path cannot score the current window; every cycle raises and the process exits | `core/dataset.py:397`, `core/windows.py:197` |
| 2 | CRITICAL | Live re-enters the same (symbol, window) every cycle — up to 12 entries where the backtest allows 1; all window risk limits non-functional live | `scripts/live.py:455`, `core/decide.py:320` |
| 3 | MEDIUM (was CRITICAL) | Coinbase key/secret in pushed git history — **read-only scope**; see the correction at the end | `b70c78c:backend/api/.env` |
| 4 | CRITICAL | Live settles on `open(t1)` with strict `>` vs trained `mean(O,H,L,C) over [t1-1,t1)` with `>=` — 3.4–3.9% of windows get the opposite label | `scripts/live.py:235,244` |
| 5 | CRITICAL | Unresolved market books a phantom position and debits the bankroll without placing an order | `scripts/live.py:511-540` |
| 6 | CRITICAL | Fills are assumed, not read back: a killed `fill_or_kill` is recorded as a full fill | `scripts/live.py:519` |
| 7 | HIGH | `--dry-run` is declared and never read; `--mode live --dry-run --place-orders` places real orders | `scripts/live.py:112` |
| 8 | HIGH | Venue settlements fetched and discarded; `revenue` is a dead local | `scripts/live.py:328-337` |
| 9 | HIGH | Calibration gate cannot protect the traded band; permitted error (2pp) is 4x the required edge (0.5pp) | `core/metrics.py:285`, `core/baseline.py:313` |
| 10 | HIGH | Limit prices truncated *below* intent in the tails, guaranteeing FOK kills that are then recorded as fills | `data_collection/kalshi_client.py:433` |

---

# Findings

## [CRITICAL] The live path cannot score the window it is asked to decide

**Location:** `backend/trader/core/dataset.py:397-404`; mechanism in `backend/trader/core/windows.py:197,205,273`

**What happens:** `score_live` calls `build_window_panel` and then slices for the
window being decided. `build_windows` trims the minute grid to a *whole* number
of windows (`n_windows = (len(grid) - offset_into) // window`) and takes the
settlement value from `means[:, window - 1]` — minute 14 of the window. A window
that is 3, 6, 9 or 12 minutes old has neither. It is therefore absent from the
panel twice over, the slice is empty, and `DatasetError` is raised.

**Why it matters:** `run_cycle` does not catch it (`scripts/live.py:389`) and
`main()`'s loop catches only `KeyboardInterrupt` (`scripts/live.py:611-619`), so
the process exits on its first cycle. Paper and live trading have never worked.
Under `docker-compose`'s `restart: unless-stopped` this is a crash loop.

**Evidence:**
```
window_open = 2026-08-23 04:30, settles 04:45
  bars up to 04:33 (offset + 3m)  ->  window 04:30 in panel? False   newest = 04:15
  bars up to 04:36 (offset + 6m)  ->  window 04:30 in panel? False   newest = 04:15
  bars up to 04:39 (offset + 9m)  ->  window 04:30 in panel? False   newest = 04:15
  bars up to 04:42 (offset +12m)  ->  window 04:30 in panel? False   newest = 04:15
  bars up to 04:45 (offset +15m)  ->  window 04:30 in panel? True    newest = 04:30
```
Independently confirmed end-to-end against a real fitted bundle:
`score_live RAISED DatasetError: no window opens at ...` for 12 of 12 simulated
cycles, while the previous settled window returns 3 rows.

**Example scenario:** The container starts at 04:33. `choose_offset(3)` returns 3.
`score_live` raises. The process exits. Docker restarts it. At 04:34 the same
thing happens. No order is ever placed and no prediction is ever recorded.

**Recommended fix:** Extend `build_windows` with `include_unsettled: bool = False`.
When set, pad the minute grid with NaN rows up to the end of the window
containing the last bar so the trailing partial window survives the reshape;
require `strike` to be present (it is the *previous* window's minute-14 mean and
does exist) but allow `settle_price`, `settle_return` and `outcome` to be NaN for
the final row; count `minutes_missing` only over `[0, offset)` for that row.
`score_live` passes `include_unsettled=True`. Do **not** write a second window
builder — the bit-exact backtest/live parity measured in this audit is a direct
consequence of both paths sharing this arithmetic.

Add a freshness precondition in the same change: require a non-NaN bar at
`event_time == window_open + offset - 1` and abstain otherwise, so a stale feed
cannot forward-fill the strike into a fabricated `last_price`.

**Regression test:** Build bars ending at `window_open + offset` for each
configured offset, call `score_live`, and assert one row per symbol with finite
`displacement`, `sigma_remaining` and `baseline_probability` and NaN `outcome`.
Assert `last_price` equals `close(window_open + offset - 1)` exactly. Assert that
bars ending at `window_open + offset - 1` (one minute short) abstain rather than
returning a row. The two existing tests
(`tests/test_features_and_model.py:203,216`) feed `window_index[-3]` — a fully
settled window — and so cannot fail for the only case live ever asks for; the
second is named `test_score_live_reports_no_outcome_for_an_unsettled_window`
while its fixture guarantees a settled one. Both should be rewritten.

## [CRITICAL] Live re-enters the same (symbol, window) on every cycle

**Location:** `backend/trader/scripts/live.py:455`; guard at `backend/trader/core/decide.py:320-321`

**What happens:** `run_cycle` constructs `exposure = WindowExposure()` fresh on
every invocation and never seeds it from the database. `decide()`'s
`ALREADY_ENTERED`, `POSITION_LIMIT` and `WINDOW_EXPOSURE` gates all read only
that in-memory object. `choose_offset` returns the latest offset the clock has
reached, so with `decision_offsets = (3,6,9,12)` and `--cycle-seconds 60` a single
window is decided twelve times per symbol.

**Why it matters:** `max_positions_per_window` (2) and
`max_window_exposure_fraction` (0.08) are enforced only in the backtest. The
backtest therefore simulates a strategy the live path cannot execute — it sizes
one entry per window where live takes up to twelve. Under `--place-orders` the
only thing preventing duplicate real orders is Kalshi honouring
`client_order_id`, an unasserted and untested venue behaviour.

**Evidence:**
```
live.py: fresh WindowExposure() each cycle
  cycle  1  elapsed= 3m  offset=+3m   -> traded  contracts=8  stake=$4.98
  cycle  2  elapsed= 4m  offset=+3m   -> traded  contracts=8  stake=$4.98
  ... 12 of 12 traded ...
backtest: one shared WindowExposure (decide_window)
  offset=+ 3m -> traded            offset=+ 6m -> already_entered
  offset=+ 9m -> already_entered   offset=+12m -> already_entered

open_position() twice for one (symbol, window):
  1st id = 1
  2nd RAISED IntegrityError: UNIQUE constraint failed: positions.symbol, positions.window_open
write_ticket() twice: ids 1 then 1  -> get-or-create, so act_on() proceeds again
```
Independently measured on real bars: backtest 124 entries → live 618 (4.98x);
worst single window 2 → 21.

**Why the crash is not a mitigation:** `open_position` raises *after*
`place_order` has already gone to the wire (`scripts/live.py:511-527` precedes
`:532`), the exception is uncaught, and `restart: unless-stopped` re-runs the
cycle.

**Recommended fix:** Seed exposure from durable state at the top of every cycle —
`WindowExposure` built from `writer.open_positions()` filtered to the current
`window_open`, plus any `order_tickets` row for it. Make `open_position` an
idempotent get-or-create on `(symbol, window_open)` like `write_ticket` already
is, and have `act_on` refuse when a ticket for the window already has status
`placed`/`filled`. Add a process singleton (a Postgres advisory lock, mirroring
`backend/api/app.py:60`) so a compose `trader` and a hand-run `scripts.live`
cannot both trade one account.

**Regression test:** Two consecutive `run_cycle` calls at elapsed 3m and 4m
against a mocked venue must produce exactly one position, one ticket and one
order. A third call at elapsed 6m (a different offset) must also produce no
second entry. Restart recovery: construct a position in the DB, then call
`run_cycle` for the same window and assert `ALREADY_ENTERED`.

## [CRITICAL] Live credentials in pushed git history

**Location:** `b70c78c:backend/api/.env` (deleted in `c49e9c0`), `6097ed1:frontend/.env.local` (deleted in `dc4310b`)

**What happens:** A `.env` holding `COINBASE_API_KEY` (95 chars),
`COINBASE_API_SECRET` (234-char EC private key) and a `DATABASE_URL` with an
embedded password was committed, then deleted in the following commit. Deleting a
file does not remove the blob. `git branch -r --contains b70c78c` confirms both
commits are on `origin/main`.

**Why it matters:** Anyone with a clone, fork or read access to the GitHub repo
can recover the credentials permanently with `git show`. The value lengths match
the shape of the credentials currently in the operator's working `.env`, so these
are very likely genuine.

**Recommended fix:** Rotate the Coinbase key/secret, the Postgres password and
`API_TOKEN` **first** — that is the only control that works, because history
rewriting cannot reach existing clones. Then `git rm --cached frontend/.env`
(currently tracked) and broaden `.gitignore:109` from `.env` to `.env*`:
`git check-ignore` reports `frontend/.env` and `frontend/.env.local` as NOT
IGNORED, and the documented dev workflow instructs putting `VITE_API_TOKEN` in
exactly those files. Optionally rewrite history with `git filter-repo` afterwards.

**Regression test:** A pre-commit hook or CI step that fails on any staged path
matching `.env*`, `*.pem`, `*.key`, and on any diff containing `BEGIN * PRIVATE KEY`.

## [CRITICAL] Live settlement uses a different rule than training

**Location:** `backend/trader/scripts/live.py:229-244`

**What happens:** Three deviations at once from the trained label
(`core/windows.py:216,241`) and from the venue's published rule:

| | training / venue | live |
|---|---|---|
| bar | minute `[t1-1, t1)` | minute `[t1, t1+1)` — one minute late |
| estimator | `(O+H+L+C)/4` | `open` — a single print |
| comparison | `>=` (`strike_type: greater_or_equal`) | `>` |

**Why it matters:** Paper-mode PnL, the equity curve, the win rate and every
realised-edge number computed live are measured against a label the model was
never trained on and the venue does not use. Ties resolve DOWN locally and UP at
the venue.

**Evidence:** Measured on real BTC bars, two agents independently:
**3.38% (5,863 of 173,933)** and **3.88% (149 of 3,838)** of windows settle the
opposite way. `settle_due` also runs unconditionally at `scripts/live.py:382`
even with `--reconcile`.

**The docstring argues for the bug.** `scripts/live.py:216-220` says *"the strike
was read the same way, so the window's return is open-to-open"* — verified false:
`strike == open(t0)` is False, `strike == bar_mean(prior window minute 14)` is
True. It then warns that *"anchoring one end on a last trade and the other on a
first trade is how this project once manufactured 98% of an apparent edge"*,
which is precisely what the code below it now does. The comment is a leftover
from the pre-averaging version.

**Recommended fix:** Replace with `bar_mean` of the bar at
`settle_time - 1min` and a `>=` comparison — i.e. call the same helper
`core/windows.py:bar_mean` rather than re-deriving. Delete the dead
`price_reference` branch at `:236-238`, which is overwritten at `:244`.

**Regression test:** A property test asserting that for a set of synthetic
windows, `settle_due`'s outcome equals `build_windows`' `outcome` for the same
window, for every window including exact ties.

## [CRITICAL] An unresolved market books a phantom position

**Location:** `backend/trader/scripts/live.py:511-540`

**What happens:** When `resolve_window_market` returns `None`, `market_ticker` is
`None`, so `ask_up`/`ask_down` are NaN, `has_book` is False, and `decide()` falls
back to the *backtest counterfactual* price (`core/decide.py:217-221`) — still
returning `TRADED`. In `act_on`, the guard
`if args.place_orders and kalshi is not None and decision.market_ticker:` is
False so no order is sent, but control **falls through** to `writer.open_position`
and `writer.update_account(bankroll - stake)`.

**Why it matters:** The system records a position it does not hold and debits
money it did not spend, then `settle_due` settles it into fictional PnL. This is
the exact failure `core/pg_writer.py:82-87` says the `price_source` column exists
to prevent. It is also the opposite of the behaviour CLAUDE.md celebrates
("that abstention was the resolution logic working").

**Evidence:** Measured under `--mode live --place-orders` with an unresolved
market: 0 orders sent, 5-contract position written, $3.10 debited.

**Recommended fix:** Add `Reason.NO_MARKET` and refuse in `decide()` whenever the
row carries no `market_ticker` while running live. Structurally, `act_on` must
never reach `open_position` unless an order was actually acknowledged — invert it
so the position is written from the *fill*, not from the decision.

**Regression test:** `run_cycle` with `resolve_window_market` mocked to `None`
must yield zero positions, zero bankroll movement, and a recorded prediction with
`reason == 'no_market'`.

## [CRITICAL] Fills are assumed, not read back

**Location:** `backend/trader/scripts/live.py:519`

**What happens:** `filled = int(order.get('count', decision.contracts))`. `count`
is the size *requested*. `status`, `remaining_count` and `taker_fill_count` are
never read. A killed `fill_or_kill`, a partial fill, and an HTTP 200 with an empty
body all record a full fill and debit the bankroll. `Position.price` and
`filled_price` are set to `decision.price` while the limit actually sent is
`price + edge`.

**Why it matters:** Combined with the limit-rounding defect below, the common case
is a *guaranteed* kill recorded as a complete fill. The book then holds positions
that do not exist and settles them into invented PnL, and reconciliation is
one-directional so it never looks for the discrepancy in this direction.
CLAUDE.md's claim *"a `fill_or_kill` that killed leaves a ticket and no
position"* is false.

**Recommended fix:** Read `status` and the fill counts; treat anything other than
a confirmed fill as zero contracts; write the position only for the filled
quantity at the filled price; on a partial fill record the partial. Then
re-verify the venue's view with `GET /portfolio/positions` before writing.

**Regression test:** Mocked responses for filled / killed / partial / empty-body /
malformed-JSON, each asserting the resulting position count, price and bankroll
delta.

## [CRITICAL] Pagination off-by-one destroys one minute in every 301

**Location:** `backend/trader/data_collection/coinbase_connector.py:177,179,181,217,219,226-228`

**What happens:** `batch_duration = tf_seconds * 300` and `batch_end = current_start + 300min`,
with both `start` and `end` sent to the API inclusive, names **301** one-minute
candles for a `limit` of 300. The API returns the newest 300; the oldest —
`current_start` itself — is dropped, and the loop then advances past it.

**Why it matters:** 0.72% of windows are dropped outright and 4.7% are
forward-filled. Worse, the paginator's phase is *fixed* during a single backfill
but *moves on every live fetch* (`end=now`), so training and serving lose
different minutes: a train/serve skew on the barrier's only input, measured
nowhere. ~5 minutes vanish per live fetch; 0.332% of live cycles lose the exact
`offset-1` minute that becomes `last_price`.

**Evidence (five independent confirmations on the real store):**
- 98.9% of inter-hole spacings are *exactly* 301 minutes
- observed rate 0.003318 vs 1/301 = 0.003322 (four significant figures)
- flat across UTC hour (358-370 vs 363 expected), weekday, and year (1,752 / 1,750 / 1,746 / 1,744 for 2022-25, despite regime change)
- **5,121 of BTC's 8,721 holes are the identical minute in ETH, against 28.9 expected under independence — a 177x excess.** Two independent order books cannot go untraded in the same 8,700 minutes.
- 86.1% of BTC's missing minutes and 85.8% of ETH's are isolated single-minute holes

**The repair tool is configured to skip exactly these holes.**
`scripts/scrape.py:84` defaults `--min-gap-minutes` to `2`, documented as *"a
single missing minute is usually a minute in which nothing traded, which no
amount of re-requesting will produce."* That premise is false. It is asserted as
fact in `CLAUDE.md` ("Most of the shortfall is minutes in which nothing traded")
and locked in by a passing test (`tests/test_backfill_windows.py:499`), so
correcting the bug breaks CI.

**Recommended fix:** Make the batch exclusive at one end — request
`batch_end = current_start + (limit - 1) * tf_seconds`, or subtract one timeframe
from `end` before sending. Assert in a test that a 300-limit request spans exactly
300 candle starts. Then change `--min-gap-minutes` to default `1`, fix the
docstring and `CLAUDE.md`, correct the test, and run `--fill-gaps` to recover the
~10,100 minutes per symbol.

**Regression test:** With a mocked API that returns exactly the candles whose
`start` falls in `[start, end]`, assert `get_candles_range` over a 3,000-minute
span returns 3,000 distinct consecutive minutes with no holes. Assert the
requested `(start, end)` pair never spans more than `limit` candle starts.

## [CRITICAL] NaN predictions are pooled into the most-confident bin, and one gate fails open

**Location:** `backend/trader/core/baseline.py:97,313-317,361`; `backend/trader/core/metrics.py:191`; `backend/trader/scripts/baseline.py:103,119`

**What happens:** `np.digitize(nan, edges[1:-1])` returns the last index, so NaN
predictions are counted in the `[0.95, 1.00]` reliability bin — the band the
strategy trades. `expected_calibration_error` then returns NaN, and three
reporting paths disagree about what to do with it:

```
np.digitize(nan, edges[1:-1])  ->  10  =  bin [0.95, 1.00]
nan > 0.02                     ->  False        # a NaN error PASSES a max-gate
pandas .max([0.015, nan])      ->  0.015        # skips NaN silently
builtin max([0.015, nan]) = 0.015 ; max([nan, 0.015]) = nan   # order-dependent
```

**Why it matters:** On real BTC data, 31 non-finite rows in 99,388 (0.031%) were
enough for `scripts/baseline.py` to print
**`gate passed: worst-fold calibration error 0.01516 <= 0.02` while five of six
folds had `log loss nan, ECE nan`.** The reliability table showed
`predicted nan / observed 0.9813 / n 57,890`. `metrics.py:346` fails closed;
`scripts/baseline.py:119` fails **open**; `metrics.py:191` depends on fold order.
"No skill" and "one data hole" produce the same output.

**Recommended fix:** Drop or explicitly bucket non-finite predictions in
`reliability` rather than letting `digitize` pool them; make every aggregation
NaN-propagating (`np.nanmax` is *not* what is wanted here — a NaN must fail);
add an explicit `non_finite_rows == 0` gate so a data hole is reported as a data
hole. Replace `pandas.max()`/builtin `max()` in gate aggregation with an
order-independent NaN-propagating helper.

**Regression test:** Inject one NaN prediction into a fold and assert that every
gate reader reports failure, that `scripts/baseline.py` exits non-zero, and that
the NaN is not counted in any reliability bin.

## [HIGH] The calibration gate cannot protect the band the system trades

**Location:** `backend/trader/core/metrics.py:285`; `backend/trader/core/baseline.py:313-317,344-371`

**What happens:** `calibration_error <= 0.02` reads a **count-weighted mean ECE
over all rows**, with 10pp-wide bins, never restricted to the traded subset.
`Reliability.max_deviation` exists at `baseline.py:320` and is **not gated**.

**Why it matters:** `min_edge_pp` is **0.5pp** (`config.py:226`) — and its own
comment says it exists to guard against calibration error and that "the right
value is whatever the measured calibration error turns out to be." The gate
permits **4x** that. The two files directly contradict each other. A uniformly
2pp-overconfident model taking 0.5-2.0pp claimed edges has mean claimed EV
+1.250pp and **true EV -0.750pp, with 0.0000% of trades genuinely positive.**

**Evidence — three independent demonstrations:**
```
(1) perfect on 190k pinned rows, 5pp overconfident on the 10k rows it trades:
    aggregate ECE 0.00443  -> PASSES ;  max_deviation 0.02049 (not gated)
    traded rows are 5% of the sample, so a 5pp error contributes only 0.0025

(2) sign-opposite errors cancel inside a bin:
    5pp overconfident at 0.94 + 5pp underconfident at 0.86
    -> ECE = 0.000078, passes by 256x. Every trade is at 0.94 and loses 5pp.

(3) predictions 6pp-miscalibrated inside every bin -> ECE 0.00181 (100-bin: 0.02815)
```
And the gate is passed by a **provably zero-signal null** (0.0154) and by the
raw baseline itself (0.0151-0.0163).

**Recommended fix:** Gate `max_deviation` on populated bins, not just mean ECE.
Compute calibration **on the traded subset** as a separate gated metric. Narrow
bins in the traded region (the tails need 2pp bins, not 10pp). Lower the gate to
at or below `min_edge_pp`, or raise `min_edge_pp` above the measured error —
the two numbers must be reconciled in one place rather than set independently in
two files.

**Regression test:** The three constructions above, each asserted to FAIL the gate.

## [HIGH] `--dry-run` is declared and never read

**Location:** `backend/trader/scripts/live.py:112`

**What happens:** `args.dry_run` appears nowhere in the file. The two "dry run"
log lines key off `args.mode` and `args.place_orders`.

**Why it matters:** `--mode live --dry-run --place-orders` parses cleanly and
**places real orders**. The documented protection at `scripts/live.py:28` does
not exist.

**Recommended fix:** Make `--dry-run` and `--place-orders` mutually exclusive at
the parser level (`add_mutually_exclusive_group`), so the combination is a usage
error rather than a silent override, and gate order placement on
`args.place_orders and not args.dry_run`.

**Regression test:** Assert the parser rejects `--mode live --dry-run
--place-orders`, and that with `--dry-run` the mocked venue receives zero POSTs.

## [HIGH] A live account renders as paper

**Location:** `backend/trader/core/pg_writer.py:520-532`

**What happens:** `ensure_account` sets `mode` only when *creating* the row; an
existing row's `mode` is never updated.

**Why it matters:** Paper is the compose default, so the first run creates a
`mode='paper'` account and every later `--mode live` run keeps rendering as paper
on every dashboard surface. `CLAUDE.md` calls this exact scenario "the worst
failure the schema could permit."

**Recommended fix:** Update `mode` on the existing row, and refuse to reuse a
`paper` account's bankroll history for a `live` run — a mode change should
require an explicit reset or a separate account row.

**Regression test:** Seed a paper account, call `ensure_account(mode='live')`,
assert `account().mode == 'live'`.

## [HIGH] Limit prices are truncated below intent, guaranteeing FOK kills

**Location:** `backend/trader/data_collection/kalshi_client.py:433`

**What happens:** `cents = int(round(limit_price / CENT))`. Python's `round` is
banker's rounding, and rounding to whole cents discards the tapered deci-cent
ladder `core/costs.py` carefully models.

**Evidence:**
```
 decide() price   round_to_tick  cents sent  wire price  vs intended  FOK outcome
         0.0450          0.0450           4      0.0400      -0.0050  KILLS
         0.0650          0.0650           6      0.0600      -0.0050  KILLS
         0.9450          0.9450          94      0.9400      -0.0050  KILLS
         0.9650          0.9650          96      0.9600      -0.0050  KILLS
round(34.5) = 34 ;  round(94.5) = 94 ;  round(4.5) = 4
```

**Why it matters:** These are the tails — exactly where `CLAUDE.md`'s own fee
table says the economics are best. With `time_in_force='fill_or_kill'` an
under-priced limit is a guaranteed kill, and the fill-readback defect above then
records the kill as a full fill.

**Recommended fix:** Round the limit **up** for a buy (`math.ceil`) so it never
lands below intent, then re-verify expected value against the rounded price and
abstain if it no longer clears. Record the rounded price as the ticket's limit so
the ledger matches the wire.

**Regression test:** For every price on the tick ladder, assert the cents sent are
`>=` the intended price, and that a price whose EV goes negative after rounding
abstains.

## [HIGH] Venue settlements are fetched and discarded

**Location:** `backend/trader/scripts/live.py:328-337`

**What happens:** `revenue = row.get('revenue_dollars', row.get('revenue'))` is a
dead local. `resolved` is used only as a membership set for an unrelated warning
at `:344`. `settle_position` is called from exactly one place repo-wide (`:245`).

**Why it matters:** `CLAUDE.md`'s "Settlement from `/portfolio/settlements` where
it knows" is unimplemented, so the wrong-rule bar settlement above is the sole
source of live PnL. Reconciliation is also one-directional: it never looks for
venue positions we lack, which is exactly what an uncaught order timeout leaves
behind.

**Recommended fix:** Apply venue settlements as authoritative, keyed on
`market_ticker`, before falling back to bars; log and persist any disagreement.
Add the reverse reconciliation pass.

**Regression test:** A venue settlement that disagrees with the bar-derived
outcome must produce the venue's PnL, and the disagreement must be recorded.

---

# End-to-End Architecture

```
                        ENTRY POINTS
  scripts.scrape ─ scripts.sync_store ─ scripts.baseline ─ scripts.train
  scripts.evaluate ─ scripts.promote ─ scripts.check_venue ─ scripts.measure_book
  scripts.live [--loop]          (the ONLY entry point that can trade)
  uvicorn app:app (4 workers)    read-only routes + POST /jobs/{job}

                        COMPOSE SERVICES
  db (Postgres 16, loopback) · backend · frontend
  trader   -> scripts.live --loop --cycle-seconds 60   [paper, restart: unless-stopped]
  scrape   -> scripts.scrape --backfill-days 1825      [one-off]
  evaluate -> scripts.evaluate                          [one-off]

                        STORES
  data/trading.db          SQLite, raw 1-minute bars      (scrape writes, sync reads)
  RESEARCH_STORE/          Parquet + DuckDB               (sync writes, dataset reads)
  models/forecast.joblib   promoted artifact              (promote writes, live reads)
  models/promotions/*.json the ledger = the trial count
  Postgres                 predictions, positions, account, equity_curve,
                           minute_prices, order_tickets, model_runs*, calibration*
                           (* schema exists, NEVER WRITTEN — see Data Quality)
```

# Data Flow — one trade, end to end

```
Coinbase REST 1m candles
 └─ data_collection/coinbase_connector.py  get_candles / get_candles_range
     ⚠ inclusive-end vs limit=300 destroys 1 minute per 301   :177,:217,:219
     ⚠ end=now stores the in-progress candle                   :186-203
 └─ pipeline.py:213 _process_and_insert_bars -> ingest.py -> storage.py
     └─ data/trading.db   (UNIQUE(symbol,timeframe,venue,event_time), INSERT OR REPLACE)
 └─ scripts/sync_store.py -> core/datastore.py:394 from_sqlite -> Parquet

core/dataset.py:55 load_minute_bars      (backtest)
scripts/live.py:162 fetch_bars           (live — bypasses the research store)
 └─ core/windows.py:161 build_windows     one row per (symbol, window, offset)
     strike = prev window's bar_mean(minute 14);  settle = this window's;  outcome = >=
     ⚠ cannot emit an unsettled window -> the live path dies here
 └─ core/vol.py:191 VolModel.fit/predict -> sigma_per_min
 └─ core/vol.py:296 sigma_remaining       (W - delta - m) + delta/3   [variance-minutes]
 └─ core/baseline.py:189 BarrierBaseline.fit / :374 attach_baseline -> THE NULL
 └─ core/features.py:303 build_features   42 columns, 4 mechanisms + clock control
 └─ core/model.py:204 fit_model / :92 predict   LightGBM on the baseline logit as init_score
     model_probability = expit(baseline_logit + alpha * raw_correction)

core/decide.py:284 decide()               THE ONLY PLACE A TRADE IS CHOSEN
 ├─ backtest: core/backtest.py:96 decide_window   ONE shared WindowExposure per window
 └─ live:     scripts/live.py:459               FRESH WindowExposure every cycle  ⚠
     price: real ask when quoted, else the calibrated baseline (price_source)

 ├─ backtest -> core/book.py:113 record / :127 settle / :145 bankroll += payout   ✅
 └─ live     -> scripts/live.py:494 act_on
                 write_ticket (get-or-create)  ⚠ caller ignores whether it was new
                 kalshi place_order            ⚠ limit rounded DOWN; FOK; fills assumed
                 open_position                 ⚠ bare insert -> IntegrityError -> exit
                 update_account(bankroll - stake)   ⚠ DEBIT ONLY, never credited

settlement
 ├─ backtest: core/backtest.py:90 settle-before-decide
 └─ live: scripts/live.py:215 settle_due   ⚠ open(t1) and strict > — wrong rule
          reconcile_with_venue :299        ⚠ venue settlements fetched and discarded

serving  core/pg_writer.py -> Postgres -> backend/api/controllers -> endpoints
frontend frontend/src/api/serving.ts -> Live | Decisions | Calibration | Model | Account
```

# Timestamp / Leakage Audit

**Verdict: the research and backtest path is leak-free, proven experimentally.**

Corrupting every bar at or after `decision_time` (5x OHLC, 37x volume, 41x trade
count, all symbols) moved **0 of 42 feature columns and 0 of 9 derived barrier
quantities**, while labels moved 400%. Spiking only the bar at
`event_time == decision_time` — the first forbidden bar — moved **all 51 columns
by exactly 0.0**. The positive control (spiking the *last allowed* bar) moved 37
of 51, so the test has power; every non-responder is individually explained as
clock-only, previous-window, scale-invariant, or sign-only.

A second agent independently hand-recomputed **all 42 features from raw bars**
for one row and got 42/42 exact matches, and confirmed point-in-time correctness
under a separate future-mutation test with its own validated positive control
(19 of 22 columns moved at `as_of = CUT + 1min`).

Fit containment, measured by diffing fitted objects under corruption from
various cut points:

| corrupt from | vol model | seasonality | baseline scale/tail |
|---|---|---|---|
| `test_start` | identical | identical | identical |
| `train_end + 15min` | identical | identical | identical |
| `train_end` | changed (coef up to 1.3e-1, nu 1.09) | ~3e-3 | changed |

The `train_end` sensitivity is `forward_realised_vol` reaching `train_end + 15min`
— harmless at the default 1,440-minute embargo, but it makes the embargo
load-bearing rather than belt-and-braces.

**CV split:** splits on `window_open`, never the row; zero shared windows; all
four offsets always on one side; measured gap 1,446 minutes. Correct.

**Residual leakage, both minor and both real:**
1. `core/model.py:242-244` — the inner validation split has **no purge and no
   embargo**; the gap is 15 minutes against the outer split's 1,440. Early
   stopping *and* `residual_scale` are fitted there. Ships weaker shrinkage than
   intended and makes the `residual_scale >= 0.25` gate easier; does **not**
   inflate reported out-of-sample skill.
2. **Selection leakage.** `core/backtest.py:228` `edge_curve` re-runs the book
   over the same out-of-sample rows at seven `min_edge_pp` values, with the
   docstring "the right value is measured, not guessed" — measured on the test
   folds. Per-offset skill "is how the offset set gets narrowed" (`metrics.py:223`).
   `cost_stress` adds a third. `trial_count()` is printed and never applied.

**Unasserted invariants:** nothing pins the bar index to UTC (an
`America/New_York` index shifts the fitted seasonal peak by 353 minutes, and a
tz-naive index is silently accepted); `assert_no_leakage` validates the embargo
against itself and passes at `--embargo-minutes 0`; the documented point-in-time
`as_of` read is never used (`core/dataset.py:68`).

# Label Correctness Audit

**Verdict: correct, and unusually well reasoned — for the offline label.**

Verified on a hand-checkable ramp and on 173,933 real BTC windows: the `[1:]`/`[:-1]`
shift pairing is consistent, **0 offsets disagreed on a label**, 0 of 3,000
sampled rows leaked, chaining is exact, no rounding anywhere on the label path.
The `>=` comparison matches the venue's `strike_type: greater_or_equal`, and both
ends use the same `(O+H+L+C)/4` estimator over the same minute, so the estimator
bias largely cancels. Timezone verified independently: mean |1-min return| peaks
at 14:00Z (US equity open) — a local-time bug would have put it at 17-19Z.

**Two of the repo's own justifications are wrong, though the code is right:**
- "Ties are not rare on a minute grid" — measured **1 exact tie in 173,937**
  (0.000575%). Wrong by four orders of magnitude. `>=` is still correct because it
  matches the venue, not because ties are common. `Config.tie_resolves_up` is dead
  code; the comparison is hardcoded.
- `windows.base_rate`'s docstring says the base rate should sit *below* 0.5. It is
  **0.5009 (BTC) / 0.5031 (ETH)** — above, exactly as `>=` implies. The docstring
  describes the superseded `>` behaviour and would flag correct output as a bug.

**Basis risk, quantified.** The strike-end error fully cancels (outcome is
`1{(settle - last) + x >= 0}` and `x` is observed), so only the settle end is real
label noise. Simulation puts `(O+H+L+C)/4` at 0.1875·σ₁ = 1.38bp from the true
60-second average → **2.9% of labels differ from the venue's**, cross-checked
empirically (swapping to `close` changes 4.97%, matching a simulated 4.09%
differential). This is a **power problem, not a false-positive risk** — the noise
is symmetric and invisible at decision time, so it can only destroy skill. Cost:
+0.0002 to +0.0018 log loss against the baseline's 0.18 gain.

Transfer function for the venue basis: **~2.1% of labels flip per bp of basis
sd.** P(|15-min return| < 1bp) = 5.3%, < 2bp = 10.6%, < 5bp = 25.4%; median
|return| 11.1bp. The Coinbase-vs-BRTI basis itself is **unmeasurable here** —
`SELECT DISTINCT venue` returns only `coinbase_spot`.

**Window drops are not predictively biased** (checked, because they could have
been): base rate on kept windows 0.5010 ±0.0012 vs 0.5065 ±0.0147 on windows
dropped for one quiet minute — 0.37σ. Volatility runs the *opposite* way to the
concern.

**The live label is a different quantity** — see the CRITICAL finding above.

# Coinbase Integration Audit

Correct and verified: `start` → `event_time` open-minute semantics on all 5.2M
rows; `available_time` exactly `event_time + 60s`; newest-first reversal and
in-run dedup; the venue-keyed UNIQUE constraint plus `INSERT OR REPLACE`; public
unauthenticated candles endpoint; `-USD` product ids matching both the Kalshi
series map and the BRTI constituents; `minute_grid` NaN reindexing; returns
NaN-ed across holes.

Broken: the **301/300 pagination off-by-one** (CRITICAL, above) and the **stored
in-progress candle** (HIGH). Also `get_candles_range` buffers all 2.6M bars in
memory and returns only at the end, so any exception discards everything —
contradicting the documented resumability, and confirmed by the single 29-second
commit observed during the live scrape. The websocket path is dead code for bars
(handlers match the legacy Exchange channel names `l2update`/`snapshot`, not
Advanced Trade's `l2_data`) and its reconnect logic **doubles the product list**
each time; harmless today only because bars are REST-only.

# Feature Engineering Audit

42 columns, all 42 formulas hand-verified against raw bars (42 match, 0 mismatch).

| group | n | verdict |
|---|---|---|
| `vol_state` | 11 | leak-free (proven); `seasonal_ramp` is a pure clock function misfiled here |
| `microstructure` | 8 | leak-free; 3 encode a missing bar as a finite measurement; 1 permanently dead |
| `cross_asset` | 6 | leak-free (proven with a planted spike); no `merge_asof`, no `pivot`+`ffill` |
| `geometry` | 9 | leak-free; every denominator guarded |
| `clock` | 8 | pure functions of offset/window_open; 2 are affine duplicates |

**Column order/name parity: PASS, by construction.** `_feature_matrix` selects
`table.loc[:, list(self.features)]` — by name, in the artifact's own order — so
table order is irrelevant. Proven: shuffling table columns gives `max|dP| = 0.0`;
dropping or renaming a feature raises `ValueError`. 41 float64 + one int32, no
object/category anywhere, so encoding drift is impossible.

Real defects:
- **`trade_count_z_15` is permanently all-NaN** — `trade_count` is never populated
  (0 of 2,617,876 rows). It only looks alive because `tests/conftest.py:84`
  fabricates the column. On the current store **7 of 42 features are all-NaN**.
- **Three features encode missing data as a measurement**: `run_length`,
  `signed_volume_15`, `zero_return_share_60` all use `r.fillna(0)`. A silent zero
  is worse than a NaN, which LightGBM can at least learn a direction for.
- **The venue-strike override leaves 8 geometry features and `peer_displacement`
  stale**, so `abs_z_score != |z_score|` — an invariant that holds to 0.0 on 100%
  of training rows. At 1bp of basis the violation is 3.70 z-units. The booster is
  scored off-manifold.
- **A missing symbol silently redefines the others' features** — `beta_1440`
  shifted 7.7x in one measurement, with no error and no NaN.
- **No finiteness gate in the live path.** `FETCH_MINUTES=1500` against a longest
  real lookback of 1,455 minutes leaves 45 minutes of margin. One silently-NaN
  column shifted probabilities by `max|dP| = 0.0087`, comparable to
  `min_edge_pp = 0.005`, with no exception.

# Model Training Audit

**The init_score design is correct and verified numerically** on lightgbm 4.6.0:
`predict(raw_score=True)` returns the tree sum only; `boost_from_average` is
suppressed when `init_score` is present; log losses base-only 0.48527,
`base + raw` (what the code does) 0.41181, `raw` alone 0.61779,
`base + 2*raw` 0.41488. Right branch, not forgotten, not double-added.

**All three fitted objects and `residual_scale` are inside the fold.**
`ForecastModel.predict` refuses to run without a pre-attached fold baseline logit.

**The pipeline does not manufacture skill.** On 400 days of iid synthetic bars
with no direction and no drift: mean log-loss skill **-0.00002 ± 0.00005**, 2 of 6
folds positive, per-offset skill all ≤ 0. The right answer.

**The baseline is real and understated.** Measured out-of-sample on real BTC:
log loss **0.44946-0.47025 vs 0.69314** for a coin flip — **32-35% better from
arithmetic alone**, against the 26% the docs claim. Base rate 0.4987-0.5014.

**But the baseline's own residual miscalibration is the size of the traded edge.**
It is monotonically **under-confident by 0.44-1.33pp** across the traded bands,
against a cost hurdle of 0.67-1.37pp and a `min_edge_pp` of 0.5pp. Since the model
is architecturally a correction to the baseline's logit, any stable part of that
bias is precisely what it will learn — and it pays in a backtest whose market
price *is* the baseline.

# Validation / Backtesting Audit

**On a provably zero-signal null, 9 of 14 gates passed.** The five that failed
reduce to two facts: skill was not positive, and the book lost money.

| gate | thr | null | assessment |
|---|---|---|---|
| `log_loss_skill` | ≥0 | -0.00002 F | works; `skill_t` is computed and never gated; NaN and "no skill" are the same output |
| `folds_skill_positive` | ≥5/6 | 2 F | 7/64 is right arithmetic on a false independence assumption — true null rate ≈34% at rho 0.7 |
| `calibration_error` | ≤0.02 | 0.0154 P | passed by the null *and* the raw baseline; 4x `min_edge_pp`; all rows, 10pp bins |
| `residual_scale` | ≥0.25 | **0.902 P** | the overfitting detector reads 0.9 on pure noise |
| `control_gain_share` | ≤0.30 | 0.281 P | cleared by 0.019; `clock` is not a control (encodes the offset) |
| `windows_evaluated` | ≥20k | 98,742 P | can never bind (a real run yields ~450,000) |
| `trades` | ≥200 | 660 P | 24-100x too few; all 660 came from one fold |
| `coverage` | ≥0.0005 | 0.0067 P | blocks total abstention; no upper bound |
| `realised_edge_pp` | ≥0 | -3.80 F | best gate in the set — caught a 5.23pp winner's-curse gap |
| `total_return` | ≥0 | -0.275 F | works |
| `sharpe` | ≥0.5 | -5.20 F | correctly annualised on trades placed; sqrt(N) assumes independence |
| `sharpe_implausible` | ≤0 | 0 P | **jointly near-infeasible with `min_edge_pp=0.5`** — a genuine 0.6pp edge at max coverage yields 5.76 and is rejected |
| `max_drawdown` | ≤0.35 | 0.304 P | passed by a null that lost 27.5% |
| `halted` | ≤0 | 0 P | reads the backtest account; live never sets it |

**The structural problem is not any single gate.** `core/decide.py:314` sets
`p_market = baseline_probability`. All eight money gates are therefore downstream
of a synthetic price, and the "edge" they measure is arithmetically the model's own
claimed correction minus costs. The decisive demonstration:

```
a "model" with ZERO forecasting content (isotonic recalibration of the
baseline against its own measured bias), through the real run_book/decide():

  MARKET = baseline (what decide.py assumes):
    4,134 trades / 65,211 windows | $100 -> $194.42 (+94.4%) | Sharpe +3.29
    >>> ALL 14 GATES PASS <<<
  MARKET = baseline + w*(truth - baseline):
    w=0.00 -> +94.42%   w=0.10 -> +44.93%   w=0.25 -> +4.37%   w=0.50 -> 0 trades
```

The venue only has to price away **half** of the null's known bias for the system
to take zero trades. Honest counterweight: an isotonic fitted on the first half
and applied out-of-sample to the second did *not* generalise ($100 → $49.80,
halted, four money gates failed), so the money gates are not a rubber stamp — but
they can only ever ask "does the correction beat the baseline", which is the same
question `log_loss_skill` asks.

**Effective sample size, honestly:** 173,933 windows kept x 4 offsets, correctly
divided out by `effective_observations`. But at rho 0.7 three symbols are ~1.25
effective, so ~219,000 effective window-observations against the 2,102,400 a row
count claims (9.6x). Ample to resolve 0.5pp in aggregate — except the reported
standard error comes from **6 fold means that share 50-83% of their training
data**, which throws the power away and biases the error bars *down*.

# Train vs Live Parity Audit

**`core/` is bit-exact. `scripts/live.py` is not.**

| diff | columns nonzero | worst abs |
|---|---|---|
| real bars, full history | **0 of 52** | **0.0** |
| synthetic, full history | **0 of 52** | **0.0** |
| real bars, 1500-min slice (`FETCH_MINUTES`) | 16 of 52 | 1.1e-12 |

`model_probability` worst diff **2.2e-16**. No warmup break, no `init_score`
asymmetry, no alpha or clipping asymmetry, no dtype or column-order break, no
timezone break. `ScoringBundle` round-trips through joblib with max|Δ| = 0.0 on
every vol coefficient and seasonality factor. `decide()` is identical across 400
rows on every field.

Every divergence is in the live orchestration, not the maths:

| break | measured magnitude |
|---|---|
| `score_live` on the current window | **raises — 12/12 cycles** |
| `floor_strike` override, 1bp basis | median Δ probability **0.82-1.46pp**, max 3.70 (gate is 0.50) |
| same, at live's 25bp warning threshold | median **16.9-24.2pp**, max 64.3pp |
| entries per (symbol, window) | backtest 124 → live **618 (4.98x)**; worst window 2 → 21 |
| settlement outcome | **3.4-3.9% of windows settle the opposite way** |
| one silent `Config` field change | ±0.50pp, unvalidated |

# Execution / Order Safety Audit

**Side mapping is correct** — traced field by field. UP → buy YES off `yes_ask`;
DOWN → buy NO off `no_ask` (the venue's own field, not `1 - yes_bid`); depth for
DOWN reads `yes_bid_size`, which is right because buying NO consumes resting YES
bids. Observed wire body: `{"action":"buy","side":"no","no_price":34,
"count":7,"type":"limit","time_in_force":"fill_or_kill","client_order_id":"ETH-USD-202608231200"}`.
Markets are genuinely resolved by asking the venue and matching on `close_time` —
no ticker is ever constructed from a pattern. Both price encodings and `_fp`
sizes are parsed. No retry on POST. Three order guards all default safe.

**Duplicate real orders: YES, deterministically.** The full chain is in the
CRITICAL finding above. The entire defence is one line —
`client_order_id = f'{symbol}-{window:%Y%m%d%H%M}'`, correctly deterministic and
correctly excluding the offset — and **nothing in this repository verifies that
Kalshi honours it as an idempotency key**; no test asserts the field is even sent,
and the client's own default is a fresh `uuid4` per attempt.

Hostile-condition matrix, all measured against the real client:

| condition | behaviour | dup risk |
|---|---|---|
| POST timeout after the server accepted | `TimeoutError` **propagates** (not a `KalshiError`) → process exits; ticket `new`, no position | **yes** |
| connection reset | same | **yes** |
| HTTP 500 then success | `KalshiError`, exactly 1 request, ticket `skipped` | no |
| HTTP 200, malformed JSON | `JSONDecodeError` is a `ValueError` → ticket `skipped`, "order refused" logged **after a 200** | **yes** |
| HTTP 200, empty body | recorded as a full fill, bankroll debited | assumed fill |
| **FOK that KILLED** | ticket `placed`, **phantom position** | — |
| partial fill 2 of 5 | records **5** | — |
| same window, 2nd cycle | 2 orders, same `client_order_id`, bankroll debited twice | **yes** |
| 2nd `open_position` | `IntegrityError` **uncaught** → exit, after the order | **yes** |
| market unresolved | **position + debit anyway**, 0 orders | phantom |
| `--mode live --dry-run` | **position + debit anyway**, and `--place-orders` still places | phantom |
| one-sided book | trades at **baseline 0.60** against a real 0.20 ask; claimed +8.04pp when truth is **-11.10pp** | mispriced |
| balance as a dollar string | reads 1.37 not 137.42 → `BANKROLL_FLOOR` | no (closed) |

**No process singleton anywhere.** Two traders against one account were measured
sizing a full position each ($9.96 vs $4.98).

# Risk Management Audit

**Present and working:** $25 per-position cap, 5% per position, 0.25 fractional
Kelly (not full), measured-depth cap, never stake > bankroll, 8% per-window
notional, 2 positions/window, 50% ruin floor, `min_edge_pp`, price band, 25pp
disagreement guard, the post-rounding fee-ceiling EV re-check (verified working),
NaN guard, sigma floor, probability clipping, `place_order` refusing without
`live=True`, two flags for real orders, `--require-gates` on by default,
`deployable` check, `fill_or_kill`, hold-to-settle, abstain on unresolved market,
atomic promotion install, API token fail-closed, arg allow-list, loopback ports.

**Absent:** max daily loss · max consecutive losses · **any drawdown kill switch
on the live account** · **any persistent kill switch** · cross-cycle exposure
accounting · total and per-symbol exposure caps · trades-per-hour cap · order
rate limit · **any stale-data guard** · clock-skew guard · **single-instance
lock** · artifact-vs-config validation · artifact↔ledger linkage
(`model_version` is always NULL) · a requirement that a live decision be priced
from a real quote · book sanity check · slippage cap · probability range check ·
exception circuit breaker · loop liveness healthcheck.

`Account.halted`/`halted_reason` are **never written by anything** — the
dashboard's safety chip is structurally incapable of turning on. The `halted`
gate reads the *backtest* account, which `scripts/live.py` never imports.

**Worst-case capital at risk, computed:** $5.00 per position (brute-forced over
the price/probability grid — `max_stake_fraction` binds, not the $25 cap);
$8.00 per window per cycle (measured $7.92 with three symbols); $768/day nominal
= 7.68x a $100 bankroll; **$50 = 50% of bankroll** in ~1h35m if the ruin floor
works. **As the live path is actually written the $8/window cap does not hold** —
`WindowExposure` is rebuilt every cycle, each of up to 12 cycles stakes another
$8, the DB constraint stops only the bookkeeping (after the order is on the wire),
and `restart: unless-stopped` retries. Honest worst case: **the entire bankroll
committed to one 15-minute window in one direction, with every documented cap
bypassed.**

# Data Quality Audit

Snapshot 2026-08-23T05:32Z, mid-scrape (SOL not yet written — that is the scrape
in progress, not a defect).

| metric | BTC-USD | ETH-USD | SOL-USD |
|---|---|---|---|
| rows | 2,617,876 | 2,617,844 | not yet scraped |
| range | 2021-08-24 → 2026-08-23 | same | — |
| missing | 10,124 (**0.385%**) | 10,156 (**0.387%**) | — |
| isolated 1-min holes | 8,721 (86.1%) | 8,714 (85.8%) | — |
| multi-minute runs | 1,403 min / 46 runs | 1,442 min / 48 runs | — |
| duplicates | **0** | **0** | — |
| NULL / NaN / Inf / zero / negative OHLCV | **0** | **0** | — |
| impossible bars (h<l, h<max(o,c), l>min(o,c), c≤0) | **0** | **0** | — |
| zero-volume bars | **0** | **0** | — |
| `available_time - event_time` | exactly 60s | exactly 60s | — |
| `quote_volume` / `trade_count` | **100% NULL** | **100% NULL** | — |
| P(UP), unique windows | **50.091%** (173,937) | **50.306%** (173,926) | — |

No asset contamination (price ranges disjoint by 3x). No legacy tables or coins
from the previous perp-era project. All price outliers are real market events
(2021-12-04 BTC flash crash, 2022-11-08 FTX, 2025-02-03 ETH tariff crash) — no
bad-data outliers. Timestamps naive-UTC and monotonic.

**86% of the missing minutes are the client-side pagination bug, not untraded
minutes** — see the CRITICAL finding. `CLAUDE.md` is wrong about its own data.

Schema tables `model_runs` and `calibration` exist and have **zero writers**
(`record_model_run` and `write_calibration` are never called), while the API and
the Model/Calibration dashboard pages read exclusively from them — so those tabs
are permanently empty regardless of real promotion activity. The `backend` service
also never mounts the `trader_models` volume, so the filesystem ledger is not a
workaround.

# Security Audit

| location | type | tree/history | rotate? |
|---|---|---|---|
| `b70c78c:backend/api/.env` | Coinbase key (95c) + EC secret (234c) + DB URL | **history, on origin/main** | **YES** |
| `6097ed1:frontend/.env.local` | `VITE_API_TOKEN` | **history, on origin/main** | **YES** |
| `./.env` (untracked, gitignored) | POSTGRES_PASSWORD (5 chars), Coinbase, Kalshi, API_TOKEN (**3 chars**) | tree only | weak — regenerate |
| `~/…/kalshi.pem` (outside repo) | RSA signing key | not in git | mode 664 — chmod 600 |

`joblib.load` on the model artifact has **zero integrity check** — no hash, no
signature, no allowlist. Not reachable from HTTP today (the API cannot control the
load path, and `validate_job_args` forbids `/` outright), but a trading process
holding exchange credentials unpickles whatever is at that path.

`POST /jobs/scripts.promote {"args":["--force","--reason","x"]}` returns 200 —
`FLAG = ^--[a-z][a-z0-9]*(-[a-z0-9]+)*$` accepts `--force`. Combined with
`_refuse_if_blocked` testing `installed` rather than `passed`, that is a remote
path from one authenticated request to a gate-failing model being traded, behind
a 3-character token that is also inlined into the client bundle.

Dependencies: `backend/api`'s web stack is exact-pinned, but `lightgbm`,
`scikit-learn`, `scipy`, `joblib`, `cryptography` and `coinbase-advanced-py` are
floating or unconstrained in both manifests, with no lock file — the packages
that deserialize the model and sign live orders. The frontend is reproducible
(`npm ci` + committed lock, 0 vulnerabilities offline).

**Positive controls confirmed:** `require_token` fails closed (503 unset, 401
wrong) with `hmac.compare_digest` on bytes; no `shell=True`, `os.system`, `eval`
or `exec` anywhere; no SQL string interpolation; no `verify=False`; timeouts on
both HTTP clients; all compose ports bound to 127.0.0.1; `POSTGRES_PASSWORD` has
no default; `.dockerignore` excludes env files in all three services; committed
compose config cannot start live order placement.

# Test Coverage Audit

Real numbers: **230 passing** in ~170s (`backend/trader`) and **28 passing** in
1.2s (`backend/api`). AGENTS.md's "207 tests in 26s" is stale.

The suite is good at the maths and blind to the orchestration. `tests/test_live.py`
is 96 lines covering `current_window`, `choose_offset`, `FETCH_MINUTES`, the series
map and two argparse guards. **There is no test of `run_cycle`, `act_on`,
`settle_due`, `fetch_quotes`, or `reconcile_with_venue`** — which is precisely the
set of functions carrying every CRITICAL finding in this report.

Two specific anti-patterns let the CRITICALs ship green:

1. **Tests that assert on source text, not behaviour.** Eight
   `inspect.getsource` substring assertions (5 in `test_kalshi.py`, 3 in
   `test_backfill_windows.py`). For example:
   ```python
   def test_reconciliation_writes_the_venues_balance_not_ours():
       source = inspect.getsource(reconcile_with_venue)
       assert 'balance drift' in source
       assert 'update_account(bankroll=venue_balance)' in source
   ```
   This passes while the settlement reconciliation it describes is unimplemented.
   `test_kalshi.py:204` asserts `'place_order' not in source`.
2. **Fixtures that guarantee the case the test claims to cover cannot occur.**
   `test_score_live_reports_no_outcome_for_an_unsettled_window`
   (`test_features_and_model.py:216`) passes `window_index[-3]` — a fully settled
   window. `test_every_declared_feature_is_produced` passes only because
   `conftest.py:84` fabricates the permanently-NULL `trade_count` column.
   `tests/test_backfill_windows.py:499` encodes the false "single missing minutes
   are untraded minutes" premise, so fixing the pagination bug breaks CI.

Genuine mutation guards do exist — `test_a_tie_resolves_up` correctly failed when
a `>=`→`>` mutation was transiently present during this audit.

# Profitability / Expected Value Audit

**The EV arithmetic is the strongest part of the codebase.** There is no `p > 0.5`
anywhere in the decision path. Derived independently and compared term by term:

```
correct:  p > (m+h) + ceil(100 * 0.07 * C * (m+h) * (1-(m+h))) / (100*C)
```

| term | code | verdict |
|---|---|---|
| crossed price `m+h` | `costs.py:160` | correct |
| fee at the **crossed** price | `costs.py:162` | correct |
| `EV = q - cost`, `(1-q)` on NO | `decide.py:222` | correct |
| real ask not double-charged the spread | `decide.py:213` | correct |
| NO ask = `no_ask`, depth = `yes_bid_size` | `kalshi_client.py:137,141` | correct |
| Kelly `(q-c)/(1-c)` | `decide.py:172` | correct (re-derived) |
| per-order ceiling re-check | `decide.py:379` | present, wrong 3 ways |

The required-edge table in CLAUDE.md matches the code to 0.006pp
(2.2498 / 1.9558 / 1.3678 / 1.1018 / 0.8008), and `trade_fee(1, 0.50) = $0.02`
against a schedule value of $0.0175 — the +14.3% per-order ceiling the docs claim.

Three real defects in the re-check: it adds the half-spread even when the price is
a real ask (measured $4.415 recorded against $4.390 charged); it prices the fee at
the mid rather than the crossed price; and it demands `EV > 0` rather than
`EV >= min_edge_pp`.

**The documented 83c fee/half-spread crossover is stale in six places.** At the
current 0.5c default it is **92.26c**; at 83c the fee ($0.0099) is still ~2x the
half-spread. `tests/test_costs.py:99` derives it correctly and contradicts the
prose.

**The band is side-symmetric but not symmetric about 0.5, and the asymmetry runs
the wrong way.** `[0.05, 0.97]`: all 11 mirror pairs agree, so the failure
CLAUDE.md warns about is absent — but the 2pp gap admits 96-97c favourites (edge
+1.8/+2.3pp, where a 1pp calibration error destroys **43%** of the gross) and
refuses 3-4c longshots (edge +5.2/+6.3pp, where it destroys 1%).

**The headline: the permitted calibration error is 4x the required edge, and the
two files contradict each other.** `min_edge_pp = 0.5pp` (`config.py:226`, whose
own comment says it exists to guard calibration error) against
`calibration_error <= 2.0pp` (`metrics.py:285`). A uniformly 2pp-overconfident
model taking 0.5-2.0pp claimed edges has mean claimed EV **+1.250pp** and true EV
**-0.750pp**, with **0.0000% of trades genuinely positive**.

**And no measurement in this repository can establish profitability.**
`decide.py:314` sets `p_market = baseline_probability`. `market_probability` is
written once per live decision and read by nothing. `measure_book.py` can record
quotes but has no compose entry, records no outcome, and has never been run. Order
tickets cover only the ~6% of windows that traded — a selected sample. **No Kalshi
quote history is stored and none is being recorded.** The system can currently
demonstrate that a model beats an analytic formula; it cannot demonstrate that the
formula beats the market.

---

# What has been fixed

Eight commits on `audit/critical-fixes`. The suite went from 230 tests to 345, and
every fix below was checked by reintroducing the bug and confirming a test fails.

| finding | status |
|---|---|
| Live path cannot score the current window | **fixed** — `build_windows(include_unsettled=True)`; verified at every offset, and withheld when the feed is short |
| Live re-enters the same (symbol, window) every cycle | **fixed** — exposure seeded from committed positions and tickets; `open_position` idempotent; Postgres advisory lock for single-writer |
| Paper bankroll never credited a win | **fixed** — payout credited inside `settle_position`'s transaction; increments via one relative `UPDATE` |
| Live settles on the wrong rule | **fixed** — `bar_mean` of the minute ending at `settle_time`, `>=`; venue settlements applied as authoritative |
| Venue settlements fetched and discarded | **fixed** — returned and applied. *The "reverse reconciliation added" half of this claim was false: it read `row['position']`, a field V2 does not send, so it could never fire. See the Addendum.* |
| Unresolved market books a phantom position | **fixed** — `Reason.NO_QUOTE`; live prices from the book or abstains |
| Fills assumed, not read back | **fixed** — `status`/`remaining_count`/`taker_fill_count` parsed; position written from the fill |
| Coinbase pagination destroys 1 minute in 301 | **fixed** — span is `(limit - 1) * tf`; `--min-gap-minutes` defaults to 1 |
| In-progress candle stored | **fixed** — dropped by `available_time <= now` |
| NaN pooled into the top reliability bin | **fixed** — excluded and counted; `non_finite_rows == 0` gate |
| `scripts/baseline.py` gate fails open | **fixed** — refuses on unmeasured folds |
| Calibration gate blind to the traded band | **partly fixed** — `calibration_max_deviation` gate, 2pp tail bins, empty-bin NaN fixed. See the caveat below |
| `residual_scale` returns scipy's bracket seed | **fixed** — non-convergence and non-finite input raise; reads 0.0000 on a null |
| Inner split unpurged; alpha shares rows with early stopping | **fixed** — embargoed, and separate blocks |
| `--dry-run` never read | **fixed** — mutually exclusive with `--place-orders` |
| Live account renders as paper | **fixed** — `ensure_account` refuses a mode change rather than silently inheriting |
| `--require-gates` tests `installed` not `passed` | **fixed** |
| `--force` reachable over HTTP | **fixed** — `FORBIDDEN_FLAGS` in `validate_job_args` |
| Limit prices truncated below intent | **fixed** — rounds up; slippage capped at a share of the edge |
| Half-spread charged on top of a real ask | **fixed** |
| Sharpe averages per-trade ratios | **fixed** — dollars per calendar day; `sharpe_per_trade` retained |
| Price band asymmetric | **fixed** — `[0.05, 0.95]`, enforced in `__post_init__` |
| Embargo validated against itself | **partly fixed** — `Config` refuses `embargo < window_minutes`; `assert_no_leakage` is still self-referential |
| No stale-data guard | **fixed** — bar age, missing symbol, and remaining-window floor |
| No probability range check | **fixed** |
| No circuit breaker; `halted` never written | **fixed** — daily loss and consecutive losses, persisted |
| Artifact loaded on trust | **fixed** — `verify()` at load: booster/feature agreement and material config drift |
| `trade_count_z_15` permanently NaN | **fixed** — removed; the fixture no longer fabricates it |
| `model_runs`/`calibration` never written | **fixed** — `scripts/promote.py` mirrors best-effort |
| Store returns local time | **fixed** — reads pinned to UTC |
| `--start`/`--end` crash on pandas 3.0 | **fixed** |
| `us_equity_hours` EDT-only | **fixed** — converted through `zoneinfo` |
| `balance()` single encoding, silent zero | **fixed** — both encodings, NaN on failure, reconcile refuses to write it |
| Migrations swallow every exception | **fixed** — tolerated on SQLite only |
| Dependencies unpinned | **fixed** — exact pins, agreement between containers enforced by test |
| Secrets in pushed history | **NOT FIXED — requires rotation by the operator** |
| Backtest prices against the baseline | **NOT FIXED — architectural, see below** |
| No market-implied benchmark | **NOT FIXED — needs quote history collected first** |
| `core/backtest.py` at 0% coverage | **NOT FIXED** |

## Three that are deliberately not fixed

**The counterparty.** `core/decide.py` prices the counterfactual market as the
calibrated baseline, and the model is fitted on that baseline's logit as
`init_score`. So the backtest's "edge" is arithmetically the model's own claimed
correction minus costs, and there is no adverse-selection term: the backtest
cannot show a loss caused by the market being smarter than the model. Changing the
price to something pessimistic would be inventing a counterparty rather than
measuring one. The fix is to collect real quotes, which is why `market_probability`
is now written on every live decision.

**The 0.5pp/2pp tension.** `min_edge_pp` is 0.5pp and the calibration gates
permit more error than that. This is not an oversight that can be closed by
tightening a threshold: 500 rows in a bin at p=0.9 carry a 1.3pp standard error,
so no calibration measurement on this sample can *resolve* 0.5pp. Either
`min_edge_pp` rises above the measurable error or the gate is understood as
bounding the damage rather than certifying the edge. That is a decision about risk
appetite, not a bug, and it is left to the operator with the numbers stated.

**Secrets rotation.** Nothing in a repository can un-leak a pushed blob.

---

# Recommended Fixes

Grouped as the plan requires. Everything marked *(done)* is on
`audit/critical-fixes`; `AUDIT_FIX_PLAN.md` carries the detail and the ordering
constraints for what remains.

### P0 — Must fix before any live trading
- Rotate the Coinbase key/secret, the Postgres password and `API_TOKEN`. **Open — operator action.**
- Score the window being decided *(done)*, one entry per (symbol, window) *(done)*, credit settlements *(done)*, settle on the trained rule *(done)*, book only confirmed fills *(done)*, honour `--dry-run` *(done)*, correct the order envelope *(done)*.
- **Open:** run `scripts.live --mode paper --loop` for a week and read the funnel before considering `--place-orders`. Every fix above is verified by tests and by construction; none has been verified by a cycle that actually ran against the venue.

### P1 — Must fix before increasing capital
- Coinbase pagination *(done)*, in-progress candle *(done)*, NaN handling *(done)*, calibration gates *(partly — see the caveat)*, inner-split purge *(done)*, stale-input refusal *(done)*, artifact verification *(done)*, circuit breakers *(done)*.
- **Open:** re-scrape or `--fill-gaps` the whole store, then retrain. Every model fitted before the pagination fix was fitted on a grid missing one minute in 301, with the loss phase fixed in training and moving in serving.
- **Open:** collect Kalshi quote history and measure the model against it. Until that exists, no number in this repository speaks to profitability.
- **Open:** decide the `min_edge_pp` question above, explicitly, and write the reasoning down.

### P2 — Important reliability improvements
- Half-spread *(done)*, side-adjusted reporting *(done)*, Sharpe *(done)*, `n_features_populated` *(done)*, migrations *(done)*, `balance()` *(done)*, serving-store writers *(done)*, `trade_count_z_15` *(done)*, two source-text tests *(done)*, `--start/--end` *(done)*, UTC pinning *(done)*, `us_equity_hours` *(done)*.
- **Open:** `core/backtest.py` has 158 statements at 0% coverage — no test imports it. It needs a `slow`-marked end-to-end walk-forward.
- **Open:** `run_cycle`, `act_on` and `reconcile_with_venue` are still untested end to end; `scripts/live.py` sits at ~19%.
- **Open:** six remaining `inspect.getsource` assertions in `test_kalshi.py` and `test_backfill_windows.py`.
- **Open:** reclassify `seasonal_ramp` — it is a deterministic function of minute-of-day and offset, filed under `vol_state`, so its gain is not counted against the control gate. And decide whether offset-dependent `clock` columns belong in a control at all, since per-offset recalibration is legitimate.
- **Open:** `max_disagreement_pp` stays at 25.0. Tightening it to 8.0 was tried and reverted — a sigma disagreement at a 0.88 quote legitimately moves P(up) to 0.70. The real protection is the price band plus requiring a real quote, both now in place.
- **Open:** a loop-liveness healthcheck. The trader healthcheck tests a Postgres connection, so a crash loop is invisible to compose.

### P3 — Cleanup / maintainability
- Dependency pinning *(done)*, documentation corrections *(done)*.
- **Open:** `Mapped[]` typing on `core/pg_writer.py` — 43 of mypy's 106 errors are in the file that carried the bankroll bug, so type checking could not see it.
- **Open:** a `ruff` config and CI. There is no CI at all, which means `test_orm_parity.py` is not actually enforced on merge.
- **Open:** delete or quarantine the perp-era code — `RedisQueue`, funding/OI models and tables, the `--live` scrape path, `write_features`/`read_features`.
- **Open:** rename one of the two `serving.py` modules so mypy can resolve imports.

---

# Final Adversarial Review

Paid to argue against deployment, here are the five strongest arguments.

**1. Nothing in this repository can measure whether the strategy is profitable,
and the number it does produce is close to tautological.** The backtest's market
price *is* the calibrated baseline, and the model is fitted on that baseline's
logit as `init_score`. So "edge" is the model's own claimed correction minus
costs, with no term for the market being right when we are wrong. Two independent
demonstrations: a model that knows the truth exactly earns +2219% against a
baseline-priced counterparty, +191% against a half-informed one and **zero
against an informed one**; and a "model" with no forecasting content at all — an
isotonic recalibration of the baseline against its own measured bias — passed all
fourteen gates with +94% and a Sharpe of 3.29. The venue needs to have priced away
only half of the null's known 0.44–1.33pp bias for the system to take no trades,
and for a liquid BTC binary that is the likely case, not the pessimistic one.

**2. The one real skill number is explainable four ways before it is alpha.** The
five-year BTC walk-forward reports +0.000897 ± 0.000240, 6/6 folds positive. But
consecutive folds share 50–83% of their training windows, so at ρ≈0.7 "6 of 6" is
a **22%** event under the null and t > 3.74 an **18.9%** event — not the 1.6% and
0.68% independence implies. Four of six folds were scored with an unfitted
shrinkage constant. The correction is largest at offset 3 and negative at offset
12, which is where the null is worst calibrated and the exact opposite of what the
barrier framing predicts. And two of the top four features by gain are the clock
control, with `hour_sin` at #2 — the repo's own documented failure mode, at a
threshold too loose to catch it.

**3. Nothing has ever run.** The live loop could not score a window, the paper
bankroll could not go up, and the paper container crashed on the second cycle of
its first window. So there is no operational history at all: no fill ever read
back, no settlement ever reconciled, no drift ever observed, no idea what the
venue actually does with a duplicate `client_order_id`. Every fix in this branch is
verified by a test and by construction, and not one is verified by a cycle that
ran. The first week of paper trading will find things this audit did not.

**4. The training data was wrong in a way that differs between training and
serving.** One minute in 301 was destroyed for five years. The paginator's phase
is fixed during a backfill and moves on every live fetch, so the two lose
different minutes — 0.332% of live cycles lose the exact minute that becomes
`last_price`. Everything fitted before this branch was fitted on that grid. The
store needs re-filling and the model refitting before any number is worth reading,
and that has not happened yet.

**5. The measurement apparatus was itself unreliable, which is worse than a
model being wrong.** A 0.03% data hole turned five of six folds' metrics into NaN
while the Phase 1 gate printed "gate passed". The overfitting detector returned
scipy's golden-section bracket point and cleared its own threshold. The
calibration gate could not see the band where the money goes. The Sharpe gate was
policing a quantity that can carry the opposite sign from the account. Those are
fixed, but they were all *simultaneously* true, and the system reported 12 of 14
gates passing in that state. The prior on further undetected measurement error
should be high.

## What would need to exist before I would call it production-ready

1. **Kalshi quote history, and skill measured against it.** Several months of
   `--mode live --dry-run` writing `market_probability` and `outcome` on every
   window, traded or refused. Positive log-loss skill *against the market's own
   price* is the only result that matters, and no amount of skill against
   `F(x/σ)` substitutes for it.
2. **A store re-filled after the pagination fix, and a model refitted on it**, with
   the walk-forward rerun and the `non_finite_rows` gate at zero on its own merits.
3. **A month of paper trading with no crash, no unexplained balance drift, and a
   funnel that matches the backtest's.** Coverage, rejection histogram and realised
   edge should agree between the two within their error bars; if live coverage is
   materially lower, the difference is the thing to understand before sizing.
4. **`min_edge_pp` reconciled with the measurable calibration error**, in writing —
   including the possibility that the answer is "this sample cannot support a
   0.5pp gate, so the strategy needs a bigger edge to be tradeable."
5. **The control question settled.** Either `hour_sin` and `quarter_of_hour` stop
   being top-four features, or the control gate is redesigned so that legitimate
   per-offset recalibration does not count against it while genuine clock
   dependence does. As it stands the gate can neither pass a good model
   confidently nor fail a clock-driven one.
6. **`core/backtest.py` covered, and `run_cycle`/`act_on` tested end to end** against
   a mocked venue, including the hostile cases in the execution audit above.
7. **CI.** There is none, so `test_orm_parity.py` — which exists because that mirror
   already drifted once by a factor of ten — is not enforced on merge.
8. **Credentials rotated**, and a pre-commit hook backing the two new tests.

None of that is a reason the hypothesis is wrong. The barrier reframing is right,
the label reproduces the venue's published rule, the feature pipeline is provably
free of lookahead, and the baseline beats a coin flip by 32–35% from arithmetic
alone — measured, and better than the docs claimed. It is a good foundation with
an unverified conclusion resting on it, and the distance between those two is
larger than the gate report suggests.

---

# Edge Investigation (2026-08-23, 326-day subset)

Run on the most recent 326 days while the five-year repair was still in flight, so
treat every number as provisional: one regime, a seasonality factor that fell back
to flat (45.3 days against a 60-day minimum), and 6 folds sharing 50-83% of their
training data. The *directions* are what matter.

## The Phase 1 null

```
pooled log loss 0.44948 against 0.69315 for a coin flip = 35.2% better
base rate 0.4996 | worst-fold ECE 0.01410 | worst max bin deviation 0.0397
86 rows (0.027%) unscoreable, from one 6.5-hour venue outage on 2026-05-08
```

The barrier arithmetic works and is understated in the docs (which claim 26%).

## Skill is real, and it is not what the docs say it is

15 of 16 gates pass with the shipped config (`max_drawdown` 58.4% fails). But
three of the passes are within a hair of their thresholds
(`calibration_max_deviation` 0.03883/0.04, `control_gain_share` 0.27852/0.30),
`residual_scale` reads 1.046 — above 1, so the fit wants to *amplify* the
correction rather than shrink it — and predicted edge +2.00pp came in at +0.99pp
realised. So the gate report alone would be misleading, which is the whole reason
for what follows.

**It is not the model recalibrating the baseline.** This was the audit's leading
hypothesis and it is refuted. Give the null a free in-fold recalibration and score
it out of sample:

```
                     mean        se        t     folds+
model_skill      +0.000287  0.000128   +2.25     6/6
platt_skill      -0.000022  0.000018   -1.21     3/6
iso_skill        -0.000274  0.000055   -4.95     0/6
model_beyond_platt +0.000309 0.000114  +2.72     6/6
model_beyond_iso   +0.000562 0.000117  +4.80     6/6
```

A 2-parameter map gains nothing and a monotone map actively loses. The baseline's
3.97pp worst-bin deviation is bin noise, not an exploitable bias.

**It is not the clock.** Measured directly rather than through a gain share:

```
groups                            skill        se       t  folds+
all five                      +0.000287  0.000128   +2.25     6/6
clock only (the CONTROL)      -0.000008  0.000029   -0.26     2/6
no clock                      +0.000315  0.000149   +2.12     6/6
cross_asset only              +0.000183  0.000054   +3.39     6/6
geometry only                 +0.000082  0.000071   +1.15     5/6
vol_state only                -0.000101  0.000101   -1.00     2/6
microstructure only           +0.000071  0.000056   +1.27     5/6
```

The control behaves exactly like a control, and removing it slightly *helps*. So
`control_gain_share` at 0.279 was a false alarm: a high LightGBM gain share means
splits were spent there, not that the feature forecasts. **The gate is measuring
the wrong quantity** — an ablation is the real test and should be what gates.

**It is cross-asset lead-lag, at the earliest offset.** `cross_asset` alone is the
strongest single group, and it is the one mechanism with independent prior
support: the archive records a cross-sectional residual signal at h=4h at +0.0186,
t=4.54, 6/6 folds over five years and four regimes. Two unrelated horizons
pointing at the same mechanism is the most encouraging result here.

Per-offset, `cross_asset` alone:

```
offset      n   mean_skill  folds+   mean_abs_correction_pp
     3  79765     0.000368     6/6                  0.782
     6  79769     0.000265     5/6                  0.630
     9  79770    -0.000004     3/6                  0.446
    12  79770     0.000103     4/6                  0.242
```

**This contradicts the design thesis stated in `CLAUDE.md` and printed by
`scripts/evaluate.py`** — that the edge should peak mid-window where
`|x|/sigma ~ 1` and decay late, because that is where `P` is most sensitive to a
sigma error. It peaks *earliest* and is dead by offset 9. That shape is wrong for
a sigma-error mechanism and right for lead-lag: a BTC move needs time for ETH and
SOL to follow, so twelve remaining minutes express it and three do not. Together
with `vol_state` alone being *negative*, the sigma story does not survive and the
lead-lag one does.

## The money numbers are noise at this sample size

Six configurations, forecast skill against realised money:

```
configuration                 skill      t   f+  trades   return  sharpe  realEdge   maxDD
(3,6,9,12) all groups     +0.000287  +2.25  6/6   4,553 +212.18%   +2.81    +0.99  58.38%
(3,6)      all groups     +0.000365  +1.94  5/6   1,123  -50.49%   -4.23    -0.14  51.85%
(3,)       all groups     +0.000419  +3.14  6/6   5,290 +193.90%   +3.11    +0.52  28.16%
(3,6,9,12) no clock       +0.000315  +2.12  6/6   3,010 +272.66%   +3.52    +3.30  43.21%
(3,6)      no clock       +0.000391  +1.81  5/6   7,060 +302.94%   +3.41    +0.84  48.43%
(3,6)      cross only     +0.000352  +2.65  6/6   4,485 +146.42%   +1.58    +0.88  38.19%
```

**Skill and money are decoupled.** `(3,6) all groups` has *higher* forecast skill
than the shipped configuration and loses half the account. At essentially constant
skill the return spans −50% to +303%, and every configuration draws down 28-58%,
with several tripping the ruin floor in April 2026.

The operational conclusion is a discipline rule: **do not select a configuration
on these money figures.** That is precisely what `core/backtest.py:edge_curve`
invites with "the right value of `min_edge_pp` is measured, not guessed" — measured
on the same out-of-sample rows, with this much dispersion. Narrowing on *skill and
mechanism* is defensible; narrowing on return is not.

## What this changes

* Offsets 9 and 12 carry no skill. `(3,)` or `(3, 6)` is the defensible set, chosen
  on skill and on the lead-lag mechanism rather than on return.
* `clock` measurably contributes nothing to the deployed model and should be kept
  as an *ablation* control rather than a feature group whose gain share is gated.
* `vol_state` contributes nothing on this subset, which is worth understanding
  before adding more volatility features.
* None of this is skill against the market. That still requires the quote history
  the new `predictions.market_probability` / `market_ask_*` / `outcome` columns
  exist to accumulate.
* Seven group trials plus six configuration trials on one subset is a search, and
  nothing here carries a multiple-testing correction. The five-year run on the
  repaired store is the test.

## The sizing knobs are not what they look like

`max_drawdown <= 0.35` was the only gate the shipped configuration failed, at
58.4%. Two measurements, and both refuted what I expected.

**`max_stake_fraction` is close to inert.** Cutting it fivefold barely moves the
drawdown and leaves return and trade count almost unchanged:

```
stake frac  window cap  trades     return  sharpe   maxDD  realEdge
      0.05        0.08   4,553  +212.18%   +2.81  58.38%    +0.99
      0.03        0.05   4,553  +214.86%   +2.96  52.79%    +0.99
      0.02        0.04   4,552  +234.46%   +3.45  48.79%    +0.98
      0.01        0.02   4,537  +187.14%   +3.64  43.47%    +0.95
```

Fractional Kelly binds first: at a 1-2pp edge, `0.25 * (q-c)/(1-c) * $100` is
roughly $0.50-$1.00 against a $5 cap.

**`kelly_fraction` is secretly an edge filter.** It is the real lever, and not for
the reason it appears:

```
kelly frac  trades     return  sharpe   maxDD  realEdge  gate
      0.25   4,553  +212.18%   +2.81  58.38%    +0.99   FAIL
      0.15   3,148  +138.28%   +3.25  32.19%    +2.23   pass
      0.10   1,941   +85.48%   +3.34  20.84%    +3.32   pass
      0.05     567   +22.87%   +2.00   8.64%    +3.73   pass
```

Sizing should change how much is staked, not how often — yet trades fall eightfold
and realised edge per contract nearly quadruples. The rejection histograms say why:

```
                     kelly 0.25  kelly 0.10  change
below_min_contracts        1813        8218   +6405
traded                     3221        1941   -1280
edge_below_gate          242571      242571       0
price_out_of_band         63954       63954       0
```

`edge_below_gate` is **identical**, so `min_edge_pp` is not doing the filtering.
`decide()` floors the stake to whole contracts, so a smaller Kelly fraction pushes
marginal trades under one contract and refuses them. The drawdown falls and the
per-contract edge rises because the *survivors are the higher-edge trades*, not
because the sizing got safer.

So `kelly_fraction` and `min_edge_pp` are coupled, and the repository documented
them as independent. Anyone lowering Kelly to control drawdown is also raising the
effective edge threshold, and would attribute the improvement to the wrong cause.
Documented in `core/config.py` and `CLAUDE.md`.

**And this is where the discipline rule bites on my own analysis.** The table above
makes `kelly_fraction = 0.10` look strictly better — it passes the drawdown gate,
Sharpe rises from 2.81 to 3.34, and realised edge triples. Choosing it on those
grounds would be exactly the selection this report argues against two sections
earlier, on the same 326 days where return spanned -50% to +303% at constant
skill. What is defensible is the *mechanism*: the integer floor makes Kelly a
selectivity control, and a lower setting trades coverage for concentration. Which
point on that curve to take is a risk-appetite decision to make deliberately, on
the full five years, and to write down.

## The pagination repair, verified

`--fill-gaps` (itself fixed, since it had been fetching nothing) recovered 8,672
minutes for BTC and 8,667 for ETH. The signature is gone:

```
             before                                after
BTC   8,721 isolated singles, 98.9% at 301   ->   49 singles, 0.0% at 301
ETH   8,714 isolated singles, 98.9% at 301   ->   47 singles, 0.0% at 301
```

The residual 1,452 (BTC) and 1,489 (ETH) minutes are genuine and unrecoverable:
46-48 multi-minute runs that are real venue outages — 391/394 min on 2026-05-08,
349/350 min on 2025-10-25, 277/278 min on 2023-03-04, each hitting all symbols
simultaneously — plus ~48 isolated minutes with no periodic structure, which is
what a genuinely untraded minute looks like.

So the original claim in `CLAUDE.md` was *directionally* right about one thing and
wrong about the proportion: untraded minutes do exist, and they were 0.5% of the
shortfall rather than most of it. The 2026-05-08 outage is also the source of the
86 unscoreable rows the `non_finite_share` gate now reports, since a 6.5-hour hole
leaves the 240-minute lookback unfillable for about two hours afterwards.

---

# Edge Investigation, five years (7,875,926 bars, 2021-08 to 2026-08)

Re-run after the store repair, on 1,825 days across four regimes. Every direction
from the 326-day subset held, and two of them strengthened enough to change what
they mean.

## Phase 1 passes, and the null is well calibrated *in the logit*

```
pooled log loss 0.45688 against 0.69313 = 34.1% better from arithmetic alone
base rate 0.5030 | worst-fold ECE 0.01268 | worst max bin deviation 0.0219
375 rows (0.021%) unscoreable | all 6 folds measured
```

The 0.0397 worst-bin deviation that nearly failed the new
`calibration_max_deviation` gate on the subset fell to **0.0219** on 16x the data,
confirming it was small-sample bin noise.

The null is also regime-dependent, which is worth holding onto: the fitted tail
runs `nu=15.03` in fold 0 (nearly Gaussian) to `nu=4.03` in fold 3 (very fat), and
the offset-3 scale 1.197 to 1.462. "The null" is not one object across five years,
and skill is measured as a difference against it.

## The skill is not baseline recalibration — settled

```
                       mean         se        t    folds+
model_skill        +0.001011   0.000177   +5.70      6/6
platt_skill        -0.000055   0.000088   -0.62      3/6
model_beyond_platt +0.001066   0.000098  +10.90      6/6
model_beyond_iso   +0.001085   0.000157   +6.93      6/6
platt_slope        +0.992872   0.008588              6/6
```

**The Platt slope is 0.9929 +/- 0.0086 — indistinguishable from 1.0.** So the
+0.5 to +0.8pp observed-above-predicted pattern visible in the reliability table
is *not* a logit-scale bias, and there is nothing for a monotone map to harvest.
A 2-parameter recalibration of the null captures **-5.4%** of the model's skill and
a monotone one **-7.3%**; the model's advantage over the *recalibrated* null is
larger than over the raw one.

This was the audit's leading hypothesis and it is now refuted on the full sample,
having also been refuted on the subset. It is closed.

## The mechanism, on four regimes

```
groups                            skill        se       t  folds+  trees
all groups                    +0.001011  0.000177   +5.70     6/6    302
all minus clock               +0.000737  0.000151   +4.88     6/6    318
cross_asset alone             +0.000427  0.000041  +10.40     6/6    188
microstructure alone          +0.000230  0.000035   +6.53     6/6    169
geometry alone                +0.000172  0.000134   +1.29     4/6    174
clock alone (the CONTROL)     +0.000004  0.000016   +0.24     3/6     37
vol_state alone               -0.000001  0.000011   -0.07     4/6     24
```

* **The control is zero.** `clock alone` scores +0.000004 on 3 of 6 folds — a coin
  flip. Removing the clock costs 27% of the total, but that is *conditioning*, not
  direction: time of day genuinely predicts volatility and spread behaviour that
  other features measure. The individual groups also sum to +0.000832 against
  +0.001011 for the full set, a **positive interaction of +0.000179**, which is the
  same story. The benign explanation of the two, and the reason the ablation reports
  each group alone rather than trusting a difference.
* **`cross_asset` is the mechanism** — 42% of total skill, t=+10.40, 6/6 folds.
  Third independent confirmation, after the subset and the archive's h=4h
  cross-sectional residual (+0.0186, t=4.54, 6/6 over five years).
* **`microstructure` is a real second contributor** — 23%, t=+6.53, 6/6. The subset
  had it at t=+1.27 and lacked the power to see it.
* **`vol_state` is exactly zero** — -0.000001 on 24 trees. The sigma-disagreement
  thesis contributes nothing to the *model*. It does its work inside the null,
  which is why the null beats a coin flip by 34%.
* **`geometry` is weak and inconsistent** — 4/6 folds, t=+1.29, unchanged from the
  subset.

Skill also tripled with more data (+0.000287 -> +0.001011) and the tree count went
51 -> 302. Overfitting shrinks with more data and three added regimes; this grew.

## What still is not established

Every number above is skill against `F(x/sigma)`, a formula in this repository.
`scripts/market_benchmark.py` exists and correctly reports that no window has both
a recorded venue quote and an outcome. A model can beat the formula by 0.001 nats
and lose to a liquid book, and nothing here distinguishes those cases.

And read every t against the fold correlation, not a normal table: six expanding
folds share 50-83% of their training windows.


---

# Correction: the credential exposure is MEDIUM, not CRITICAL

I rated this CRITICAL from the shape of the values and the fact that they sit on
`origin/main`. The operator states the Coinbase key is **read-only scope**, which
removes the loss vector the rating was built on — a view-only key cannot trade or
withdraw. That is information the repository does not contain and I could not have
inferred from it.

What I could check, by SHA-256 prefix rather than by value:

```
COINBASE_API_KEY      leaked 3d6030e8d9   current 3d6030e8d9   still live
COINBASE_API_SECRET   leaked 8e46bf5e56   current 8e46bf5e56   still live
VITE_API_TOKEN        leaked ef260e9aa3   current 1b1b068f8c   ALREADY CHANGED
DATABASE_URL          absent from the current .env (compose builds it)
```

So the API token has already been rotated, the database URL is moot, and the
Coinbase pair is byte-identical to what is live but carries no trading capability.
The residual is read access to balances, positions and order history — financial
privacy rather than financial loss, and a risk the operator is entitled to accept.

**Rotation is not even the right fix.** The candles endpoint is called with
`authenticated=False` (`data_collection/coinbase_connector.py:190`), and the only
authenticated calls in that module are the dormant perp-era funding and
open-interest paths. Demonstrated with both variables unset from the process:

```
COINBASE_API_KEY / SECRET: unset for this process
fetched 30 one-minute candles with NO credentials
  newest 2026-08-23 14:12:00  close 77511.63
```

The working data path does not need the credential, so deleting it from the
environment closes the exposure with no console visit and nothing to re-plumb. Left
in place here because removing credentials could disturb the dormant authenticated
paths, which is an operator's decision rather than an auditor's.

**What does not change.** `.gitignore` matched `.env` by basename, so it covered
`backend/api/.env` and missed `frontend/.env` — which was tracked — and
`frontend/.env.local`, the file the documented dev workflow tells you to put a
token in. That pattern is fixed, and two tests now fail on a tracked env file or on
any tracked file containing a real PEM body. The mechanism that produced this is
closed regardless of how much the particular values mattered.

---

# Addendum — the first 24 hours live (2026-08-24)

**Method:** the same as above, on a running system. Every number here is read from
the live Postgres store, the container logs, or a read-only call to the venue
constructed without `live=True`. Real orders were placed on a $100 account
throughout, at $2–5 a trade.

Point 3 of the Final Adversarial Review said the audit's blind spot was that
nothing had ever run, and listed four things nobody had observed: a fill read
back, a settlement reconciled, a drift observed, and what the venue does with a
duplicate `client_order_id`. **Those four things turned out to be the four
defects.** The prediction was right, and it was right about the specific items,
which is worth more than the general warning.

The pattern across all four is worth stating plainly: **the code that trades
worked on first contact; the code that checks the code that trades did not.**
Every fail-closed path was exercised for real within hours — a retired endpoint, a
killed fill-or-kill, a rejected duplicate, an unresolvable market — and not one of
them booked a phantom position. Every *verification* mechanism was broken, and all
four were silent.

## [CRITICAL] The order endpoint was retired, and V2 is not a renamed path

The first real order returned `410 deprecated_v1_order_endpoint`. That part was
survivable — `order refused, no position recorded`, no phantom holding, which is
one of this audit's own CRITICAL fixes working on its first real test.

The hazard was in the migration. V2 quotes a **single book from the YES side**:
`bid` buys YES, and `ask` sells YES, which is economically buying NO at
`1 - price`. `decide()` produces what we would *pay* for the chosen side, so a
DOWN order at 31c must be sent as an `ask` at **69c**. Sending 31c as an `ask`
offers to sell YES for thirty-one cents — a strictly worse error than inverting
the side, and one that looks plausible in a log line. The rounding direction
inverts with it: a bid must ceiling to the cent and an ask must floor, or a
`fill_or_kill` cannot fill at all.

**Fixed** in `data_collection/kalshi_client.py`. Three mutants now die where the
V1 side-flip had left all 230 tests passing: inverted book side (5 failures),
missing `1 - price` conversion (4), reversed ask rounding (4).

## [CRITICAL] Both directions of the position cross-check were dead

`reconcile_with_venue` filtered open positions with
`int(row.get('position') or 0) != 0`. **V2 does not send a `position` field.** It
sends `position_fp`, a fixed-point string, negative for the short-YES leg that a
NO position is held as. Read from `/portfolio/positions` while a position we had
just watched fill was open:

```json
{"ticker": "KXBTC15M-26AUG241000-00", "position_fp": "-5.00",
 "market_exposure_dollars": "2.150000", "fees_paid_dollars": "0.085800"}
```

`int(None or 0)` is `0`, so `venue_open` was the empty set on every cycle. One
empty set broke the check in both directions at once:

* **forward** — *"we hold 5 contracts the venue does not report. Most likely the
  order never filled"* fired once a minute against a position that was open and
  fine. The alarm for a killed fill-or-kill was permanently on, which is the same
  as it being off.
* **reverse** — *"the venue reports an open position we have no record of"* could
  never fire. **This is the direction the report above singles out as the one that
  costs money silently**, and the row in "What has been fixed" claiming it was
  added is corrected in place. It has been structurally incapable of firing since
  it was written.

This is the same trap as the quote fields, where `yes_bid_dollars` carries the
value and the integer-cent field the older documentation describes comes back
`null`. `_quantity` already existed for it; the positions path never used it.

**Fixed** via `KalshiClient.position_size`, which accepts both encodings and
returns `0.0` rather than raising on junk — `int('-5.00')` raises `ValueError`
inside a set comprehension where nothing would catch it, aborting the
reconciliation mid-cycle so the balance was never adopted either. Verified by the
clock: container restarted `13:59:36Z`, last false alarm `13:58:47Z`, none since.

## [HIGH] Every settlement was credited twice for one cycle

`run_cycle` adopted the venue's balance and *then* called `settle_due`. The venue
credits a settlement the instant the market settles, so the payout was already in
the balance we adopted, and `settle_position` added it again:

```
09:15:38  ours $147.03, venue $168.03 (+21.00)   <- venue credited the payout
09:16:42  ours $189.03, venue $168.03 (-21.00)   <- we credited the same payout
09:19:56  ours $160.67, venue $161.07 ( +0.41)   <- reconciled back
```

No money moved — the next cycle corrected the bankroll, and our computed PnL
matches the venue's balance change to seven cents across 155 positions. Two things
went wrong anyway. Kelly sized off a bankroll inflated by the payout for up to a
full cycle, and the inflation scales with the win, so it is largest exactly when
the account has just been most volatile. And every settlement produced a large
spurious drift warning, in a log that is the only thing standing between an
unrecorded fill and silence.

**Fixed** by splitting `adopt_venue_balance` out of `reconcile_with_venue` and
calling it after settlement. The test drives the real `run_cycle`, because
asserting a hand-written sequence of two calls only tests the sequence in the
test; restoring the old order fails it with `got 113.80 ... the payout was booked
twice`.

What the alarm reports now that it is not crying wolf: `+0.15`, `+0.26`, `+0.07` —
a persistent few cents, always in the venue's favour, i.e. we slightly
over-estimate our own fees against what they actually charge. Invisible before.

## [HIGH] The order key enforced one *attempt* per window, not one *position*

`client_order_id` was `{symbol}-{window}`. A `fill_or_kill` that kills still
consumes the id at the venue, so the first thin-volume kill locked every later
offset out of that window. From `order_tickets` after 24 hours:

| outcome | n | avg claimed edge |
|---|---:|---:|
| filled | 159 | 5.77pp |
| refused: duplicate `client_order_id` | 57 | **8.37pp** |
| refused: `fill_or_kill_insufficient_resting_volume` | 9 | 3.94pp |
| refused: V1 `410` | 3 | 5.23pp |

Read carelessly this is textbook adverse selection — the trades that fail carry a
higher edge than the trades that fill. It is not. 57 of 69 failures were **our own
key**, and the 9 genuine thin-book refusals were the *lowest*-edge group of the
three. The market was not selecting against us; we were blocking ourselves out of
the better half of our own signal.

Double-entry was never what this key protected — `entries_for_window` does that,
counting a ticket in any status but `skipped`, so a crash between sending an order
and booking a position still blocks the window while `skipped` correctly reopens
it. **Fixed** by putting the offset in the key. Observed working within minutes:
in window 14:30–14:45, ETH was refused at `+3m` for thin volume and then **filled
3 @ 0.13 at `+6m`** — a fill the old key made impossible.

The upside is bounded by `max_positions_per_window = 2`, which is correct and
unchanged: three ~0.7-correlated symbols in one window is largely one bet at 3x
size, the same logic that makes four offsets one bet rather than four.

## The market benchmark, measured for the first time

`scripts/market_benchmark.py` existed and had never had data. Over 1,109 scored
rows and 285 windows of live-recorded quotes:

| slice | n | market_ll | baseline_ll | model_ll | model − market |
|---|---:|---:|---:|---:|---:|
| all | 1109 | **+0.331** | +0.428 | +0.430 | **−0.098** |
| BTC-USD | 365 | +0.309 | +0.415 | +0.415 | −0.106 |
| ETH-USD | 373 | +0.310 | +0.408 | +0.409 | −0.100 |
| SOL-USD | 371 | +0.375 | +0.460 | +0.465 | −0.089 |

**The market is a better forecaster than the model on every symbol and every
offset, and the model is indistinguishable from its own baseline.** Restricted to
the 108 rows actually traded, against the de-spread mid `(ask_up + (1 −
ask_down))/2`: model 0.5851, baseline 0.5775, market **0.5389**, gap **−0.0461**.
Selection narrows the gap and never closes it — and on the traded subset the model
is *worse than its own baseline*, so the ML layer is actively harmful exactly where
it is used.

This is a structural indictment of the gate set, not just of this candidate. A
model like that passes `log_loss_skill` **by construction** — it does beat
`F(x/sigma)` — and passes the other thirteen gates too. `market_windows` and
`model_minus_market` are gates now, read first, and they fail as *unmeasured*
until 2,000 windows of quotes exist.

## The money, and why it does not settle the question either way

$100 → $165.23 in 24 hours: 153 settled positions, 42.5% win rate at a
contract-weighted entry of 33.3c against a 34.7% breakeven, +$65.67 realised,
$13.35 fees (4.1% of stake). Venue-confirmed, so it is real money and not an
accounting artifact.

The right test is not the win rate but a bootstrap under the null that the
market's de-spread mid is the true probability, using each trade's actual price,
size and fee. On 158 settled trades over 88 windows, with a Gaussian copula for
within-window correlation:

| ρ | mean | sd | P(net ≥ +$62.96) |
|---:|---:|---:|---:|
| 0.0 | −$23.04 | $41.17 | 0.022 |
| 0.5 | −$23.13 | $43.82 | 0.029 |
| 0.9 | −$22.97 | $46.77 | 0.038 |

Expected P&L if the market were right was **−$23**. Being up $63 is a **2–4%
event** under that null, and clustering barely matters because there are only 1.80
symbols per window. **So "it is just variance" is too strong** — this is genuine
mild evidence against the market-is-right null, and it subsumes rather than
compounds the earlier objection that five trades carry 73% of the profit: that
concentration is exactly what the $41 standard deviation is made of.

Two measurements now point opposite ways, and the reconciliation is the
interesting part. Log loss punishes confident errors severely; a binary bet at a
fixed price only cares which side of the price the truth lands on. A forecaster
can be badly calibrated in magnitude and still pick the right side of its own
disagreement. For money the bootstrap is the objective and log loss is the proxy,
so the bootstrap wins on relevance — while being far more underpowered, at one day
and 88 windows.

## Tested and rejected: the offset-3 edge is not a sigma-scale artifact

Worth recording because it was a good hypothesis with a specific mechanism, and
because re-running it would cost another afternoon.

The baseline's fitted scale declines monotonically with offset — 1.268, 1.189,
1.094, 0.965 — and the five-year edge investigation found skill peaking at offset
3 and dead by offset 9. The offset with the largest sigma inflation is the offset
with all the apparent skill and essentially all the trades. Our probabilities also
sit closer to 0.5 than the market's by 0.067 on average, which is what an inflated
sigma produces. So: is the "edge" just a too-large sigma making us price cheap
sides above the market, in the one region the venue's own calibration table shows
it already overprices?

**No.** Refitting a multiplier `a` on the current sigma with `nu` held at 5.075,
against two targets independently — the market's de-spread mid, and the realised
outcomes on the same 1,133 rows and 97 windows:

| slice | n | a_market | a_outcome | scale now → market → outcome |
|---|---:|---:|---:|---|
| all | 1133 | 1.023 | 0.950 | 1.133 → 1.159 → 1.077 |
| offset 3 | 291 | 1.178 | 1.087 | 1.268 → 1.494 → 1.378 |
| offset 6 | 291 | 1.098 | 1.098 | 1.189 → 1.307 → 1.306 |
| offset 9 | 288 | 0.987 | 0.926 | 1.094 → 1.080 → 1.014 |
| offset 12 | 263 | 0.844 | 0.664 | 0.965 → 0.815 → 0.641 |

At offset 3, where the trades are, **both the market and the outcomes want the
sigma larger, not smaller** — the opposite of the hypothesis. Pooled `a_outcome`
is 0.950 with a 90% block-bootstrap CI over windows of [0.760, 1.155], containing
1.0. And the aggregate closeness-to-0.5 turns out to be a mix across offsets, with
offset 12 wanting sigma 34% smaller while offset 3 wants it larger, so no single
scale story fits.

The decisive number is that rescaling cannot close the gap at all:

```
mean |baseline - market|, the quantity the trading edge is drawn from:
  at a=1, as traded           12.35pp
  at the market's implied a   12.08pp
  at the outcome-optimal a    11.86pp
```

Half a point of a twelve-point disagreement is attributable to sigma scale. The
result is robust to how `a_market` is defined, because the residual is ~12pp even
at the outcome-optimal scale. Log loss against outcomes agrees from the other
side: 0.42578 at a=1, 0.42606 at the market's a, 0.42551 at the outcome-optimal
a — a 0.0005 spread against the market's 0.32915. **The five-year scale fit
generalised fine and the baseline is a properly calibrated null.** It is simply
not competitive with the price.

So the market's 0.10 nats is not recalibration, and no scale parameter recovers
it. The price carries information the barrier form cannot express, which is a
harder problem than a mis-tuned sigma and the honest reading of it. It also
strengthens the case for initialising on the market rather than weakening it: if
the price holds information our functional form cannot produce, fitting a
correction *to the price* is the only way to use it.

## What this changes about the verdicts

Nothing in the Executive Summary flips. *Safe to trade real money* stays **NO**,
for a narrower and better-evidenced reason than before: the model does not beat
the price it trades against on the only sample where that has been measured, and
the gates were not asking. The plumbing is materially better than the audit found
it — four fail-closed paths held under real conditions — and the measurement
apparatus was wrong again, in four new places, which is the fifth point of the
Final Adversarial Review holding up exactly as written: *"the prior on further
undetected measurement error should be high."*

Concretely open, in priority order:

1. **`init_score` should be the market's logit, not the baseline's.** The model
   currently learns to correct the weaker of the two forecasters by 0.10 nats.
   Correcting the price is the residual that pays, and it is now recordable.
2. **2,000 windows of quotes**, then re-read `model_minus_market`. At ~96 windows
   a day across three symbols that is about a week. If it is still negative the
   strategy is falsified on its own terms.
3. **The same fixed-point trap, everywhere else.** Found in quotes, then fills,
   then positions. `settlements.revenue` is safe only by accident — it is used
   solely as a sign test, never as a magnitude — which is the kind of safety that
   stops holding the moment someone reads the field for its value.
