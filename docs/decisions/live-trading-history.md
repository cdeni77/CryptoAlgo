# Live trading: the history behind the rules

`backend/trader/scripts/live.py` was split into the `backend/trader/livetrade/`
package (October 2026). Its comments and docstrings had accumulated incident
history — what was measured on which night, what an earlier version did wrong —
alongside the rules that history produced. In the moved code each one was
condensed to what the code does and the reason that still constrains it.

This file keeps the original text, verbatim, of every docstring and comment
block that was condensed, organised by the module and function it now lives in.
Nothing was dropped: a block that survived unchanged in the code is not
repeated here. Line wrapping and the leading indentation of docstrings are as
they were, minus the common indent. Code that stayed in `scripts/live.py` (the
CLI, the loop, `run_cycle`, scheduling) kept its comments in place and does not
appear here.

Read this when a rule in `livetrade/` looks arbitrary and you are about to
change it: the measurement that justified it is usually below.

## `livetrade/artifact.py`

### `config_for_artifact`

Original docstring:

```text
Adopt the artifact's init-score source, refusing the one mode that cannot
supply it.

The artifact records which forecaster its correction was fitted on top of.
Live has no legitimate choice here — scoring a market-fitted residual on the
baseline logit is exactly the silent failure `ForecastModel.verify` exists to
catch — so the artifact wins and the mismatch becomes impossible rather than
merely detected.

The refusal is the other half. A market-init model needs the book to score:
`fetch_quotes` returns {} when no Kalshi client was opened, which is every
paper run. Adopting silently would swap a loud crash for a NaN prediction
every cycle, which reads as a model that has decided to abstain.
```

Original comment:

```text
# The economic policy the artifact was EVALUATED under. `scripts/live.py`
# has no flag for any of these, so without this the loop traded whatever
# Config defaulted to — 0.25 Kelly against the promoted 0.10, a 1.50pp gate
# against 3.00pp. `verify` does not catch it because these change what the
# strategy DOES, not what the model SAYS, and that is exactly why it went
# unnoticed: the model was scored correctly and then acted on wrongly.
#
# CLAUDE.md measures the cost of getting Kelly wrong: 0.25 -> 0.10 moved
# realised edge per contract +0.99pp -> +3.32pp and drawdown 58% -> 21%,
# because a smaller fraction also floors marginal trades under one contract
# and refuses them. It is an edge filter, not only a size.
```

Original comment:

```text
# Anything the command line set explicitly wins. Adoption exists so an
# artifact trades under the policy it was MEASURED under, not to overrule
# the operator — and silently reverting a deliberate setting at the next
# promotion is exactly the failure that would be hardest to notice.
```

Original comment:

```text
# An artifact that reads the cross-venue group refuses a row without the
# peer book. `--complete-cases` put the peer on every training row, so a
# missing one sends LightGBM down NaN branches it never fitted — and the
# flag that guards this was off in the deployed command. Explicit wins.
```

### `_installed_version`

Original docstring:

```text
Which staged artifact is the one on disk, by CONTENT.

**Not "the newest ledger entry".** Those diverge in two ordinary cases: a
promotion that fails installs nothing and leaves the previous artifact
trading, and a rollback deliberately writes no ledger entry at all (the
ledger is the multiple-testing denominator, and a rollback is not a trial).
Both happened on 2026-09-16.

Comparing the bytes is exact and needs no new bookkeeping: `promote` stages
each candidate under `models/.staging/<version>/` and then renames it into
place, so the installed file is byte-identical to exactly one of them.
```

### `_refuse_if_blocked`

Original comment:

```text
# **The artifact ON DISK, not the newest attempt.** Reading `iloc[0]` meant
# a promotion that failed on Sunday -- installing nothing, leaving a passing
# model trading -- would refuse to start the loop on the next restart, and a
# rollback to a passing artifact would too. The question this gate asks is
# "did the model I am about to trade pass?", and that is a property of the
# file, not of whatever was attempted most recently.
```

Original comment:

```text
# `installed` and `passed` are different questions. `--force` installs an
# artifact whose gates FAILED and records installed=True, so testing
# `installed` alone let a gate-failing model trade silently — and
# `POST /jobs/scripts.promote {"args": ["--force", "--reason", "x"]}` reaches
# that from a single HTTP request. Test the gates.
```

## `livetrade/bars.py`

### `FETCH_MINUTES`

Original comment:

```text
# How much history each cycle fetches. The longest feature lookback is 1,440
# minutes, and the seasonal factor is a fitted lookup rather than a rolling
# window, so a day plus a margin is sufficient and a week is waste.
```

### `fetch_bars`

Original docstring:

```text
One-minute bars for the universe, straight from the venue.

**Only the tail is fetched.** `log_rv_1440` and `beta_1440` need a day of
history, so the window is 1,500 minutes — but the cycle runs every 60
seconds, and re-downloading twenty-five hours to learn one new minute was
measured at 3.2s, the largest single cost in the cycle. `get_candles_range`
pages at 300 candles, so a full window is ~6 sequential round trips per
symbol, which is why fetching the three symbols concurrently barely helped.

The window is kept between cycles and only the newest few minutes are
re-asked. A symbol with no usable cache — first cycle, or a gap wider than
the window after an outage — falls back to the full fetch, so this is an
optimisation and never a source of missing history.

`prefer_stream` (decision cycles) first tries the Coinbase trade tape for
the newest minutes — see `stream_bars` — and asks REST only for symbols
the tape cannot vouch for. That takes the REST round trip off the decision
path, and with it the risk of reading a candle Coinbase has not finished.
```

### `_fetch_rest_bars`

Original comment:

```text
# **Concurrently, not one symbol after another.** Measured at 3.17s for
# three symbols — three independent round trips to the same host, run in
# series for no reason.
#
# `return_exceptions` so one symbol's failure costs only that symbol.
# Losing all three because SOL timed out would turn a partial outage into
# a total one, and the loop already handles a missing symbol.
```

## `livetrade/quotes.py`

### `_ladder_touch`

Original docstring:

```text
The best price the LADDER actually shows, in the quote's denomination.

The depth buckets must be measured against the touch of the same snapshot
that supplies the levels. Taking it from a separately-fetched quote made the
1c bucket read zero whenever the quote was a cent better than the cached
ladder — measured at 31.24% of live rows, against 0.00% in the backfill,
which reads both from one snapshot and so cannot produce that state.

`fallback` is used only when the ladder is empty, where there is nothing to
measure and the NaN is the honest answer.
```

### `_cumulative`

Original docstring:

```text
Resting size at prices at least as good as `best +/- within`.

This, not the touch, is what decides a `fill_or_kill`. Our limit sits a cent
or two past the touch, so the order fills only if the cumulative size up to
that limit covers it. Measured on SOL: the touch held 6 contracts against a
7-contract order, while cumulative within 2c was ~19 — the touch alone says
"no fill" where the ladder says "fills".
```

### `ladder_from_cache`

Original docstring:

```text
(yes_levels, no_levels) from the stream's in-process book, or empty.

**This is why the full feature set costs no latency.** `run_live` supervises
`stream` and `trade` as asyncio tasks in ONE process, so `record_stream.CACHE`
already holds every subscribed ladder folded from the socket — no fetch, and
fresher than one: a frame arrives ~34ms after the venue stamps it against
~73ms for a REST round trip, at 100% top-of-book agreement and zero ladder
drift against REST snapshots.

Refused in three cases, all of which mean the book in hand may not be the
book on the venue:

  * no cache — `scripts.live` also runs standalone, without the stream task;
  * stale — ten seconds of silence at 400+ frames a second is a sick
    transport, not a quiet market, and pricing against it is worse than
    having no depth;
  * gapped — `seq` is global per SUBSCRIPTION, so a miss condemns every book
    on the connection, this one included.

Empty levels give NaN depth features, which `_warn_unscoreable_features`
reports. That is the honest answer, not a silent substitution.
```

### `depth_dollars`

Original docstring:

```text
(depth_up, depth_down) in DOLLARS, or (None, None) with no ladder.

`decide()` caps the stake at `measured_depth * depth_fraction` — "a measured
depth beats the standing guess" — but nothing outside the tests ever set
`depth_up`/`depth_down`, so the cap never fired and sizing was bounded only
by `max_stake_dollars`, an acknowledged guess. Harmless at a $3 stake; at the
$25 cap it is ~100 contracts against a median 212 resting at the touch.

Which side is crossed is the part that inverts. Kalshi's book is two BID
stacks: buying YES crosses the YES ask, whose size comes from the NO stack;
buying NO crosses the NO ask, whose size comes from the YES stack. Dollars,
not contracts, because that is what `decide()` compares against the stake.

None, not zero, when there is no ladder — zero would refuse every trade,
where the honest answer is "unmeasured, fall back to the standing cap".
```

### `_resting_total`

Original docstring:

```text
Every resting contract on one side, at any price.

`depth_bid_total` / `depth_ask_total` were the one thing `_record_touch`
never derived, so `depth_ratio` could not be computed live at all.
```

### `_record_touch`

Original docstring:

```text
Write the venue's book — top of book plus cumulative depth — to the store.

Best effort: a failure here must never stop a trading cycle, because this is
measurement and the cycle is money.

The ladder comes from `GET /markets/{ticker}/orderbook`, which works while a
market is open and returns empty once it settles. That asymmetry is the whole
reason this runs live: depth is not in any historical endpoint, so a book not
written down now is gone.
```

Original comment:

```text
# **From the STREAM CACHE, the same ladder scoring read, not a
# fresh REST fetch after the order.**
#
# The call site says moving this "changes only WHEN the archive is
# written, never what it contains: `quotes` is the same object the
# decision priced against". True of the four touch fields; false of
# the six ladder-derived columns below, which were re-fetched --
# AFTER `act_on`, so our own fill had already removed resting size
# from the side we bought. `depth_bid_1c`, `depth_ask_1c`,
# `depth_bid_5c`, `depth_ask_5c` and the totals feed `imbalance_5c`,
# `depth_ratio` and `book_convexity`: three of the five
# `market_state` features, systematically understated on the traded
# side, in rows `QUOTE_SOURCE_PRIORITY` ranks FIRST.
#
# The cache read is local and free, so this also removes three REST
# round trips from the post-order path. The REST fetch stays only as
# a fallback for when the stream is not running -- with a flag on
# the row saying so, because a reconstruction and an observation are
# different claims.
# The ladder the decision priced from comes first: it is the same
# snapshot as the touch, whichever transport supplied it.
```

Original comment:

```text
# Which sampler produced the LADDER half of this row. A
# reconstruction taken after our own fill and an observation
# taken at the decision are different claims, and the column
# that could not tell them apart is how this went unnoticed.
```

Original comment:

```text
# ONE sampler, as in `book_feature_row`: the ladder's own
# touch, not the quote's. Measured, the mixed version wrote
# "1c empty while 5c is full" on 31.24% of these rows against
# 0.00% in the backfill, so a third of every recorded row's
# convexity was a bookkeeping artifact rather than a thin book.
```

Original comment:

```text
# The whole resting ladder, not just the levels near the
# touch. `depth_ratio` and `book_convexity` read these, and
# `book_feature_row` already derives them for the DECISION —
# recording anything less would trade a feature the next
# retrain cannot see.
```

Original comment:

```text
# **Name the observer.** `source` is in
# `EVENT_KEY_EXTRA['venue_depth']` so that independent
# observations of one minute survive a read instead of
# collapsing to whichever was published last. This writer left
# it unset, so every row it has ever written sits under a NULL
# observer — invisible to `_validate_depth`, which compares by
# source, and indistinguishable from a row whose source the
# store failed to record.
#
# It is deliberately NOT 'live': that belongs to the ladder
# recorder's REST poll, and this is a different sample at a
# different instant — the quote the trading loop actually
# decided against. Colliding them would discard one.
```

## `livetrade/features.py`

### `_venue_caches`

Original comment:

```text
# The SOCKET first: it is contemporaneous with the Kalshi book, which is
# what training assumes. The REST recorder writes at a median 32s past the
# minute against decisions at ~+2s, and measured over 516,680 rows a gap
# against a peer one minute stale correlates only 0.672 with the
# contemporaneous one. Fall back to it only when the socket has nothing —
# `cross_venue_row` refuses a stale reading either way.
```

### `_GAP_HISTORY`

Original comment:

```text
# The gap at each decision offset of the window in progress, per symbol.
#
# The backtest computes `venue_gap_change_5` as `shift(1)` over rows ordered by
# OFFSET and grouped by (symbol, window_open) — the previous decision offset in
# the SAME window. Live has to reproduce that exactly or it is fitting one
# feature and scoring another.
```

### `gap_change`

Original comment:

```text
# Evict by WINDOW, never by arrival. Three symbols are scored in every
# cycle of the same window, so clearing whenever an unseen key turns up
# meant BTC's reading was destroyed by ETH's and ETH's by SOL's — every
# symbol then reported NaN on every cycle, forever.
```

Original comment:

```text
# **The IMMEDIATELY previous decision offset, or nothing.** Training is
# `shift(1)` over a panel that always carries all four offsets, so a missing
# gap at the previous offset propagates as NaN. Taking `max(o < offset)`
# instead SKIPS the hole and returns a six- or nine-minute change under a
# column named for three — a different feature wearing the same name.
#
# Measured on the store: of 97,416 decision rows past offset 3 that carry a
# gap, 19,548 (20.1%) are NaN in training because the previous offset had
# none. Live exposure is larger still, because a cycle that misses
# DECISION_TOLERANCE_SECONDS records nothing for its offset at all.
```

### `cross_venue_row`

Original comment:

```text
# **The same staleness bar the backtest applies to BOTH venues.**
#
# Training aligns them: `event_time` is identical on 100% of 693,805 matched
# pairs and Polymarket is 4.0s STALER on average, so `venue_prob_gap` there
# is a contemporaneous disagreement. Live it was not — `record_pm_ladder`
# wrote at a median 32s past the minute against decisions at ~+2s, so the
# freshest reading was ~30s old and nothing checked. A 30s-old peer price
# makes the gap partly a measure of KALSHI'S OWN MOVE over that interval,
# which is not the quantity the model was fitted on.
#
# It matters most here: `cross_venue` is the only load-bearing group —
# dropping it takes skill +0.00282 -> -0.00015.
#
# A reading with no stamp is kept, matching `attach_quotes`: absence of a
# stamp is not evidence of staleness.
# **The WINDOW, before the age.** Staleness cannot catch a rollover:
# observed live, Kalshi rebuilt its subscription at 22:45:16 and Polymarket
# at 22:45:46, because each venue's silence begins when ITS markets settle.
# For those thirty seconds the cache held the previous window's book with a
# recent stamp — data that is genuinely fresh and about the wrong fifteen
# minutes. A reading with no window is kept: the REST fallback publishes
# none under EITHER key — the socket writes `window`, the REST recorder
# writes `window_open`, and reading only the first left the guard inert
# on the fallback, which is precisely when a rollover is most likely.
```

### `latest_fit`

Original docstring:

```text
The freshest usable ladder fit for `symbol`, or None.

The in-process cache first — the recorder runs in this process and its
reading is the newest thing that exists. Then the STORE, which is what the
backtest reads: `attach_implied_vol` does an as-of join tolerating a fit up
to `MAX_FIT_AGE_MINUTES` old, and live consulting only the cache meant every
restart blanked all five iv_* features until the recorder next succeeded.

That gap is not cosmetic. `--complete-cases` fitted the artifact on rows
that ALL carry a fit, so it has never seen a NaN there and scoring without
one is out of distribution.
```

### `_attach_book_features`

Original comment:

```text
# The other two book groups. Both were declared in FEATURE_GROUPS and
# fitted into the artifact, and neither was ever attached here — nine of
# forty-nine features scored NaN every cycle. Dropping them instead is
# not available: refitted without them, log_loss_skill goes +0.00307 ->
# -0.00023 and folds positive 6/6 -> 3/6.
# The stake cap `decide()` has always had and never received. Both are
# dollars at the touch on the side actually crossed.
```

### `_warn_unscoreable_features`

Original docstring:

```text
Say loudly when the model wants a feature this row cannot supply.

A NaN feature does not raise; the booster substitutes the direction it
learned in training. So the only signal that a live model is not the model
that was measured is this line.

**Which is why it must not cry wolf.** It fired 96 times in 24 hours --
once per window, always `venue_gap_change_5`, which is NaN at a window's
first offset by construction. A genuine failure, a dead stream leaving
`imbalance_5c`, `depth_ratio` and `book_convexity` empty, appends three
names to the identical line at the identical rate, and was therefore
indistinguishable at a glance from the moment this feature shipped.

Expected absences are still REPORTED, at debug, rather than dropped: the
day one of them stops being expected, silence would be worse than noise.
```

### `book_feature_row`

Original docstring:

```text
The model's book features, from the book the fill will price against.

**Live computed none of these.** `_record_touch` derives the depth every
cycle and writes it to the store; the scoring row never saw it. That does
not raise — LightGBM scores a NaN feature with the default direction it
learned — so the loop would run silently as a DIFFERENT model from the one
whose gates were measured.

Built through `core.book_features.market_state_features`, the same function
the backtest uses, off the same quote the order is priced against, so the
two cannot drift. Prices go in as CENTS because that is what the snapshot
shape uses; a one-sided book yields no probability, exactly as in the
backtest.
```

Original comment:

```text
# ONE sampler. `best` is the ladder's own touch, not the quote's — see
# `_ladder_touch`. Mixing the two zeroed the 1c bucket on 31% of rows
# while 5c stayed full, which is a NaN convexity from a bookkeeping
# mismatch rather than from a thin book.
```

## `livetrade/risk.py`

### `check_circuit_breakers`

Original docstring:

```text
Halt the account on a bad day or a bad streak. Returns the reason, or None.

`Account.halted` and `halted_reason` existed as columns, were rendered on the
dashboard as a safety chip, and **were never written by anything** — the only
code that set them was `core/book.py`, the backtest's in-memory account,
which `scripts/live.py` does not import. So the indicator was structurally
incapable of turning on, and the `halted` promotion gate read the simulated
account rather than the real one.

The ruin floor in `decide` was the only live limit, and it fires after half
the account is gone. At 96 windows a day across three symbols that is a long
way to bleed: the nominal worst case was $768/day against a $100 bankroll.

A halt is sticky. Clearing it is a manual decision, because a breaker that
resets itself at midnight is a speed bump.
```

Original comment:

```text
# Peak-to-current drawdown on realised equity.
#
# **The daily rule cannot see this shape, and it is the shape that happened.**
# Equity ran $100 -> $166.86 by 13:00 UTC on the second day and gave back
# $63.92 over the next ten hours — all inside one UTC day, so that day's
# realised was +$3.81 against a -$15.00 limit. The daily rule saw a good day
# while the account sat 38.3% below its high. `max_drawdown <= 0.35` was
# already a promotion gate on the *backtest*; a threshold worth enforcing on a
# simulation is worth enforcing on the money.
#
# Realised basis on both sides: `realised_high_water()` is the running maximum
# of the same series `account.realized_pnl` currently holds, derived from the
# settlements rather than stored, so there is no second source of truth to
# drift.
```

Original comment:

```text
# **Only a halt once it is also real capital loss.** The first
# version halted on the drawdown alone, and immediately stopped an
# account sitting at $107.94 on a $100 start — up 8%, having never
# cost anything, because it had once been up 67%. That is a fund's
# rule, for protecting banked gains. This account exists to find out
# whether an edge is real, and what it has to protect is the stake.
#
# The stake is already guarded twice: `ruin_floor_fraction` refuses
# every trade below 50% of the starting bankroll, and the daily rule
# bounds one day's loss at 15% of it. So above water this is a signal
# rather than a stop, and it says so in the log.
```

### `MIN_USABLE_BARS`

Original comment:

```text
# The longest lookback any feature needs, plus the mean it is taken over.
# `volume_z_15` is a 1,440-minute z-score of a 15-minute rolling mean, so
# bit-parity needs 1,440 + 15 = 1,455 contiguous bars; `log_rv_1440` needs
# 1,441. `FETCH_MINUTES` is 1,500, leaving a 45-bar margin that nothing guarded.
```

### `short_symbols`

Original docstring:

```text
Symbols whose history is too SHORT to reproduce the training features.

`stale_symbols` checks that the newest bar is recent. Nothing checked how
many bars came back, and `get_candles_range` returns a short or holed series
after logging an ERROR — so a partial Coinbase answer produced confident
wrong numbers rather than the NaN that makes `decide()` abstain.

Measured against the walk-forward value:

    bars   volume_z_15   rv_surprise   rv_slope_long
    1440        105%          16%           3.5%
    1200         46x          67x           6.8x
     800        130x         285x            12x

Below ~721 bars `log_rv_1440` fails its `min_periods` and the NaN reaches
`sigma_remaining`, so the cycle abstains on its own — safe. Between 721 and
1,454 it trades on wrong numbers, and that band is exactly where a partial
fetch lands.

Invisible without this: a short-but-unholed grid spans what was RETURNED, so
`Dataset.coverage()` reports 100.0000% and logs at DEBUG.

Rows, not span, because a 1,500-minute span can hold 400 bars — and it is
the row count the rolling windows consume.
```

### `stale_symbols`

Original docstring:

```text
Symbols whose feed is too old, or absent, with the reason.

There was no staleness guard anywhere. The last `event_time` was logged
(`fetch_bars`) and never compared to the wall clock or to the decision
minute, so a delayed or partial fetch was scored as though it were current.
Two distinct hazards:

* **An old feed.** The forward-fill inside `build_windows` is right for a
  minute in which nothing traded and cannot tell that from a minute not yet
  fetched, so a stale feed produces a fabricated `last_price` and the barrier
  treats it as a measurement.
* **A missing symbol.** `fetch_bars` logged and continued, which silently
  redefines the *other* symbols' `cross_asset` features — measured,
  `beta_1440` moved 7.7x with no error and no NaN. The universe is part of
  the feature definition, so a short universe is a different model.
```

## `livetrade/execution.py`

### `order_limit_price`

Original docstring:

```text
The worst price still worth paying for this decision.

**Pay down to the gate, not to a fraction.** `min_edge_pp` is the threshold
that admitted this trade, so any fill leaving at least that much is one the
system has already said yes to — while no fill at all earns zero. Everything
above the gate is therefore spendable.

The rule this replaced was `min(edge * 0.25, 1c)`, which allowed the live
BTC decision of 2026-08-26 just 0.92c of give against a book that moves ~2c
in the seconds between reading it and the order landing. Measured over the
first live nights: 26 of 42 attempts did not fill, and they missed on price
rather than on size.

Crossing further costs little: the order is `immediate_or_cancel` with
`taker_at_cross`, so it fills against resting size from the touch outward
and stops. The limit bounds the worst fill; it does not set the price. It
only binds where the touch was too thin to fill at the old limit — which is
precisely the case that used to return nothing at all.

**The gate is the one in force, not the default.** This read
`DEFAULT_CONFIG.min_edge_pp` (1.5) while live runs 3.0, so on a 5pp edge it
spent 3.5c where the rule allows 2.0c — and a fill at that price leaves less
edge than the threshold that admitted the trade, which is the one outcome
"pay down to the gate" is meant to exclude.

`max_slippage_cents` stays as a rail against a book so thin the order walks
it absurdly far. That is not hypothetical: the original `price + edge` limit
sent 0.7832 against a 0.60 ask, 18c of tolerance on a 1c measured spread.
```

Original comment:

```text
# `snap_to_tick` used to live here, rounding the limit DOWN onto the venue's
# tick grid before `place_order` saw it. It is deleted rather than left unused,
# because the next person to find it would reasonably wire it back in.
#
# It was added to stop the venue snapping an off-grid limit against us: 4 of 143
# fills came back above `max_price` by 0.08c to 0.71c, about 0.011c per
# contract. It cost far more than it saved. `place_order` already puts the limit
# on the grid and rounds in the direction that can still fill (`ceil` for a bid,
# `floor` for an ask); snapping down first turned any sub-tick allowance into
# exactly zero, so the limit landed on the touch. Three of the ten kills on
# 2026-09-04 had a zero-cent allowance for that reason, and a kill earns zero
# against the hundredth of a cent the rail was protecting.
#
# The bound that actually matters is the gate in `decide()`, which a fill one
# tick above `max_price` still clears by `min_edge_pp - 1c`.
```

Original comment:

```text
# Never below the price we must actually pay. Real venue prices are on the
# grid, so snapping down from `price + allowance` cannot land under `price`
# — but an off-grid price would, and a limit below the ask is an order that
# cannot fill.
# **No grid snapping here.** `place_order` already puts the limit on the
# venue's grid, and it rounds in the direction that can still FILL — `ceil`
# for a bid, `floor` for an ask — because "rounding the wrong way under
# fill_or_kill is a guaranteed kill". Snapping DOWN first destroyed the
# allowance before that rounding could see it: a sub-cent allowance became
# exactly zero, so the limit landed on the touch and any move killed the
# order. Measured on the ten kills of 2026-09-04, three had a zero-cent
# allowance for this reason.
#
# What the down-snap was protecting is real but tiny: 4 of 143 fills came
# back above `max_price` by 0.08c to 0.71c, about 0.011c per contract. That
# buys nothing against a lost fill, which earns zero. And the bound that
# matters is not this number — it is the gate in `decide()`, which a fill one
# tick above `max_price` still clears by `min_edge_pp - 1c`, at least 2pp at
# the 3pp gate in force.
```

### `diagnose_kill`

Original docstring:

```text
Read the book and the venue's own order record, right after a kill.

A non-fill costs nothing, so it never had to explain itself — and by
2026-09-04 that turned into ten consecutive kills with no evidence to
reason from. Our log said `status None` because the POST reply's shape is
not the GET order record's; the recorded ladder is sampled 1-3s off the
order, which is enough for a fast book to move cents, so it could not
settle a single case either. Both readable in the second after the kill.

Never raises. A diagnosis that breaks the loop is worse than no diagnosis,
and it runs inside the path that has just failed to buy something.
```

Original comment:

```text
# **Filter by TICKER rather than scanning a page.** One 15-minute market
# carries at most a handful of our orders, so this is exact and returns
# one row instead of fifty.
#
# It is NOT a fix for paging, which was the first guess and was wrong:
# measured against the live account, `/portfolio/orders` is
# newest-first and a 50-row page did contain the order. The `null` seen
# on the 15:42 kill is therefore most likely a race — the diagnosis
# reads the listing in the same second the order was posted. The ticker
# filter does not cure that; it just makes the read exact, and the
# message below say what happened instead of printing `null`.
#
# No retry and no sleep. The POST reply already carries the counts this
# function needs, so the venue record is corroboration rather than
# evidence, and a diagnosis that stalls the trading loop to get it would
# cost more than it is worth.
```

### `filled_from_order`

Original docstring:

```text
Contracts actually filled, and at what price, from the venue's reply.

This used to be `int(order.get('count', decision.contracts))`. `count` is the
size *requested*, and `status`, `remaining_count` and `taker_fill_count` were
never read — so a killed `fill_or_kill`, a partial fill, and an HTTP 200 with
an empty body all recorded a full fill and debited the bankroll for contracts
nobody held. The documented claim that "a fill_or_kill that killed leaves a
ticket and no position" was false.

Returns `(0, nan)` for anything that is not a confirmed fill. Assuming a fill
is the one error here that cannot be reconciled later: the position is
invented, `settle_due` settles it, and the PnL is fiction.
```

Original comment:

```text
# `int(...)`, not `int(str)`: V2 returns these as fixed-point decimal strings
# like "10.00", and `int("10.00")` raises ValueError — which this function
# caught and turned into "nothing filled". A real, paid-for fill would have
# been recorded as no fill and the position dropped on the floor, which is the
# most expensive way to be wrong here. Float, not int, because the venue fills
# fractionally: one live order came back as 0.43 + 0.57 on the same ticket.
```

Original comment:

```text
# **`fill_count_fp` first, because it is the only count a V2 order carries.**
# Read off the live account: an order has `fill_count_fp`, `initial_count_fp`
# and `remaining_count_fp`, and none of the three names below. So this loop
# never matched, and every fill was decided by `status` alone — all-or-nothing,
# which was harmless only while the client sent `fill_or_kill`. Under
# `immediate_or_cancel` a partial fill is `status='canceled'` with a non-zero
# `fill_count_fp`, and the status fallback books it as nothing.
```

Original comment:

```text
# **Never infer a fill from `remaining` on a dead order.** The live kill
# reported `remaining_count_fp='0.00'` against `initial_count_fp='5.00'` and
# nothing filled: remaining hits zero because the order left the book, not
# because it traded, so `requested - remaining` would invent five contracts
# out of a total miss. Remaining only means anything while the order is live.
```

Original comment:

```text
# **The price actually paid, not the one we hoped for.** No price key the old
# list looked for exists on a V2 order either, so this returned nan every time
# and the caller fell back to the decision price. `taker_fill_cost_dollars` is
# the money that actually left the account, so cost / count is the true
# average fill — and it already accounts for filling better than the limit.
```

### `place_and_book`

Original docstring:

```text
Record the ticket, place the order when asked to twice, book the fill.

**A position is written only when money actually moved.** An unresolved
market or a dry run books nothing — both once produced holdings the venue
had never heard of, which `settle_due` then settled into invented PnL.

**`quote_time` is when the book was read**, checked against
`max_quote_age_seconds`. None makes no staleness claim.

The ticket write happens BEFORE the POST, per order, because it is what
`entries_for_window` counts: a crash between sending and booking must still
block the window. It runs in a thread so the other orders keep moving.

A POST that times out, or that the venue answers with a 5xx, may have been
accepted. The order is looked up by `client_order_id` and booked from the
venue's own record — filled, partial or nothing. Only if that lookup fails
is the ticket left `unknown`, which still blocks the window.
```

Original comment:

```text
# **The offset belongs in this key.** Keyed on (symbol, window)
# alone it enforced one order ATTEMPT per window, and the first
# kill locked every later offset out with a 409.
```

## `livetrade/ledger.py`

### `as_utc`

Original docstring:

```text
A UTC timestamp, whether the source remembered the timezone or not.

The ORM columns are `DateTime(timezone=True)`, so Postgres hands back
tz-aware values and SQLite hands back naive ones — it has no timezone type.
`pd.Timestamp(x).tz_convert('UTC')` raises on the naive case, so any code
path that reads a timestamp back out of the store and assumes awareness works
against Postgres and raises against SQLite. Everything stored is UTC, so
localise when the tz is missing rather than guessing.
```

### `resolve_window`

Original docstring:

```text
Did this window settle up, per the trained rule? Plus the value used.

One implementation, because `settle_due` and `settle_predictions` must agree —
the whole reason the live settlement drifted from the training label in the
first place was a second copy of this arithmetic.

The rule is `core/windows.py`'s: the `(O+H+L+C)/4` mean of the minute *ending*
at `settle_time`, compared with `>=` because the venue's `strike_type` is
`greater_or_equal`. Returns `(None, nan)` when the bar is not there yet.
```

### `settle_due`

Original docstring:

```text
Settle every matured position, on the venue's rule.

**This must agree with `core/windows.py:build_windows` exactly**, because the
model is trained against that label and the money is booked against this one.
It did not. This function read the *open* of the bar starting at `settle_time`
and compared with a strict `>`, while the target is the `(O+H+L+C)/4` mean of
the minute *ending* at `settle_time` compared with `>=`. Three deviations at
once — wrong minute, wrong estimator, wrong comparison — and measured on real
bars they disagreed on 3.4-8.2% of windows. Its docstring justified the
`open` because "the strike was read the same way", which stopped being true
when the averaged target landed and left the comment arguing for the bug.

The venue's own settlement wins wherever it is available. Ours is an OHLC
mean of Coinbase standing in for sixty seconds of CF Benchmarks BRTI, which
is a close proxy and not the same number; every disagreement is logged
because a persistent one is the basis risk becoming measurable.
```

### `VenueState`

Original docstring:

```text
What the venue says, read before we settle anything ourselves.

The balance travels with the settlements rather than being written on the
spot, because the order of those two operations turned out to matter — see
`adopt_venue_balance`.
```

### `adopt_venue_balance`

Original docstring:

```text
Make the venue's balance the one we report. **Call this after settling.**

Our running figure against theirs: a gap that grows is an unrecorded fill, a
partial, or a fee we mispriced. Logged rather than silently overwritten,
because the venue is right either way and a silent overwrite hides how wrong
we were.

**The ordering is the whole point, and getting it backwards cost the alarm its
meaning.** This used to run at the top of the cycle, before `settle_due`. But
the venue credits a settlement the moment it settles, so by the time we read
the balance the payout is already in it — and then `settle_due` credited the
same payout again. Measured over the first live night, on real money:

    09:15:38  ours $147.03, venue $168.03 (+21.00)   <- venue credited
    09:16:42  ours $189.03, venue $168.03 (-21.00)   <- we credited it again
    09:19:56  ours $160.67, venue $161.07 ( +0.41)   <- reconciled back

The bankroll self-healed on the next cycle, so no money moved. Two things did
go wrong. Kelly sized off a bankroll inflated by the payout for up to a full
cycle; and the drift log filled with benign +/-$21 pairs, which is exactly the
noise a genuine unrecorded fill would hide in. An alarm that cries wolf every
time a position settles is not an alarm.

Settling first makes the drift mean what it says: a disagreement between our
bookkeeping and the venue's, with the ordering artifact removed.
```

Original comment:

```text
# **A payout we have booked and the venue has not yet is not a drift.**
# `venue_balance` is sampled at the TOP of the cycle by
# `reconcile_with_venue`; `settle_due` then credits our side, and the venue's
# own balance reflects it a cycle later. So every winning settlement produced
# a matched pair of warnings -- measured:
#
#     15:15:07  ours $527.98, venue $511.98  (-16.00)   <- we credited
#     15:16:09  ours $511.98, venue $527.98  (+16.00)   <- venue caught up
#
# against ETH +$8 and SOL +$8 settling at 19:15 UTC. This function's own
# docstring records the same cry-wolf pattern being fixed once before, in the
# other direction; moving the call site fixed the sign and not the sample.
#
# The honest comparison subtracts what we just booked. A drift that survives
# THAT is the thing this alarm exists for: an unrecorded fill, a partial, or
# a mispriced fee.
```

Original comment:

```text
# Sampled *before* the overwrite, so the row keeps both figures as they stood
# at the same instant. Recording it afterwards would store our bankroll as
# the venue's and report a drift of zero forever — a self-fulfilling alarm.
# A single log line cannot show a trend, and the trend is the diagnosis:
# a drift that stays put is a starting-balance mismatch, one that grows is an
# unrecorded fill.
```

### `_LAST_EXCHANGE_INDEX`

Original comment:

```text
# The last shard actually observed on a book. See `run_cycle`: the balance query
# needs it, and the book is now read after reconciliation, so it cannot come from
# this cycle's quotes.
```

### `remember_exchange_index`

Original docstring:

```text
Learn the shard from the book that was just read.

**A change here is loud, not silent.** Balances are local to a shard, so a
stale one reads the wrong balance — and that is exactly the failure that once
had every order refused `insufficient_balance` while the funds sat elsewhere.
A move is rare (the venue re-categorising a series), costs at most the one
cycle that already reconciled, and is corrected before the next.
```

### `venue_exchange_index`

Original docstring:

```text
Which exchange shard the traded markets live on, per the venue.

Kalshi shards its exchange by category and **balances are local to a shard**,
so the only balance an order can draw on is the one on the shard holding its
market. On 2026-08-25 the KX*15M series moved to shard 2 while this account's
funds sat on shard 0, and every order was refused `insufficient_balance`
against an apparently healthy $107.96 total.

Read off the quotes rather than configured, for the same reason markets are
resolved by asking rather than by building a ticker: a constant is a guess
that keeps working until the venue moves, and then it is silently wrong.
Returns None when no book was read, which means the whole-account total.
```

Original comment:

```text
# **`fetch_quotes` stores `(Quote, ticker)`, not a bare Quote**, and reading
# `.exchange_index` straight off the tuple silently yielded nothing: this
# returned None on every live cycle, so the balance query always fell back to
# the whole-account total instead of the shard the crypto markets are on.
# The unit test passed because it built `{symbol: Quote}` — a shape the live
# path never produces. Accept both, and let the test assert the real one.
```

### `persist_venue_ledger`

Original docstring:

```text
Store the venue's own fills and settlements. Idempotent on the venue's keys.

These arrive on every reconcile and were read, compared, and dropped. That is
why the dashboard's P&L was our arithmetic rather than the venue's: nothing
kept the rows that would have made the venue's version computable. The account
curve is drawn from what this writes.

Free, in API terms — `reconcile` already fetched both, so this adds no request
to the cycle. It covers only what the live tier returned, which for a loop
running every minute is everything that has happened since it started;
`scripts.sync_venue` does the deep paginated backfill across the historical
tier.

Exceptions are caught and logged rather than raised. A cycle that cannot write
telemetry must still trade and settle — the store is a record, not the
account — and the previous behaviour of an unhandled write killing the loop
was the wrong trade-off in a process that holds positions.
```

Original comment:

```text
# Two queries, then only the rows that are actually new. The venue returns
# the same ~200 fills and ~200 settlements every cycle, and upserting all
# of them would be four hundred read-then-write round trips a minute to
# change nothing. A settlement already stored with a null `pnl` is *not*
# in the skip set, so an incomplete parse is retried and heals.
```

Original comment:

```text
# **Only markets this system traded.** The Kalshi account is a
# person's and may hold anything; `core/venue_ledger.py` has no
# series filter, so a manually-taken NFL position settling would
# add its revenue and fees to `venue_settlements` and land in the
# realised P&L the dashboard shows for this strategy. Two
# strategies summed into one equity curve is the same class of
# error as a balance-difference curve counting a deposit as
# profit.
```

### `_is_ours`

Original docstring:

```text
Delegates to `core.venue_ledger.is_ours`.

Kept as a name because this module and its tests use it widely; the
IMPLEMENTATION moved so `scripts/sync_venue.py` filters the same way.
Two copies of an ownership test is how one of them stops being applied.
```

### `reconcile_with_venue`

Original docstring:

```text
Read the venue's ledger, and keep it.

Three comparisons, each of which has a specific failure it catches:

* **balance** — carried out in `VenueState` and written by
  `adopt_venue_balance` *after* settlement, not here.
* **settlements** — what a position actually paid. Ours are settled from an
  OHLC mean of Coinbase standing in for CF Benchmarks BRTI, which will
  sometimes disagree.
* **open positions** — a position we think is open and the venue does not is
  an order that never filled.

It also **stores** the fills and settlements now, via `persist_venue_ledger`.
It used to compare them and drop them, which is why the account page could
only ever show our own arithmetic: the venue's numbers passed through this
function every cycle and nothing kept them.
```

Original comment:

```text
# Settle from the venue where it knows, keyed on the market ticker we stored.
# This dict used to be built, logged, and dropped on the floor — `revenue` was
# assigned and never read (ruff F841), `resolved` was used only as a
# membership set for the warning below, and `settle_position` was called from
# exactly one place: `settle_due`, off our own bars. So the documented
# "settlement from /portfolio/settlements where it knows" did not exist. It is
# returned now, and `run_cycle` hands it to `settle_due`.
```

Original comment:

```text
# `KalshiClient.position_size` and not `p['position']`: the field is
# `position_fp`, a fixed-point string, and reading the name from the older
# documentation made `venue_open` the empty set on every cycle. That broke
# this check in both directions at once — the forward one warned that every
# real position "the venue does not report", and the reverse one, for a
# position the venue holds and we do not, could never fire at all. The
# reverse is the case the audit called the one that costs money silently.
# **Only the series this loop trades.** The Kalshi account is a person's
# account and may hold positions nobody here placed — an NFL market, a
# college football market, anything. Those are not unbooked fills, and
# reporting them as "an order was filled and not booked, reconcile by hand
# before trading again" once a minute is the alarm that cries wolf: the
# same failure `adopt_venue_balance` records, where a drift alarm firing on
# every settlement hid the drift it existed to catch. Measured
# 2026-09-10, one manual football position produced that ERROR on every
# cycle for hours.
#
# Scoped by SERIES PREFIX rather than by an ignore-list, so the check stays
# strict where it matters: a genuinely unbooked fill in KXBTC15M /
# KXETH15M / KXSOL15M still fires, which is the case the audit called the
# one that costs money silently.
```

Original comment:

```text
# The reverse direction, which was never checked: a position the venue holds
# and we do not. That is what an order POST that timed out after the venue
# accepted it leaves behind, and it is the one discrepancy that costs money
# silently.
```

## `livetrade/state.py`

### `_RUNNING_CONFIG`

Original comment:

```text
# The configuration the running loop is trading under, set by `run_cycle`.
```

### `_LAST_RECONCILED`

Original comment:

```text
# time.monotonic() of the last reconciliation the venue answered. An entry
# cycle sizes off the balance it adopted; see `run_cycle`.
```
