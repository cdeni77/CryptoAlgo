# Next research — ranked

## 1. Accumulate order-flow data forward (highest value, zero cleverness)

The leader needs ~5,000 evaluable windows; it has 987. At ~88/day that is **8 weeks**.
`record_stream.py` is already running and `venue_book_events` is current. The only
action is to run `build_orderflow.py` on a schedule and re-score the **frozen**
specification in `BEST_CANDIDATE.md` without refitting anything.

**Blocker to fix first:** `venue_book_events` has a 14-day retention tier, so the tape
is being discarded as fast as it accrues. Either extend retention or have
`build_orderflow.py` run daily so the derived features survive the raw frames. **This
is urgent — every day it is not done is a day of evidence lost permanently.**

## 2. Repair the dead ladder summariser

`venue_ladder` stopped on **2026-09-16** and has been dead a month while
`venue_book_events` (its source) is current at 1.18B rows. Nothing noticed. Fixing it
restores a second independent book record; replaying the raw stream can backfill
2026-09-16 → now.

## 3. Rerun the registered batch with early stopping

Measured, not speculated: E5 (no early stopping) scores −0.003791 against arm A's
+0.000195 on the same architecture — **~0.004 nats, larger than every effect in this
project.** E4–E11 were all fitted without it. This is the cheapest known improvement
and it was deliberately not done mid-campaign because the pre-registration fixed the
protocol.

## 4. Separate quote momentum from order-flow information

`d_book_mid_1/3` is the quote's own recent return and may be carrying the result that
`imbalance` is credited with. Orthogonalise the flow signals against lagged quote
returns and re-screen. Cheap; sharpens the interpretation of the leader.

## 5. Hypotheses not reached within budget

* **H4** state-conditional miscalibration (volatility regime × time-to-settle ×
  quote velocity). Tier 1, large sample, cheap.
* **H5** passive maker fills with a queue-aware model. Requires queue-position
  reconstruction from the tape — now feasible since the fold exists, but expensive.
  Kalshi's maker fee is 0.25% against 7%·p(1−p) taker, so the economics are worth it
  *if* adverse selection can be measured.
* **H7** Polymarket lead-lag. E11 failed as a feature set, but lead-lag at
  sub-minute resolution was never tested.

## 6. Obtain an execution record

Nothing in the store distinguishes a cancellation from a fill — there is no trade
channel. `KalshiClient.market_trades` exists and is unused for recording. Capturing
the public tape would make every removal-derived signal interpretable instead of
`*_removed_ambig`, and would unlock H1/H3 properly rather than leaving them null by
measurement limitation.

## What NOT to do

* Do not re-test H1, H3, H6 or H8 without materially new evidence. H8 in particular is
  a Simpson's paradox, not a near miss.
* Do not tune the leader further on the 15-day sample. It is frozen for a reason.
* Do not treat July–October as an untouched holdout for anything. It has been
  inspected repeatedly across two campaigns.
