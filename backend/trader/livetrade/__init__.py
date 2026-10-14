"""The live trading engine, split by responsibility.

`scripts.live` keeps the CLI, the loop and `run_cycle`, and re-exports the
public names below so existing imports keep working.

    artifact    load, version-stamp and hot-reload the promoted model; gates
    bars        one-minute bars: REST fetch, rolling cache, trade-stream splice
    quotes      market resolution, stream-or-REST quotes, ladder/depth helpers,
                the top-of-book archive
    features    book, cross-venue, implied-vol and gap features for scoring
    risk        circuit breakers and the bar-feed guards that stop a decision
    execution   limit pricing, order placement and booking, kill diagnosis
    ledger      venue reconciliation, settlement, balance and equity
    state       mutable state shared between these modules
    util        small shared helpers

The history behind the rules in these modules is in
`docs/decisions/live-trading-history.md`.
"""
