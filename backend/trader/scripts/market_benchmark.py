"""Score the model against the venue's own probability.

`scripts/evaluate.py` measures skill against `F(x/sigma)`, the null. This
measures it against the price the venue quotes. It reads what the live loop
records on every window, traded or not: `market_probability` (the venue's
mid), `baseline_probability`, `model_probability` and `outcome`.

It needs recorded live rows. Run `scripts.live --mode live --dry-run` to
collect them; with too few, this prints how many are missing and exits.

    python -m scripts.live --mode live --dry-run --loop --cycle-seconds 60
    python -m scripts.market_benchmark
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from core.baseline import reliability
from core.metrics import (MIN_MARKET_WINDOWS, market_comparison,
                          market_frame)

# Below this many windows the aggregate log loss cannot separate. Shared with
# the gate in `core/metrics.py`, so there is one threshold.
MIN_WINDOWS = MIN_MARKET_WINDOWS


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--database-url', type=str, default=None,
                        help='Defaults to $DATABASE_URL')
    parser.add_argument('--entry-offsets', type=int, nargs='+', default=None,
                        help='restrict to the offsets that can OPEN a position, '
                             'e.g. --entry-offsets 12. Unset pools every '
                             'recorded offset, which is a policy nobody runs: '
                             'measured, +12m reads +0.000921 against a pooled '
                             '-0.000311.')
    args = parser.parse_args()

    url = args.database_url or os.getenv('DATABASE_URL')
    if not url:
        print('DATABASE_URL is not set, and this reads the serving store.')
        print('Outside docker: export DATABASE_URL=postgresql+psycopg2://...')
        return 2

    from core.pg_writer import PgWriter

    rows = PgWriter(database_url=url).scored_against_market()
    # Only offsets that can open a position. A pooled headline measures a
    # policy nobody runs.
    if getattr(args, 'entry_offsets', None):
        wanted = {int(o) for o in args.entry_offsets}
        rows = [r for r in rows if int(r[2]) in wanted]
        print(f'Scored at offset(s) {sorted(wanted)} — the ones that can open '
              f'a position.')
    else:
        # Say the headline is pooled, so it is not read as the gate.
        print('Scored across EVERY recorded offset. The loop records four and '
              'trades one, so the pooled line below is not the gate — read the '
              'per-offset rows, or pass --entry-offsets 12.')
    if not rows:
        print('No settled window has both a recorded market quote and an outcome.')
        print()
        print('That is expected until the dry-run loop has been running: quotes are')
        print('recorded by `scripts.live --mode live --dry-run`, and outcomes are')
        print('filled in by `settle_predictions` on the cycle after each window')
        print('settles. Nothing here can be answered from backtest data — the')
        print('backtest has no quotes, which is the whole reason this exists.')
        return 1

    frame = market_frame(rows)
    windows = frame.drop_duplicates(['symbol', 'window_open']).shape[0]

    # The quote lag is a bias: features are built for the nominal offset, but
    # the book is read when the cycle reaches it.
    if 'decision_time' in frame.columns:
        lag = ((pd.to_datetime(frame['decision_time'], utc=True)
                - pd.to_datetime(frame['window_open'], utc=True)).dt.total_seconds() / 60.0
               - frame['offset'].astype(float))
        frame = frame.assign(lag_minutes=lag)
        finite = frame['lag_minutes'].dropna()
        if len(finite):
            print(f'quote lag past the nominal offset: median {finite.median():.2f}m, '
                  f'p90 {finite.quantile(0.9):.2f}m, max {finite.max():.2f}m')
            print(f'  one minute of information is worth ~0.027 nats (measured by '
                  f'shifting the offset), so this lag favours the market.')
            print()

    print(f'{len(frame):,} scored rows over {windows:,} windows, '
          f'{frame["window_open"].min()} to {frame["window_open"].max()}')
    print(f'symbols: {", ".join(sorted(frame["symbol"].unique()))}')
    print()

    summary = market_comparison(frame)

    pd.set_option('display.width', 220)
    print(summary[['slice', 'n', 'market_ll', 'baseline_ll', 'model_ll',
                   'model_minus_market', 'baseline_minus_market']].to_string(
        index=False, float_format=lambda v: f'{v:+.6f}'))
    print()
    print('  model_minus_market > 0 means the model beats the price on offer.')
    print('  baseline_minus_market > 0 would mean the arithmetic alone beats it,')
    print('  which would be remarkable and should be disbelieved first.')
    print()
    print('the venue\'s own calibration, which is the thing to beat:')
    print(reliability(frame['outcome'].to_numpy(dtype=float),
                      frame['market'].to_numpy(dtype=float)).table())

    if windows < MIN_WINDOWS:
        print()
        print(f'  {windows:,} windows is under the {MIN_WINDOWS:,} this starts to '
              f'mean anything at. Keep the dry-run loop going; nothing above is a '
              f'conclusion yet.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
