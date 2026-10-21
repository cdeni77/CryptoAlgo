"""Does replaying the archive reproduce the book the live recorder sampled?

`scripts/replay_ladder.py` claims a 31-day hole in `venue_ladder` is partly
recoverable, because `venue_book_events` holds every frame the live `BookCache`
folded. That is a claim about an irrecoverable dataset, which is the worst kind
to get wrong: a reconstruction that is merely CLOSE would look like recorded
truth and quietly become training data.

So it is checked where both exist. `record_ladder` came back on 2026-10-19 and
the raw archive runs to the present, so every live `transport='ws'` row from
there on has a frame-by-frame answer behind it. This folds the archive, stops
at each live row's own observation instant, and compares the two ladders level
for level.

**Measured 2026-10-20 over the whole overlap — 2026-10-19 03:50 to 2026-10-21
01:55, 8,169 live stream-sampled minutes across BTC, ETH and SOL: the rebuilt
ladder is EXACTLY the recorded one on 8,169 of 8,169.** Every price and every
size on both sides, zero price-set drift, size ratio 1.000000, no minute where
the archive failed to produce a book at all. Not "agrees at the touch" — the
whole ladder, bit for bit. So `venue_ladder` is recoverable wherever
`venue_book_events` still reaches, and `scripts/replay_ladder.py` is not an
approximation of the recorder, it is the recorder run late.

**`available_time` is a few microseconds EARLY, and the comparison has to say
so out loud.** `record_ladder` stamps a stream row at `read_at - age_seconds`,
where `read_at` is `datetime.now()` taken in the recorder and `age_seconds` is
measured from a SECOND clock read inside `BookCache.ladder()`. The difference
between the two reads lands entirely in that subtraction, so the recorded
instant sits just before the frame it describes. Measured, that is the whole of
the story:

    tolerance     0 us: exact whole-ladder match     0/720 =   0.0000%
    tolerance    20 us: exact whole-ladder match   718/720 =  99.7222%
    tolerance    50 us: exact whole-ladder match   720/720 = 100.0000%
    tolerance   100 us: exact whole-ladder match   720/720 = 100.0000%
    tolerance  1000 us: exact whole-ladder match   720/720 = 100.0000%
    tolerance  5000 us: exact whole-ladder match   632/720 =  87.7778%

At zero the fold is one frame short on exactly one side of the book every time
— the side the boundary delta touched — which is what a one-event offset looks
like and nothing else does. The plateau then holds flat across a twentyfold
range of tolerance and only breaks at 5ms, where the window is finally wide
enough to swallow genuinely later frames at ~300 a second. A fold that was
merely approximate has no reason to produce a plateau, and no reason for the
plateau to end where the frame rate says it should.

So `DEFAULT_TOLERANCE_US` is 200: an order of magnitude above the clock skew it
absorbs and an order of magnitude below the inter-frame gap it must not cross.

This is the CHEAP check, and the definitive one is `tests/test_stream_kalshi.py`
— real frames against REST snapshots captured at the same instants, where the
timing question does not arise at all. What this adds is scale and a different
question: not "does the cache fold correctly" but "is what is on disk enough to
rebuild what the cache held".
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from core.datastore import ResearchStore
from core.replay_book import archive_events, hour_files
from core.stream_book import BookCache
from data_collection.stream.kalshi import VENUE

#: See the module docstring. Microseconds.
DEFAULT_TOLERANCE_US = 200.0


def _levels(raw) -> list[tuple[float, float]]:
    try:
        return [(round(float(p), 6), round(float(s), 6))
                for p, s in json.loads(raw or '[]')]
    except (TypeError, ValueError):
        return []


def _top(levels) -> float:
    return max((p for p, _ in levels), default=float('nan'))


def live_rows(store: ResearchStore, *, start=None, end=None) -> pd.DataFrame:
    """The live stream-sampled ladders, in observation order.

    Read with `min_quality=None`: a row the recorder flagged is still a row the
    replay must reproduce, and silently dropping the awkward ones is how a
    reconstruction gets graded on the easy half of its own evidence.
    """
    rows = store.read('venue_ladder', start=start, end=end, min_quality=None)
    if rows.empty or 'transport' not in rows.columns:
        return pd.DataFrame()
    rows = rows[rows['transport'] == 'ws']
    return rows.sort_values('available_time').reset_index(drop=True)


def compare(store: ResearchStore | None = None, *, start=None, end=None,
            tolerance_us: float = DEFAULT_TOLERANCE_US,
            on_file=None) -> pd.DataFrame:
    """One row per live sample: what the archive rebuilds against what was stored."""
    store = store or ResearchStore(os.getenv('RESEARCH_STORE'))
    live = live_rows(store, start=start, end=end)
    if live.empty:
        return pd.DataFrame()

    # **Via an explicit `datetime64[ns]`, never `.astype('int64')`.** A
    # tz-aware column's integer view is in its OWN unit, which pandas 3.0
    # stores as microseconds here — so dividing by 1e9 put every sample in
    # January 1970, `hour_files` matched nothing, and the comparison reported
    # 100% "no book rebuilt" rather than raising. A silent 1000x on a
    # timestamp is the shape of error this whole check exists to catch.
    instants = (live['available_time'].dt.tz_convert('UTC').dt.tz_localize(None)
                .to_numpy().astype('datetime64[ns]').astype('int64') / 1e9
                + tolerance_us / 1e6)
    tickers = live['market_ticker'].tolist()
    stored_yes = [_levels(x) for x in live['yes_levels']]
    stored_no = [_levels(x) for x in live['no_levels']]

    paths = hour_files(store.root,
                       start=pd.Timestamp(instants[0], unit='s', tz='UTC'),
                       end=pd.Timestamp(instants[-1], unit='s', tz='UTC')
                       + pd.Timedelta(hours=1))
    cache, out, cursor = BookCache(), [], 0

    def settle(index: int) -> None:
        ladder = cache.ladder(tickers[index])
        got_yes = [] if ladder is None else [(round(p, 6), round(s, 6))
                                             for p, s in ladder.yes]
        got_no = [] if ladder is None else [(round(p, 6), round(s, 6))
                                            for p, s in ladder.no]
        want_yes, want_no = stored_yes[index], stored_no[index]
        total = sum(s for _, s in want_yes) + sum(s for _, s in want_no)
        out.append({
            'available_time': live['available_time'].iloc[index],
            'market_ticker': tickers[index],
            'symbol': live['symbol'].iloc[index],
            # No book at all is a DISTINCT outcome from an empty one, and the
            # headline must not quietly absorb it: it means the archive does
            # not reach back far enough to bootstrap this market.
            'no_book': ladder is None,
            'exact': got_yes == want_yes and got_no == want_no,
            'exact_yes': got_yes == want_yes,
            'exact_no': got_no == want_no,
            'top_yes': _top(got_yes) == _top(want_yes)
            or (np.isnan(_top(got_yes)) and np.isnan(_top(want_yes))),
            'top_no': _top(got_no) == _top(want_no)
            or (np.isnan(_top(got_no)) and np.isnan(_top(want_no))),
            'price_drift': (len({p for p, _ in got_yes} ^ {p for p, _ in want_yes})
                            + len({p for p, _ in got_no} ^ {p for p, _ in want_no})),
            'size_ratio': ((sum(s for _, s in got_yes)
                            + sum(s for _, s in got_no)) / total
                           if total else np.nan),
            'book_age_ms': live['book_age_ms'].iloc[index],
            'quality': live['quality'].iloc[index],
        })

    for event in archive_events(paths, venue=VENUE, on_file=on_file):
        while cursor < len(instants) and instants[cursor] < event.received:
            settle(cursor)
            cursor += 1
        cache.apply(event)
    while cursor < len(instants):
        settle(cursor)
        cursor += 1
    return pd.DataFrame(out)


def report(both: pd.DataFrame, *, tolerance_us: float) -> None:
    n = len(both)
    print(f'live stream-sampled minutes with archive behind them: {n}')
    print(f'fold tolerance: {tolerance_us:.0f} us (clock skew in the live '
          f"recorder's own stamp — see this module's docstring)")
    missing = int(both['no_book'].sum())
    if missing:
        print(f'  no book rebuilt at all: {missing} '
              f'({missing / n:.2%}) — the archive does not reach these')
    scored = both[~both['no_book']]
    if scored.empty:
        return
    m = len(scored)
    print(f'\n  EXACT whole-ladder match, every price and every size, both '
          f'sides: {scored["exact"].mean():.4%}  ({int(scored["exact"].sum())}/{m})')
    print(f'    YES side exact: {scored["exact_yes"].mean():.4%}   '
          f'NO side exact: {scored["exact_no"].mean():.4%}')
    print(f'    best bid identical  YES {scored["top_yes"].mean():.4%}   '
          f'NO {scored["top_no"].mean():.4%}')
    print(f'    price-set drift: median {scored["price_drift"].median():.0f}, '
          f'mean {scored["price_drift"].mean():.4f} prices')
    print(f'    total resting size ratio: median '
          f'{scored["size_ratio"].median():.6f}')
    bad = scored[~scored['exact']]
    if not bad.empty:
        print(f'\n  {len(bad)} mismatch(es); worst by price drift:')
        print(bad.nlargest(min(10, len(bad)), 'price_drift')[
            ['available_time', 'market_ticker', 'price_drift', 'size_ratio',
             'book_age_ms', 'quality']].to_string(index=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    parser.add_argument('--start', help='UTC bound on the live rows compared')
    parser.add_argument('--end', help='UTC bound, exclusive')
    parser.add_argument('--tolerance-us', type=float,
                        default=DEFAULT_TOLERANCE_US)
    parser.add_argument('--quiet', action='store_true',
                        help='do not announce each hour file as it is folded')
    args = parser.parse_args(argv)

    store = ResearchStore(args.research_store)
    both = compare(store, start=args.start, end=args.end,
                   tolerance_us=args.tolerance_us,
                   on_file=None if args.quiet
                   else lambda p: print(f'folding {p.name}', flush=True))
    if both.empty:
        print('no live stream-sampled rows in range — nothing to compare '
              'against. The replay is unverified here, which is not the same '
              'as verified.')
        return 1
    report(both, tolerance_us=args.tolerance_us)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
