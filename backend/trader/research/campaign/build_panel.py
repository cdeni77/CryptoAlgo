"""Build the canonical July-October 2026 research panel, ONCE, and cache it.

Deliberately permissive at build time: `max_quote_age_seconds` is set wide open
and `quote_age_seconds` is kept as a COLUMN, so the age cut is an analysis
decision rather than a build decision. That is what makes D3 answerable without
rebuilding, and it is why no experiment may read this file without stating its
own age cut.

Writes an immutable, hashed parquet under research/campaign/datasets/.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

from core.config import Config, DEFAULT_CONFIG            # noqa: E402
from core.datastore import ResearchStore                  # noqa: E402
from core.dataset import Dataset, load_minute_bars        # noqa: E402

OUT = Path(__file__).resolve().parent / 'datasets'

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-06-01')   # a month of warmup before Jul 1
    ap.add_argument('--end',   default='2026-10-18')
    ap.add_argument('--max-age', type=float, default=900.0)
    ap.add_argument('--version', default='v1')
    ap.add_argument('--window-start', default='2026-07-01')
    ap.add_argument('--window-end',   default='2026-10-18')
    args = ap.parse_args()

    cfg = Config(
        max_quote_age_seconds=args.max_age,
        peer_max_quote_age_seconds=args.max_age,
        null_quote_age='stale',
        decision_latency_seconds=3.0,
        decision_offsets=tuple(range(1, 15)),     # every minute, not just 3/6/9/12
        entry_offsets=tuple(range(1, 15)),
    )
    store = ResearchStore()
    start = pd.Timestamp(args.start, tz='UTC'); end = pd.Timestamp(args.end, tz='UTC')
    bars = load_minute_bars(cfg, store=store, start=start, end=end)
    for s, b in bars.items():
        print(f'bars {s}: {len(b):,} {b.index.min()} -> {b.index.max()}', flush=True)

    depth = store.read('venue_depth')
    print(f'venue_depth: {len(depth):,} rows', flush=True)
    iv = store.read('venue_implied_vol')
    print(f'venue_implied_vol: {len(iv):,} rows', flush=True)

    ds = Dataset.build(bars, cfg, depth=depth, ladder_fits=iv)
    w = ds.windows
    print(f'panel: {len(w):,} rows, {w.window_open.nunique():,} windows', flush=True)

    # the venue's own label, joined explicitly and kept SEPARATE from the proxy
    sett = store.read('venue_settlements')
    k = sett[(sett.venue == 'kalshi') & sett.settled_up.notna()][
        ['symbol', 'window_open', 'settled_up', 'market_ticker']].copy()
    k = k.rename(columns={'settled_up': 'venue_settled_up',
                          'market_ticker': 'venue_market_ticker'})
    k['window_open'] = pd.to_datetime(k.window_open, utc=True)
    k = k.drop_duplicates(['symbol', 'window_open'])
    w['window_open'] = pd.to_datetime(w.window_open, utc=True)
    w = w.merge(k, on=['symbol', 'window_open'], how='left')
    print(f'venue label attached to {w.venue_settled_up.notna().sum():,} of {len(w):,}', flush=True)

    w = w[(w.window_open >= pd.Timestamp(args.window_start, tz='UTC')) &
          (w.window_open <  pd.Timestamp(args.window_end, tz='UTC'))].reset_index(drop=True)
    print(f'panel window: {len(w):,} rows, {w.window_open.nunique():,} windows', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'panel_{args.version}.parquet'
    w.to_parquet(path, index=False)
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    meta = {
        'version': args.version, 'sha256_16': h, 'rows': int(len(w)),
        'windows': int(w.window_open.nunique()),
        'start': str(w.window_open.min()), 'end': str(w.window_open.max()),
        'build_args': vars(args), 'columns': list(w.columns),
        'config': {k_: str(v) for k_, v in vars(cfg).items()},
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
    }
    (OUT / f'panel_{args.version}.meta.json').write_text(json.dumps(meta, indent=2))
    print(f'wrote {path}  sha256[:16]={h}  rows={len(w):,}')

if __name__ == '__main__':
    main()
