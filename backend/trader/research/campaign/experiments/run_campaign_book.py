"""E7 / E8 / E11 — the candidates that need the venue's book attached.

Separated from the bar-only batch because attaching depth to `Dataset.build` is the
expensive step (0.5s without it, >20 minutes with the full 2.4M-row table). Depth is
pre-filtered to the benchmark era here, which is the only span where a trustworthy
book exists anyway.

Implied vol: SOL is excluded, not by preference but because the store holds 7,446
SOL fits against 79,540 BTC and 51,759 ETH -- too thin to train on.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import harness as H
import runner as R
from run_campaign import Fold, make_folds
from core.config import Config
from core.datastore import ResearchStore
from core.dataset import Dataset, load_minute_bars


def build_with_book():
    cfg = Config(decision_offsets=R.OFFSETS, entry_offsets=R.OFFSETS,
                 max_quote_age_seconds=900.0, peer_max_quote_age_seconds=900.0)
    store = ResearchStore()
    bars = load_minute_bars(cfg, store=store, start=R.BARS_START,
                            end=pd.Timestamp('2026-10-15', tz='UTC'))
    depth = store.read('venue_depth')
    depth['window_open'] = pd.to_datetime(depth.window_open, utc=True)
    # the benchmark era only, and the trustworthy samplers only
    depth = depth[(depth.window_open >= H.BENCH_START) & (depth.window_open < H.BENCH_END)]
    kal = depth[(depth.venue == 'kalshi') & depth.source.isin(H.STREAM_SOURCES)]
    pm  = depth[depth.venue == 'polymarket']
    depth = pd.concat([kal, pm], ignore_index=True)
    print(f'depth restricted to the benchmark era: {len(depth):,} rows '
          f'(kalshi stream {len(kal):,}, polymarket {len(pm):,})', flush=True)
    iv = store.read('venue_implied_vol')
    iv['event_time'] = pd.to_datetime(iv.event_time, utc=True)
    iv = iv[(iv.event_time >= H.BENCH_START) & (iv.event_time < H.BENCH_END)]
    print(f'implied vol fits in the era: {len(iv):,}', flush=True)
    t = time.time()
    ds = Dataset.build(bars, cfg, depth=depth, ladder_fits=iv)
    print(f'Dataset.build with book: {time.time()-t:.1f}s', flush=True)
    return ds, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=2400)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    t0 = time.time()
    ds, cfg = build_with_book()
    px, lab = H.stream_prices(), H.venue_labels()
    folds = make_folds(ds.window_index)
    for f in folds: print('  ' + f.label(), flush=True)

    BAR = ['geometry', 'vol_state', 'microstructure', 'cross_asset', 'offset']
    SPECS = [
      dict(name='E7_implied_vol', hypothesis="the market's own forward sigma, inverted "
           'from the KXBTCD ladder, beats bar-only features', groups=BAR + ['implied_vol'],
           fitter='lgbm', init='market', symbols=['BTC-USD', 'ETH-USD']),
      dict(name='E8_book_features', hypothesis='verified order-book structure (imbalance, '
           'depth, ladder slope) carries information the bar features cannot express',
           groups=BAR + ['market_state', 'book_flow'], fitter='lgbm', init='market'),
      dict(name='E8b_book_only', hypothesis='the book ALONE beats the market -- run '
           'separately because a group that forecasts nothing still absorbs splits',
           groups=['offset', 'market_state', 'book_flow'], fitter='lgbm', init='market'),
      dict(name='E11_cross_venue', hypothesis="Polymarket's price identifies Kalshi "
           'mispricing', groups=BAR + ['cross_venue'], fitter='lgbm', init='market'),
    ]
    if args.only:
        want = set(args.only.split(',')); SPECS = [s for s in SPECS if s['name'] in want]

    results = {}
    for spec in SPECS:
        print(f"\n=== {spec['name']} ===", flush=True)
        res, oof = R.run(ds, cfg, folds, px, lab, budget_s=args.budget, **spec)
        if res['status'] != 'OK':
            print(f"  FAILED: {res['reason']}", flush=True); results[spec['name']] = res; continue
        print(f"  n={res['n_oof']:,} over {res['windows']:,} windows, "
              f"{res['n_features']} features  [{res['runtime_s']}s]")
        print(f"  log loss  model {res['log_loss_model']:.6f}  market {res['log_loss_market']:.6f}")
        print(f"  model_minus_market {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f}, {res['ci95'][1]:+.6f}] p={res['p_one_sided']:.4f}")
        print(f"  VERDICT: {'BEATS the market' if res['beats_market'] else 'does NOT beat the market'}")
        for gate in (1.0, 1.5, 2.5):
            sim = H.simulate(oof, 'p_model', min_edge_pp=gate, offsets=(3, 6, 9, 12))
            sim.pop('_trades', None)
            if sim['trades']:
                print(f"    gate {gate:>4.1f}pp: {sim['trades']:>5,} trades  pnl ${sim['pnl']:>8.2f}  "
                      f"{sim['pp_per_contract']:>+7.3f}pp/ct  t={sim['t_stat']:>5.2f}")
            else:
                print(f"    gate {gate:>4.1f}pp: no trades")
            res.setdefault('money', {})[f'gate_{gate}'] = sim
        results[spec['name']] = res
        if oof is not None:
            oof[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
                 'yes_bid','yes_ask','split']].to_parquet(HERE / f"oof_{spec['name']}.parquet", index=False)
    H.record('campaign_batch_2_book', {'results': results,
                                       'folds': [f.label() for f in folds],
                                       'total_runtime_s': round(time.time()-t0, 1)})
    print(f"\nrecorded. total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
