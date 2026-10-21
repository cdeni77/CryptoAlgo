"""STAGE 7 — the surviving order-flow signals, BACKFILLED from venue_depth.

The tape gave 15 days. But the six signals that survived Stage 1 screening --
`imbalance`, `d_imbalance_1/3`, `d_touch_imb_3`, `d_book_mid_1/3` -- are all
functions of book SNAPSHOTS, not of the delta stream. The add/remove decomposition
is the only thing that needs raw deltas, and every one of those screened null.

`venue_depth` holds per-minute snapshots with `depth_bid_total`, `depth_ask_total`,
`yes_bid_size`, `yes_ask_size` and the touch prices from 2026-01 onward, populated
on 96.7% of rows. So the same signals can be built over the whole July-October
window -- roughly 7x the evaluation sample.

**Caveat carried forward, not hidden:** Predexon's snapshot is a coarser projection
of the venue's ladder (CLAUDE.md measures `levels_bid`/`levels_ask` at a 0.579 ratio
across sources), so these totals are not numerically the same quantity as the
tape-reconstructed ones. They are self-consistent WITHIN the backfill, which is what
a feature needs. The tape result and this result are therefore related but not
identical experiments, and both are reported.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
from stage1_screen import log
from stage2_models import folds_of, _sig, _logit, _ll, window_boot_delta
OUT = HERE.parent
STORE = HERE.parents[2] / 'data' / 'research'


def build_snapshot_flow() -> pd.DataFrame:
    """Read the Kalshi backfill snapshots straight from the month partitions.

    **No DuckDB here, deliberately.** The first version ran
    `read_parquet(..., union_by_name=true)` over every partition and then a
    `row_number()` window, filtering last. DuckDB cannot push a WHERE through a
    window function, and `union_by_name` across 60 partitions with differing
    schemas is pathological, so it buffered its way to 20.7 GB and the kernel
    killed it twice. The files are 7 MB each and ~100 MB in total -- pandas reads
    the twelve that matter directly and peaks at a few hundred MB.
    """
    root = STORE / 'venue_depth' / 'venue=kalshi'
    months = ('2026-01','2026-02','2026-03','2026-04','2026-05','2026-06',
              '2026-07','2026-08','2026-09','2026-10')
    cols = ['symbol', 'window_open', 'offset_minutes', 'event_time', 'available_time',
            'source', 'yes_bid', 'yes_ask', 'yes_bid_size', 'yes_ask_size',
            'depth_bid_total', 'depth_ask_total', 'levels_bid', 'levels_ask',
            'quote_age_seconds']
    parts = []
    for sym_dir in sorted(root.glob('symbol=*')):
        for mo in months:
            f = sym_dir / f'month={mo}' / 'data.parquet'
            if not f.exists():
                continue
            d = pd.read_parquet(f)
            d = d[[c for c in cols if c in d.columns]]
            d = d[(d.source == 'backfill') & d.offset_minutes.between(0, 15)]
            parts.append(d)
    if not parts:
        raise SystemExit('no kalshi backfill partitions found')
    df = pd.concat(parts, ignore_index=True)
    df['window_open'] = pd.to_datetime(df.window_open, utc=True)
    df = df[(df.window_open >= pd.Timestamp('2026-01-01', tz='UTC')) &
            (df.window_open < pd.Timestamp('2026-10-19', tz='UTC'))]
    # *** AS-OF RULE. This was the worst bug of 2026-10-17. ***
    # `drop_duplicates(keep='last')` alone takes the newest revision of each
    # (symbol, event_time, offset) regardless of whether it was knowable at the
    # decision instant. Measured: 25.3% of Kalshi rows and 48.0% of Polymarket rows
    # carried quotes from the FUTURE, median 25-30s ahead. Everything built on it
    # (Stages 9/11/12/14) was invalid.
    df['available_time'] = pd.to_datetime(df.available_time, utc=True)
    decision = (df.window_open + pd.to_timedelta(df.offset_minutes, unit='m')
                + pd.Timedelta(seconds=3))
    df = df[df.available_time <= decision]
    df = (df.sort_values('available_time')
            .drop_duplicates(['symbol', 'event_time', 'offset_minutes'], keep='last'))
    df = df.sort_values(['symbol', 'window_open', 'offset_minutes'])
    eps = 1e-9
    bt, at = df.depth_bid_total.fillna(0), df.depth_ask_total.fillna(0)
    df['imbalance'] = (bt - at) / (bt + at + eps)
    tb, ta = df.yes_bid_size.fillna(0), df.yes_ask_size.fillna(0)
    df['touch_imbalance'] = (tb - ta) / (tb + ta + eps)
    df['book_mid'] = (df.yes_bid + df.yes_ask) / 2
    g = df.groupby(['symbol', 'window_open'], sort=False)
    for k in (1, 3):
        df[f'd_imbalance_{k}'] = df.imbalance - g.imbalance.shift(k)
        df[f'd_touch_imb_{k}'] = df.touch_imbalance - g.touch_imbalance.shift(k)
        df[f'd_book_mid_{k}'] = df.book_mid - g.book_mid.shift(k)
    return df.reset_index(drop=True)



def _boot_ci(v, B=10000, seed=0, batch=200):
    """Bootstrap CI without materialising the whole resample matrix.

    **This is the bug that OOM-killed the machine three times.**
    `rng.integers(0, n, size=(B, n))` with B=10,000 and n=226,576 rows asks for a
    10,000 x 226,576 int64 array -- 16.9 GiB in a single allocation. It is
    harmless at the window level (n ~ 9,500) and fatal at the row level, which is
    why it only showed up once a row-weighted CI was added. Drawing in batches
    costs nothing and caps the peak at `batch * n` indices.
    """
    v = np.asarray(v, float); n = len(v)
    rng = np.random.default_rng(seed)
    means = np.empty(B, float)
    for i in range(0, B, batch):
        k = min(batch, B - i)
        means[i:i + k] = v[rng.integers(0, n, size=(k, n))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


FEATS = ['imbalance', 'd_imbalance_1', 'd_imbalance_3', 'd_touch_imb_3',
         'd_book_mid_1', 'd_book_mid_3']


def fit_oof(m, feats, seed=17, n_folds=5):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    preds = np.full(len(m), np.nan); iters = []
    for i, trw, tew in folds_of(m.window_open.to_numpy(), n_folds=n_folds):
        tr = m[m.window_open.isin(trw)]; te = m[m.window_open.isin(tew)]
        if len(tr) < 2000 or len(te) == 0: continue
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 1000 or len(iva) < 500: itr, iva = tr, tr
        imp = SimpleImputer(strategy='median').fit(itr[feats].to_numpy(float))
        p = dict(objective='binary', metric='binary_logloss', verbosity=-1,
                 learning_rate=0.02, num_leaves=7, max_depth=3,
                 min_child_samples=500, bagging_fraction=0.8, bagging_freq=1,
                 feature_fraction=0.8, lambda_l2=20.0, deterministic=True, seed=seed)
        dtr = lgb.Dataset(imp.transform(itr[feats].to_numpy(float)),
                          label=itr.y.to_numpy(int),
                          init_score=_logit(itr.p_mkt), free_raw_data=False)
        dva = lgb.Dataset(imp.transform(iva[feats].to_numpy(float)),
                          label=iva.y.to_numpy(int), init_score=_logit(iva.p_mkt),
                          reference=dtr, free_raw_data=False)
        bst = lgb.train(p, dtr, num_boost_round=500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
        corr = np.asarray(bst.predict(imp.transform(te[feats].to_numpy(float)),
                                      raw_score=True), float)
        preds[m.window_open.isin(tew).to_numpy()] = _sig(_logit(te.p_mkt) + corr)
        iters.append(bst.best_iteration)
    return preds, iters


if __name__ == '__main__':
    t0 = time.time(); res = {}
    flow = build_snapshot_flow()
    print(f"snapshot flow built: {len(flow):,} rows [{time.time()-t0:.0f}s]")
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable]
    m = flow.drop(columns=['yes_bid','yes_ask','quote_age_seconds']).merge(
        mv, on=['symbol','window_open','offset_minutes'], how='inner')
    m = m[m.offset_minutes.between(1,14)].reset_index(drop=True)
    print(f"joined to verified market: {len(m):,} rows, {m.window_open.nunique():,} windows, "
          f"{m.groupby(['symbol','window_open']).ngroups:,} symbol-windows")
    print(f"  span {m.window_open.min()} -> {m.window_open.max()}")

    folds = folds_of(m.window_open.to_numpy(), n_folds=5)
    print(f"  folds: {len(folds)}")
    preds, iters = fit_oof(m, FEATS)
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    print(f"\n=== model_minus_market, Jul-Oct backfilled flow (n={ok.sum():,}) ===")
    for lbl, v in (('row', s.d.to_numpy()),
                   ('symbol-window', s.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f"  {lbl:>14} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}"
              f"{'   BEATS' if lo>0 else ''}")
        res[lbl] = {'mean': float(v.mean()), 'ci': [float(lo), float(hi)], 'n': int(len(v))}
    print(f"  best_iterations: {iters}")
    res['best_iters'] = iters

    print("\n=== stability ===")
    for by in ('symbol','month'):
        s['month'] = s.window_open.dt.to_period('M').astype(str)
        rows=[]
        for k,g in s.groupby(by):
            if len(g)<1000: continue
            mean,lo,hi,nw = window_boot_delta(g.d.to_numpy(), g.window_open.to_numpy())
            rows.append({by:str(k),'n':len(g),'w':nw,'mmm':mean,'ci':[lo,hi]})
            print(f"  {str(k):>10} n={len(g):>7,} w={nw:>5,} {mean:+.6f} "
                  f"[{lo:+.6f},{hi:+.6f}]{'  <<<' if lo>0 else ''}")
        res[f'by_{by}'] = rows
    s[['symbol','window_open','offset_minutes','p_model','p_mkt','y','yes_bid','yes_ask']].to_parquet(
        OUT / 'oof_stage7_backfilled_flow.parquet', index=False)
    (OUT / 'stage7_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 7, 'hypothesis': 'H2 backfilled from venue_depth snapshots',
         'result': res, 'runtime_s': round(time.time()-t0,1)})
    print(f"\nstage 7 done in {time.time()-t0:.0f}s")
