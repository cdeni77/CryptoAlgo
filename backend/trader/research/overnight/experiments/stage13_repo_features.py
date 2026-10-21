"""STAGE 13 — can the REPO'S OWN feature groups reproduce Stage 12's result?

Stage 12 scored model_minus_market +0.001050 (row, CI excluding zero on all three
weightings) using a research feature set. Mapping that set onto `core/features.py`
shows most of it already exists:

    mine                 repo                      group
    touch_imbalance  ->  imbalance_touch           market_state
    k_depth_imb_5c   ->  imbalance_5c              market_state
    k_spread         ->  spread                    market_state
    imbalance        ~~   depth_ratio              market_state  (monotone transform:
                                                   log(a/b) vs (a-b)/(a+b) -- identical
                                                   to a tree, which is invariant to a
                                                   monotone map of a single feature)
    d_touch_imb_3    ->  imbalance_change_3        book_flow
    pm_mid/pm_spread ->  venue_prob_gap, …         cross_venue
    iv_sigma         ->  implied_sigma_per_min     implied_vol

Genuinely absent: d_imbalance_1/3, d_book_mid_1/3, d_touch_imb_1, k_levels_ratio.

If the repo's groups reproduce the result under WEEKLY folds with EARLY STOPPING --
the two things that actually changed tonight -- then the model is promotable today
with no new feature code, and the port is unnecessary. If they fall short, the gap
measures exactly what the six missing features are worth, which is the number that
justifies writing them.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[2]))
from stage7_backfilled_flow import _boot_ci
from stage1_screen import log
from core.config import Config
from core.datastore import ResearchStore
from core.dataset import Dataset, load_minute_bars, fit_fold, apply_fold
from core.features import FEATURE_GROUPS
OUT = HERE.parent
LO = pd.Timestamp('2026-07-01', tz='UTC'); HI = pd.Timestamp('2026-10-18', tz='UTC')

GROUPS = ['geometry', 'vol_state', 'microstructure', 'cross_asset', 'offset',
          'market_state', 'book_flow', 'cross_venue', 'implied_vol']


def _sig(x): return 1/(1+np.exp(-x))
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); return np.log(p/(1-p))
def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


if __name__ == '__main__':
    t0 = time.time()
    store = ResearchStore()
    cfg = Config(decision_offsets=tuple(range(1, 15)), entry_offsets=tuple(range(1, 15)),
                 max_quote_age_seconds=900.0, peer_max_quote_age_seconds=900.0)
    bars = load_minute_bars(cfg, store=store,
                            start=pd.Timestamp('2026-01-01', tz='UTC'), end=HI)
    depth = store.read('venue_depth')
    depth['window_open'] = pd.to_datetime(depth.window_open, utc=True)
    depth = depth[(depth.window_open >= LO) & (depth.window_open < HI)]
    iv = store.read('venue_implied_vol')
    print(f'depth {len(depth):,}  iv {len(iv):,}  [{time.time()-t0:.0f}s]', flush=True)
    ds = Dataset.build(bars, cfg, depth=depth, ladder_fits=iv)
    print(f'dataset built [{time.time()-t0:.0f}s]', flush=True)

    idx = ds.window_index; bench = idx[(idx >= LO) & (idx < HI)]
    wk = pd.Series(bench).dt.to_period('W').astype(str).to_numpy()
    blocks = [pd.DatetimeIndex(bench[wk == w]) for w in pd.unique(wk)]
    blocks = [b for b in blocks if len(b) >= 200]
    print(f'weekly blocks: {len(blocks)}', flush=True)

    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable][['symbol','window_open','offset_minutes','p_mkt','y',
                        'yes_bid','yes_ask','ask_at_touch','bid_at_touch']]

    import lightgbm as lgb
    parts = []
    for i, b in enumerate(blocks):
        cut = pd.Timestamp(b.min()) - pd.Timedelta(days=1)
        tr_w = idx[idx < cut]
        if len(tr_w) < 1000: continue
        fit, trt = fit_fold(ds, pd.DatetimeIndex(tr_w), cfg, groups=GROUPS)
        tet = apply_fold(ds, fit, b, cfg, groups=GROUPS)
        for d in (trt, tet):
            d.rename(columns={'offset': 'offset_minutes'}, inplace=True)
            d['window_open'] = pd.to_datetime(d.window_open, utc=True)
        tr = trt.merge(mv, on=['symbol','window_open','offset_minutes'], how='inner',
                       suffixes=('', '_mv'))
        te = tet.merge(mv, on=['symbol','window_open','offset_minutes'], how='inner',
                       suffixes=('', '_mv'))
        feats = sorted({c for g in GROUPS for c in FEATURE_GROUPS[g] if c in tr.columns})
        if len(tr) < 2000 or len(te) == 0: continue
        iw = np.array(sorted(pd.unique(tr.window_open))); icut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < icut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= icut]
        if len(itr) < 1000 or len(iva) < 500: itr, iva = tr, tr
        par = dict(objective='binary', metric='binary_logloss', verbosity=-1,
                   learning_rate=0.02, num_leaves=15, max_depth=4,
                   min_child_samples=500, bagging_fraction=0.8, bagging_freq=1,
                   feature_fraction=0.7, lambda_l2=20.0, deterministic=True, seed=17)
        dtr = lgb.Dataset(itr[feats], label=itr.y.to_numpy(int),
                          init_score=_logit(itr.p_mkt), free_raw_data=False)
        dva = lgb.Dataset(iva[feats], label=iva.y.to_numpy(int),
                          init_score=_logit(iva.p_mkt), reference=dtr, free_raw_data=False)
        bst = lgb.train(par, dtr, num_boost_round=600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        te = te.copy()
        te['p_model'] = _sig(_logit(te.p_mkt)
                             + np.asarray(bst.predict(te[feats], raw_score=True), float))
        te['_fold'] = i
        parts.append(te)
        print(f'  fold {i}: train {len(tr):,} -> test {len(te):,}  '
              f'iters {bst.best_iteration}  feats {len(feats)}', flush=True)

    s = pd.concat(parts, ignore_index=True)
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    print(f'\n=== REPO FEATURES ONLY ({len(feats)} features) ===')
    res = {'features': feats, 'n_features': len(feats), 'n_oof': int(len(s))}
    for lbl, v in (('row', s.d.to_numpy()),
                   ('sym-win', s.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f'  {lbl:>8} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}'
              f"{'   BEATS THE MARKET' if lo > 0 else ''}")
        res[lbl] = {'mean': float(v.mean()), 'ci': [lo, hi], 'n': int(len(v))}
    print('\n  Stage 12 (research features) was: row +0.001050 [+0.000879,+0.001226]')

    # saved BEFORE the money loop: the dataset build costs ~11 minutes and a
    # crash in the simulator must not throw it away (it did, once)
    s.to_parquet(OUT / 'oof_stage13_repo.parquet', index=False)
    print('  (OOF saved)')
    rows = []
    d = s.rename(columns={'_fold': 'foldid'}).sort_values(
        ['symbol','window_open','offset_minutes'])
    for (sym, w), g in d.groupby(['symbol','window_open'], sort=False):
        for r in g.itertuples():
            pm = r.p_model
            if not np.isfinite(pm): continue
            yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
            ev_y, ev_n = pm - yes_px, (1.0 - pm) - no_px
            side, px, ev = ('yes', yes_px, ev_y) if ev_y >= ev_n else ('no', no_px, ev_n)
            if not (0.0 < px < 1.0): continue
            fee = np.ceil(0.07*px*(1-px)*10_000)/10_000
            if ev - fee < 0.015: continue
            vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
            if not np.isfinite(vol) or vol < 1: continue
            won = (r.y == 1) if side == 'yes' else (r.y == 0)
            rows.append({'fold': r.foldid, 'pnl': (1.0 if won else 0.0) - px - fee,
                         'won': won, 'window_open': w})
            break
    t = pd.DataFrame(rows)
    if len(t):
        gw = t.groupby('window_open').pnl.sum()
        ts = gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw)))
        silent = sum(1 for i in sorted(s._fold.unique()) if not len(t[t.fold == i]))
        late = sorted(s._fold.unique())[-5:]
        lt = t[t.fold.isin(late)]
        print(f'\n  @1.5pp: {len(t):,} trades  ${t.pnl.sum():.2f}  '
              f'{t.pnl.sum()/len(t)*100:+.3f}pp/ct  t={ts:.2f}  silent {silent}/{len(blocks)}')
        print(f'  last 5 folds: {len(lt):,} trades  ${lt.pnl.sum():.2f}  silent '
              f'{sum(1 for i in late if not len(t[t.fold==i]))}/5')
        res['money'] = {'trades': int(len(t)), 'pnl': float(t.pnl.sum()),
                        'pp_per_contract': float(t.pnl.sum()/len(t)*100),
                        't_stat': float(ts), 'silent_folds': int(silent),
                        'late_trades': int(len(lt)), 'late_pnl': float(lt.pnl.sum())}
    (OUT / 'stage13_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 13, 'experiment': 'repo feature groups only, weekly folds + early stop',
         'result': {k: v for k, v in res.items() if k != 'features'},
         'runtime_s': round(time.time()-t0, 1)})
    print(f'\nstage 13 done in {time.time()-t0:.0f}s')
