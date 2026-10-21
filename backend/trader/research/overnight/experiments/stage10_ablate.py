"""STAGE 10 — the two ablations that decide whether Stage 9 is real.

Stage 9 reported model_minus_market +0.000534 (row) / +0.000598 (symbol-window),
0 silent folds, and +$102.75 at t=3.33. Two findings make that provisional:

  1. the time_of_day CONTROL is 27.72% of total gain -- the largest single group.
     CLAUDE.md: "a high LightGBM gain share means splits were spent there, not that
     the feature forecasts anything. Run the group alone before believing the share
     either way," and records clock alone at -0.000008.
  2. fold 4 (October) holds 3,050 of 3,098 trades and $99.99 of the $102.75.

So: (A) drop the control entirely; (B) drop it AND score folds 1-3 only. If the
edge survives both, it is not the clock and not one month. If it collapses, Stage 9
measured its own control.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[2]))
import stage9_full_training as S9
from stage9_full_training import (bar_features, book_and_peer, implied_vol,
                                  FLOW, _sig, _logit, _ll, LO, HI)
from stage7_backfilled_flow import build_snapshot_flow, _boot_ci
from stage1_screen import log
OUT = HERE.parent
CONTROL = ['quarter_of_hour', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
           'us_equity_hours']


def assemble():
    bars = bar_features()
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable][['symbol','window_open','offset_minutes','p_mkt','y',
                        'yes_bid','yes_ask','ask_at_touch','bid_at_touch']]
    m = bars.merge(mv, on=['symbol','window_open','offset_minutes'], how='inner')
    flow = build_snapshot_flow()[['symbol','window_open','offset_minutes']+FLOW]
    m = m.merge(flow, on=['symbol','window_open','offset_minutes'], how='left')
    k, p = book_and_peer()
    if len(k): m = m.merge(k, on=['symbol','window_open','offset_minutes'], how='left')
    if len(p): m = m.merge(p, on=['symbol','window_open','offset_minutes'], how='left')
    iv = implied_vol()
    if len(iv): m = m.merge(iv, on=['symbol','window_open','offset_minutes'], how='left')
    return m[m.offset_minutes.between(1,14)].reset_index(drop=True)


def run(m, feats, label, folds=None):
    import lightgbm as lgb
    preds = np.full(len(m), np.nan); iters = []
    use = sorted(m._fold.unique()) if folds is None else folds
    for i in use:
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 5000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 2000 or len(iva) < 1000: itr, iva = tr, tr
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
        preds[te_m] = _sig(_logit(te.p_mkt)
                           + np.asarray(bst.predict(te[feats], raw_score=True), float))
        iters.append(bst.best_iteration)
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    out = {'label': label, 'n_features': len(feats), 'n_oof': int(ok.sum()),
           'iters': iters}
    print(f"\n  {label}")
    for lbl, v in (('row', s.d.to_numpy()),
                   ('sym-win', s.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        flag = '  BEATS' if lo > 0 else ''
        print(f"    {lbl:>8} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}{flag}")
        out[lbl] = {'mean': float(v.mean()), 'ci': [lo, hi], 'n': int(len(v))}
    return out, s


if __name__ == '__main__':
    t0 = time.time(); res = {}
    m = assemble()
    from core.features import FEATURE_GROUPS
    bar_cols = [c for g in ('geometry','vol_state','microstructure','cross_asset',
                            'offset','time_of_day')
                for c in FEATURE_GROUPS[g] if c in m.columns]
    extra = [c for c in (FLOW + ['k_spread','k_depth_imb_1c','k_depth_imb_5c',
                                 'k_levels_ratio','pm_mid','pm_spread','iv_sigma'])
             if c in m.columns]
    full = bar_cols + extra
    noctrl = [c for c in full if c not in CONTROL]
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}")
    print(f"full {len(full)} features; without control {len(noctrl)}")
    print("\n=== ABLATION ===")
    res['A_full'], _ = run(m, full, 'A  full (control included)   [Stage 9 repro]')
    res['B_noctrl'], s_b = run(m, noctrl, 'B  control REMOVED')
    res['C_noctrl_f123'], _ = run(m, noctrl, 'C  control REMOVED, folds 1-3 only (no October)',
                                  folds=[1, 2, 3])
    res['D_ctrl_only'], _ = run(m, [c for c in CONTROL if c in m.columns],
                                'D  control ALONE (must be ~0)')
    (OUT / 'stage10_ablation.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 10, 'experiment': 'control + October ablation', 'result': res,
         'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 10 done in {time.time()-t0:.0f}s")
