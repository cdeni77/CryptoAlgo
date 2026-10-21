"""STAGE 14 — WHICH of the six new features carries the 10x gap?

Stage 13: the repo's own 53 features score +0.000100 (row). Stage 12's set, which
adds six columns, scores +0.001050. Six features producing ten times the edge is a
large claim, and porting production code for the wrong one would be wasted work --
or worse, would ship a feature that turns out to be an artifact.

The six: d_imbalance_1, d_imbalance_3, d_touch_imb_1, d_book_mid_1, d_book_mid_3,
k_levels_ratio.

`d_book_mid_*` deserves particular suspicion: it is the market's OWN recent price
change, and the model already receives the market logit as its init score. If that
one carries the result, the model is trading quote momentum, which is a real and
well-documented effect but a different claim from "the order book predicts
settlement" -- and it would need its own validation.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[2]))
from stage9_full_training import FLOW, _sig, _logit, _ll
from stage7_backfilled_flow import _boot_ci
from stage11_neural import assemble, CONTROL
from stage1_screen import log
OUT = HERE.parent

NEW = ['d_imbalance_1', 'd_imbalance_3', 'd_touch_imb_1',
       'd_book_mid_1', 'd_book_mid_3', 'k_levels_ratio']


def fit(m, feats, label):
    import lightgbm as lgb
    preds = np.full(len(m), np.nan)
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 2000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
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
        preds[te_m] = _sig(_logit(te.p_mkt)
                           + np.asarray(bst.predict(te[feats], raw_score=True), float))
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    d = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    out = {}
    print(f"  {label:<40}", end='')
    for lbl, v in (('row', d),
                   ('win', pd.DataFrame({'d': d, 'w': s.window_open})
                            .groupby('w')['d'].mean().to_numpy())):
        lo, hi = _boot_ci(np.asarray(v))
        out[lbl] = {'mean': float(np.mean(v)), 'ci': [lo, hi]}
        print(f"  {lbl} {np.mean(v):+.6f} [{lo:+.6f},{hi:+.6f}]", end='')
    print()
    return out


if __name__ == '__main__':
    t0 = time.time()
    m = assemble()
    from core.features import FEATURE_GROUPS
    bar = [c for g in ('geometry','vol_state','microstructure','cross_asset','offset')
           for c in FEATURE_GROUPS[g] if c in m.columns]
    base = [c for c in bar + [x for x in FLOW if x not in NEW] +
            ['k_spread','k_depth_imb_1c','k_depth_imb_5c','pm_mid','pm_spread','iv_sigma']
            if c in m.columns and c not in NEW]
    print(f"rows {len(m):,}  base features {len(base)}  new {len(NEW)}\n")
    res = {}
    res['base_only'] = fit(m, base, 'BASE (no new features)')
    res['base_plus_all'] = fit(m, base + [c for c in NEW if c in m.columns],
                               'BASE + all six new')
    print()
    for c in NEW:
        if c not in m.columns: continue
        res[f'base_plus_{c}'] = fit(m, base + [c], f'BASE + {c}')
    print()
    res['all_minus_bookmid'] = fit(
        m, base + [c for c in NEW if c in m.columns and not c.startswith('d_book_mid')],
        'BASE + new MINUS d_book_mid_*')
    (OUT / 'stage14_isolate.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 14, 'experiment': 'isolate which new feature carries the gap',
         'result': res, 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 14 done in {time.time()-t0:.0f}s")
