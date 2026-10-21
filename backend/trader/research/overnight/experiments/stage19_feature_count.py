"""STAGE 19 — are 65 features too many for ~20,400 effective observations?

The effective sample is WINDOWS, not rows: four offsets share a settlement and the
three symbols are ~0.7 correlated inside a window. 473,714 rows is ~20,429 windows,
so 65 features is ~314 independent observations each -- thin for resolving an effect
of ~0.0008 nats.

**Selection happens INSIDE each training fold.** Ranking features by importance on
the whole sample and then scoring would leak the test block into the feature set,
which is the same class of error as the availability bug. Here each fold fits on all
features, ranks by gain on its OWN training data, refits on the top K, and only then
touches its test block.
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
from stage15_nn_candidate import assemble, simulate, CONTROL
from stage1_screen import log
OUT = HERE.parent
PAR = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
           learning_rate=0.02, max_depth=4, min_child_weight=125.0, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=20.0, seed=17, nthread=8)


def run(m, allfeats, topk, label):
    import xgboost as xgb
    preds = np.full(len(m), np.nan); kept = {}
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 3000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 1500 or len(iva) < 800: itr, iva = tr, tr
        feats = allfeats
        if topk is not None and topk < len(allfeats):
            # rank on TRAINING data only
            d0 = xgb.DMatrix(itr[allfeats], label=itr.y.to_numpy(int),
                             base_margin=_logit(itr.p_mkt))
            b0 = xgb.train(PAR, d0, num_boost_round=120, verbose_eval=False)
            g = b0.get_score(importance_type='gain')
            feats = [f for f, _ in sorted(g.items(), key=lambda kv: -kv[1])][:topk]
            if len(feats) < 2: feats = allfeats[:topk]
        kept[int(i)] = feats
        dtr = xgb.DMatrix(itr[feats], label=itr.y.to_numpy(int), base_margin=_logit(itr.p_mkt))
        dva = xgb.DMatrix(iva[feats], label=iva.y.to_numpy(int), base_margin=_logit(iva.p_mkt))
        dte = xgb.DMatrix(te[feats], base_margin=_logit(te.p_mkt))
        bst = xgb.train(PAR, dtr, num_boost_round=600, evals=[(dva, 'v')],
                        early_stopping_rounds=50, verbose_eval=False)
        it = bst.best_iteration
        full = np.asarray(bst.predict(dte, output_margin=True, iteration_range=(0, it+1)), float)
        preds[te_m] = _sig(full)
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s = s.rename(columns={'_fold': 'foldid'})
    d = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    wv = pd.DataFrame({'d': d, 'w': s.window_open}).groupby('w')['d'].mean().to_numpy()
    lo, hi = _boot_ci(wv)
    t = simulate(s)
    if len(t):
        gw = t.groupby('window_open').pnl.sum()
        ts = gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw)))
        pnl, pp, nct = t.pnl.sum(), t.pnl.sum()/t.contracts.sum()*100, int(t.contracts.sum())
        folds = sorted(s.foldid.unique()); last5 = folds[-5:]
        tr5 = sum(1 for f in last5 if len(t[t.fold == f]))
    else:
        pnl=pp=ts=0.0; nct=0; tr5=0
    print(f"  {label:<22} mmm {wv.mean():+.6f} [{lo:+.6f},{hi:+.6f}]   "
          f"${pnl:>+9.2f}  {pp:>+6.3f}pp/ct  t={ts:>+5.2f}  last5 {tr5}/5")
    return {'label': label, 'topk': topk, 'mmm': float(wv.mean()), 'ci': [lo, hi],
            'pnl': float(pnl), 'pp_per_contract': float(pp), 't_stat': float(ts),
            'contracts': nct, 'traded_last5': tr5,
            'features_last_fold': kept.get(max(kept), []) if kept else []}


if __name__ == '__main__':
    t0 = time.time(); m = assemble()
    from core.features import FEATURE_GROUPS
    bar = [c for g in ('geometry','vol_state','microstructure','cross_asset','offset',
                       'time_of_day') for c in FEATURE_GROUPS[g] if c in m.columns]
    extra = [c for c in (FLOW + ['k_spread','k_depth_imb_1c','k_depth_imb_5c',
                                 'k_levels_ratio','pm_mid','pm_spread','iv_sigma'])
             if c in m.columns]
    feats = [c for c in bar + extra if c not in CONTROL]
    sparse = [c for c in extra if m[c].notna().mean() < 0.99]
    for c in sparse: m[f'{c}__miss'] = m[c].isna().astype(float)
    feats = feats + [f'{c}__miss' for c in sparse]
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}  "
          f"all features {len(feats)}  (~{m.window_open.nunique()//len(feats)} windows each)\n")
    res = []
    for k in (5, 10, 15, 25, 40, None):
        res.append(run(m, feats, k, f'top {k}' if k else f'ALL {len(feats)}'))
    (OUT / 'stage19_feature_count.json').write_text(json.dumps(res, indent=2, default=str))
    best = max(res, key=lambda r: r['mmm'])
    print(f"\n  best forecast at: {best['label']}  (mmm {best['mmm']:+.6f})")
    bp = max(res, key=lambda r: r['pnl'])
    print(f"  best money at:    {bp['label']}  (${bp['pnl']:+.2f}, t={bp['t_stat']:+.2f})")
    if res[-1]['features_last_fold']:
        pass
    log({'stage': 19, 'experiment': 'feature-count ablation, selection inside fold',
         'result': res, 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 19 done in {time.time()-t0:.0f}s")
