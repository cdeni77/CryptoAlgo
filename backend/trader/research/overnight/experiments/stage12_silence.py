"""STAGE 12 — WHICH weeks are silent, and would a model deployed today trade?

The tree sat out 6 of 15 folds. That number is only alarming if the silence is
scattered or late. In an EXPANDING walk-forward the first folds train on almost
nothing -- fold 1 sees 5,746 rows against fold 15's 272,695 -- so an early silent
fold says "a model trained on one week has no opinion", which is true and not a
defect of the thing you would actually deploy.

A model promoted today is trained on ALL history to date. Its live analogue is the
LAST fold, not the average. So this reports silence per fold against training size,
and separately scores the deploy-today case.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[2]))
from stage9_full_training import (bar_features, book_and_peer, implied_vol,
                                  FLOW, _sig, _logit, _ll)
from stage7_backfilled_flow import build_snapshot_flow, _boot_ci
from stage11_neural import assemble, sim_per_fold, CONTROL
from stage1_screen import log
OUT = HERE.parent

if __name__ == '__main__':
    t0 = time.time()
    m = assemble()
    from core.features import FEATURE_GROUPS
    bar_cols = [c for g in ('geometry','vol_state','microstructure','cross_asset',
                            'offset','time_of_day')
                for c in FEATURE_GROUPS[g] if c in m.columns]
    extra = [c for c in (FLOW + ['k_spread','k_depth_imb_1c','k_depth_imb_5c',
                                 'k_levels_ratio','pm_mid','pm_spread','iv_sigma'])
             if c in m.columns]
    feats = [c for c in bar_cols + extra if c not in CONTROL]
    sparse = [c for c in extra if m[c].notna().mean() < 0.99]
    for c in sparse:
        m[f'{c}__miss'] = m[c].isna().astype(float)
    feats = feats + [f'{c}__miss' for c in sparse]

    import lightgbm as lgb
    preds = np.full(len(m), np.nan); trainsz = {}
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 2000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        trainsz[int(i)] = int(tr_m.sum())
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
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)

    print("=== TREE: silence vs TRAINING SIZE, per fold ===")
    res = {}
    for gate in (1.0, 1.5):
        t = sim_per_fold(s, gate)
        print(f"\n  gate {gate}pp:")
        print(f"  {'fold':>5} {'train rows':>11} {'test win':>9} {'trades':>8} "
              f"{'pnl$':>9} {'pp/ct':>8}")
        rows=[]
        for i in sorted(s._fold.unique()):
            g = t[t.fold == i] if len(t) else t
            w = s[s._fold == i].window_open.nunique()
            tz = trainsz.get(int(i), 0)
            if not len(g):
                print(f"  {int(i):>5} {tz:>11,} {w:>9,} {'SILENT':>8}")
                rows.append({'fold':int(i),'train_rows':tz,'trades':0}); continue
            print(f"  {int(i):>5} {tz:>11,} {w:>9,} {len(g):>8,} "
                  f"{g.pnl.sum():>9.2f} {g.pnl.sum()/len(g)*100:>+8.3f}")
            rows.append({'fold':int(i),'train_rows':tz,'trades':int(len(g)),
                         'pnl':float(g.pnl.sum())})
        res[f'gate_{gate}'] = rows
        # the deploy-today analogue: the LAST third of folds
        late = [r for r in rows if r['fold'] >= max(rr['fold'] for rr in rows) - 4]
        lt = sum(r['trades'] for r in late); lp = sum(r.get('pnl',0) for r in late)
        ls = sum(1 for r in late if r['trades'] == 0)
        print(f"    LAST 5 FOLDS (the deploy-today analogue): {lt:,} trades  "
              f"${lp:.2f}  silent {ls}/5")
        res[f'gate_{gate}_late'] = {'trades': lt, 'pnl': lp, 'silent': ls}
    s[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
       'yes_bid','yes_ask','ask_at_touch','bid_at_touch','_fold']].to_parquet(
        OUT / 'oof_stage12_tree.parquet', index=False)
    (OUT / 'stage12_silence.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 12, 'question': 'which folds are silent and why', 'result': res,
         'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 12 done in {time.time()-t0:.0f}s")
