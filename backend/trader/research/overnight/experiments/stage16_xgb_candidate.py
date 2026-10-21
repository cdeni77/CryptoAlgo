"""STAGE 16 — the same specification, XGBoost instead of the net.

  XGBoost | July 1 -> today | ALL features | min edge 0.5pp | Kelly 0.25
  PROMOTE IF: profitable AND trades in at least 4 of the 5 most recent folds.

Identical data, folds, labels, sizing and gates as Stage 15, so the two are directly
comparable and the only thing that varies is the estimator.

`base_margin` carries the market logit. Verified in the boosting audit: XGBoost's
`predict(output_margin=True)` INCLUDES base_margin while LightGBM's
`predict(raw_score=True)` EXCLUDES init_score -- a naive port that adds it back is
off by 0.105 in probability. The wrapper here returns the CORRECTION alone, so
`sigmoid(margin + correction)` is the model, exactly as in Stage 15.
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
from stage15_nn_candidate import assemble, simulate, MIN_EDGE_PP, KELLY, BANKROLL, CONTROL
from stage1_screen import log
OUT = HERE.parent


def fit_xgb(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte, seed=17):
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=mtr)
    dva = xgb.DMatrix(Xva, label=yva, base_margin=mva)
    dte = xgb.DMatrix(Xte, base_margin=mte)
    par = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
               learning_rate=0.02, max_depth=4, min_child_weight=125.0,
               subsample=0.8, colsample_bytree=0.7, reg_lambda=20.0, reg_alpha=0.0,
               gamma=0.0, seed=seed, nthread=8)
    bst = xgb.train(par, dtr, num_boost_round=600, evals=[(dva, 'inner')],
                    early_stopping_rounds=50, verbose_eval=False)
    it = bst.best_iteration
    rng = (0, it + 1)
    full = np.asarray(bst.predict(dte, output_margin=True, iteration_range=rng), float)
    return full - mte, it + 1          # the CORRECTION alone


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
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}  features {len(feats)}")
    print(f"  span {m.window_open.min().date()} -> {m.window_open.max().date()}")

    preds = np.full(len(m), np.nan); iters = []
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 3000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 1500 or len(iva) < 800: itr, iva = tr, tr
        corr, nit = fit_xgb(itr[feats].to_numpy(float), itr.y.to_numpy(int), _logit(itr.p_mkt),
                            iva[feats].to_numpy(float), iva.y.to_numpy(int), _logit(iva.p_mkt),
                            te[feats].to_numpy(float), _logit(te.p_mkt))
        preds[te_m] = _sig(_logit(te.p_mkt) + corr); iters.append(nit)
        print(f"  fold {i}: train {tr_m.sum():,} -> test {te_m.sum():,}  iters {nit}",
              flush=True)

    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s = s.rename(columns={'_fold': 'foldid'})
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    res = {'n_features': len(feats), 'iters': iters, 'min_edge_pp': MIN_EDGE_PP,
           'kelly': KELLY, 'bankroll': BANKROLL}
    print("\n=== forecast (reported, not a veto) ===")
    for lbl, v in (('row', s.d.to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f"  model_minus_market {lbl:>7}: {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]")
        res[f'mmm_{lbl}'] = {'mean': float(v.mean()), 'ci': [lo, hi]}

    t = simulate(s)
    folds = sorted(s.foldid.unique())
    print(f"\n=== THE PROMOTION GATES (min edge {MIN_EDGE_PP}pp, Kelly {KELLY}) ===")
    if not len(t):
        print("  NO TRADES -> FAILS"); res['verdict'] = 'FAIL: no trades'
    else:
        print(f"  {'fold':>5} {'windows':>8} {'trades':>7} {'contracts':>10} {'pnl$':>10} {'pp/ct':>8}")
        per = []
        for f in folds:
            g = t[t.fold == f]; w = s[s.foldid == f].window_open.nunique()
            if not len(g):
                print(f"  {int(f):>5} {w:>8,} {'SILENT':>7}")
                per.append({'fold': int(f), 'trades': 0}); continue
            print(f"  {int(f):>5} {w:>8,} {len(g):>7,} {g.contracts.sum():>10,} "
                  f"{g.pnl.sum():>10.2f} {g.pnl.sum()/g.contracts.sum()*100:>+8.3f}")
            per.append({'fold': int(f), 'trades': int(len(g)),
                        'contracts': int(g.contracts.sum()), 'pnl': float(g.pnl.sum())})
        gw = t.groupby('window_open').pnl.sum()
        ts = float(gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw))))
        lo, hi = _boot_ci(gw.to_numpy())
        last5 = folds[-5:]
        traded5 = sum(1 for f in last5 if len(t[t.fold == f]))
        profitable = t.pnl.sum() > 0
        print(f"\n  PROFITABLE?  ${t.pnl.sum():+.2f} on {t.contracts.sum():,} contracts  "
              f"{t.pnl.sum()/t.contracts.sum()*100:+.3f}pp/ct  t={ts:+.2f}")
        print(f"               window-bootstrap 95% CI [${lo*len(gw):+.2f}, ${hi*len(gw):+.2f}]")
        print(f"  TRADES 4/5?  traded in {traded5} of the last 5 folds {last5}")
        verdict = 'PROMOTE' if (profitable and traded5 >= 4) else 'DO NOT PROMOTE'
        print(f"\n  VERDICT: {verdict}")
        res.update({'per_fold': per, 'trades': int(len(t)),
                    'contracts': int(t.contracts.sum()), 'pnl': float(t.pnl.sum()),
                    'pp_per_contract': float(t.pnl.sum()/t.contracts.sum()*100),
                    't_stat': ts, 'pnl_ci': [lo*len(gw), hi*len(gw)],
                    'traded_last5': traded5, 'profitable': bool(profitable),
                    'verdict': verdict})
    s.to_parquet(OUT / 'oof_stage16_xgb.parquet', index=False)
    (OUT / 'stage16_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 16, 'experiment': 'user spec with XGBoost', 'result': res,
         'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 16 done in {time.time()-t0:.0f}s")
