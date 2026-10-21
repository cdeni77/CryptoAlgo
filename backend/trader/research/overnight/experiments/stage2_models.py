"""STAGE 2 — train simple models on the Stage-1 survivors and measure them against
the Kalshi price out of fold.

Survivors (uncorrected 95%): imbalance, d_imbalance_1, d_imbalance_3,
d_touch_imb_3, d_book_mid_1, d_book_mid_3.

Design notes that matter:
* **The market logit is the init score / offset**, so an untrained model reproduces
  the market exactly and every tree is incremental skill by construction.
* Folds are chronological, purged by one day, and split on the WINDOW.
* Early stopping uses a chronological inner split -- the prior campaign measured
  that omitting it costs ~0.004 nats, which is larger than any effect here.
* The benchmark to beat is the Kalshi mid. The noise floor established by this
  project's negative controls is model_minus_market ~ -0.001.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'experiments'))
sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
from stage1_screen import load_flow, derive, log   # reuse the exact feature code

SURVIVORS = ['imbalance', 'd_imbalance_1', 'd_imbalance_3', 'd_touch_imb_3',
             'd_book_mid_1', 'd_book_mid_3']
ALLSIG = SURVIVORS + ['touch_imbalance', 'yes_depletion', 'no_depletion',
                      'flow_asym', 'd_total_size_1', 'd_touch_size_1',
                      'log_event_rate', 'd_levels_1']


def _sig(x): return 1/(1+np.exp(-x))
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); return np.log(p/(1-p))
def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def window_boot_delta(d, windows, B=10000, seed=0):
    g = pd.DataFrame({'d': d, 'w': windows}).groupby('w')['d'].mean()
    v = g.to_numpy(); n = len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs,.025)), float(np.quantile(bs,.975)), n


def build():
    flow = derive(load_flow())
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable]
    m = flow.merge(mv, left_on=['symbol','window_open','minute'],
                   right_on=['symbol','window_open','offset_minutes'], how='inner')
    return m[m.minute.between(1,14)].reset_index(drop=True)


def folds_of(windows, n_folds=4, embargo_days=1):
    w = np.array(sorted(pd.unique(windows)))
    blocks = [b for b in np.array_split(w, n_folds) if len(b)]
    out = []
    for i, b in enumerate(blocks):
        cut = pd.Timestamp(b.min()) - pd.Timedelta(days=embargo_days)
        tr = w[w < cut]
        if len(tr) < 150: continue
        out.append((i, tr, b))
    return out


def run_model(m, feats, kind, name, params=None, seed=17, budget_s=900):
    t0 = time.time()
    preds = np.full(len(m), np.nan)
    iters, tr_ll = [], []
    for i, trw, tew in folds_of(m.window_open.to_numpy()):
        tr = m[m.window_open.isin(trw)]; te = m[m.window_open.isin(tew)]
        if len(tr) < 1000 or len(te) == 0: continue
        # chronological inner split for early stopping
        iw = np.array(sorted(pd.unique(tr.window_open)))
        cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 500 or len(iva) < 200: itr, iva = tr, tr
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median').fit(itr[feats].to_numpy(float))
        Xi, Xv = imp.transform(itr[feats].to_numpy(float)), imp.transform(iva[feats].to_numpy(float))
        Xt = imp.transform(te[feats].to_numpy(float))
        mi, mv_, mt = _logit(itr.p_mkt), _logit(iva.p_mkt), _logit(te.p_mkt)
        if kind == 'lgbm':
            import lightgbm as lgb
            p = dict(objective='binary', metric='binary_logloss', verbosity=-1,
                     learning_rate=0.02, num_leaves=7, max_depth=3,
                     min_child_samples=500, bagging_fraction=0.8, bagging_freq=1,
                     feature_fraction=0.8, lambda_l2=20.0, deterministic=True, seed=seed)
            p.update(params or {})
            dtr = lgb.Dataset(Xi, label=itr.y.to_numpy(int), init_score=mi, free_raw_data=False)
            dva = lgb.Dataset(Xv, label=iva.y.to_numpy(int), init_score=mv_,
                              reference=dtr, free_raw_data=False)
            bst = lgb.train(p, dtr, num_boost_round=500, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(40, verbose=False)])
            corr = np.asarray(bst.predict(Xt, raw_score=True), float)
            iters.append(bst.best_iteration)
        elif kind == 'logit':
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xi)
            lr = LogisticRegression(C=(params or {}).get('C', 0.1), max_iter=2000)
            # fit the RESIDUAL: offset the market logit by fitting on it as a fixed term
            lr.fit(np.column_stack([sc.transform(Xi), mi]), itr.y.to_numpy(int))
            raw = lr.decision_function(np.column_stack([sc.transform(Xt), mt]))
            corr = raw - mt
            iters.append(0)
        preds[m.window_open.isin(tew).to_numpy()] = _sig(mt + corr)
        tr_ll.append(float(np.mean(_ll(_sig(mi + (
            np.asarray(bst.predict(Xi, raw_score=True), float) if kind=='lgbm'
            else lr.decision_function(np.column_stack([sc.transform(Xi), mi])) - mi)),
            itr.y))))
        if time.time()-t0 > budget_s: break
    ok = np.isfinite(preds)
    if ok.sum() < 500:
        return {'name': name, 'status': 'FAILED', 'reason': 'too few OOF preds'}
    s = m[ok]
    d = _ll(s.p_mkt, s.y) - _ll(preds[ok], s.y)
    mean, lo, hi, nw = window_boot_delta(d, s.window_open.to_numpy())
    res = {'name': name, 'status': 'OK', 'kind': kind, 'features': feats,
           'n_oof': int(ok.sum()), 'windows': nw,
           'log_loss_model': float(np.mean(_ll(preds[ok], s.y))),
           'log_loss_market': float(np.mean(_ll(s.p_mkt, s.y))),
           'model_minus_market': mean, 'ci95': [lo, hi],
           'beats_market': bool(lo > 0), 'best_iters': iters,
           'train_log_loss': tr_ll, 'seed': seed,
           'runtime_s': round(time.time()-t0, 1)}
    print(f"  {name:<28} mmm {mean:+.6f} CI [{lo:+.6f},{hi:+.6f}] n={ok.sum():,} "
          f"w={nw:,} iters={iters} {'BEATS' if lo>0 else 'no'}", flush=True)
    return res, preds


if __name__ == '__main__':
    t0 = time.time()
    m = build()
    print(f"STAGE 2: {len(m):,} rows, {m.window_open.nunique():,} windows, "
          f"{m.groupby(['symbol','window_open']).ngroups:,} symbol-windows")
    print(f"folds: {len(folds_of(m.window_open.to_numpy()))}")
    out = {}
    specs = [
        ('S2a_logit_survivors', 'logit', SURVIVORS, {'C': 0.1}),
        ('S2b_logit_imbalance_only', 'logit', ['imbalance'], {'C': 0.1}),
        ('S2c_lgbm_survivors', 'lgbm', SURVIVORS, None),
        ('S2d_lgbm_allflow', 'lgbm', ALLSIG, None),
    ]
    for name, kind, feats, params in specs:
        feats = [f for f in feats if f in m.columns]
        r = run_model(m, feats, kind, name, params)
        if isinstance(r, tuple):
            res, preds = r
            out[name] = res
            np.save(HERE.parent / f'preds_{name}.npy', preds)
        else:
            out[name] = r
            print(f"  {name}: FAILED {r['reason']}")
        log({'stage': 2, 'experiment': name, **(out[name] if isinstance(out[name], dict) else {})})
    (HERE.parent / 'stage2_results.json').write_text(json.dumps(out, indent=2, default=str))
    print(f"\nstage 2 done in {time.time()-t0:.0f}s")
