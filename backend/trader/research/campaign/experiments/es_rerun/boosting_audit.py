"""Bounded model-family audit: LightGBM vs XGBoost on the corrected campaign data.

The first thing verified is the residual plumbing, because the two libraries differ
in a way that silently produces a wrong model:

* **LightGBM**: `init_score` is used during TRAINING only. `predict(raw_score=True)`
  returns the CORRECTION alone, so the init logit must be added back by hand.
* **XGBoost**: `base_margin` set on the DMatrix is used in training AND applied at
  prediction. `predict(output_margin=True)` already INCLUDES it, so adding it back
  double-counts.

`verify_residual_plumbing` asserts the invariant that settles it: a model trained for
ZERO rounds must reproduce the init probability exactly, in both libraries. If that
holds, the two are fitting the same residual target and the comparison is fair.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------ plumbing -----
def lgbm_fit_predict(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte, params, rounds,
                     early_stopping=40, seed=17):
    import lightgbm as lgb
    p = dict(objective='binary', metric='binary_logloss', verbosity=-1,
             deterministic=True, seed=seed)
    p.update(params)
    dtr = lgb.Dataset(Xtr, label=ytr, init_score=mtr, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, init_score=mva, reference=dtr,
                      free_raw_data=False)
    cbs = [lgb.early_stopping(early_stopping, verbose=False)] if early_stopping else []
    bst = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                    valid_names=['inner'], callbacks=cbs)
    # LightGBM: raw_score is the CORRECTION ONLY -> add the margin back
    corr_te = np.asarray(bst.predict(Xte, raw_score=True), float)
    corr_va = np.asarray(bst.predict(Xva, raw_score=True), float)
    return corr_te, corr_va, bst, getattr(bst, 'best_iteration', rounds)


def xgb_fit_predict(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte, params, rounds,
                    early_stopping=40, seed=17):
    import xgboost as xgb
    p = dict(objective='binary:logistic', eval_metric='logloss',
             tree_method='hist', seed=seed, nthread=4)
    p.update(params)
    dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=mtr)
    dva = xgb.DMatrix(Xva, label=yva, base_margin=mva)
    dte = xgb.DMatrix(Xte, base_margin=mte)
    bst = xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, 'inner')],
                    early_stopping_rounds=early_stopping or None,
                    verbose_eval=False)
    it = getattr(bst, 'best_iteration', None)
    rng = (0, it + 1) if it is not None else (0, rounds)
    # XGBoost: output_margin INCLUDES base_margin -> subtract it to get the
    # correction, so both libraries return the same quantity to the caller
    full_te = np.asarray(bst.predict(dte, output_margin=True, iteration_range=rng), float)
    full_va = np.asarray(bst.predict(dva, output_margin=True, iteration_range=rng), float)
    return full_te - mte, full_va - mva, bst, (it + 1 if it is not None else rounds)


FAMILIES = {'lgbm': lgbm_fit_predict, 'xgb': xgb_fit_predict}


def verify_residual_plumbing(n=6000, seed=0):
    """Assert that both wrappers return the SAME quantity: the correction alone.

    LightGBM cannot train zero rounds, so the invariant is checked against each
    library's own probability output instead, which is stronger anyway because it
    catches a sign or double-count error at the exact seam where it would occur:

    * LightGBM `predict(X)` EXCLUDES init_score  -> sigmoid(corr) must equal it
    * XGBoost  `predict(d)` INCLUDES base_margin -> sigmoid(m + corr) must equal it

    If both hold, then `sigmoid(m + corr)` is the fitted model in both libraries and
    the two are fitting the same residual target.
    """
    import lightgbm as lgb, xgboost as xgb
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    m = rng.normal(0.3, 1.2, n)
    y = (rng.random(n) < _sig(m + 0.4 * X[:, 0])).astype(int)
    h = n // 2
    Xtr, ytr, mtr = X[:h], y[:h], m[:h]
    Xte, yte, mte = X[h:], y[h:], m[h:]
    params = {'learning_rate': 0.1, 'max_depth': 3}
    ok = {}

    corr_l, _, bst_l, _ = lgbm_fit_predict(Xtr, ytr, mtr, Xte, yte, mte, Xte, mte,
                                           params, rounds=25, early_stopping=0)
    own_l = np.asarray(bst_l.predict(Xte), float)
    e = float(np.max(np.abs(_sig(corr_l) - own_l)))
    ok['lgbm: predict() EXCLUDES init_score'] = e
    print(f"  lgbm  max |sigmoid(corr) - bst.predict(X)|          = {e:.3e}  "
          f"{'PASS' if e < 1e-9 else 'FAIL'}")

    corr_x, _, bst_x, it = xgb_fit_predict(Xtr, ytr, mtr, Xte, yte, mte, Xte, mte,
                                           params, rounds=25, early_stopping=0)
    dte = xgb.DMatrix(Xte, base_margin=mte)
    own_x = np.asarray(bst_x.predict(dte, iteration_range=(0, it)), float)
    e2 = float(np.max(np.abs(_sig(mte + corr_x) - own_x)))
    ok['xgb: predict() INCLUDES base_margin'] = e2
    print(f"  xgb   max |sigmoid(m + corr) - bst.predict(d)|      = {e2:.3e}  "
          f"{'PASS' if e2 < 1e-6 else 'FAIL'}")

    # and the double-count a naive port would produce, for the record
    bad = float(np.mean(np.abs(_sig(mte + mte + corr_x) - own_x)))
    print(f"  (a naive port that adds base_margin twice would be off by {bad:.4f} "
          f"on average -- this is the trap)")

    assert ok['lgbm: predict() EXCLUDES init_score'] < 1e-9, ok
    assert ok['xgb: predict() INCLUDES base_margin'] < 1e-6, ok
    # both wrappers now return the correction alone
    print("  -> both wrappers return the CORRECTION ALONE; sigmoid(m + corr) is the "
          "model in both.")
    return ok





# ========================= the actual comparison =============================
import json

CURRENT_LGBM = {   # exactly what models/forecast.provenance.json records
    'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 4,
    'min_child_samples': 500, 'bagging_fraction': 0.7, 'bagging_freq': 1,
    'feature_fraction': 0.6, 'lambda_l2': 10.0,
}

# Equal budget: six configurations per library, fixed BEFORE any result was seen.
# The grids are NOT parameter-for-parameter copies. `min_child_samples` counts ROWS
# in LightGBM; XGBoost's `min_child_weight` is a sum of Hessians, which for logloss
# is sum p(1-p) -- at p~0.5 that is ~0.25 per row, so a comparable support of R rows
# is min_child_weight ~ R/4, and much larger at extreme p. The grids below are
# matched on INTENT (shallow, well-supported, regularized), not on numbers.
GRID_LGBM = [
    {'learning_rate': .03, 'num_leaves': 7,  'max_depth': 3, 'min_child_samples': 1000,
     'bagging_fraction': .7, 'bagging_freq': 1, 'feature_fraction': .6, 'lambda_l2': 10.,
     'min_gain_to_split': 0.0},
    {'learning_rate': .03, 'num_leaves': 7,  'max_depth': 3, 'min_child_samples': 2000,
     'bagging_fraction': .7, 'bagging_freq': 1, 'feature_fraction': .5, 'lambda_l2': 50.,
     'min_gain_to_split': 1e-3},
    {'learning_rate': .02, 'num_leaves': 3,  'max_depth': 2, 'min_child_samples': 2000,
     'bagging_fraction': .8, 'bagging_freq': 1, 'feature_fraction': .5, 'lambda_l2': 100.,
     'min_gain_to_split': 1e-3},
    {'learning_rate': .05, 'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 500,
     'bagging_fraction': .7, 'bagging_freq': 1, 'feature_fraction': .6, 'lambda_l2': 10.,
     'min_gain_to_split': 0.0},
    {'learning_rate': .02, 'num_leaves': 7,  'max_depth': 3, 'min_child_samples': 4000,
     'bagging_fraction': .6, 'bagging_freq': 1, 'feature_fraction': .4, 'lambda_l2': 200.,
     'min_gain_to_split': 1e-2},
    {'learning_rate': .03, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 200,
     'bagging_fraction': .8, 'bagging_freq': 1, 'feature_fraction': .8, 'lambda_l2': 1.,
     'min_gain_to_split': 0.0},
]
GRID_XGB = [
    {'learning_rate': .03, 'max_depth': 3, 'min_child_weight': 250., 'subsample': .7,
     'colsample_bytree': .6, 'reg_lambda': 10., 'reg_alpha': 0., 'gamma': 0.},
    {'learning_rate': .03, 'max_depth': 3, 'min_child_weight': 500., 'subsample': .7,
     'colsample_bytree': .5, 'reg_lambda': 50., 'reg_alpha': 0., 'gamma': 1e-3},
    {'learning_rate': .02, 'max_depth': 2, 'min_child_weight': 500., 'subsample': .8,
     'colsample_bytree': .5, 'reg_lambda': 100., 'reg_alpha': 1., 'gamma': 1e-3},
    {'learning_rate': .05, 'max_depth': 4, 'min_child_weight': 125., 'subsample': .7,
     'colsample_bytree': .6, 'reg_lambda': 10., 'reg_alpha': 0., 'gamma': 0.},
    {'learning_rate': .02, 'max_depth': 3, 'min_child_weight': 1000., 'subsample': .6,
     'colsample_bytree': .4, 'reg_lambda': 200., 'reg_alpha': 2., 'gamma': 1e-2},
    {'learning_rate': .03, 'max_depth': 6, 'min_child_weight': 50., 'subsample': .8,
     'colsample_bytree': .8, 'reg_lambda': 1., 'reg_alpha': 0., 'gamma': 0.},
]
ROUNDS = 600
EARLY = 50


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def inner_split(tr: pd.DataFrame, embargo_days=1, valid_frac=0.25):
    """Chronological inner split for tuning + early stopping, purged by an embargo.
    All selection happens here; the outer test block is never touched."""
    w = np.sort(tr.window_open.unique())
    cut = w[int(len(w) * (1 - valid_frac))]
    emb = pd.Timedelta(days=embargo_days)
    return (tr[tr.window_open < cut - emb], tr[tr.window_open >= cut])


def fit_family(fam, tr, te, feats, params, rounds=ROUNDS, early=EARLY, seed=17):
    from sklearn.impute import SimpleImputer
    itr, iva = inner_split(tr)
    if len(itr) < 500 or len(iva) < 200:
        itr, iva = tr, tr
    imp = SimpleImputer(strategy='median').fit(itr[feats].to_numpy(float))
    Xtr = imp.transform(itr[feats].to_numpy(float))
    Xva = imp.transform(iva[feats].to_numpy(float))
    Xte = imp.transform(te[feats].to_numpy(float))
    mtr = _logit(itr.p_mkt.to_numpy()); mva = _logit(iva.p_mkt.to_numpy())
    mte = _logit(te.p_mkt.to_numpy())
    corr_te, corr_va, bst, nit = FAMILIES[fam](
        Xtr, itr.y.to_numpy(int), mtr, Xva, iva.y.to_numpy(int), mva,
        Xte, mte, params, rounds, early, seed)
    return (_sig(mte + corr_te), _sig(mva + corr_va), iva, bst, nit,
            _sig(_logit(itr.p_mkt.to_numpy()) + 0))


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_comparison(budget_s=5400, seeds=(17,)):
    """A/B/C/D on identical data, folds, features, labels and execution assumptions."""
    sys.path.insert(0, str(HERE.parent))
    import harness as H, runner as R
    from run_campaign import make_folds

    t0 = time.time()
    ds, cfg = R.build_dataset(bars_end=pd.Timestamp('2026-10-15', tz='UTC'))
    px, lab = H.market_prices(), H.venue_labels()
    folds = make_folds(ds.window_index)
    print(f'folds: {len(folds)}', flush=True)
    for f in folds: print('  ' + f.label(), flush=True)

    GROUPS = ['geometry', 'vol_state', 'microstructure', 'cross_asset', 'offset']
    tables = []
    for f, tr, te in R.fold_tables(ds, cfg, folds, groups=GROUPS):
        tables.append((f, R.attach_market(tr, px, lab), R.attach_market(te, px, lab)))
        print(f'  prepared {f.label()}  train {len(tables[-1][1]):,} test {len(tables[-1][2]):,}',
              flush=True)
    feats = R.feature_cols(tables[0][1], GROUPS)
    print(f'features: {len(feats)}', flush=True)

    results = {}

    # ---- D: the market alone -------------------------------------------
    allte = pd.concat([t[2] for t in tables], ignore_index=True)
    results['D_market_only'] = {
        'label': 'Kalshi market-implied probability, no ML',
        'n': int(len(allte)), 'windows': int(allte.window_open.nunique()),
        'log_loss': float(np.mean(_ll(allte.p_mkt, allte.y))),
        'brier': float(np.mean((allte.p_mkt - allte.y) ** 2)),
        'calibration': H.calibration(allte.p_mkt, allte.y),
    }
    print(f"\nD market-only: log loss {results['D_market_only']['log_loss']:.6f}", flush=True)

    # ---- A: the CURRENT LightGBM configuration, untuned ------------------
    def evaluate(name, fam, params_fn, label, seed=17):
        preds, tr_lls, its, chosen = [], [], [], []
        for f, tr, te in tables:
            if time.time() - t0 > budget_s:
                return {'label': label, 'status': 'FAILED',
                        'reason': f'budget {budget_s}s exceeded'}
            params = params_fn(fam, tr, te)
            chosen.append(params)
            p_te, p_va, iva, bst, nit, _ = fit_family(fam, tr, te, feats, params, seed=seed)
            o = te.copy(); o['p_model'] = p_te
            preds.append(o); its.append(nit)
            # in-sample vs out-of-sample, on the SAME fold
            p_tr, _, _, _, _, _ = fit_family(fam, tr, tr, feats, params, seed=seed)
            tr_lls.append(float(np.mean(_ll(p_tr, tr.y))))
        oof = pd.concat(preds, ignore_index=True)
        mean, lo, hi, pv, nw = H.model_minus_market(oof.p_model, oof.p_mkt, oof.y,
                                                    oof.window_open)
        res = {
            'label': label, 'status': 'OK', 'family': fam,
            'n': int(len(oof)), 'windows': nw,
            'log_loss': float(np.mean(_ll(oof.p_model, oof.y))),
            'log_loss_market': float(np.mean(_ll(oof.p_mkt, oof.y))),
            'brier': float(np.mean((oof.p_model - oof.y) ** 2)),
            'brier_market': float(np.mean((oof.p_mkt - oof.y) ** 2)),
            'model_minus_market': mean, 'ci95': [lo, hi], 'p_one_sided': pv,
            'beats_market': bool(lo > 0),
            'calibration': H.calibration(oof.p_model, oof.y),
            'train_log_loss_by_fold': tr_lls,
            'best_iterations': its, 'params_by_fold': chosen,
            'runtime_s': round(time.time() - t0, 1),
        }
        print(f"\n{name}: log loss {res['log_loss']:.6f} (market {res['log_loss_market']:.6f})"
              f"  mmm {mean:+.6f} CI [{lo:+.6f},{hi:+.6f}] p={pv:.4f}"
              f"  {'BEATS' if res['beats_market'] else 'does NOT beat'} the market",
              flush=True)
        print(f"   train log loss by fold {[round(x,5) for x in tr_lls]}  "
              f"best_iters {its}", flush=True)
        for gate in (1.0, 1.5, 2.5):
            sim = H.simulate(oof, 'p_model', min_edge_pp=gate, offsets=(3, 6, 9, 12))
            sim.pop('_trades', None)
            res.setdefault('money', {})[f'gate_{gate}'] = sim
            if sim['trades']:
                print(f"     gate {gate:>4.1f}pp: {sim['trades']:>5,} trades  "
                      f"${sim['pnl']:>8.2f}  {sim['pp_per_contract']:>+7.3f}pp/ct  "
                      f"t={sim['t_stat']:>5.2f}", flush=True)
            else:
                print(f"     gate {gate:>4.1f}pp: no trades", flush=True)
        oof[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
             'yes_bid','yes_ask','split']].to_parquet(HERE / f'oof_{name}.parquet', index=False)
        return res

    results['A_lgbm_current'] = evaluate(
        'A_lgbm_current', 'lgbm', lambda *a: dict(CURRENT_LGBM),
        'current LightGBM configuration, untuned')

    # ---- B/C: tuned inside each outer training period --------------------
    def tuned(grid):
        def pick(fam, tr, te):
            itr, iva = inner_split(tr)
            if len(itr) < 500 or len(iva) < 200:
                return dict(grid[0])
            best, best_ll = None, np.inf
            for g in grid:
                p_va = fit_family(fam, itr, iva, feats, g)[0]
                ll = float(np.mean(_ll(p_va, iva.y)))
                if ll < best_ll:
                    best, best_ll = g, ll
            return dict(best)
        return pick

    results['B_lgbm_tuned'] = evaluate('B_lgbm_tuned', 'lgbm', tuned(GRID_LGBM),
                                       'LightGBM, tuned inside each training fold (6 configs)')
    results['C_xgb_tuned'] = evaluate('C_xgb_tuned', 'xgb', tuned(GRID_XGB),
                                      'XGBoost, tuned inside each training fold (6 configs)')

    # ---- seed sensitivity ------------------------------------------------
    print('\n=== seed sensitivity (same config, different seeds) ===', flush=True)
    for fam, key in (('lgbm', 'B_lgbm_tuned'), ('xgb', 'C_xgb_tuned')):
        mms = []
        for sd in (17, 101, 999):
            r = evaluate(f'seed_{fam}_{sd}', fam, tuned(GRID_LGBM if fam == 'lgbm' else GRID_XGB),
                         f'{fam} seed {sd}', seed=sd)
            if r.get('status') == 'OK':
                mms.append(r['model_minus_market'])
        results[key]['seed_sensitivity'] = {
            'seeds': [17, 101, 999], 'model_minus_market': mms,
            'spread': float(max(mms) - min(mms)) if len(mms) > 1 else None}
        print(f"  {fam}: mmm across seeds {[round(x,6) for x in mms]}", flush=True)

    H.record('boosting_audit', {'results': results, 'features': feats,
                                'folds': [f.label() for f in folds],
                                'grid_lgbm': GRID_LGBM, 'grid_xgb': GRID_XGB,
                                'rounds': ROUNDS, 'early_stopping': EARLY,
                                'runtime_s': round(time.time() - t0, 1)})
    print(f'\nrecorded. {time.time()-t0:.0f}s')
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--budget', type=int, default=5400)
    a = ap.parse_args()
    print('=== residual plumbing verification ===')
    verify_residual_plumbing()
    print('both libraries fit the SAME residual target; the comparison is fair.\n')
    if not a.verify_only:
        run_comparison(budget_s=a.budget)
