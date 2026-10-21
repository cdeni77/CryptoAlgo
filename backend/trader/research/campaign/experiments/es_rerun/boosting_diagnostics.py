"""Residual-plumbing, persistence, and UNDERFITTING diagnostics for both libraries.

Three things are tested here that the comparison alone cannot establish:

1. **Plumbing and persistence.** `init_score` / `base_margin` must behave the same
   way after a model is saved and reloaded, and identical baseline margins must
   produce equivalent zero-correction probabilities. A library that silently drops
   the margin on reload would produce a model that scores correctly in the research
   process and wrongly in production.
2. **Can the models LEARN AT ALL on data of this shape?** A synthetic signal of known
   strength is injected into a real feature matrix, built only from information
   available at the decision instant. If neither library recovers a signal that is
   present by construction, a null result on the real data means "no signal in the
   data", not "the models underfit". If they recover it easily, the null is about the
   data. This is the control that separates the two explanations.
3. **Underfitting evidence**: learning curves, train-validation gaps, leaf
   utilization, and the DISTRIBUTION of best_iteration over folds and seeds.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
from boosting_audit import (_sig, _logit, _ll, FAMILIES, CURRENT_LGBM,
                            GRID_LGBM, GRID_XGB, inner_split, ROUNDS, EARLY)


# ===================== 1. plumbing, persistence, equivalence ================
def verify_plumbing_and_persistence(tmp: Path, n=6000, seed=0):
    import lightgbm as lgb, xgboost as xgb
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    m = rng.normal(0.3, 1.2, n)
    y = (rng.random(n) < _sig(m + 0.4 * X[:, 0])).astype(int)
    h = n // 2
    Xtr, ytr, mtr = X[:h], y[:h], m[:h]
    Xte, yte, mte = X[h:], y[h:], m[h:]
    params = {'learning_rate': 0.1, 'max_depth': 3}
    out = {}
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- LightGBM -------------------------------------------------------
    corr, _, bst, _ = FAMILIES['lgbm'](Xtr, ytr, mtr, Xte, yte, mte, Xte, mte,
                                       params, ROUNDS // 6, 0)
    e_live = float(np.max(np.abs(_sig(corr) - bst.predict(Xte))))
    p_live = _sig(mte + corr)
    f = tmp / 'lgbm.txt'; bst.save_model(str(f))
    re = lgb.Booster(model_file=str(f))
    corr_re = np.asarray(re.predict(Xte, raw_score=True), float)
    e_reload = float(np.max(np.abs(_sig(mte + corr_re) - p_live)))
    out['lgbm'] = {'predict_excludes_init_score': e_live,
                   'reload_identical': e_reload}
    print(f"  lgbm  predict() excludes init_score : {e_live:.3e}  "
          f"{'PASS' if e_live < 1e-9 else 'FAIL'}")
    print(f"  lgbm  save -> reload identical      : {e_reload:.3e}  "
          f"{'PASS' if e_reload < 1e-9 else 'FAIL'}")

    # ---- XGBoost --------------------------------------------------------
    corr_x, _, bx, it = FAMILIES['xgb'](Xtr, ytr, mtr, Xte, yte, mte, Xte, mte,
                                        params, ROUNDS // 6, 0)
    dte = xgb.DMatrix(Xte, base_margin=mte)
    e_livex = float(np.max(np.abs(_sig(mte + corr_x)
                                  - bx.predict(dte, iteration_range=(0, it)))))
    p_livex = _sig(mte + corr_x)
    fx = tmp / 'xgb.json'; bx.save_model(str(fx))
    rex = xgb.Booster(); rex.load_model(str(fx))
    full_re = np.asarray(rex.predict(dte, output_margin=True,
                                     iteration_range=(0, it)), float)
    e_reloadx = float(np.max(np.abs(_sig(full_re) - p_livex)))
    # Does the RELOADED model still honour base_margin at all?
    #
    # **The naive form of this check FAILS for a reason that is not a bug**, and it
    # is worth recording. Since XGBoost 2.0 `base_score` is ESTIMATED FROM THE DATA
    # rather than fixed at 0.5, so a prediction with no base_margin carries
    # `logit(base_score)` instead of zero. Comparing `with - without` against the
    # margin therefore reads ~0.24 (the estimated base_score logit) and looks like a
    # dropped margin. Measured here: 0.2425.
    #
    # `base_margin`, when supplied, REPLACES `base_score` -- which is exactly what
    # the equivalence test above confirms to 8e-8. So the correct assertion is that
    # `with - without - margin` is a CONSTANT (the base_score logit), not zero.
    d_nomargin = xgb.DMatrix(Xte)
    full_nm = np.asarray(rex.predict(d_nomargin, output_margin=True,
                                     iteration_range=(0, it)), float)
    delta = full_re - full_nm - mte
    margin_honoured = float(np.std(delta))
    base_score_logit = float(np.mean(-delta))
    out_extra = {'implied_base_score_logit': base_score_logit}
    print(f"  xgb   (base_score is estimated, not 0.5: implied logit "
          f"{base_score_logit:+.4f} -> base_score {_sig(base_score_logit):.4f})")
    out['xgb'] = {'predict_includes_base_margin': e_livex,
                  'reload_identical': e_reloadx,
                  'reloaded_margin_offset_is_constant_std': margin_honoured,
                  **out_extra}
    print(f"  xgb   predict() includes base_margin: {e_livex:.3e}  "
          f"{'PASS' if e_livex < 1e-6 else 'FAIL'}")
    print(f"  xgb   save -> reload identical      : {e_reloadx:.3e}  "
          f"{'PASS' if e_reloadx < 1e-6 else 'FAIL'}")
    print(f"  xgb   reloaded model applies margin : std {margin_honoured:.3e}  "
          f"{'PASS' if margin_honoured < 1e-6 else 'FAIL'}")

    # ---- equivalence: identical margins -> identical zero-correction ----
    # Train each library on PURE NOISE labels so no correction can be learned,
    # then check both reduce to the same probability as the margin alone.
    yn = (rng.random(n) < 0.5).astype(int)
    eq = {}
    for fam in ('lgbm', 'xgb'):
        c, _, _, _ = FAMILIES[fam](Xtr, yn[:h], mtr, Xte, yn[h:], mte, Xte, mte,
                                   {'learning_rate': 1e-8, 'max_depth': 2}, 5, 0)
        eq[fam] = float(np.max(np.abs(_sig(mte + c) - _sig(mte))))
    out['zero_correction_equivalence'] = eq
    print(f"  identical margins -> zero-correction probability:")
    for k, v in eq.items():
        print(f"    {k:>5}: max deviation {v:.3e}  {'PASS' if v < 1e-6 else 'FAIL'}")
    agree = float(np.max(np.abs(np.array(list(eq.values())) - 0)))
    assert e_live < 1e-9 and e_reload < 1e-9, out
    assert e_livex < 1e-6 and e_reloadx < 1e-6 and margin_honoured < 1e-6, out
    assert all(v < 1e-6 for v in eq.values()), out
    return out


# ===================== 2. can they learn at all? ============================
def synthetic_recovery(tr, te, feats, strengths=(0.0, 0.05, 0.10, 0.25, 0.50),
                       seed=0):
    """Inject a signal of KNOWN strength and see whether each library recovers it.

    The injected feature is pure noise drawn independently of everything else; the
    LABEL is then re-drawn so the signal is present by construction. No future
    information is used -- the synthetic label replaces the real one, so there is
    nothing to leak from.

    strength 0.0 is the null: both libraries must score ~0 against the market.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in strengths:
        tr2, te2 = tr.copy(), te.copy()
        z_tr = rng.standard_normal(len(tr2)); z_te = rng.standard_normal(len(te2))
        tr2['_synth'] = z_tr; te2['_synth'] = z_te
        mtr = _logit(tr2.p_mkt.to_numpy()); mte = _logit(te2.p_mkt.to_numpy())
        tr2['y'] = (rng.random(len(tr2)) < _sig(mtr + s * z_tr)).astype(int)
        te2['y'] = (rng.random(len(te2)) < _sig(mte + s * z_te)).astype(int)
        f2 = feats + ['_synth']
        row = {'strength': s}
        for fam, grid in (('lgbm', CURRENT_LGBM), ('xgb', GRID_XGB[0])):
            from boosting_audit import fit_family
            p, _, _, bst, nit, _ = fit_family(fam, tr2, te2, f2, dict(grid))
            # skill against the market margin, which is the TRUE null here
            d = _ll(_sig(mte), te2.y) - _ll(p, te2.y)
            row[f'{fam}_skill'] = float(np.mean(d))
            row[f'{fam}_iters'] = int(nit)
        out.append(row)
    return pd.DataFrame(out)


# ===================== 3. underfitting evidence =============================
def learning_curve(fam, tr, te, feats, params, rounds=ROUNDS, seed=17):
    """Train/validation log loss as a function of boosting round, no early stop."""
    from sklearn.impute import SimpleImputer
    itr, iva = inner_split(tr)
    imp = SimpleImputer(strategy='median').fit(itr[feats].to_numpy(float))
    Xtr, Xva = imp.transform(itr[feats].to_numpy(float)), imp.transform(iva[feats].to_numpy(float))
    mtr, mva = _logit(itr.p_mkt.to_numpy()), _logit(iva.p_mkt.to_numpy())
    ytr, yva = itr.y.to_numpy(int), iva.y.to_numpy(int)
    curve = []
    if fam == 'lgbm':
        import lightgbm as lgb
        p = dict(objective='binary', metric='binary_logloss', verbosity=-1,
                 deterministic=True, seed=seed); p.update(params)
        dtr = lgb.Dataset(Xtr, label=ytr, init_score=mtr, free_raw_data=False)
        dva = lgb.Dataset(Xva, label=yva, init_score=mva, reference=dtr, free_raw_data=False)
        ev = {}
        bst = lgb.train(p, dtr, num_boost_round=rounds,
                        valid_sets=[dtr, dva], valid_names=['train', 'valid'],
                        callbacks=[lgb.record_evaluation(ev)])
        curve = pd.DataFrame({'round': range(1, len(ev['train']['binary_logloss']) + 1),
                              'train': ev['train']['binary_logloss'],
                              'valid': ev['valid']['binary_logloss']})
        leaves = [t['num_leaves'] for t in bst.dump_model()['tree_info']]
    else:
        import xgboost as xgb
        p = dict(objective='binary:logistic', eval_metric='logloss',
                 tree_method='hist', seed=seed, nthread=4); p.update(params)
        dtr = xgb.DMatrix(Xtr, label=ytr, base_margin=mtr)
        dva = xgb.DMatrix(Xva, label=yva, base_margin=mva)
        ev = {}
        bst = xgb.train(p, dtr, num_boost_round=rounds,
                        evals=[(dtr, 'train'), (dva, 'valid')],
                        evals_result=ev, verbose_eval=False)
        curve = pd.DataFrame({'round': range(1, len(ev['train']['logloss']) + 1),
                              'train': ev['train']['logloss'],
                              'valid': ev['valid']['logloss']})
        df = bst.trees_to_dataframe()
        leaves = df[df.Feature == 'Leaf'].groupby('Tree').size().tolist()
    return curve, leaves
