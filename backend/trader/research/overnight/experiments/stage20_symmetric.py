"""STAGE 20 — fix the one-sided sharpening, on all data, with the top 15 features.

Measured 2026-10-17 on the live book: the promoted model sharpens UP-favourites by
+1.705pp and DOWN-favourites by -0.035pp, a 49x asymmetry whose net is the whole
+0.883pp bullish tilt (0.508*1.705 + 0.492*0.035 = 0.883, reproduced exactly). The
baseline F(x/sigma) and the market are both calibrated here; only the model's
correction is not. Nothing in the pipeline requires the correction to be symmetric,
`intercept` is 0.0 and `init_score_source` is `market`, so nothing pins the mean
prediction to the mean label out of sample.

Three arms, identical folds, identical features:

  A  plain             the current recipe, as a control
  B  mirror-augmented  every TRAINING row is duplicated reflected about the strike.
                       Enforces p(M x) = 1 - p(x) by construction rather than hope.
  C  intercept de-bias a scalar fitted on the inner validation block so the mean
                       prediction matches the mean label. Kills the net tilt but
                       leaves the asymmetry's shape, so it separates "the mean is
                       wrong" from "the rule is one-sided".

The mirror negates the ODD features and leaves the EVEN ones. Each was checked
against its definition in `core/features.py`, not assumed from its name --
`excursion_asymmetry = (exc_up+exc_down)/(exc_up-exc_down)` is odd because
reflection negates the numerator and leaves the span. A feature whose parity is
not established must NOT be mirrored: a wrong sign corrupts half the training set
silently. `_assert_parity` fails the run rather than trusting this docstring.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[2]))
from stage9_full_training import _sig, _logit, _ll
from stage7_backfilled_flow import _boot_ci
from stage15_nn_candidate import assemble, simulate
from stage1_screen import log
OUT = HERE.parent

TOP15 = ['z_score', 'displacement_vs_elapsed', 'peer_return_15', 'pm_mid',
         'abs_z_score', 'signed_volume_15', 'rv_slope_long', 'excursion_asymmetry',
         'rv_slope_short', 'pm_spread', 'volume_z_15', 'rv_surprise', 'log_rv_15',
         'k_spread', 'k_depth_imb_1c']
ODD = {'z_score', 'displacement_vs_elapsed', 'peer_return_15', 'signed_volume_15',
       'excursion_asymmetry', 'k_depth_imb_1c'}        # negate
ONE_MINUS = {'pm_mid'}                                  # probability -> 1 - p
EVEN = set(TOP15) - ODD - ONE_MINUS                     # leave alone
PAR = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
           learning_rate=0.02, max_depth=4, min_child_weight=125.0, subsample=0.8,
           colsample_bytree=0.7, reg_lambda=20.0, seed=17, nthread=8)


def _assert_parity():
    unclassified = set(TOP15) - (ODD | ONE_MINUS | EVEN)
    assert not unclassified, f'unclassified parity: {unclassified}'
    assert not (ODD & ONE_MINUS), 'a feature cannot be both'
    print(f'  parity map: {len(ODD)} odd, {len(ONE_MINUS)} one-minus, {len(EVEN)} even'
          f'  (= {len(TOP15)} features)')


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect a block about the strike: up and down swap roles.

    The label, the market price and every odd feature flip; the even ones do not.
    A mirrored row is a real row of a market that genuinely exists -- the same
    fifteen minutes seen from the NO side -- so it is not synthetic data.
    """
    m = df.copy()
    m['y'] = 1 - m['y']
    m['p_mkt'] = 1.0 - m['p_mkt']
    for c in ODD:
        if c in m.columns: m[c] = -m[c]
    for c in ONE_MINUS:
        if c in m.columns: m[c] = 1.0 - m[c]
    return m


def debias_shift(margin: np.ndarray, y: np.ndarray) -> float:
    """Scalar b with mean(sigmoid(margin + b)) == mean(y). Bisection, 60 steps."""
    target = float(np.mean(y)); lo, hi = -2.0, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.mean(_sig(margin + mid))) < target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def run(m, arm):
    import xgboost as xgb
    feats = [c for c in TOP15 if c in m.columns]
    preds = np.full(len(m), np.nan)
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 3000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 1500 or len(iva) < 800: itr, iva = tr, tr
        if arm == 'B':
            itr = pd.concat([itr, mirror(itr)], ignore_index=True)
            iva = pd.concat([iva, mirror(iva)], ignore_index=True)
        elif arm == 'D':
            # **The control for B.** Same row COUNT, no reflection. If B's gain
            # is really the extra rows rather than the symmetry constraint, this
            # matches it; if symmetry is doing the work, this reproduces A.
            itr = pd.concat([itr, itr], ignore_index=True)
            iva = pd.concat([iva, iva], ignore_index=True)
        dtr = xgb.DMatrix(itr[feats], label=itr.y.to_numpy(int), base_margin=_logit(itr.p_mkt))
        dva = xgb.DMatrix(iva[feats], label=iva.y.to_numpy(int), base_margin=_logit(iva.p_mkt))
        bst = xgb.train(PAR, dtr, num_boost_round=600, evals=[(dva, 'v')],
                        early_stopping_rounds=50, verbose_eval=False)
        it = bst.best_iteration
        dte = xgb.DMatrix(te[feats], base_margin=_logit(te.p_mkt))
        mt = np.asarray(bst.predict(dte, output_margin=True, iteration_range=(0, it+1)), float)
        if arm == 'C':
            # fitted on the inner VALIDATION block only -- never on the test fold
            mv = np.asarray(bst.predict(xgb.DMatrix(iva[feats], base_margin=_logit(iva.p_mkt)),
                                        output_margin=True, iteration_range=(0, it+1)), float)
            mt = mt + debias_shift(mv, iva.y.to_numpy(int))
        preds[te_m] = _sig(mt)
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s = s.rename(columns={'_fold': 'foldid'})
    d = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    wv = pd.DataFrame({'d': d, 'w': s.window_open}).groupby('w')['d'].mean().to_numpy()
    lo, hi = _boot_ci(wv)
    # the diagnostic this whole stage exists for
    corr = (s.p_model - s.p_mkt) * 100
    up = s.p_mkt >= 0.5
    cu, cd = float(corr[up].mean()), float(corr[~up].mean())
    t = simulate(s)
    if len(t):
        gw = t.groupby('window_open').pnl.sum()
        ts = float(gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw))))
        pnl = float(t.pnl.sum()); pp = pnl/t.contracts.sum()*100
        nup = int((t.side == 'yes').sum())
        ndn = int((t.side == 'no').sum())
        folds = sorted(s.foldid.unique()); tr5 = sum(1 for f in folds[-5:] if len(t[t.fold == f]))
        pos = sum(1 for f in folds if len(t[t.fold==f]) and t[t.fold==f].pnl.sum() > 0)
    else:
        pnl=pp=ts=0.0; nup=ndn=tr5=pos=0; folds=[]
    print(f"  {arm}  mmm {wv.mean():+.6f} [{lo:+.6f},{hi:+.6f}]  tilt {corr.mean():+.3f}pp "
          f"(up {cu:+.2f} / down {cd:+.2f})  ${pnl:>+9.2f} {pp:+.3f}pp/ct t={ts:+.2f}  "
          f"up:dn {nup}:{ndn}  last5 {tr5}/5  +folds {pos}/{len(folds)}")
    return dict(arm=arm, mmm=float(wv.mean()), ci=[lo, hi], tilt_pp=float(corr.mean()),
                corr_up_fav=cu, corr_down_fav=cd, pnl=pnl, pp_per_contract=float(pp),
                t_stat=ts, trades_up=nup, trades_down=ndn, traded_last5=tr5,
                folds_positive=pos, folds=len(folds), n_features=len(feats))


if __name__ == '__main__':
    t0 = time.time(); _assert_parity(); m = assemble()
    missing = [c for c in TOP15 if c not in m.columns]
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}  "
          f"{m.window_open.min().date()} -> {m.window_open.max().date()}"
          + (f"  MISSING {missing}" if missing else ""))
    # the mirror must leave the EVEN features untouched and negate the ODD ones
    chk = mirror(m.head(5000))
    for c in (set(TOP15) & EVEN):
        assert np.allclose(chk[c].fillna(0), m.head(5000)[c].fillna(0)), f'{c} moved'
    for c in (set(TOP15) & ODD):
        assert np.allclose(chk[c].fillna(0), -m.head(5000)[c].fillna(0)), f'{c} not negated'
    print("  mirror verified on 5,000 rows\n")
    import os
    arms = os.environ.get('ARMS', 'A,B,C').split(',')
    res = [run(m, a) for a in arms]
    import os as _os
    _tag = _os.environ.get('TAG', '')
    (OUT / f'stage20_symmetric{_tag}.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 20, 'experiment': 'symmetry fix, top-15, all data',
         'result': res, 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 20 done in {time.time()-t0:.0f}s")
