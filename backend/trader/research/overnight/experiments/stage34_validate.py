"""STAGE 3/4 — estimator comparison, negative controls, stability, and trading.

S2c (LightGBM on the order-flow survivors) is the only candidate above the noise
floor. Before any more compute goes into it, the things that would KILL it:

 1. a negative control (features permuted, label+price kept together)
 2. period stability -- the subsample's market residual mean is -0.00763, so a model
    could be learning a 15-day directional bias rather than conditional information
 3. symbol / offset concentration
 4. an estimator cross-check (XGBoost on identical data and folds)
 5. a trading simulation at the venue's real fee schedule with depth constraints
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
from stage2_models import (build, folds_of, run_model, SURVIVORS, ALLSIG,
                           _sig, _logit, _ll, window_boot_delta)
from stage1_screen import log

OUT = HERE.parent


def trade_sim(m, preds, min_edge_pp, fee_rate=0.07, stake=1, require_depth=True):
    """One entry per (symbol, window): walk offsets, take the first clearing the gate.
    Buy at the live ask (YES) or 1-bid (NO). Venue fee. Held to settlement."""
    d = m.copy(); d['p_model'] = preds
    d = d[np.isfinite(d.p_model)].sort_values(['symbol', 'window_open', 'minute'])
    rows = []
    for (sym, w), g in d.groupby(['symbol', 'window_open'], sort=False):
        for r in g.itertuples():
            p = r.p_model
            yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
            ev_y, ev_n = p - yes_px, (1.0 - p) - no_px
            side, px, ev = ('yes', yes_px, ev_y) if ev_y >= ev_n else ('no', no_px, ev_n)
            if not (0.0 < px < 1.0): continue
            fee = np.ceil(fee_rate * stake * px * (1-px) * 10_000)/10_000
            if ev * stake - fee < min_edge_pp/100.0 * stake: continue
            if require_depth:
                vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
                if not np.isfinite(vol) or vol < stake: continue
            won = (r.y == 1) if side == 'yes' else (r.y == 0)
            rows.append({'symbol': sym, 'window_open': w, 'offset': r.minute,
                         'side': side, 'price': px, 'fee': fee,
                         'pnl': (stake if won else 0.0) - px*stake - fee,
                         'won': won, 'p_model': p, 'p_mkt': r.p_mkt})
            break
    t = pd.DataFrame(rows)
    if not len(t): return {'trades': 0}
    gw = t.groupby('window_open')['pnl'].sum()
    rng = np.random.default_rng(0); v = gw.to_numpy(); n = len(v)
    bs = v[rng.integers(0, n, size=(10000, n))].sum(axis=1)
    return {'trades': int(len(t)), 'contracts': int(len(t)),
            'pnl': float(t.pnl.sum()),
            'pp_per_contract': float(t.pnl.sum()/len(t)*100),
            'win_rate': float(t.won.mean()), 'mean_price': float(t.price.mean()),
            'fees': float(t.fee.sum()), 'independent_windows': int(n),
            't_stat': float(gw.mean()/(gw.std(ddof=1)/np.sqrt(n))) if n > 1 else float('nan'),
            'pnl_ci95': [float(np.quantile(bs,.025)), float(np.quantile(bs,.975))],
            'max_drawdown': float((gw.cumsum().cummax()-gw.cumsum()).max()),
            'by_symbol': t.groupby('symbol')['pnl'].agg(['size','sum']).to_dict('index'),
            'by_offset': t.groupby('offset')['pnl'].agg(['size','sum']).to_dict('index')}


if __name__ == '__main__':
    t0 = time.time(); results = {}
    m = build()
    feats = [f for f in SURVIVORS if f in m.columns]
    print(f"base: {len(m):,} rows, {m.window_open.nunique():,} windows")

    # ---- 3.1 estimator cross-check -------------------------------------
    print('\n=== 3.1 estimator comparison on identical data/folds ===')
    r_l, preds_l = run_model(m, feats, 'lgbm', 'S3_lgbm_survivors')
    results['S3_lgbm'] = r_l

    def xgb_model(m, feats, name, seed=17):
        import xgboost as xgb
        from sklearn.impute import SimpleImputer
        preds = np.full(len(m), np.nan); iters = []
        for i, trw, tew in folds_of(m.window_open.to_numpy()):
            tr = m[m.window_open.isin(trw)]; te = m[m.window_open.isin(tew)]
            if len(tr) < 1000 or len(te) == 0: continue
            iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
            itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]; iva = tr[tr.window_open >= cut]
            if len(itr) < 500 or len(iva) < 200: itr, iva = tr, tr
            imp = SimpleImputer(strategy='median').fit(itr[feats].to_numpy(float))
            dtr = xgb.DMatrix(imp.transform(itr[feats].to_numpy(float)),
                              label=itr.y.to_numpy(int), base_margin=_logit(itr.p_mkt))
            dva = xgb.DMatrix(imp.transform(iva[feats].to_numpy(float)),
                              label=iva.y.to_numpy(int), base_margin=_logit(iva.p_mkt))
            dte = xgb.DMatrix(imp.transform(te[feats].to_numpy(float)),
                              base_margin=_logit(te.p_mkt))
            p = dict(objective='binary:logistic', eval_metric='logloss',
                     tree_method='hist', learning_rate=0.02, max_depth=3,
                     min_child_weight=125., subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=20., seed=seed, nthread=6)
            bst = xgb.train(p, dtr, num_boost_round=500, evals=[(dva,'v')],
                            early_stopping_rounds=40, verbose_eval=False)
            it = bst.best_iteration; iters.append(it+1)
            full = bst.predict(dte, output_margin=True, iteration_range=(0,it+1))
            preds[m.window_open.isin(tew).to_numpy()] = _sig(full)
        ok = np.isfinite(preds); s = m[ok]
        d = _ll(s.p_mkt, s.y) - _ll(preds[ok], s.y)
        mean, lo, hi, nw = window_boot_delta(d, s.window_open.to_numpy())
        print(f"  {name:<28} mmm {mean:+.6f} CI [{lo:+.6f},{hi:+.6f}] n={ok.sum():,} "
              f"w={nw:,} iters={iters} {'BEATS' if lo>0 else 'no'}")
        return {'name': name, 'status': 'OK', 'kind': 'xgb', 'features': feats,
                'n_oof': int(ok.sum()), 'windows': nw,
                'model_minus_market': mean, 'ci95': [lo, hi],
                'beats_market': bool(lo > 0), 'best_iters': iters}, preds

    r_x, preds_x = xgb_model(m, feats, 'S3_xgb_survivors')
    results['S3_xgb'] = r_x

    # ---- 3.2 negative control ------------------------------------------
    print('\n=== 3.2 negative control: features permuted within 2-day blocks ===')
    for seed in (0, 1):
        mp = m.copy(); rng = np.random.default_rng(seed)
        blk = mp.window_open.dt.floor('2D')
        for _, idx in mp.groupby(blk).groups.items():
            perm = rng.permutation(np.asarray(idx))
            mp.loc[idx, feats] = mp.loc[perm, feats].to_numpy()
        rr = run_model(mp, feats, 'lgbm', f'NC_permuted_s{seed}')
        results[f'NC_permuted_s{seed}'] = rr[0] if isinstance(rr, tuple) else rr

    # ---- 3.3 stability --------------------------------------------------
    print('\n=== 3.3 stability of the LightGBM candidate ===')
    ok = np.isfinite(preds_l); s = m[ok].copy(); s['p_model'] = preds_l[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    stab = {}
    for by, lbl in (('symbol','symbol'), ('minute','offset')):
        rows = []
        for k, g in s.groupby(by):
            if len(g) < 400: continue
            mean, lo, hi, nw = window_boot_delta(g.d.to_numpy(), g.window_open.to_numpy())
            rows.append({lbl: str(k), 'n': len(g), 'w': nw, 'mmm': mean, 'lo': lo, 'hi': hi})
        stab[lbl] = rows
        print(f"  --- by {lbl} ---")
        for r in rows:
            print(f"    {r[lbl]:>10} n={r['n']:>6,} w={r['w']:>4,} "
                  f"mmm {r['mmm']:+.6f} CI [{r['lo']:+.6f},{r['hi']:+.6f}]")
    s['half'] = np.where(s.window_open < s.window_open.median(), 'first', 'second')
    rows = []
    for k, g in s.groupby('half'):
        mean, lo, hi, nw = window_boot_delta(g.d.to_numpy(), g.window_open.to_numpy())
        rows.append({'half': k, 'n': len(g), 'w': nw, 'mmm': mean, 'lo': lo, 'hi': hi})
        print(f"  {k:>7} half: n={len(g):>6,} w={nw:>4,} mmm {mean:+.6f} "
              f"CI [{lo:+.6f},{hi:+.6f}]")
    stab['half'] = rows
    results['stability'] = stab

    # ---- 4 trading ------------------------------------------------------
    print('\n=== 4 trading simulation, venue fee schedule, depth-constrained ===')
    money = {}
    for gate in (0.5, 1.0, 1.5, 2.5):
        sim = trade_sim(m, preds_l, gate)
        money[f'gate_{gate}'] = sim
        if sim.get('trades'):
            print(f"  gate {gate:>4.1f}pp: {sim['trades']:>5,} trades  ${sim['pnl']:>8.2f}  "
                  f"{sim['pp_per_contract']:>+7.3f}pp/ct  t={sim['t_stat']:>5.2f}  "
                  f"win {sim['win_rate']*100:>5.1f}%  maxDD ${sim['max_drawdown']:.2f}")
        else:
            print(f"  gate {gate:>4.1f}pp: no trades")
    results['money_lgbm'] = money

    (OUT / 'stage34_results.json').write_text(json.dumps(results, indent=2, default=str))
    log({'stage': '3-4', 'results': {k: (v if not isinstance(v, dict) or 'name' in v else '...')
                                     for k, v in results.items()},
         'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 3-4 done in {time.time()-t0:.0f}s")
