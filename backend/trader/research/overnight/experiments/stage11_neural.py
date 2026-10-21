"""STAGE 11 — a residual neural network on the full feature set.

Why a net might differ from the trees, stated before the result: LightGBM and
XGBoost agreed to three decimals, which is evidence about the DATA rather than the
estimator -- but both build axis-aligned splits, and the underlying object here is
smooth (`F(x/sigma)` is a CDF). A network can represent smooth interactions across
many weak inputs that a depth-4 tree cannot, and it extrapolates instead of forming
a constant outside the training support. That is a real difference in inductive
bias, not a hope.

**Residual by construction.** The net outputs a CORRECTION to the market logit:

    p = sigmoid( logit(p_mkt) + correction(x) )

and the final layer is ZERO-INITIALISED, so an untrained network reproduces the
Kalshi price exactly and every unit of training is incremental skill -- the same
guarantee `init_score` gives the trees. The correction is bounded by a scaled tanh
so the net cannot produce the 34x-overconfident predictions that sank an earlier
incarnation of this project.

Folds are purged chronological and split on the WINDOW. Early stopping on an inner
chronological split. Per-fold trade counts are reported because a model that sits
out whole weeks is not usable.
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


def train_fold(Xtr, ytr, mtr, Xva, yva, mva, Xte, mte, *, seed=17, hidden=(128, 64),
               dropout=0.3, wd=1e-3, lr=1e-3, epochs=200, patience=20,
               max_corr=2.0, device='cuda'):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    def T(a, dt=torch.float32): return torch.tensor(np.asarray(a), dtype=dt, device=dev)
    Xtr_t, ytr_t, mtr_t = T(Xtr), T(ytr), T(mtr)
    Xva_t, yva_t, mva_t = T(Xva), T(yva), T(mva)
    Xte_t, mte_t = T(Xte), T(mte)

    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    head = nn.Linear(d, 1)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)   # start AT the market
    net = nn.Sequential(*layers, head).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float('inf'), None, 0
    n = len(Xtr_t); bs = 4096
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            corr = max_corr * torch.tanh(net(Xtr_t[idx]).squeeze(-1))
            loss = lossf(mtr_t[idx] + corr, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vcorr = max_corr * torch.tanh(net(Xva_t).squeeze(-1))
            vl = lossf(mva_t + vcorr, yva_t).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        corr = max_corr * torch.tanh(net(Xte_t).squeeze(-1)).cpu().numpy()
    return corr, ep + 1, best


def sim_per_fold(s, gate, fee_rate=0.07, stake=1):
    rows = []
    # itertuples mangles a leading-underscore column name, so alias it first
    d = s.rename(columns={'_fold': 'foldid'}).sort_values(['symbol','window_open','offset_minutes'])
    for (sym, w), g in d.groupby(['symbol','window_open'], sort=False):
        for r in g.itertuples():
            pm = r.p_model
            if not np.isfinite(pm): continue
            yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
            ev_y, ev_n = pm - yes_px, (1.0 - pm) - no_px
            side, px, ev = ('yes', yes_px, ev_y) if ev_y >= ev_n else ('no', no_px, ev_n)
            if not (0.0 < px < 1.0): continue
            fee = np.ceil(fee_rate*stake*px*(1-px)*10_000)/10_000
            if ev*stake - fee < gate/100.0*stake: continue
            vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
            if not np.isfinite(vol) or vol < stake: continue
            won = (r.y == 1) if side == 'yes' else (r.y == 0)
            rows.append({'fold': r.foldid, 'window_open': w,
                         'pnl': (stake if won else 0.0) - px*stake - fee, 'won': won})
            break
    return pd.DataFrame(rows)


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
    feats = [c for c in bar_cols + extra if c not in CONTROL]
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}  features {len(feats)}")
    print(f"  (time_of_day control EXCLUDED: it was 27.7% of tree gain and is known null)")

    # missingness indicators for the sparse book/flow columns
    sparse = [c for c in extra if m[c].notna().mean() < 0.99]
    for c in sparse:
        m[f'{c}__miss'] = m[c].isna().astype(float)
    feats = feats + [f'{c}__miss' for c in sparse]
    print(f"  + {len(sparse)} missingness indicators -> {len(feats)} inputs")

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    preds = np.full(len(m), np.nan); eps_used = []
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 5000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 2000 or len(iva) < 1000: itr, iva = tr, tr
        imp = SimpleImputer(strategy='median').fit(itr[feats])
        sc = StandardScaler().fit(imp.transform(itr[feats]))
        prep = lambda d: np.clip(sc.transform(imp.transform(d[feats])), -8, 8)
        corr, ep, vl = train_fold(
            prep(itr), itr.y.to_numpy(float), _logit(itr.p_mkt),
            prep(iva), iva.y.to_numpy(float), _logit(iva.p_mkt),
            prep(te), _logit(te.p_mkt))
        preds[te_m] = _sig(_logit(te.p_mkt) + corr)
        eps_used.append(ep)
        print(f"  fold {i}: train {tr_m.sum():,} -> test {te_m.sum():,}  "
              f"epochs {ep}  inner BCE {vl:.6f}", flush=True)

    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    print(f"\n=== NEURAL RESIDUAL MODEL ===")
    res = {'n_features': len(feats), 'epochs': eps_used, 'n_oof': int(ok.sum())}
    for lbl, v in (('row', s.d.to_numpy()),
                   ('sym-win', s.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f"  {lbl:>8} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}"
              f"{'   BEATS THE MARKET' if lo > 0 else ''}")
        res[lbl] = {'mean': float(v.mean()), 'ci': [lo, hi], 'n': int(len(v))}
    print(f"\n  log loss  model {np.mean(_ll(s.p_model, s.y)):.6f}   "
          f"market {np.mean(_ll(s.p_mkt, s.y)):.6f}")

    print("\n=== TRADES PER FOLD ===")
    money = {}
    for gate in (1.0, 1.5, 2.5):
        t = sim_per_fold(s, gate)
        if not len(t):
            print(f"  gate {gate}pp: NO TRADES AT ALL"); money[f'gate_{gate}']={'trades':0}; continue
        per, silent = [], 0
        print(f"  gate {gate}pp:")
        for i in sorted(s._fold.unique()):
            g = t[t.fold == i]; w = s[s._fold == i].window_open.nunique()
            if not len(g):
                silent += 1; print(f"    fold {i}: {w:>5,} windows -> **0 TRADES**")
                per.append({'fold': int(i), 'trades': 0}); continue
            print(f"    fold {i}: {w:>5,} windows -> {len(g):>5,} trades  "
                  f"${g.pnl.sum():>8.2f}  {g.pnl.sum()/len(g)*100:>+7.3f}pp/ct  "
                  f"win {g.won.mean()*100:>5.1f}%")
            per.append({'fold': int(i), 'trades': int(len(g)), 'pnl': float(g.pnl.sum()),
                        'pp_per_contract': float(g.pnl.sum()/len(g)*100)})
        gw = t.groupby('window_open').pnl.sum()
        ts = float(gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw)))) if len(gw) > 1 else float('nan')
        print(f"    TOTAL: {len(t):,} trades  ${t.pnl.sum():.2f}  "
              f"{t.pnl.sum()/len(t)*100:+.3f}pp/ct  t={ts:.2f}  silent {silent}/{len(per)}")
        money[f'gate_{gate}'] = {'per_fold': per, 'silent_folds': silent,
                                 'trades': int(len(t)), 'pnl': float(t.pnl.sum()),
                                 'pp_per_contract': float(t.pnl.sum()/len(t)*100),
                                 't_stat': ts}
    res['money'] = money
    print("\n=== by month ===")
    s['month'] = s.window_open.dt.to_period('M').astype(str)
    for mo, g in s.groupby('month'):
        wv = g.groupby('window_open').d.mean().to_numpy(); lo, hi = _boot_ci(wv)
        print(f"  {mo}  w={len(wv):>5,}  {wv.mean():+.6f} [{lo:+.6f},{hi:+.6f}]")
    s[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
       'yes_bid','yes_ask','ask_at_touch','bid_at_touch','_fold']].to_parquet(
        OUT / 'oof_stage11_neural.parquet', index=False)
    (OUT / 'stage11_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 11, 'experiment': 'residual neural network, all sources',
         'result': {k: v for k, v in res.items() if k != 'money'},
         'money': money, 'runtime_s': round(time.time()-t0, 1)})
    # ---- head-to-head: trees on the IDENTICAL feature set ---------------
    print("\n=== SAME FEATURES, LightGBM instead of the net ===")
    import lightgbm as lgb
    tp = np.full(len(m), np.nan)
    for i in sorted(m._fold.unique()):
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
        tp[te_m] = _sig(_logit(te.p_mkt)
                        + np.asarray(bst.predict(te[feats], raw_score=True), float))
    okt = np.isfinite(tp); st = m[okt].copy(); st['p_model'] = tp[okt]
    st['d'] = _ll(st.p_mkt, st.y) - _ll(st.p_model, st.y)
    tree = {}
    for lbl, v in (('row', st.d.to_numpy()),
                   ('sym-win', st.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', st.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f"  {lbl:>8} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}"
              f"{'   BEATS THE MARKET' if lo > 0 else ''}")
        tree[lbl] = {'mean': float(v.mean()), 'ci': [lo, hi], 'n': int(len(v))}
    res['tree_same_features'] = tree
    t2 = sim_per_fold(st, 1.5)
    if len(t2):
        gw2 = t2.groupby('window_open').pnl.sum()
        print(f"  tree @1.5pp: {len(t2):,} trades  ${t2.pnl.sum():.2f}  "
              f"{t2.pnl.sum()/len(t2)*100:+.3f}pp/ct  "
              f"t={gw2.mean()/(gw2.std(ddof=1)/np.sqrt(len(gw2))):.2f}  "
              f"silent folds {sum(1 for i in sorted(st._fold.unique()) if not len(t2[t2.fold==i]))}")
    (OUT / 'stage11_results.json').write_text(json.dumps(res, indent=2, default=str))
    print(f"\nstage 11 done in {time.time()-t0:.0f}s")
