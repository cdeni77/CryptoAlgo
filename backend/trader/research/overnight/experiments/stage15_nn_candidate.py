"""STAGE 15 — the user's specification, exactly.

  neural network | July 1 -> today | ALL features | min edge 0.5pp | Kelly 0.25
  PROMOTE IF: profitable AND trades in at least 4 of the 5 most recent folds.

Built on the CORRECTED feature path. The previous neural run (Stage 11) is invalid:
`build_snapshot_flow` and `book_and_peer` took the newest revision of each row
regardless of availability, so 25.3% of Kalshi and 48.0% of Polymarket quotes were
future-dated. Both now enforce `available_time <= window_open + offset + 3s`.

Residual by construction: the net outputs a correction to the market logit with a
zero-initialised head, so an untrained network reproduces the Kalshi price exactly.
Correction bounded by a scaled tanh so it cannot produce the 34x-overconfident
predictions an earlier incarnation of this project shipped.

Kelly sizing is applied for the dollar P&L; per-contract edge is reported alongside
because it is size-independent and therefore the number that survives a bankroll
change.
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
from stage11_neural import train_fold
from stage1_screen import log
OUT = HERE.parent

MIN_EDGE_PP = 0.5
KELLY = 0.25
BANKROLL = 500.0
CONTROL = ['quarter_of_hour','hour_sin','hour_cos','dow_sin','dow_cos','us_equity_hours']


def assemble():
    bars = bar_features()
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v3.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable][['symbol','window_open','offset_minutes','p_mkt','y',
                        'yes_bid','yes_ask','ask_at_touch','bid_at_touch']]
    m = bars.merge(mv, on=['symbol','window_open','offset_minutes'], how='inner')
    flow = build_snapshot_flow()
    flow = flow[['symbol','window_open','offset_minutes'] + [c for c in FLOW if c in flow.columns]]
    m = m.merge(flow, on=['symbol','window_open','offset_minutes'], how='left')
    k, p = book_and_peer()
    if len(k): m = m.merge(k, on=['symbol','window_open','offset_minutes'], how='left')
    if len(p): m = m.merge(p, on=['symbol','window_open','offset_minutes'], how='left')
    iv = implied_vol()
    if len(iv): m = m.merge(iv, on=['symbol','window_open','offset_minutes'], how='left')
    return m[m.offset_minutes.between(1,14)].reset_index(drop=True)


def simulate(s, gate=MIN_EDGE_PP, kelly=KELLY, bankroll=BANKROLL):
    """One entry per (symbol, window). Kelly-sized, floored to whole contracts."""
    d = s.sort_values(['symbol','window_open','offset_minutes'])
    rows = []
    for (sym, w), g in d.groupby(['symbol','window_open'], sort=False):
        for r in g.itertuples():
            pm = r.p_model
            if not np.isfinite(pm): continue
            yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
            ev_y, ev_n = pm - yes_px, (1.0 - pm) - no_px
            side, px, ev, q = (('yes', yes_px, ev_y, pm) if ev_y >= ev_n
                               else ('no', no_px, ev_n, 1.0 - pm))
            if not (0.0 < px < 1.0): continue
            fee1 = np.ceil(0.07 * px * (1-px) * 10_000) / 10_000
            if ev - fee1 < gate / 100.0: continue
            # Kelly for a binary costing px and paying 1: f* = (q - px) / (1 - px)
            fstar = max(0.0, (q - px) / (1.0 - px))
            n = int(np.floor(kelly * fstar * bankroll / px))
            if n < 1: continue
            vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
            if not np.isfinite(vol): continue
            n = int(min(n, vol))
            if n < 1: continue
            fee = np.ceil(0.07 * n * px * (1-px) * 10_000) / 10_000
            if ev * n - fee < gate / 100.0 * n: continue
            won = (r.y == 1) if side == 'yes' else (r.y == 0)
            rows.append({'fold': r.foldid, 'window_open': w, 'symbol': sym,
                         'side': side,
                         'contracts': n, 'price': px, 'fee': fee, 'won': won,
                         'pnl': (n if won else 0.0) - px*n - fee,
                         'offset': r.offset_minutes})
            break
    return pd.DataFrame(rows)


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
    print(f"rows {len(m):,}  windows {m.window_open.nunique():,}  "
          f"span {m.window_open.min().date()} -> {m.window_open.max().date()}")
    print(f"features {len(feats)}  (ALL sources; control excluded as known-null)")
    print("  population of the book/flow/peer/iv columns:")
    for c in extra:
        print(f"    {c:<20} {m[c].notna().mean()*100:5.1f}%")

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    preds = np.full(len(m), np.nan); eps = []
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy(); tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 3000 or te_m.sum() == 0: continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 1500 or len(iva) < 800: itr, iva = tr, tr
        imp = SimpleImputer(strategy='median').fit(itr[feats])
        sc = StandardScaler().fit(imp.transform(itr[feats]))
        prep = lambda d: np.clip(sc.transform(imp.transform(d[feats])), -8, 8)
        corr, ep, vl = train_fold(prep(itr), itr.y.to_numpy(float), _logit(itr.p_mkt),
                                  prep(iva), iva.y.to_numpy(float), _logit(iva.p_mkt),
                                  prep(te), _logit(te.p_mkt))
        preds[te_m] = _sig(_logit(te.p_mkt) + corr); eps.append(ep)
        print(f"  fold {i}: train {tr_m.sum():,} -> test {te_m.sum():,}  epochs {ep}",
              flush=True)

    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s = s.rename(columns={'_fold': 'foldid'})
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    res = {'n_features': len(feats), 'features': feats, 'epochs': eps,
           'min_edge_pp': MIN_EDGE_PP, 'kelly': KELLY, 'bankroll': BANKROLL}
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
        print("  NO TRADES AT ALL -> FAILS"); res['verdict'] = 'FAIL: no trades'
    else:
        print(f"  {'fold':>5} {'windows':>8} {'trades':>7} {'contracts':>10} {'pnl$':>9} {'pp/ct':>8}")
        per = []
        for f in folds:
            g = t[t.fold == f]; w = s[s.foldid == f].window_open.nunique()
            if not len(g):
                print(f"  {int(f):>5} {w:>8,} {'SILENT':>7}")
                per.append({'fold': int(f), 'trades': 0}); continue
            print(f"  {int(f):>5} {w:>8,} {len(g):>7,} {g.contracts.sum():>10,} "
                  f"{g.pnl.sum():>9.2f} {g.pnl.sum()/g.contracts.sum()*100:>+8.3f}")
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
                    'verdict': verdict,
                    'max_drawdown': float((gw.cumsum().cummax()-gw.cumsum()).max())})
    s.to_parquet(OUT / 'oof_stage15_nn.parquet', index=False)
    (OUT / 'stage15_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 15, 'experiment': "user spec: NN, Jul-today, all features, "
                                    "min_edge 0.5, kelly 0.25",
         'result': {k: v for k, v in res.items() if k != 'features'},
         'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 15 done in {time.time()-t0:.0f}s")
