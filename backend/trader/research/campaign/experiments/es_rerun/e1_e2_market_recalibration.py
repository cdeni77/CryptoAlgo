"""E1 / E2 — is the Kalshi mid itself miscalibrated in a correctable way?

E1: a single monotone (isotonic) recalibration of the mid, fitted out-of-fold.
E2: the same, conditioned on the decision offset.

These need no features at all, only the price and the venue's label, so they are the
cheapest possible way to beat the market -- and if they work, every later experiment
must beat THEM, not the raw mid. If they do not, the market is already calibrated and
any edge has to come from information, not from a correction.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import harness as H

px = H.stream_prices(); lab = H.venue_labels()
df = px.merge(lab, on=['symbol','window_open'], how='inner')
df = df[(df.window_open >= H.BENCH_START) & (df.window_open < H.BENCH_END)]
df = df[df.offset_minutes.between(1,14)].copy()
df['y'] = df.settled_up.astype(int)
df['split'] = np.where(df.window_open < H.DEV_END, 'dev', 'val')
df = df.sort_values(['window_open','symbol','offset_minutes']).reset_index(drop=True)
print(f"benchmark frame: {len(df):,} rows, {df.window_open.nunique():,} windows, "
      f"{df.groupby(['symbol','window_open']).ngroups:,} symbol-windows")
print(f"  dev {(df.split=='dev').sum():,}   val {(df.split=='val').sum():,}")
print(f"  span {df.window_open.min()} -> {df.window_open.max()}")
print(f"  base rate {df.y.mean():.4f}")

print("\n--- R0: the raw Kalshi mid, the thing to beat ---")
for s in ('dev','val','all'):
    g = df if s=='all' else df[df.split==s]
    c = H.calibration(g.p_mkt, g.y)
    print(f"  {s:>4}  n={len(g):>7,}  log_loss {H.logloss(g.p_mkt,g.y):.6f}  "
          f"calib_err {c['error']:.5f}  max_dev {c['max_dev']:.4f}")

print("\n--- the market's reliability table (all rows) ---")
c = H.calibration(df.p_mkt, df.y, bins=12)
print(f"  {'n':>7} {'mean price':>11} {'realised':>10} {'dev pp':>8}")
for r in c['table']:
    print(f"  {r['n']:>7,} {r['p_mean']:>11.4f} {r['y_mean']:>10.4f} "
          f"{(r['p_mean']-r['y_mean'])*100:>+8.3f}")

# ---- out-of-fold recalibration, folds on the WINDOW ----------------------
folds = H.window_folds(df.window_open, n_folds=5)
print(f"\nfolds: {len(folds)}")
for f in folds: print("  "+f.label())

def oof(fit_predict):
    out = np.full(len(df), np.nan)
    for f in folds:
        tr = df.window_open.isin(set(f.train)).to_numpy()
        te = df.window_open.isin(set(f.test)).to_numpy()
        if tr.sum() < 200 or te.sum() == 0: continue
        out[te] = fit_predict(df[tr], df[te])
    return out

def iso(tr, te):
    m = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
    m.fit(tr.p_mkt, tr.y)
    return m.predict(te.p_mkt)

def iso_by_offset(tr, te):
    out = np.full(len(te), np.nan)
    for o in sorted(te.offset_minutes.unique()):
        s_tr = tr[tr.offset_minutes == o]; s_te = (te.offset_minutes == o).to_numpy()
        if len(s_tr) < 300:
            out[s_te] = te.p_mkt.to_numpy()[s_te]; continue
        m = IsotonicRegression(out_of_bounds='clip', y_min=0.001, y_max=0.999)
        m.fit(s_tr.p_mkt, s_tr.y)
        out[s_te] = m.predict(te.p_mkt.to_numpy()[s_te])
    return out

def platt(tr, te):
    lg = np.log(np.clip(tr.p_mkt,1e-6,1-1e-6)/(1-np.clip(tr.p_mkt,1e-6,1-1e-6)))
    m = LogisticRegression(C=1e6).fit(lg.to_numpy().reshape(-1,1), tr.y)
    lt = np.log(np.clip(te.p_mkt,1e-6,1-1e-6)/(1-np.clip(te.p_mkt,1e-6,1-1e-6)))
    return m.predict_proba(lt.to_numpy().reshape(-1,1))[:,1]

results = {}
for name, fn in [('E1_isotonic', iso), ('E1b_platt', platt), ('E2_isotonic_by_offset', iso_by_offset)]:
    p = oof(fn)
    ok = np.isfinite(p)
    sub = df[ok]
    mean, lo, hi, pval, nw = H.model_minus_market(p[ok], sub.p_mkt, sub.y, sub.window_open)
    c = H.calibration(p[ok], sub.y)
    print(f"\n--- {name} ---")
    print(f"  n={ok.sum():,} over {nw:,} windows")
    print(f"  log loss          model {H.logloss(p[ok],sub.y):.6f}   market {H.logloss(sub.p_mkt,sub.y):.6f}")
    print(f"  model_minus_market {mean:+.6f}  95% CI [{lo:+.6f}, {hi:+.6f}]  p(<=0)={pval:.4f}")
    print(f"  calib_err {c['error']:.5f} (market {H.calibration(sub.p_mkt,sub.y)['error']:.5f})")
    print(f"  VERDICT: {'BEATS the market' if lo>0 else 'does NOT beat the market'}")
    results[name] = {'n': int(ok.sum()), 'windows': nw,
                     'log_loss_model': H.logloss(p[ok],sub.y),
                     'log_loss_market': H.logloss(sub.p_mkt,sub.y),
                     'model_minus_market': mean, 'ci95': [lo,hi], 'p_one_sided': pval,
                     'calibration': c, 'beats_market': bool(lo>0)}

H.record('E1_E2_market_recalibration', {
    'hypothesis': 'the Kalshi mid carries a stable (and offset-varying) miscalibration '
                  'correctable without any new information',
    'dataset': 'stream ladder (live, live_ws) x venue settlement, 2026-08-25..09-16',
    'rows': int(len(df)), 'windows': int(df.window_open.nunique()),
    'symbol_windows': int(df.groupby(['symbol','window_open']).ngroups),
    'market_log_loss_all': H.logloss(df.p_mkt, df.y),
    'market_calibration': H.calibration(df.p_mkt, df.y, bins=12),
    'results': results,
})
print("\nrecorded -> research/campaign/experiments/E1_E2_market_recalibration.json")
