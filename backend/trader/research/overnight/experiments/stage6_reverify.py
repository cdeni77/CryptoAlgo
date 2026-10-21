"""STAGE 6 — re-verify the order-flow candidate under BOTH weightings.

Stage 5b exposed a flaw in `boot_windows`: averaging per-window means gives sparse
windows the same weight as dense ones, and when group size correlates with price
composition that produces a Simpson's paradox (the tight-spread "edge" was
+4.63pp window-weighted and **-1.74pp row-weighted**).

`model_minus_market` should be immune, because model and market are scored on the
SAME rows and the statistic is a PAIRED difference -- any row weight applies to both
sides. This checks that rather than assuming it.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage2_models import build, folds_of, run_model, SURVIVORS, _ll, _sig, _logit
from stage1_screen import log

OUT = HERE.parent

def boot(v, B=10000, seed=0):
    v = np.asarray(v); n = len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs,.025)), float(np.quantile(bs,.975)), n

if __name__ == '__main__':
    t0 = time.time()
    m = build()
    feats = [f for f in SURVIVORS if f in m.columns]
    res, preds = run_model(m, feats, 'lgbm', 'S6_lgbm_reverify')
    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    print(f"\n=== model_minus_market under three weightings (n={len(s):,}) ===")
    out = {}
    rm, rlo, rhi, rn = boot(s.d.to_numpy())
    print(f"  ROW weighted           {rm:+.6f} CI [{rlo:+.6f},{rhi:+.6f}]  n={rn:,}")
    out['row'] = {'mean': rm, 'ci': [rlo, rhi], 'n': rn}
    wv = s.groupby('window_open').d.mean().to_numpy()
    wm, wlo, whi, wn = boot(wv)
    print(f"  WINDOW weighted        {wm:+.6f} CI [{wlo:+.6f},{whi:+.6f}]  w={wn:,}")
    out['window'] = {'mean': wm, 'ci': [wlo, whi], 'n': wn}
    sv = s.groupby(['symbol','window_open']).d.mean().to_numpy()
    sm, slo, shi, sn = boot(sv)
    print(f"  SYMBOL-WINDOW weighted {sm:+.6f} CI [{slo:+.6f},{shi:+.6f}]  sw={sn:,}")
    out['symbol_window'] = {'mean': sm, 'ci': [slo, shi], 'n': sn}
    agree = (np.sign(rm) == np.sign(wm) == np.sign(sm))
    print(f"\n  all three weightings agree in sign: {agree}")
    print("  (the tight-spread finding did NOT: -1.74 row vs +4.63 window)")

    # row-count composition check, the thing that caused the paradox
    cnt = s.groupby('window_open').size()
    s2 = s.merge(cnt.rename('rows_in_window'), left_on='window_open', right_index=True)
    print(f"\n  rows per window: min {cnt.min()} median {int(cnt.median())} max {cnt.max()}")
    print(f"  corr(rows_in_window, p_mkt) = {s2.rows_in_window.corr(s2.p_mkt):+.4f}")
    print("  (a large correlation here is what makes the weightings diverge)")
    out['rows_per_window'] = {'min': int(cnt.min()), 'median': int(cnt.median()),
                              'max': int(cnt.max()),
                              'corr_with_price': float(s2.rows_in_window.corr(s2.p_mkt))}
    (OUT / 'stage6_reverify.json').write_text(json.dumps(out, indent=2, default=str))
    log({'stage': 6, 'check': 'weighting robustness of model_minus_market',
         'result': out, 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 6 done in {time.time()-t0:.0f}s")
