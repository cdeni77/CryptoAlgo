"""STAGE 8 — was the tape result real signal, or a coarser book destroying it?

Stage 7 backfilled the surviving signals from venue_depth over Jul-Oct and found
nothing: window-weighted -0.000027 on 7,600 windows, against +0.000661 on the
tape's 987. Two explanations, very different consequences:

  (a) the tape result was noise, and 987 windows was never enough; or
  (b) Predexon's 21-level snapshot is too coarse to carry the signal -- CLAUDE.md
      measures level counts at a 0.579 ratio across sources, so `imbalance` from
      the backfill is not numerically the same quantity as from the full ladder.

The test holds the WINDOWS fixed and varies only the feature source: both built on
exactly the overlap, 2026-09-30..10-14. Same folds, same model, same labels.
If the tape wins there, it is (b) and the backfill cannot be used for this.
If both are null there, it is (a) and the earlier number was small-sample noise.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
from stage2_models import folds_of, _sig, _logit, _ll
from stage1_screen import load_flow, derive, log
from stage7_backfilled_flow import build_snapshot_flow, FEATS, fit_oof, _boot_ci
OUT = HERE.parent

OVERLAP_LO = pd.Timestamp('2026-09-30', tz='UTC')
OVERLAP_HI = pd.Timestamp('2026-10-15', tz='UTC')


def score(m, feats, label):
    preds, iters = fit_oof(m, feats, n_folds=3)
    ok = np.isfinite(preds)
    if ok.sum() < 500:
        print(f"  {label:<22} too few OOF preds ({ok.sum()})"); return None
    s = m[ok].copy(); s['p_model'] = preds[ok]
    d = (_ll(s.p_mkt, s.y) - _ll(s.p_model, s.y))
    wv = pd.DataFrame({'d': d, 'w': s.window_open}).groupby('w')['d'].mean().to_numpy()
    lo, hi = _boot_ci(wv)
    print(f"  {label:<22} mmm {wv.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  "
          f"n={ok.sum():,} w={len(wv):,} iters={iters}")
    return {'label': label, 'mmm': float(wv.mean()), 'ci': [lo, hi],
            'n': int(ok.sum()), 'windows': int(len(wv)), 'iters': iters}


if __name__ == '__main__':
    t0 = time.time(); res = {}
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable]

    # --- TAPE features (full-ladder reconstruction) ---
    tape = derive(load_flow())
    tape = tape.merge(mv, left_on=['symbol', 'window_open', 'minute'],
                      right_on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    tape = tape[tape.minute.between(1, 14)]
    tape = tape[(tape.window_open >= OVERLAP_LO) & (tape.window_open < OVERLAP_HI)]

    # --- BACKFILL features (Predexon 21-level snapshot) ---
    bf = build_snapshot_flow()
    bf = bf.drop(columns=['yes_bid', 'yes_ask', 'quote_age_seconds'], errors='ignore')
    bf = bf.merge(mv, on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    bf = bf[bf.offset_minutes.between(1, 14)]
    bf = bf[(bf.window_open >= OVERLAP_LO) & (bf.window_open < OVERLAP_HI)]

    # --- hold the WINDOWS fixed: intersect ---
    common = sorted(set(tape.window_open.unique()) & set(bf.window_open.unique()))
    tape = tape[tape.window_open.isin(common)].reset_index(drop=True)
    bf = bf[bf.window_open.isin(common)].reset_index(drop=True)
    print(f"overlap windows held fixed: {len(common):,}")
    print(f"  tape rows {len(tape):,}   backfill rows {len(bf):,}")
    print(f"  span {min(common)} -> {max(common)}\n")

    feats = [f for f in FEATS if f in tape.columns and f in bf.columns]
    print(f"features common to both sources: {feats}\n")
    print("=== same windows, same model, same folds — only the BOOK SOURCE differs ===")
    res['tape'] = score(tape, feats, 'TAPE full ladder')
    res['backfill'] = score(bf, feats, 'BACKFILL 21-level')

    # how different are the two measurements of the same quantity?
    j = tape[['symbol','window_open','minute','imbalance']].rename(
        columns={'minute':'offset_minutes','imbalance':'imb_tape'}).merge(
        bf[['symbol','window_open','offset_minutes','imbalance']].rename(
            columns={'imbalance':'imb_bf'}),
        on=['symbol','window_open','offset_minutes'], how='inner')
    if len(j):
        c = j.imb_tape.corr(j.imb_bf)
        print(f"\ncorrelation of `imbalance` between the two sources: {c:.4f} (n={len(j):,})")
        print(f"  mean |difference|: {(j.imb_tape - j.imb_bf).abs().mean():.4f}")
        res['imbalance_corr'] = float(c)
        res['imbalance_mean_abs_diff'] = float((j.imb_tape - j.imb_bf).abs().mean())
    (OUT / 'stage8_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 8, 'question': 'tape vs backfill book source on identical windows',
         'result': res, 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 8 done in {time.time()-t0:.0f}s")
