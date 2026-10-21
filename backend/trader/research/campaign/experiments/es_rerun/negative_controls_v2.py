"""Negative controls, CORRECTED.

**The first version was wrong and its NC1 result is invalid.** It permuted the
LABEL within time blocks and left the market price in place. For a MARKET-RELATIVE
model that does not test what it claims: once the label is shuffled the market quote
no longer predicts it either, so the market becomes a confidently-WRONG forecast and
any model that merely shrinks toward 0.5 "beats" it. Measured: NC1 read
model_minus_market +0.359158, which is not a leak — it is the control measuring the
market's destruction rather than the model's.

The fix: permute the FEATURES against the (label, market price) pair, keeping label
and price together. The market keeps exactly its real predictive power, and the only
thing destroyed is the relationship between the features and the outcome. A model
that still beats the market under this permutation is reading something it should
not.

NC2 (noise features) and NC3 (shuffled price) are unaffected by this and are kept.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import harness as H, runner as R
from run_campaign import make_folds

BAR = ['geometry', 'vol_state', 'microstructure', 'cross_asset', 'offset']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--budget', type=int, default=2400)
    a = ap.parse_args()
    t0 = time.time()
    ds, cfg = R.build_dataset(bars_end=pd.Timestamp('2026-10-15', tz='UTC'))
    px, lab = H.market_prices(), H.venue_labels()
    folds = make_folds(ds.window_index)
    out = {}

    # ---- NC1 (corrected): permute FEATURES, keep (label, price) together ----
    print('\n=== NC1c features permuted within 2-day blocks '
          '(label and price kept together) ===', flush=True)
    orig_attach = R.attach_market

    def make_permuting_attach(seed):
        def attach(table, px_, lab_):
            t = orig_attach(table, px_, lab_)
            rng = np.random.default_rng(seed)
            feats = R.feature_cols(t, BAR)
            blk = t.window_open.dt.floor('2D')
            for _, idx in t.groupby(blk).groups.items():
                perm = rng.permutation(np.asarray(idx))
                t.loc[idx, feats] = t.loc[perm, feats].to_numpy()
            return t
        return attach

    for seed in (0, 1, 2):
        R.attach_market = make_permuting_attach(seed)
        try:
            res, _ = R.run(ds, cfg, folds, px, lab, name=f'NC1c_featperm_s{seed}',
                           hypothesis='features permuted within 2-day blocks; the market '
                                      'keeps its real predictive power, so any remaining '
                                      'skill is manufactured',
                           groups=BAR, fitter='lgbm', init='market', budget_s=a.budget)
        finally:
            R.attach_market = orig_attach
        if res.get('status') != 'OK':
            print(f'  seed {seed}: FAILED {res.get("reason")}', flush=True); continue
        print(f"  seed {seed}: mmm {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f},{res['ci95'][1]:+.6f}]  "
              f"{'LEAK SUSPECTED' if res['ci95'][0] > 0 else 'clean'}", flush=True)
        out[f'NC1c_s{seed}'] = res

    # ---- NC2: noise-only features ----
    # The first attempt monkeypatched `feature_cols` alone, which added the noise
    # columns to the TRAIN frame only -- `te[feats]` then raised KeyError. The
    # columns have to be attached by `attach_market`, which both frames go through.
    print('\n=== NC2 noise-only features ===', flush=True)
    NOISE = [f'_noise_{i}' for i in range(8)]

    def noise_attach(table, px_, lab_):
        t = orig_attach(table, px_, lab_)
        rng = np.random.default_rng(abs(hash(str(t.window_open.min()))) % 2**31)
        for c in NOISE:
            t[c] = rng.standard_normal(len(t))
        return t

    orig_fc = R.feature_cols
    R.attach_market = noise_attach
    R.feature_cols = lambda table, groups: list(NOISE)
    try:
        res, _ = R.run(ds, cfg, folds, px, lab, name='NC2_noise_only',
                       hypothesis='eight iid gaussian features must not beat the market',
                       groups=BAR, fitter='lgbm', init='market', budget_s=a.budget)
    finally:
        R.attach_market = orig_attach
        R.feature_cols = orig_fc
    if res.get('status') == 'OK':
        print(f"  mmm {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f},{res['ci95'][1]:+.6f}]  "
              f"{'LEAK SUSPECTED' if res['ci95'][0] > 0 else 'clean'}", flush=True)
        out['NC2_noise'] = res
    else:
        print(f"  FAILED {res.get('reason')}", flush=True)

    H.record('negative_controls_v2', {'results': out,
                                      'runtime_s': round(time.time() - t0, 1)})
    print(f'\nrecorded. {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
