"""Mandatory negative controls (RESEARCH_PROTOCOL.md section 5).

Any candidate that appears to beat the market must pass all of these. They do not
prove the absence of leakage -- they catch the kinds that are easy to create by
accident, and an unusually strong result on any of them is a reason to audit the
pipeline rather than to celebrate.
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


def shuffle_labels_in_blocks(lab: pd.DataFrame, block_days: int = 2, seed: int = 0):
    """Permute the settlement WITHIN time blocks. Destroys any real relationship
    between features and outcome while preserving the base rate and the block
    structure. Skill must vanish; if it does not, something is leaking."""
    rng = np.random.default_rng(seed)
    out = lab.copy()
    blk = out.window_open.dt.floor(f'{block_days}D')
    out['settled_up'] = (out.groupby(blk)['settled_up']
                            .transform(lambda s: rng.permutation(s.to_numpy())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=1800)
    args = ap.parse_args()
    t0 = time.time()
    ds, cfg = R.build_dataset(bars_end=pd.Timestamp('2026-10-15', tz='UTC'))
    px, lab = H.stream_prices(), H.venue_labels()
    folds = make_folds(ds.window_index)
    print(f'folds: {len(folds)}', flush=True)

    out = {}

    # --- control 1: shuffled labels -------------------------------------
    print('\n=== NC1 shuffled labels (skill must vanish) ===', flush=True)
    for seed in (0, 1, 2):
        lab_s = shuffle_labels_in_blocks(lab, seed=seed)
        res, _ = R.run(ds, cfg, folds, px, lab_s, name=f'NC1_shuffled_s{seed}',
                       hypothesis='labels permuted within 2-day blocks; any apparent '
                                  'skill is manufactured', groups=BAR, fitter='lgbm',
                       init='market', budget_s=args.budget)
        if res['status'] != 'OK':
            print(f'  seed {seed}: FAILED {res["reason"]}'); continue
        print(f"  seed {seed}: model_minus_market {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f},{res['ci95'][1]:+.6f}]  "
              f"{'LEAK SUSPECTED' if res['ci95'][0] > 0 else 'clean'}", flush=True)
        out[f'NC1_s{seed}'] = res

    # --- control 2: noise-only features ---------------------------------
    print('\n=== NC2 noise-only features (must score ~0) ===', flush=True)
    import runner
    orig = runner.feature_cols
    def noise_cols(table, groups):
        n = len(table)
        rng = np.random.default_rng(abs(hash(str(table.window_open.iloc[0]))) % 2**31)
        for i in range(8):
            table[f'_noise_{i}'] = rng.standard_normal(n)
        return [f'_noise_{i}' for i in range(8)]
    runner.feature_cols = noise_cols
    res, _ = R.run(ds, cfg, folds, px, lab, name='NC2_noise_only',
                   hypothesis='eight iid gaussian features; a model fitted on them '
                              'must not beat the market', groups=BAR, fitter='lgbm',
                   init='market', budget_s=args.budget)
    runner.feature_cols = orig
    if res['status'] == 'OK':
        print(f"  model_minus_market {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f},{res['ci95'][1]:+.6f}]  "
              f"{'LEAK SUSPECTED' if res['ci95'][0] > 0 else 'clean'}", flush=True)
        out['NC2_noise'] = res
    else:
        print(f"  FAILED {res['reason']}")

    # --- control 3: shuffled market price -------------------------------
    print('\n=== NC3 market price shuffled across windows (sanity on the benchmark) ===',
          flush=True)
    rng = np.random.default_rng(7)
    px_s = px.copy()
    px_s['p_mkt'] = rng.permutation(px_s.p_mkt.to_numpy())
    res, _ = R.run(ds, cfg, folds, px_s, lab, name='NC3_shuffled_price',
                   hypothesis='against a MEANINGLESS price any real model must win '
                              'by a large margin; if it does not, the model carries '
                              'no information of its own', groups=BAR, fitter='lgbm',
                   init='market', budget_s=args.budget)
    if res['status'] == 'OK':
        print(f"  model_minus_market {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f},{res['ci95'][1]:+.6f}]", flush=True)
        out['NC3_shuffled_price'] = res

    H.record('negative_controls', {'results': out,
                                   'runtime_s': round(time.time()-t0, 1)})
    print(f'\nrecorded. {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
