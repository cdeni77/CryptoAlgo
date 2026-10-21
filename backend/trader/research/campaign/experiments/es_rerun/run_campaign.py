"""Drive the pre-registered experiments. One process, one shared Dataset.

Folds: the TEST blocks tile the benchmark era (2026-08-25..09-16). The TRAIN set is
every window strictly before `test_start - embargo`, which may reach back to
2026-01 -- so a candidate's training window length is a parameter (`trailing_days`)
rather than an accident of where the benchmark era starts.
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import harness as H
import runner as R


@dataclass
class Fold:
    index: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex
    def label(self):
        return (f'fold {self.index}: train {len(self.train):,}w (to {self.train.max():%m-%d}) '
                f'-> test {len(self.test):,}w [{self.test.min():%m-%d} .. {self.test.max():%m-%d}]')


def make_folds(all_windows, n_folds=5, embargo_days=1, *, price_era_only=True,
               min_train=1000):
    """Test blocks tile the evaluation era; training is everything before the block
    minus the embargo.

    `price_era_only` restricts TRAINING to the span where a point-in-time market
    price exists (2026-07-01 onward). On by default so every candidate is fitted and
    scored on the same rows -- a comparison sweep must fix the sample deliberately,
    or configurations get ranked partly on which rows they happened to see.
    """
    bench = all_windows[(all_windows >= H.BENCH_START) & (all_windows < H.BENCH_END)]
    blocks = [b for b in np.array_split(bench, n_folds) if len(b)]
    folds = []
    for i, b in enumerate(blocks):
        cut = b.min() - pd.Timedelta(days=embargo_days)
        tr = (bench[bench < cut] if price_era_only
              else all_windows[all_windows < cut])
        if len(tr) < min_train:
            continue
        folds.append(Fold(i, pd.DatetimeIndex(tr), pd.DatetimeIndex(b)))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='comma-separated experiment names')
    ap.add_argument('--budget', type=int, default=1200)
    args = ap.parse_args()

    t0 = time.time()
    print('building dataset (bars only; the market price is joined separately)...', flush=True)
    ds, cfg = R.build_dataset(bars_end=pd.Timestamp('2026-10-15', tz='UTC'))
    px, lab = H.stream_prices(), H.venue_labels()
    print(f'  windows {len(ds.window_index):,}  stream quotes {len(px):,}  labels {len(lab):,}'
          f'  [{time.time()-t0:.1f}s]', flush=True)
    folds = make_folds(ds.window_index)
    print('  folds (identical for every candidate):', flush=True)
    for f in folds: print('    ' + f.label(), flush=True)

    BAR = ['geometry', 'vol_state', 'microstructure', 'cross_asset', 'offset']
    CTRL = ['time_of_day']

    SPECS = [
      dict(name='E3_baseline_only', hypothesis='the barrier baseline F(x/sigma) carries '
           'information the market does not', groups=['offset'], fitter='lgbm',
           init='baseline', params={'n_estimators': 1}),
      dict(name='E4_logistic_market_resid', hypothesis='a regularized logistic model on '
           'the market logit plus geometry and vol_state beats the market',
           groups=['geometry', 'vol_state'], fitter='logistic', init='market', params={'C': 0.05}),
      dict(name='E5_lgbm_bars_market_init', hypothesis='LightGBM on reliable bar-only '
           'features with the market logit as init_score beats the market',
           groups=BAR, fitter='lgbm', init='market'),
      dict(name='E5c_lgbm_bars_plus_control', hypothesis='the clock control must not carry '
           'the model', groups=BAR + CTRL, fitter='lgbm', init='market'),
      dict(name='E6_lgbm_short_window', hypothesis='a short recent training window (30d) '
           'beats the long window', groups=BAR, fitter='lgbm', init='market',
           trailing_days=30),
      dict(name='E6b_lgbm_very_short', hypothesis='an even shorter window (14d) beats both',
           groups=BAR, fitter='lgbm', init='market', trailing_days=14),
      dict(name='E9_catboost_bars', hypothesis="CatBoost's ordered boosting beats LightGBM "
           'at this sample size', groups=BAR, fitter='catboost', init='market'),
      dict(name='E10_direct_target', hypothesis='targeting settlement with the market as a '
           'plain feature beats using it as an offset', groups=BAR, fitter='lgbm',
           init='none'),
    ]
    if args.only:
        want = set(args.only.split(','))
        SPECS = [s for s in SPECS if s['name'] in want]

    results = {}
    for spec in SPECS:
        print(f"\n=== {spec['name']} ===", flush=True)
        res, oof = R.run(ds, cfg, folds, px, lab, budget_s=args.budget, **spec)
        if res['status'] != 'OK':
            print(f"  FAILED: {res['reason']}", flush=True)
            results[spec['name']] = res; continue
        print(f"  n={res['n_oof']:,} over {res['windows']:,} windows  [{res['runtime_s']}s]")
        print(f"  log loss  model {res['log_loss_model']:.6f}  market {res['log_loss_market']:.6f}"
              + (f"  baseline {res['log_loss_baseline']:.6f}" if res['log_loss_baseline'] else ''))
        print(f"  model_minus_market {res['model_minus_market']:+.6f} "
              f"CI [{res['ci95'][0]:+.6f}, {res['ci95'][1]:+.6f}] p={res['p_one_sided']:.4f}")
        print(f"  calib {res['calibration']['error']:.5f} (market {res['market_calibration']['error']:.5f})")
        print(f"  VERDICT: {'BEATS the market' if res['beats_market'] else 'does NOT beat the market'}")
        # money, at the pre-registered gate, on the validation split only
        for gate in (1.0, 1.5, 2.5):
            sim = H.simulate(oof, 'p_model', min_edge_pp=gate, offsets=(3, 6, 9, 12))
            if sim['trades']:
                print(f"    gate {gate:>4.1f}pp: {sim['trades']:>5,} trades  "
                      f"pnl ${sim['pnl']:>8.2f}  {sim['pp_per_contract']:>+7.3f}pp/ct  "
                      f"t={sim['t_stat']:>5.2f}  CI[{sim['pnl_ci95'][0]:.1f},{sim['pnl_ci95'][1]:.1f}]")
                sim.pop('_trades', None)
            else:
                print(f"    gate {gate:>4.1f}pp: no trades")
                sim.pop('_trades', None)
            res.setdefault('money', {})[f'gate_{gate}'] = sim
        results[spec['name']] = res
        if oof is not None:
            oof[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
                 'yes_bid','yes_ask','split']].to_parquet(
                HERE / f"oof_{spec['name']}.parquet", index=False)
    H.record('campaign_batch_1', {'results': results,
                                  'folds': [f.label() for f in folds],
                                  'total_runtime_s': round(time.time()-t0, 1)})
    print(f"\nrecorded. total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
