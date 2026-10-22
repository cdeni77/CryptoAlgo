"""Fit and report the barrier baseline alone. The Phase 1 gate.

Shows how much of a 15-minute up/down market arithmetic already settles.
Every model result is measured as a difference against this baseline.

The gate: the baseline must be calibrated out of sample. Otherwise a model's
skill against it is partly the baseline's own miscalibration.

    python -m scripts.baseline
    python -m scripts.baseline --baseline-distribution normal   # the comparison
"""

from __future__ import annotations

import argparse
import sys

import logging
import numpy as np
import pandas as pd

from core.baseline import log_loss, reliability
from core.cv import assert_no_leakage, effective_observations, folds_for_config
from core.dataset import apply_fold, fit_fold
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
)


logger = logging.getLogger('baseline')


def _graded_label(test, config):
    """The outcome to score against, honouring `label_source`.

    Falls back to the proxy row by row where the venue has no settlement, so
    a partially covered span is still scored rather than dropped.
    """
    proxy = test['outcome'].to_numpy(dtype=float)
    if getattr(config, 'label_source', 'proxy') != 'venue':
        return proxy
    if 'venue_outcome' not in test.columns:
        logger.warning('label_source=venue but venue_outcome never joined; '
                       'grading on the proxy label')
        return proxy
    venue = pd.to_numeric(test['venue_outcome'], errors='coerce').to_numpy(dtype=float)
    return np.where(np.isfinite(venue), venue, proxy)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--compare-distributions', action='store_true',
                        help='Fit both a Gaussian and a Student-t barrier and report '
                             'both. Which tail calibrates better is a measurement.')
    args = parser.parse_args()
    setup_logging(args.verbose)
    config = config_from_args(args)
    print_header('Barrier baseline — the null hypothesis', config)

    dataset = load_dataset(args, config)
    folds = folds_for_config(dataset.window_index, config)
    groups = groups_from_args(args)

    distributions = (['normal', 'student_t'] if args.compare_distributions
                     else [config.baseline_distribution])
    results: dict[str, list[dict]] = {}

    for distribution in distributions:
        variant = config.with_overrides(baseline_distribution=distribution)
        rows = []
        pooled_y, pooled_p = [], []
        for fold in folds:
            assert_no_leakage(fold)
            fit, _ = fit_fold(dataset, fold.train, variant, groups=groups)
            test = apply_fold(dataset, fit, fold.test, variant, groups=groups)
            # **`--label-source venue` must be honoured HERE, or a strike
            # comparison grades each arm on its own label.** `outcome` is
            # `settle_price >= strike`, so an arm given the venue's strike is
            # scored against a label derived from that same strike, and the
            # two arms are not graded on the same question. The campaign
            # ledger records the identical trap: "grading on our Coinbase
            # label hands F(x/sigma) free skill, because the label and the
            # baseline share a source while the market prices on BRTI".
            y_all = _graded_label(test, config)
            p_all = test['baseline_probability'].to_numpy(dtype=float)
            # A row with no volatility estimate has no forecast. Left in, it
            # makes the fold's mean NaN.
            keep = np.isfinite(y_all) & np.isfinite(p_all)
            y, p = y_all[keep], p_all[keep]
            unscoreable = int((~keep).sum())
            pooled_y.append(y)
            pooled_p.append(p)
            rel = reliability(y, p)
            rows.append({
                'fold': fold.index,
                'test_start': fold.test_start, 'test_end': fold.test_end,
                'windows': effective_observations(test.loc[keep]),
                'n': int(keep.sum()),
                'log_loss': log_loss(y, p),
                'coin_flip': log_loss(y, np.full_like(p, float(y.mean()))),
                'ece': rel.expected_calibration_error,
                'max_dev': rel.max_deviation,
                'n_non_finite': unscoreable,
                'nu': fit.baseline.nu,
                'scale': ' '.join(f'{o}m={s:.3f}' for o, s in sorted(fit.baseline.scale.items())),
            })
            print(f'  fold {fold.index} [{fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d}] '
                  f'{rows[-1]["windows"]:>7,}w  log loss {rows[-1]["log_loss"]:.5f}  '
                  f'ECE {rows[-1]["ece"]:.5f}  max dev {rows[-1]["max_dev"]:.4f}  '
                  f'nu {fit.baseline.nu:.1f}')
        results[distribution] = rows
        y = np.concatenate(pooled_y)
        p = np.concatenate(pooled_p)
        rel = reliability(y, p)
        frame = pd.DataFrame(rows)
        print()
        print(f'  {distribution}: pooled log loss {log_loss(y, p):.5f} against a '
              f'{log_loss(y, np.full_like(p, float(y.mean()))):.5f} coin flip '
              f'({1 - log_loss(y, p) / log_loss(y, np.full_like(p, float(y.mean()))):.1%} '
              f'better, from arithmetic alone)')
        print(f'  worst-fold ECE {frame["ece"].max():.5f}, worst max deviation '
              f'{frame["max_dev"].max():.4f}, base rate {y.mean():.4f}')
        print('\n  out-of-sample reliability:')
        print(rel.table())
        print()

    if len(distributions) > 1:
        print('  distribution comparison (pooled out of sample):')
        for distribution, rows in results.items():
            frame = pd.DataFrame(rows)
            print(f'    {distribution:<10} log loss {frame["log_loss"].mean():.5f}  '
                  f'ECE {frame["ece"].max():.5f}')
        print('\n  Fat tails matter here because the system trades its confident\n'
              '  predictions: a Gaussian barrier assigns 0.999 where 0.99 is right,\n'
              '  and a model that merely knew that would look skilful.')

    print()
    # Refuse on unmeasured folds first: `Series.max()` skips NaN and
    # `nan > 0.02` is False, so the gate would otherwise fail open.
    all_ece = np.concatenate([
        pd.DataFrame(rows)['ece'].to_numpy(dtype=float) for rows in results.values()])
    all_non_finite = int(sum(
        int(pd.DataFrame(rows)['n_non_finite'].sum()) for rows in results.values()
        if 'n_non_finite' in pd.DataFrame(rows).columns))
    unmeasured = int((~np.isfinite(all_ece)).sum())
    if unmeasured:
        print(f'  GATE FAILED: {unmeasured} of {all_ece.size} folds could not be '
              f'measured at all (calibration came back NaN).')
        print('  That is a hole in the data, not a verdict on the baseline, and it '
              'is not the same answer as "no skill".')
        print('  Find the missing bars before reading anything below.')
        return 1
    total_rows = int(sum(int(pd.DataFrame(rows)['n'].sum()) for rows in results.values()
                         if 'n' in pd.DataFrame(rows).columns)) or all_ece.size
    share = all_non_finite / max(total_rows + all_non_finite, 1)
    if all_non_finite:
        # Reported, not fatal. A row with no volatility estimate has no
        # forecast, for example after a data outage. Live, `decide` refuses
        # those as NOT_FINITE.
        print(f'  {all_non_finite:,} row(s) ({share:.4%}) carried no forecast — a '
              f'NaN sigma, which is the tail of a data outage. Excluded from every '
              f'number above and counted here.')
        if share > 0.001:
            print(f'  GATE FAILED: {share:.3%} of rows are unscoreable, over the '
                  f'0.1% allowance.')
            print('  An outage costs a couple of hours. This much means a lookback '
                  'or an embargo is wrong, not the venue.')
            return 1

    worst_ece = float(np.max(all_ece))
    if worst_ece > 0.02:
        print(f'  GATE FAILED: worst-fold calibration error {worst_ece:.5f} exceeds 0.02.')
        print('  Skill measured against a miscalibrated baseline is partly the '
              'baseline\'s error.')
        print('  Fix this before training anything.')
        return 1
    print(f'  gate passed: worst-fold calibration error {worst_ece:.5f} <= 0.02 '
          f'across {all_ece.size} folds, none unmeasured')
    print('  next: python -m scripts.evaluate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
