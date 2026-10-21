"""Assemble the campaign's final deliverables from the recorded experiment JSONs.

Produces:
  1. the model leaderboard, ranked by Holm-corrected lower bound of model_minus_market
  2. overfitting diagnostics (train vs OOF, seed spread, best_iteration distribution)
  3. the trading-edge assessment, with realised and simulated P&L kept separate
Every candidate appears, including failures.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

def holm(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]; m = len(order); prev = 0.0
    for i, j in enumerate(order):
        prev = min(1.0, max(prev, (m - i) * p[j])); out[j] = prev
    return out

def collect():
    rows = []
    for f in sorted(HERE.glob('*.json')):
        if f.name in ('leaderboard.json', 'stability.json', 'plumbing_persistence.json'):
            continue
        try: blob = json.loads(f.read_text())
        except Exception: continue
        res = blob.get('results')
        if not isinstance(res, dict): continue
        for name, r in res.items():
            if not isinstance(r, dict): continue
            money = (r.get('money') or {})
            g = money.get('gate_1.5') or money.get('gate_1.0') or {}
            rows.append({
                'candidate': name, 'source': f.name,
                'status': r.get('status', 'OK' if 'model_minus_market' in r else 'FAILED'),
                'reason': r.get('reason', ''),
                'family': r.get('family') or r.get('fitter'),
                'n_oof': r.get('n_oof') or r.get('n'), 'windows': r.get('windows'),
                'n_features': r.get('n_features'),
                'll_model': r.get('log_loss_model') or r.get('log_loss'),
                'll_market': r.get('log_loss_market'),
                'mmm': r.get('model_minus_market'),
                'ci_lo': (r.get('ci95') or [None, None])[0],
                'ci_hi': (r.get('ci95') or [None, None])[1],
                'p_one': r.get('p_one_sided'),
                'calib': (r.get('calibration') or {}).get('error'),
                'calib_mkt': (r.get('market_calibration') or {}).get('error'),
                'trades': g.get('trades'), 'pnl': g.get('pnl'),
                'pp_ct': g.get('pp_per_contract'), 'pnl_t': g.get('t_stat'),
                'pnl_lo': (g.get('pnl_ci95') or [None, None])[0],
                'pnl_hi': (g.get('pnl_ci95') or [None, None])[1],
                'max_dd': g.get('max_drawdown'), 'eligible': g.get('eligible_windows'),
                'train_ll': r.get('train_log_loss_by_fold'),
                'best_iters': r.get('best_iterations'),
                'seed_spread': (r.get('seed_sensitivity') or {}).get('spread'),
            })
    return pd.DataFrame(rows)

def main():
    df = collect()
    if not len(df):
        print('no experiment results recorded'); return
    ok = df[df.status == 'OK'].copy()
    if len(ok):
        ok['p_holm'] = holm(ok.p_one.to_numpy())
        ok['supported'] = (ok.ci_lo > 0) & (ok.p_holm < 0.05)
        ok = ok.sort_values(['supported', 'ci_lo'], ascending=[False, False])
    print('=' * 124)
    print('MODEL LEADERBOARD — ranked by the Holm-corrected lower bound of model_minus_market')
    print('(a candidate whose CI includes zero is NOT ranked above one whose does not, '
          'however much its backtest made)')
    print('=' * 124)
    print(f"{'candidate':<30}{'fam':<10}{'n_oof':>9}{'win':>6}{'ll_model':>10}{'ll_mkt':>9}"
          f"{'mmm':>11}{'CI low':>11}{'p_holm':>8}{'trades':>7}{'pnl$':>9}{'pp/ct':>8}{'sup':>5}")
    print('-' * 124)
    for r in ok.itertuples():
        f = lambda v, d='{:>11.6f}': (d.format(v) if v is not None and np.isfinite(v) else ' ' * 11)
        print(f"{r.candidate:<30}{str(r.family or '-'):<10}{(r.n_oof or 0):>9,}{(r.windows or 0):>6,}"
              f"{(r.ll_model if r.ll_model else float('nan')):>10.6f}"
              f"{(r.ll_market if r.ll_market else float('nan')):>9.6f}"
              f"{f(r.mmm)}{f(r.ci_lo)}{(r.p_holm if np.isfinite(r.p_holm) else 1):>8.4f}"
              f"{(r.trades or 0):>7,}{(r.pnl if r.pnl is not None else 0):>9.2f}"
              f"{(r.pp_ct if r.pp_ct is not None and np.isfinite(r.pp_ct) else float('nan')):>8.3f}"
              f"{'YES' if r.supported else 'no':>5}")
    bad = df[df.status != 'OK']
    if len(bad):
        print('\nFAILED / NOT RUN:')
        for r in bad.itertuples(): print(f"  {r.candidate:<34} {r.reason}")

    print('\n' + '=' * 90); print('OVERFITTING DIAGNOSTICS'); print('=' * 90)
    for r in ok.itertuples():
        if not r.train_ll: continue
        tl = np.mean([x for x in r.train_ll if x is not None])
        print(f"  {r.candidate:<30} train {tl:.6f}  OOF {r.ll_model:.6f}  "
              f"gap {r.ll_model - tl:+.6f}"
              + (f"  best_iters {r.best_iters}" if r.best_iters else '')
              + (f"  seed spread {r.seed_spread:.6f}" if r.seed_spread else ''))

    print('\n' + '=' * 90); print('TRADING-EDGE ASSESSMENT (simulated fills only)'); print('=' * 90)
    print(f"{'candidate':<30}{'trades':>8}{'pnl$':>10}{'pp/ct':>9}{'t':>7}{'CI low':>10}{'CI high':>10}{'maxDD':>9}")
    for r in ok.itertuples():
        if not r.trades: continue
        print(f"{r.candidate:<30}{r.trades:>8,}{r.pnl:>10.2f}{r.pp_ct:>9.3f}"
              f"{(r.pnl_t if r.pnl_t is not None else float('nan')):>7.2f}"
              f"{(r.pnl_lo if r.pnl_lo is not None else float('nan')):>10.1f}"
              f"{(r.pnl_hi if r.pnl_hi is not None else float('nan')):>10.1f}"
              f"{(r.max_dd if r.max_dd is not None else float('nan')):>9.2f}")
    print("\nNOTE: all P&L above is SIMULATED on modelled fills. The only REALISED P&L in"
          "\nthis project is the venue ledger's +$17.53 over 6,692 contracts, which D1 showed"
          "\nis indistinguishable from zero (t=0.171, CI [-$181, +$220]). The two are never summed.")
    df.to_csv(HERE / 'leaderboard.csv', index=False)
    (HERE / 'leaderboard.json').write_text(json.dumps(
        {'ranked': ok.to_dict('records') if len(ok) else [],
         'all': df.to_dict('records')}, indent=2, default=str))
    print(f"\ncandidates: {len(df)}   supported after Holm: "
          f"{int(ok.supported.sum()) if len(ok) else 0}")
    print('wrote leaderboard.csv / leaderboard.json')

if __name__ == '__main__':
    main()
