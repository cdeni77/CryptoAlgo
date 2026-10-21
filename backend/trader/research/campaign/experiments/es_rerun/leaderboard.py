"""Build the model leaderboard from every recorded experiment.

Ranked by EVIDENTIAL STRENGTH, not return (RESEARCH_PROTOCOL.md section 9). The
ranking key is the lower bound of the model_minus_market confidence interval after a
Holm correction across every candidate actually run -- a candidate whose CI includes
zero is not ranked above one whose does not, however much money its backtest made.

Every candidate is listed, including the failures. A leaderboard that lists only the
winners re-creates the selection bias the protocol exists to prevent.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent


def collect():
    rows = []
    for p in sorted(HERE.glob('*.json')):
        if p.name in ('stability.json',): continue
        try: blob = json.loads(p.read_text())
        except Exception: continue
        res = blob.get('results', {})
        if not isinstance(res, dict): continue
        for name, r in res.items():
            if not isinstance(r, dict): continue
            if r.get('status') != 'OK' and 'model_minus_market' not in r:
                rows.append({'candidate': name, 'status': r.get('status', 'FAILED'),
                             'reason': r.get('reason', ''), 'source': p.name})
                continue
            money = r.get('money', {}) or {}
            g15 = money.get('gate_1.5', {}) or {}
            rows.append({
                'candidate': name, 'status': 'OK', 'source': p.name,
                'n_oof': r.get('n_oof'), 'windows': r.get('windows'),
                'n_features': r.get('n_features'),
                'trailing_days': r.get('trailing_days'),
                'll_model': r.get('log_loss_model'), 'll_market': r.get('log_loss_market'),
                'mmm': r.get('model_minus_market'),
                'ci_lo': (r.get('ci95') or [None, None])[0],
                'ci_hi': (r.get('ci95') or [None, None])[1],
                'p_one_sided': r.get('p_one_sided'),
                'calib': (r.get('calibration') or {}).get('error'),
                'calib_market': (r.get('market_calibration') or {}).get('error'),
                'trades': g15.get('trades'), 'pnl': g15.get('pnl'),
                'pp_per_ct': g15.get('pp_per_contract'), 'pnl_t': g15.get('t_stat'),
                'max_dd': g15.get('max_drawdown'),
                'eligible': g15.get('eligible_windows'),
            })
    return pd.DataFrame(rows)


def holm(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 0.0
    for i, j in enumerate(order):
        adj = min(1.0, max(prev, (m - i) * p[j]))
        out[j] = adj; prev = adj
    return out


def main():
    df = collect()
    if not len(df):
        print('no experiments recorded yet'); return
    ok = df[df.status == 'OK'].copy()
    if len(ok):
        ok['p_holm'] = holm(ok.p_one_sided.to_numpy())
        ok['supported'] = (ok.ci_lo > 0) & (ok.p_holm < 0.05)
        ok = ok.sort_values(['supported', 'ci_lo'], ascending=[False, False])

    print('=' * 120)
    print('MODEL LEADERBOARD — ranked by the LOWER BOUND of model_minus_market, Holm-corrected')
    print('=' * 120)
    hdr = (f"{'candidate':<30}{'n_oof':>8}{'win':>6}{'mmm':>11}{'CI low':>11}"
           f"{'CI high':>11}{'p_holm':>8}{'calib':>8}{'trades':>7}{'pnl$':>9}{'pp/ct':>8}{'sup':>5}")
    print(hdr); print('-' * 120)
    for r in ok.itertuples():
        print(f"{r.candidate:<30}{(r.n_oof or 0):>8,}{(r.windows or 0):>6,}"
              f"{(r.mmm if r.mmm is not None else np.nan):>+11.6f}"
              f"{(r.ci_lo if r.ci_lo is not None else np.nan):>+11.6f}"
              f"{(r.ci_hi if r.ci_hi is not None else np.nan):>+11.6f}"
              f"{r.p_holm:>8.4f}{(r.calib or np.nan):>8.5f}"
              f"{(r.trades or 0):>7,}{(r.pnl or 0):>9.2f}"
              f"{(r.pp_per_ct if r.pp_per_ct is not None else np.nan):>8.3f}"
              f"{'YES' if r.supported else 'no':>5}")
    fail = df[df.status != 'OK']
    if len(fail):
        print('\nFAILED / not run:')
        for r in fail.itertuples():
            print(f"  {r.candidate:<34} {r.reason}")
    print(f"\ncandidates run: {len(df)}   supported after Holm: "
          f"{int(ok.supported.sum()) if len(ok) else 0}")
    (HERE / 'leaderboard.json').write_text(
        json.dumps({'candidates': df.to_dict('records'),
                    'ranked': ok.to_dict('records') if len(ok) else []},
                   indent=2, default=str))
    df.to_csv(HERE / 'leaderboard.csv', index=False)
    print('wrote leaderboard.json / leaderboard.csv')


if __name__ == '__main__':
    main()
