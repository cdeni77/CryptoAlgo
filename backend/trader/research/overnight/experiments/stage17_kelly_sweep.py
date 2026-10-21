"""STAGE 17 — Kelly sweep. Vectorised, because the row-loop took >10 minutes a cell.

Kelly is NOT only a sizing dial here. `decide()` floors the stake to whole
contracts, so a smaller fraction pushes marginal trades under one contract and
REFUSES them -- the `below_min_contracts` path. CLAUDE.md measured exactly this:
0.25 -> 0.10 left `edge_below_gate` identical while `below_min_contracts` went
1,813 -> 8,218, and realised edge per contract rose +0.99pp -> +3.32pp, "because the
survivors were the higher-edge trades rather than because the sizing was safer".

So the sweep reports contracts AND trades AND edge per contract, not just dollars:
a fraction that improves pp/contract while collapsing trade count has not found an
edge, it has just become more selective.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent


def sweep(path, label, gates=(0.5, 1.0, 1.5, 2.5),
          kellys=(None, 0.05, 0.10, 0.25, 0.50, 1.00), bankroll=500.0):
    s = pd.read_parquet(path)
    s['window_open'] = pd.to_datetime(s.window_open, utc=True)
    s = s[np.isfinite(s.p_model)].copy()
    # --- vectorised side / price / edge ---
    yes_px = s.yes_ask.to_numpy(float)
    no_px = 1.0 - s.yes_bid.to_numpy(float)
    pm = s.p_model.to_numpy(float)
    ev_y, ev_n = pm - yes_px, (1.0 - pm) - no_px
    take_yes = ev_y >= ev_n
    s['px'] = np.where(take_yes, yes_px, no_px)
    s['ev'] = np.where(take_yes, ev_y, ev_n)
    s['q'] = np.where(take_yes, pm, 1.0 - pm)
    s['won'] = np.where(take_yes, s.y.to_numpy() == 1, s.y.to_numpy() == 0)
    s['vol'] = np.where(take_yes, s.ask_at_touch.to_numpy(float),
                        s.bid_at_touch.to_numpy(float))
    s = s[(s.px > 0) & (s.px < 1) & np.isfinite(s.vol)]
    s['fee1'] = np.ceil(0.07 * s.px * (1 - s.px) * 10_000) / 10_000
    s = s.sort_values(['symbol', 'window_open', 'offset_minutes'])

    rows = []
    for gate in gates:
        elig = s[s.ev - s.fee1 >= gate / 100.0]
        if not len(elig):
            continue
        for kf in kellys:
            e = elig.copy()
            if kf is None:
                e['n'] = 1
            else:
                fstar = np.maximum(0.0, (e.q - e.px) / (1.0 - e.px))
                e['n'] = np.floor(kf * fstar * bankroll / e.px)
            e['n'] = np.minimum(e.n, e.vol)
            e = e[e.n >= 1]
            if not len(e):
                rows.append({'gate': gate, 'kelly': kf, 'trades': 0}); continue
            # ONE entry per (symbol, window): first qualifying offset
            e = e.groupby(['symbol', 'window_open'], as_index=False).first()
            fee = np.ceil(0.07 * e.n * e.px * (1 - e.px) * 10_000) / 10_000
            pnl = np.where(e.won, e.n, 0.0) - e.px * e.n - fee
            gw = pd.Series(pnl).groupby(e.window_open.to_numpy()).sum()
            ts = gw.mean() / (gw.std(ddof=1) / np.sqrt(len(gw))) if len(gw) > 1 else np.nan
            folds = sorted(s._fold.unique()) if '_fold' in s.columns else \
                    sorted(s.foldid.unique())
            fc = e['_fold'] if '_fold' in e.columns else e['foldid']
            silent = sum(1 for f in folds if not (fc == f).any())
            last5 = folds[-5:]
            traded5 = sum(1 for f in last5 if (fc == f).any())
            rows.append({'gate': gate, 'kelly': kf, 'trades': int(len(e)),
                         'contracts': int(e.n.sum()), 'pnl': float(pnl.sum()),
                         'pp_per_contract': float(pnl.sum() / e.n.sum() * 100),
                         't_stat': float(ts), 'silent': silent,
                         'traded_last5': traded5,
                         'max_dd': float((gw.cumsum().cummax() - gw.cumsum()).max())})
    df = pd.DataFrame(rows)
    print(f"\n{'='*96}\n{label}\n{'='*96}")
    print(f"  {'gate':>5} {'kelly':>7} {'trades':>7} {'contracts':>10} {'pnl$':>10} "
          f"{'pp/ct':>8} {'t':>6} {'last5':>6} {'maxDD':>8}")
    for r in df.itertuples():
        if not r.trades:
            print(f"  {r.gate:>5.1f} {str(r.kelly):>7} {'none':>7}"); continue
        print(f"  {r.gate:>5.1f} {str(r.kelly):>7} {r.trades:>7,} {r.contracts:>10,} "
              f"{r.pnl:>+10.2f} {r.pp_per_contract:>+8.3f} {r.t_stat:>+6.2f} "
              f"{r.traded_last5:>5}/5 {r.max_dd:>8.2f}")
    return df


if __name__ == '__main__':
    out = {}
    for path, label in (('research/overnight/oof_stage16_xgb.parquet',
                         'XGBOOST  — July->today, all features, clean (no lookahead)'),
                        ('research/overnight/oof_stage15_nn.parquet',
                         'NEURAL NET — July->today, all features, clean (no lookahead)')):
        p = Path(path)
        if not p.exists():
            print(f'missing {path}'); continue
        out[label] = sweep(p, label).to_dict('records')
    (OUT / 'stage17_kelly_sweep.json').write_text(json.dumps(out, indent=2, default=str))
    print("\n  'kelly None' = flat 1 contract, the size-independent reference.")
    print("  A row where pp/ct RISES while trades COLLAPSE is selectivity, not edge.")
