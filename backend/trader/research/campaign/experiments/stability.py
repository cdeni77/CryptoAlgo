"""Stability and concentration analysis (RESEARCH_PROTOCOL.md sections 4 and 7).

Reads an OOF parquet written by the campaign and reports where the result lives.
The point is not to find the good segment -- it is to QUANTIFY concentration, so a
candidate that works in one coin, one week, one offset or one price band is visible
as such rather than reported as a general edge.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H


def segment_table(oof: pd.DataFrame, by, label: str) -> pd.DataFrame:
    rows = []
    for key, g in oof.groupby(by, observed=True):
        if len(g) < 200:
            continue
        mean, lo, hi, p, nw = H.model_minus_market(g.p_model, g.p_mkt, g.y, g.window_open)
        rows.append({label: str(key), 'n': len(g), 'windows': nw,
                     'mmm': mean, 'lo': lo, 'hi': hi, 'p': p,
                     'll_model': H.logloss(g.p_model, g.y),
                     'll_market': H.logloss(g.p_mkt, g.y)})
    return pd.DataFrame(rows)


def report(path: Path):
    oof = pd.read_parquet(path)
    oof['window_open'] = pd.to_datetime(oof.window_open, utc=True)
    name = path.stem.replace('oof_', '')
    print('=' * 84); print(f'STABILITY — {name}'); print('=' * 84)
    mean, lo, hi, p, nw = H.model_minus_market(oof.p_model, oof.p_mkt, oof.y, oof.window_open)
    print(f'overall  n={len(oof):,}  windows={nw:,}  model_minus_market {mean:+.6f} '
          f'CI [{lo:+.6f},{hi:+.6f}] p={p:.4f}\n')

    oof['week'] = oof.window_open.dt.to_period('W').astype(str)
    oof['price_band'] = pd.cut(oof.p_mkt, [0, .1, .3, .5, .7, .9, 1.01])
    oof['offset_band'] = pd.cut(oof.offset_minutes, [0, 3, 6, 9, 14])

    out = {}
    for by, label in [('symbol', 'symbol'), ('week', 'week'),
                      ('offset_minutes', 'offset'), ('price_band', 'price band')]:
        t = segment_table(oof, by, label)
        if not len(t): continue
        out[label] = t.to_dict('records')
        print(f'--- by {label} ---')
        print(f"{label:>14} {'n':>7} {'win':>6} {'mmm':>10} {'CI low':>10} {'CI high':>10} {'p':>7}")
        for r in t.itertuples():
            print(f'{getattr(r, label if label != "price band" else "_1"):>14} '
                  f'{r.n:>7,} {r.windows:>6,} {r.mmm:>+10.6f} {r.lo:>+10.6f} '
                  f'{r.hi:>+10.6f} {r.p:>7.4f}')
        print()

    # concentration: how much of the total improvement comes from the best week?
    oof['d'] = (H._ll(oof.p_mkt, oof.y) - H._ll(oof.p_model, oof.y))
    byw = oof.groupby('week')['d'].sum().sort_values(ascending=False)
    tot = byw.sum()
    if abs(tot) > 1e-12:
        print(f'concentration: best week is {byw.iloc[0]/tot*100:>6.1f}% of the total '
              f'improvement; top 2 weeks {byw.iloc[:2].sum()/tot*100:.1f}%')
        print(f'  weeks positive: {(byw > 0).sum()} of {len(byw)}')
    bys = oof.groupby('symbol')['d'].sum()
    print(f'  symbols positive: {(bys > 0).sum()} of {len(bys)}  '
          f'({dict(bys.round(3))})')
    return out


if __name__ == '__main__':
    paths = [Path(a) for a in sys.argv[1:]] or sorted(HERE.glob('oof_*.parquet'))
    allout = {}
    for p in paths:
        allout[p.stem] = report(p)
        print()
    (HERE / 'stability.json').write_text(json.dumps(allout, indent=2, default=str))
