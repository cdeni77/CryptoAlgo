"""STAGE 5 — H6 (entry-time selection) and H8 (selective abstention) on TIER 1.

These test the DECISION POLICY rather than the forecast, and nothing in this project
has tested them. They use the full `market_v2` verified sample (28,594 symbol-windows)
rather than the 15-day order-flow tier, so they carry ~7x the power.

**H8 is deliberately model-free.** Instead of asking "can a model beat the price",
it asks: *are there observable states in which buying at the venue's ask has positive
realised edge?* If the market is efficient the answer is no, everywhere. If some
state is systematically mispriced, it shows up without any forecast at all -- and
such a finding cannot be an artifact of a fitting procedure.

All inference resamples WINDOWS. Everything is computed out-of-sample by a
chronological split: states are RANKED on the development period and the ranking is
applied unchanged to validation.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage1_screen import log

OUT = HERE.parent
DEV_END = pd.Timestamp('2026-09-16', tz='UTC')


def boot_windows(x, w, B=10000, seed=0):
    g = pd.DataFrame({'x': x, 'w': w}).groupby('w')['x'].mean()
    v = g.to_numpy(); n = len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs,.025)), float(np.quantile(bs,.975)), n


def load():
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable & mv.offset_minutes.between(1, 14)].copy()
    # realised edge per contract of BUYING AT THE ASK, in probability points,
    # net of the venue's fee. This is what a trade actually pays.
    fee_y = np.ceil(0.07 * mv.yes_ask * (1 - mv.yes_ask) * 10_000) / 10_000
    mv['edge_yes_pp'] = (mv.y - mv.yes_ask - fee_y) * 100
    no_px = 1.0 - mv.yes_bid
    fee_n = np.ceil(0.07 * no_px * (1 - no_px) * 10_000) / 10_000
    mv['edge_no_pp'] = ((1 - mv.y) - no_px - fee_n) * 100
    mv['spread_pp'] = (mv.yes_ask - mv.yes_bid) * 100
    mv['dist_half'] = (mv.p_mkt - 0.5).abs()
    mv['split'] = np.where(mv.window_open < DEV_END, 'dev', 'val')
    return mv


if __name__ == '__main__':
    t0 = time.time(); res = {}
    m = load()
    print(f"TIER 1: {len(m):,} rows, {m.window_open.nunique():,} windows, "
          f"{m.groupby(['symbol','window_open']).ngroups:,} symbol-windows")
    print(f"  dev {(m.split=='dev').sum():,}  val {(m.split=='val').sum():,}")

    # ---------- H6: is any single ENTRY OFFSET systematically better? -----
    print("\n=== H6 entry-time selection: realised net edge of buying at the ask ===")
    print(f"{'offset':>7} {'n':>8} {'win':>6} {'YES edge pp':>13} {'CI':>24} "
          f"{'NO edge pp':>12} {'CI':>24}")
    h6 = []
    for off, g in m.groupby('offset_minutes'):
        ey, lo_y, hi_y, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
        en, lo_n, hi_n, _ = boot_windows(g.edge_no_pp.to_numpy(), g.window_open.to_numpy())
        h6.append({'offset': int(off), 'n': len(g), 'windows': nw,
                   'yes_edge_pp': ey, 'yes_ci': [lo_y, hi_y],
                   'no_edge_pp': en, 'no_ci': [lo_n, hi_n]})
        print(f"{off:>7} {len(g):>8,} {nw:>6,} {ey:>+13.4f} "
              f"[{lo_y:>+8.4f},{hi_y:>+8.4f}] {en:>+12.4f} [{lo_n:>+8.4f},{hi_n:>+8.4f}]")
    res['H6_by_offset'] = h6

    # ---------- H8: model-free state screen -------------------------------
    print("\n=== H8 selective abstention: states where buying at the ask has edge ===")
    m['price_band'] = pd.cut(m.p_mkt, [0,.05,.1,.2,.3,.5,.7,.8,.9,.95,1.01])
    m['spread_band'] = pd.cut(m.spread_pp, [-.01,1,2,3,5,100])
    m['age_band'] = pd.cut(m.quote_age_s, [-.01,2,3,4,1e9])
    states = {}
    for key, lbl in (('price_band','market price'), ('spread_band','quoted spread pp'),
                     ('symbol','symbol')):
        rows = []
        for k, g in m.groupby(key, observed=True):
            if len(g) < 1000: continue
            for side, col in (('yes','edge_yes_pp'), ('no','edge_no_pp')):
                e, lo, hi, nw = boot_windows(g[col].to_numpy(), g.window_open.to_numpy())
                rows.append({lbl: str(k), 'side': side, 'n': len(g), 'windows': nw,
                             'edge_pp': e, 'ci': [lo, hi], 'sig': bool(lo > 0)})
        states[lbl] = rows
        print(f"\n  --- by {lbl} ---")
        print(f"  {'state':>14} {'side':>4} {'n':>8} {'win':>6} {'net edge pp':>13} {'CI':>24} {'':>4}")
        for r in rows:
            print(f"  {r[lbl]:>14} {r['side']:>4} {r['n']:>8,} {r['windows']:>6,} "
                  f"{r['edge_pp']:>+13.4f} [{r['ci'][0]:>+8.4f},{r['ci'][1]:>+8.4f}]"
                  f" {'<<<' if r['sig'] else ''}")
    res['H8_states'] = states

    # ---------- H8 out-of-sample: rank on dev, apply to val ---------------
    print("\n=== H8 OUT OF SAMPLE: states ranked on DEV, applied unchanged to VAL ===")
    dev, val = m[m.split=='dev'], m[m.split=='val']
    cells = []
    for (pb, side), g in [((pb, s), m[(m.price_band==pb)]) for pb in m.price_band.cat.categories
                          for s in ('yes','no')]:
        pass
    rows = []
    for pb in m.price_band.cat.categories:
        for side, col in (('yes','edge_yes_pp'), ('no','edge_no_pp')):
            d = dev[dev.price_band == pb]
            v = val[val.price_band == pb]
            if len(d) < 500 or len(v) < 500: continue
            de, dlo, dhi, dnw = boot_windows(d[col].to_numpy(), d.window_open.to_numpy())
            ve, vlo, vhi, vnw = boot_windows(v[col].to_numpy(), v.window_open.to_numpy())
            rows.append({'price_band': str(pb), 'side': side,
                         'dev_edge_pp': de, 'dev_ci': [dlo, dhi], 'dev_n': len(d),
                         'val_edge_pp': ve, 'val_ci': [vlo, vhi], 'val_n': len(v),
                         'dev_positive': bool(dlo > 0)})
    sel = [r for r in rows if r['dev_positive']]
    print(f"  cells with a POSITIVE dev edge (CI excludes 0): {len(sel)} of {len(rows)}")
    print(f"  {'cell':>22} {'dev edge':>10} {'dev CI':>22} {'VAL edge':>10} {'VAL CI':>22}")
    for r in sel:
        print(f"  {r['price_band']+' '+r['side']:>22} {r['dev_edge_pp']:>+10.4f} "
              f"[{r['dev_ci'][0]:>+7.4f},{r['dev_ci'][1]:>+7.4f}] {r['val_edge_pp']:>+10.4f} "
              f"[{r['val_ci'][0]:>+7.4f},{r['val_ci'][1]:>+7.4f}]")
    if sel:
        held = [r for r in sel if r['val_ci'][0] > 0]
        print(f"\n  of {len(sel)} dev-positive cells, {len(held)} remain positive "
              f"(CI excludes 0) on VALIDATION")
    res['H8_oos'] = rows
    (OUT / 'stage5_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 5, 'hypotheses': ['H6','H8'], 'runtime_s': round(time.time()-t0,1),
         'n_rows': int(len(m)), 'n_windows': int(m.window_open.nunique())})
    print(f"\nstage 5 done in {time.time()-t0:.0f}s")
