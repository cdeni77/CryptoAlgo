"""STAGE 5b — the tight-spread YES finding, tested OUT OF SAMPLE.

H8 found: when the quoted spread is <= 1pp, buying YES at the ask realises
+4.6251pp net of fees, CI [+3.98, +5.28], over 50,753 rows / 8,835 windows -- while
NO on the same rows realises -7.5394pp. The two sum to -2.91pp, which is about the
round-trip cost, so the asymmetry is a DIRECTIONAL claim: in tight-spread markets
the outcome favoured YES more than the quote implied.

That is either a real, exploitable bias or a 3.5-month artifact. The test is
chronological: measure on DEV (2026-07-01..09-15), then apply the SAME rule
unchanged to VAL (2026-09-16..10-14). No tuning between the two.

Also checked here, because each would explain the finding away:
  * is it a price-level effect in disguise (tight spreads cluster at extremes)?
  * does it survive conditioning on the price band?
  * is it stable month by month?
  * does it survive a depth constraint (could you actually have been filled)?
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage1_screen import log
from stage5_policy import load, boot_windows

OUT = HERE.parent
DEV_END = pd.Timestamp('2026-09-16', tz='UTC')

if __name__ == '__main__':
    t0 = time.time(); res = {}
    m = load()
    m['tight'] = m.spread_pp <= 1.0
    m['price_band'] = pd.cut(m.p_mkt, [0,.05,.1,.2,.3,.5,.7,.8,.9,.95,1.01])
    # **The tail-quantisation hazard, stated before the result is read.** Kalshi's
    # tick is a TENTH of a cent below 0.10 and above 0.90 and a full cent between,
    # so a spread <= 1pp is nearly automatic in the tails and informative only in
    # the middle. And CLAUDE.md records that the Predexon backfill cannot represent
    # a tail price to better than a cent: 91.2% of quotes above 90c sit off the
    # true grid with a mean rounding error of 0.250c. A rounded-DOWN ask would
    # manufacture exactly this edge. Hence the [0.10, 0.90] restriction below.
    m['mid_band'] = m.p_mkt.between(0.10, 0.90)
    print(f"rows {len(m):,}  tight-spread rows {int(m.tight.sum()):,} "
          f"({m.tight.mean()*100:.1f}%)")

    # ---------- 1. the headline, split chronologically -------------------
    print("\n=== 1. tight spread (<=1pp), YES at the ask — DEV then VAL ===")
    rows = []
    for name, g in (('DEV', m[(m.split=='dev') & m.tight]),
                    ('VAL', m[(m.split=='val') & m.tight])):
        e, lo, hi, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
        en, lon, hin, _ = boot_windows(g.edge_no_pp.to_numpy(), g.window_open.to_numpy())
        rows.append({'period': name, 'n': len(g), 'windows': nw,
                     'yes_edge_pp': e, 'yes_ci': [lo, hi],
                     'no_edge_pp': en, 'no_ci': [lon, hin]})
        print(f"  {name}: n={len(g):>7,} w={nw:>5,}  YES {e:>+8.4f}pp "
              f"[{lo:>+7.4f},{hi:>+7.4f}]   NO {en:>+8.4f}pp [{lon:>+7.4f},{hin:>+7.4f}]")
    res['headline'] = rows
    held = rows[1]['yes_ci'][0] > 0
    print(f"  -> the DEV finding {'HOLDS' if held else 'DOES NOT HOLD'} on validation")

    # ---------- 2. is it just the price level? ---------------------------
    print("\n=== 2. conditioned on price band (tight spreads cluster at extremes) ===")
    print(f"  {'price band':>14} {'n tight':>9} {'YES edge':>10} {'CI':>22} "
          f"{'n wide':>9} {'YES edge':>10} {'CI':>22}")
    cond = []
    for pb, g in m.groupby('price_band', observed=True):
        gt, gw = g[g.tight], g[~g.tight]
        if len(gt) < 500 or len(gw) < 500: continue
        et, lot, hit, _ = boot_windows(gt.edge_yes_pp.to_numpy(), gt.window_open.to_numpy())
        ew, low, hiw, _ = boot_windows(gw.edge_yes_pp.to_numpy(), gw.window_open.to_numpy())
        cond.append({'price_band': str(pb), 'tight_n': len(gt), 'tight_edge': et,
                     'tight_ci': [lot, hit], 'wide_n': len(gw), 'wide_edge': ew,
                     'wide_ci': [low, hiw]})
        print(f"  {str(pb):>14} {len(gt):>9,} {et:>+10.4f} [{lot:>+7.4f},{hit:>+7.4f}] "
              f"{len(gw):>9,} {ew:>+10.4f} [{low:>+7.4f},{hiw:>+7.4f}]")
    res['by_price_band'] = cond

    # ---------- 2b. the decisive restriction: [0.10, 0.90] only ----------
    print("\n=== 2b. restricted to [0.10, 0.90] where the tick is a full cent ===")
    print("    (outside this band the backfill cannot resolve the price to better")
    print("     than a cent, and a rounded-down ask manufactures edge)")
    sub = []
    for name, g in (('ALL  tight, mid-band', m[m.tight & m.mid_band]),
                    ('DEV  tight, mid-band', m[m.tight & m.mid_band & (m.split=='dev')]),
                    ('VAL  tight, mid-band', m[m.tight & m.mid_band & (m.split=='val')]),
                    ('ALL  tight, TAILS   ', m[m.tight & ~m.mid_band])):
        if len(g) < 200:
            print(f"  {name}: n={len(g):,} too few"); continue
        e, lo, hi, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
        sub.append({'cell': name.strip(), 'n': len(g), 'windows': nw,
                    'edge_pp': e, 'ci': [lo, hi]})
        print(f"  {name}: n={len(g):>7,} w={nw:>5,} {e:>+8.4f}pp "
              f"[{lo:>+7.4f},{hi:>+7.4f}]{'  <<<' if lo > 0 else ''}")
    res['mid_band_restriction'] = sub

    # ---------- 3. month by month ----------------------------------------
    print("\n=== 3. month by month, tight spread YES ===")
    m['month'] = m.window_open.dt.to_period('M').astype(str)
    mon = []
    for mo, g in m[m.tight].groupby('month'):
        if len(g) < 300: continue
        e, lo, hi, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
        mon.append({'month': mo, 'n': len(g), 'windows': nw, 'edge_pp': e, 'ci': [lo, hi]})
        print(f"  {mo}: n={len(g):>7,} w={nw:>5,}  {e:>+8.4f}pp [{lo:>+7.4f},{hi:>+7.4f}]"
              f"{'  <<<' if lo > 0 else ''}")
    res['by_month'] = mon

    # ---------- 4. depth-constrained: could it have been filled? ---------
    print("\n=== 4. with a depth constraint at the touch (>=1 contract, >=10, >=50) ===")
    dep = []
    for req in (1, 10, 50):
        g = m[m.tight & (m.ask_at_touch >= req)]
        if len(g) < 300: continue
        e, lo, hi, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
        dep.append({'min_depth': req, 'n': len(g), 'windows': nw,
                    'edge_pp': e, 'ci': [lo, hi]})
        print(f"  ask depth >= {req:>3}: n={len(g):>7,} w={nw:>5,} {e:>+8.4f}pp "
              f"[{lo:>+7.4f},{hi:>+7.4f}]")
    res['by_depth'] = dep

    # ---------- 5. by symbol and offset ----------------------------------
    print("\n=== 5. tight-spread YES by symbol and by offset ===")
    for by in ('symbol', 'offset_minutes'):
        print(f"  --- {by} ---")
        rr = []
        for k, g in m[m.tight].groupby(by):
            if len(g) < 300: continue
            e, lo, hi, nw = boot_windows(g.edge_yes_pp.to_numpy(), g.window_open.to_numpy())
            rr.append({str(by): str(k), 'n': len(g), 'edge_pp': e, 'ci': [lo, hi]})
            print(f"    {str(k):>8} n={len(g):>7,} {e:>+8.4f}pp [{lo:>+7.4f},{hi:>+7.4f}]"
                  f"{'  <<<' if lo > 0 else ''}")
        res[f'by_{by}'] = rr

    (OUT / 'stage5b_spread.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': '5b', 'hypothesis': 'H8 tight-spread YES bias',
         'dev_val_holds': bool(held), 'runtime_s': round(time.time()-t0, 1)})
    print(f"\nstage 5b done in {time.time()-t0:.0f}s")
