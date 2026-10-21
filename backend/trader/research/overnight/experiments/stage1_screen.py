"""STAGE 1 — fast, model-free screening of order-flow signals (H1, H2, H3).

No model is trained here. For each candidate signal the question is simply:
**does it predict the MARKET RESIDUAL `y - p_mkt`?** If the Kalshi quote already
contains the information, that correlation is zero by construction. Anything that
survives goes to Stage 2; anything that does not is dropped without further compute.

Inference clusters on the WINDOW, never the row: four offsets share a settlement and
the three symbols are ~0.7 correlated inside a window.

Point-in-time: flow in minute m covers [m-1, m); the book is folded to <= m; the
decision is at m + 3s. Nothing here reads past the decision instant.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
FEAT = HERE.parent / 'features'
OUT = HERE.parent
LEDGER = OUT / 'EXPERIMENT_LEDGER.jsonl'


def log(entry: dict):
    entry['ts'] = pd.Timestamp.utcnow().isoformat()
    with LEDGER.open('a') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def load_flow() -> pd.DataFrame:
    files = sorted(FEAT.glob('flow_*.parquet'))
    if not files:
        raise SystemExit('no flow features built')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['window_open'] = pd.to_datetime(df.window_open, utc=True)
    return df.sort_values(['market_ticker', 'minute'])


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Signals. Every one uses only data at or before its own minute mark."""
    eps = 1e-9
    d = df.copy()
    tot = d.yes_total.fillna(0) + d.no_total.fillna(0)
    d['imbalance'] = (d.yes_total.fillna(0) - d.no_total.fillna(0)) / (tot + eps)
    d['touch_imbalance'] = ((d.yes_touch_size.fillna(0) - d.no_touch_size.fillna(0))
                            / (d.yes_touch_size.fillna(0) + d.no_touch_size.fillna(0) + eps))
    # regroup AFTER the derived columns exist, so the shifts see them
    g = d.groupby('market_ticker', sort=False)
    # H2: the CHANGE in imbalance, not the level
    for k in (1, 3, 5):
        d[f'd_imbalance_{k}'] = d.imbalance - g.imbalance.shift(k)
        d[f'd_touch_imb_{k}'] = d.touch_imbalance - g.touch_imbalance.shift(k)
    # H1: depletion at the touch, net of replenishment (removals are AMBIGUOUS)
    ya, yr = d.yes_added.fillna(0), d.yes_removed_ambig.fillna(0)
    na, nr = d.no_added.fillna(0), d.no_removed_ambig.fillna(0)
    d['yes_depletion'] = yr / (ya + yr + eps)
    d['no_depletion'] = nr / (na + nr + eps)
    d['depletion_asym'] = d.yes_depletion - d.no_depletion
    d['net_flow_yes'] = (ya - yr) / (ya + yr + eps)
    d['net_flow_no'] = (na - nr) / (na + nr + eps)
    d['flow_asym'] = d.net_flow_yes - d.net_flow_no
    # H3: liquidity withdrawal
    d['total_size'] = tot
    g = d.groupby('market_ticker', sort=False)
    d['d_total_size_1'] = np.log1p(tot) - np.log1p(g.total_size.shift(1).fillna(0))
    d['d_total_size_3'] = np.log1p(tot) - np.log1p(g.total_size.shift(3).fillna(0))
    d['touch_size_tot'] = d.yes_touch_size.fillna(0) + d.no_touch_size.fillna(0)
    g = d.groupby('market_ticker', sort=False)
    d['d_touch_size_1'] = (np.log1p(d.touch_size_tot)
                           - np.log1p(g.touch_size_tot.shift(1).fillna(0)))
    d['event_rate'] = (d.n_adds.fillna(0) + d.n_removes_ambig.fillna(0))
    d['log_event_rate'] = np.log1p(d.event_rate)
    g = d.groupby('market_ticker', sort=False)
    d['d_event_rate_1'] = d.log_event_rate - g.log_event_rate.shift(1)
    d['levels_tot'] = d.yes_levels.fillna(0) + d.no_levels.fillna(0)
    g = d.groupby('market_ticker', sort=False)
    d['d_levels_1'] = d.levels_tot - g.levels_tot.shift(1)
    # book-implied mid, for a quote-velocity signal
    d['book_mid'] = (d.yes_best.fillna(np.nan) + (1 - d.no_best.fillna(np.nan))) / 2
    g = d.groupby('market_ticker', sort=False)
    for k in (1, 3):
        d[f'd_book_mid_{k}'] = d.book_mid - g.book_mid.shift(k)
    return d


SIGNALS = ['imbalance', 'touch_imbalance',
           'd_imbalance_1', 'd_imbalance_3', 'd_imbalance_5',
           'd_touch_imb_1', 'd_touch_imb_3', 'd_touch_imb_5',
           'yes_depletion', 'no_depletion', 'depletion_asym',
           'net_flow_yes', 'net_flow_no', 'flow_asym',
           'd_total_size_1', 'd_total_size_3', 'd_touch_size_1',
           'log_event_rate', 'd_event_rate_1', 'd_levels_1',
           'd_book_mid_1', 'd_book_mid_3']


def window_boot(x, resid, windows, B=5000, seed=0):
    """Correlation of a signal with the market residual, resampled on WINDOWS."""
    ok = np.isfinite(x) & np.isfinite(resid)
    if ok.sum() < 500:
        return np.nan, np.nan, np.nan, 0
    x, resid, w = x[ok], resid[ok], np.asarray(windows)[ok]
    df = pd.DataFrame({'x': x, 'r': resid, 'w': w})
    # standardise so the statistic is comparable across signals
    df['x'] = (df.x - df.x.mean()) / (df.x.std(ddof=0) + 1e-12)
    per = df.groupby('w').apply(lambda g: (g.x * g.r).mean(), include_groups=False)
    v = per.to_numpy(); n = len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), n


def main():
    t0 = time.time()
    flow = derive(load_flow())
    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable]
    m = flow.merge(mv, left_on=['symbol', 'window_open', 'minute'],
                   right_on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    m = m[m.minute.between(1, 14)].copy()
    m['resid'] = m.y - m.p_mkt
    print(f'STAGE 1 screen: {len(m):,} rows, {m.window_open.nunique():,} windows, '
          f'{m.groupby(["symbol","window_open"]).ngroups:,} symbol-windows')
    print(f'  span {m.window_open.min()} -> {m.window_open.max()}')
    print(f'  mean market residual {m.resid.mean():+.5f} (should be ~0 if calibrated)')

    rows = []
    print(f"\n{'signal':>20} {'n':>8} {'win':>6} {'corr*resid':>12} {'CI low':>11} "
          f"{'CI high':>11} {'verdict':>10}")
    for s in SIGNALS:
        if s not in m.columns: continue
        mean, lo, hi, nw = window_boot(m[s].to_numpy(float), m.resid.to_numpy(float),
                                       m.window_open.to_numpy())
        if not np.isfinite(mean): continue
        sig = (lo > 0) or (hi < 0)
        rows.append({'signal': s, 'stat': mean, 'ci_lo': lo, 'ci_hi': hi,
                     'windows': nw, 'significant': bool(sig)})
        print(f"{s:>20} {int(m[s].notna().sum()):>8,} {nw:>6,} {mean:>+12.6f} "
              f"{lo:>+11.6f} {hi:>+11.6f} {'SIGNAL' if sig else 'null':>10}")

    res = pd.DataFrame(rows).sort_values('stat', key=abs, ascending=False)
    res.to_csv(OUT / 'stage1_screen.csv', index=False)
    hits = res[res.significant]
    print(f"\nsignals tested {len(res)}   significant {len(hits)}   "
          f"(Bonferroni alpha {0.05/max(len(res),1):.4f})")
    # multiplicity: a 95% CI on 22 signals gives ~1 false positive by chance
    print("\nTop by |stat| (multiplicity NOT yet corrected):")
    print(res.head(8).to_string(index=False))
    log({'stage': 1, 'hypotheses': ['H1', 'H2', 'H3'],
         'rows': int(len(m)), 'windows': int(m.window_open.nunique()),
         'symbol_windows': int(m.groupby(['symbol', 'window_open']).ngroups),
         'span': [str(m.window_open.min()), str(m.window_open.max())],
         'signals_tested': int(len(res)), 'significant_uncorrected': int(len(hits)),
         'top': res.head(10).to_dict('records'), 'runtime_s': round(time.time()-t0, 1)})
    print(f'\nledger appended. {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
