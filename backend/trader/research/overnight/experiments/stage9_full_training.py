"""STAGE 9 — ONE model, EVERY feature source, 2026-07-01 to today.

Rationale (the user's, and it is right): every experiment so far isolated a single
family -- bars, or book, or implied vol, or cross-venue, or order flow -- and each
came back near zero. Isolation is the wrong test if the edge is an interaction. This
trains one model on all of them together over the full verified period.

Sources combined:
  bar features      geometry, vol_state, microstructure, cross_asset, offset,
                    time_of_day (the control, kept so its share can be read)
  venue book        market_state: spread, depth buckets, touch sizes
  order flow        imbalance, d_imbalance_1/3, d_touch_imb_3, d_book_mid_1/3
                    (snapshot-derived; Stage 8 showed these track the full-ladder
                     versions at r=0.985, so the coarser source is not the issue)
  implied vol       the KXBTCD ladder inverted to forward sigma
  cross venue       Polymarket's quote on the same window

**No `--complete-cases`.** That filter is what collapsed the incumbent to 3 folds:
demanding book AND peer venue AND ladder fit AND settlement on every row keeps only
the intersection. LightGBM handles NaN natively, so every row contributes what it
has. The honest cost is that the fit is partly a blend across availability regimes,
so `population` below reports what fraction of rows actually carries each feature --
read it before reading the result.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'campaign'))
sys.path.insert(0, str(HERE.parents[1] / 'campaign' / 'experiments'))
sys.path.insert(0, str(HERE.parents[2]))
from stage1_screen import log
from stage7_backfilled_flow import build_snapshot_flow, _boot_ci

OUT = HERE.parent
STORE = HERE.parents[2] / 'data' / 'research'
LO = pd.Timestamp('2026-01-01', tz='UTC')
HI = pd.Timestamp('2026-10-19', tz='UTC')

FLOW = ['imbalance', 'touch_imbalance', 'd_imbalance_1', 'd_imbalance_3',
        'd_touch_imb_1', 'd_touch_imb_3', 'd_book_mid_1', 'd_book_mid_3']


def _sig(x): return 1/(1+np.exp(-x))
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); return np.log(p/(1-p))
def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6); y = np.asarray(y, float)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def bar_features() -> pd.DataFrame:
    """Bar-derived features, built INSIDE folds by core.dataset, then pooled.

    Fitting the vol model / seasonality / baseline per fold is what keeps this
    honest; here the per-fold tables are concatenated after the fact so one
    feature matrix exists, and every row carries only its own fold's fit.
    """
    from core.config import Config
    from core.datastore import ResearchStore
    from core.dataset import Dataset, load_minute_bars, fit_fold, apply_fold
    cfg = Config(decision_offsets=tuple(range(1, 15)),
                 entry_offsets=tuple(range(1, 15)))
    bars = load_minute_bars(cfg, store=ResearchStore(),
                            start=pd.Timestamp('2025-10-01', tz='UTC'), end=HI)
    ds = Dataset.build(bars, cfg)
    idx = ds.window_index
    bench = idx[(idx >= LO) & (idx < HI)]
    # **Weekly blocks, not an arbitrary split into five.** The repo's own scheme is
    # `fold_scheme='calendar'` with `fold_block_days=7.0`, because the live model
    # retrains weekly (`cron 0 5 * * 0`) -- so a weekly fold simulates the system
    # INCLUDING its retrain policy, and a 21-day block simulates nothing that runs.
    # It also matters for the gate: `min_folds >= 6` is what the incumbent failed
    # at 3, and five blocks minus the one consumed for training leaves only four.
    # July 1 -> Oct 14 is 15 weeks, so 14 are evaluable.
    wk = pd.Series(bench).dt.to_period('W').astype(str).to_numpy()
    blocks = [pd.DatetimeIndex(bench[wk == w]) for w in pd.unique(wk)]
    blocks = [b for b in blocks if len(b) >= 200]
    out = []
    for i, b in enumerate(blocks):
        cut = pd.Timestamp(b.min()) - pd.Timedelta(days=1)
        tr = idx[idx < cut]
        if len(tr) < 1000:
            continue
        fit, _ = fit_fold(ds, pd.DatetimeIndex(tr), cfg)
        te = apply_fold(ds, fit, pd.DatetimeIndex(b), cfg)
        te['_fold'] = i
        out.append(te)
        print(f"  bar fold {i}: train {len(tr):,}w -> test {len(b):,}w "
              f"({len(te):,} rows)", flush=True)
    d = pd.concat(out, ignore_index=True)
    d = d.rename(columns={'offset': 'offset_minutes'})
    d['window_open'] = pd.to_datetime(d.window_open, utc=True)
    return d


def book_and_peer() -> tuple[pd.DataFrame, pd.DataFrame]:
    """market_state from the Kalshi book, cross_venue from Polymarket."""
    def read(venue):
        root = STORE / 'venue_depth' / f'venue={venue}'
        keep = ['symbol', 'window_open', 'offset_minutes', 'available_time',
                'yes_bid', 'yes_ask', 'yes_bid_size', 'yes_ask_size',
                'depth_bid_1c', 'depth_ask_1c', 'depth_bid_5c', 'depth_ask_5c',
                'depth_bid_total', 'depth_ask_total', 'levels_bid', 'levels_ask']
        parts = []
        for sd in sorted(root.glob('symbol=*')):
            for mo in ('2026-01','2026-02','2026-03','2026-04','2026-05',
                       '2026-06','2026-07','2026-08','2026-09','2026-10'):
                f = sd / f'month={mo}' / 'data.parquet'
                if f.exists():
                    x = pd.read_parquet(f)
                    parts.append(x[[c for c in keep if c in x.columns]])
        if not parts:
            return pd.DataFrame()
        d = pd.concat(parts, ignore_index=True)
        d['window_open'] = pd.to_datetime(d.window_open, utc=True)
        d['available_time'] = pd.to_datetime(d.available_time, utc=True)
        # *** AS-OF RULE -- see the note in stage7.build_snapshot_flow. Without
        # this, 48% of Polymarket rows were future-dated. ***
        decision = (d.window_open + pd.to_timedelta(d.offset_minutes, unit='m')
                    + pd.Timedelta(seconds=3))
        d = d[d.available_time <= decision]
        d = (d.sort_values('available_time')
               .drop_duplicates(['symbol', 'window_open', 'offset_minutes'], keep='last'))
        return d.drop(columns=['available_time'])
    k = read('kalshi'); p = read('polymarket')
    if len(k):
        eps = 1e-9
        k['k_spread'] = k.yes_ask - k.yes_bid
        k['k_depth_imb_1c'] = ((k.depth_bid_1c - k.depth_ask_1c)
                               / (k.depth_bid_1c + k.depth_ask_1c + eps))
        k['k_depth_imb_5c'] = ((k.depth_bid_5c - k.depth_ask_5c)
                               / (k.depth_bid_5c + k.depth_ask_5c + eps))
        k['k_levels_ratio'] = k.levels_bid / (k.levels_ask + eps)
        k = k[['symbol', 'window_open', 'offset_minutes', 'k_spread',
               'k_depth_imb_1c', 'k_depth_imb_5c', 'k_levels_ratio']]
    if len(p):
        p['pm_mid'] = (p.yes_bid + p.yes_ask) / 2
        p['pm_spread'] = p.yes_ask - p.yes_bid
        p = p[['symbol', 'window_open', 'offset_minutes', 'pm_mid', 'pm_spread']]
    return k, p


def implied_vol() -> pd.DataFrame:
    root = STORE / 'venue_implied_vol'
    parts = []
    for f in root.rglob('*.parquet'):
        x = pd.read_parquet(f)
        parts.append(x[[c for c in ['symbol', 'event_time', 'implied_sigma_per_min',
                                    'r2', 'minutes_to_close'] if c in x.columns]])
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    d['event_time'] = pd.to_datetime(d.event_time, utc=True)
    d = d[(d.event_time >= LO) & (d.event_time < HI) & (d.r2 >= 0.90)]
    d['window_open'] = d.event_time.dt.floor('15min')
    d['offset_minutes'] = ((d.event_time - d.window_open).dt.total_seconds() // 60).astype(int)
    d = d.sort_values('event_time').drop_duplicates(
        ['symbol', 'window_open', 'offset_minutes'], keep='last')
    return d[['symbol', 'window_open', 'offset_minutes',
              'implied_sigma_per_min']].rename(columns={'implied_sigma_per_min': 'iv_sigma'})


if __name__ == '__main__':
    t0 = time.time()
    print('building bar features (fitted inside folds)...', flush=True)
    bars = bar_features()
    print(f'  bars: {len(bars):,} rows [{time.time()-t0:.0f}s]', flush=True)

    mv = pd.read_parquet(HERE.parents[1] / 'campaign' / 'datasets' / 'market_v2.parquet')
    mv['window_open'] = pd.to_datetime(mv.window_open, utc=True)
    mv = mv[mv.usable][['symbol', 'window_open', 'offset_minutes', 'p_mkt', 'y',
                        'yes_bid', 'yes_ask', 'ask_at_touch', 'bid_at_touch',
                        'quote_age_s', 'book_age_s']]
    m = bars.merge(mv, on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    print(f'  + market_v2: {len(m):,} rows', flush=True)

    flow = build_snapshot_flow()[['symbol', 'window_open', 'offset_minutes'] + FLOW]
    m = m.merge(flow, on=['symbol', 'window_open', 'offset_minutes'], how='left')
    k, p = book_and_peer()
    if len(k): m = m.merge(k, on=['symbol', 'window_open', 'offset_minutes'], how='left')
    if len(p): m = m.merge(p, on=['symbol', 'window_open', 'offset_minutes'], how='left')
    iv = implied_vol()
    if len(iv): m = m.merge(iv, on=['symbol', 'window_open', 'offset_minutes'], how='left')
    m = m[m.offset_minutes.between(1, 14)].reset_index(drop=True)
    print(f'  + flow/book/peer/iv: {len(m):,} rows, {m.window_open.nunique():,} windows, '
          f'{m.groupby(["symbol","window_open"]).ngroups:,} symbol-windows', flush=True)

    from core.features import FEATURE_GROUPS
    bar_cols = [c for g in ('geometry', 'vol_state', 'microstructure', 'cross_asset',
                            'offset', 'time_of_day')
                for c in FEATURE_GROUPS[g] if c in m.columns]
    extra = [c for c in (FLOW + ['k_spread', 'k_depth_imb_1c', 'k_depth_imb_5c',
                                 'k_levels_ratio', 'pm_mid', 'pm_spread', 'iv_sigma'])
             if c in m.columns]
    feats = bar_cols + extra
    print(f'\nFEATURES: {len(feats)} total  ({len(bar_cols)} bar + {len(extra)} book/flow/peer/iv)')
    pop = {c: float(m[c].notna().mean()) for c in feats}
    print('  population (fraction of rows carrying the feature):')
    for c in extra:
        print(f'    {c:<20} {pop[c]*100:5.1f}%')
    thin = [c for c in bar_cols if pop[c] < 0.95]
    print(f'    bar features below 95% populated: {thin if thin else "none"}')

    import lightgbm as lgb
    preds = np.full(len(m), np.nan); iters = []; gains = {}
    for i in sorted(m._fold.unique()):
        te_m = (m._fold == i).to_numpy()
        tr_m = (m._fold < i).to_numpy()
        if tr_m.sum() < 5000 or te_m.sum() == 0:
            continue
        tr, te = m[tr_m], m[te_m]
        iw = np.array(sorted(pd.unique(tr.window_open))); cut = iw[int(len(iw)*0.75)]
        itr = tr[tr.window_open < cut - pd.Timedelta(days=1)]
        iva = tr[tr.window_open >= cut]
        if len(itr) < 2000 or len(iva) < 1000:
            itr, iva = tr, tr
        par = dict(objective='binary', metric='binary_logloss', verbosity=-1,
                   learning_rate=0.02, num_leaves=15, max_depth=4,
                   min_child_samples=500, bagging_fraction=0.8, bagging_freq=1,
                   feature_fraction=0.7, lambda_l2=20.0, deterministic=True, seed=17)
        dtr = lgb.Dataset(itr[feats], label=itr.y.to_numpy(int),
                          init_score=_logit(itr.p_mkt), free_raw_data=False)
        dva = lgb.Dataset(iva[feats], label=iva.y.to_numpy(int),
                          init_score=_logit(iva.p_mkt), reference=dtr, free_raw_data=False)
        bst = lgb.train(par, dtr, num_boost_round=600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        corr = np.asarray(bst.predict(te[feats], raw_score=True), float)
        preds[te_m] = _sig(_logit(te.p_mkt) + corr)
        iters.append(bst.best_iteration)
        for nme, gv in zip(bst.feature_name(), bst.feature_importance('gain')):
            gains[nme] = gains.get(nme, 0.0) + float(gv)
        print(f'  fold {i}: train {tr_m.sum():,} -> test {te_m.sum():,}  '
              f'iters {bst.best_iteration}', flush=True)

    ok = np.isfinite(preds); s = m[ok].copy(); s['p_model'] = preds[ok]
    s['d'] = _ll(s.p_mkt, s.y) - _ll(s.p_model, s.y)
    print(f'\n=== FULL MODEL, all sources, {LO.date()} -> {HI.date()} ===')
    res = {'n_features': len(feats), 'features': feats, 'population': pop,
           'best_iters': iters, 'n_oof': int(ok.sum())}
    for lbl, v in (('row', s.d.to_numpy()),
                   ('symbol-window', s.groupby(['symbol','window_open']).d.mean().to_numpy()),
                   ('window', s.groupby('window_open').d.mean().to_numpy())):
        lo, hi = _boot_ci(v)
        print(f'  {lbl:>14} {v.mean():+.6f}  CI [{lo:+.6f},{hi:+.6f}]  n={len(v):,}'
              f"{'   BEATS THE MARKET' if lo > 0 else ''}")
        res[lbl] = {'mean': float(v.mean()), 'ci': [lo, hi], 'n': int(len(v))}
    print(f'\n  log loss  model {np.mean(_ll(s.p_model, s.y)):.6f}   '
          f'market {np.mean(_ll(s.p_mkt, s.y)):.6f}')
    tot = sum(gains.values()) or 1.0
    print('\n  top 12 features by gain share:')
    for nme, gv in sorted(gains.items(), key=lambda kv: -kv[1])[:12]:
        print(f'    {nme:<24} {gv/tot*100:5.2f}%')
    res['gain_share'] = {k2: v2/tot for k2, v2 in gains.items()}
    print('\n=== by month ===')
    s['month'] = s.window_open.dt.to_period('M').astype(str)
    for mo, g in s.groupby('month'):
        wv = g.groupby('window_open').d.mean().to_numpy()
        lo, hi = _boot_ci(wv)
        print(f'  {mo}  n={len(g):>7,} w={len(wv):>5,}  {wv.mean():+.6f} '
              f'[{lo:+.6f},{hi:+.6f}]')
    # ---- per-fold trade accounting -------------------------------------
    # A fold that sits out a whole week is a defect, not caution: candidate A had
    # 5 of 18 silent folds, several at alpha=1.000. Reported per fold, per gate.
    def sim(df, gate, fee_rate=0.07, stake=1):
        d = df.sort_values(['symbol', 'window_open', 'offset_minutes'])
        rows = []
        for (sym, w), g in d.groupby(['symbol', 'window_open'], sort=False):
            for r in g.itertuples():
                pm = r.p_model
                if not np.isfinite(pm):
                    continue
                yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
                ev_y, ev_n = pm - yes_px, (1.0 - pm) - no_px
                side, px, ev = (('yes', yes_px, ev_y) if ev_y >= ev_n
                                else ('no', no_px, ev_n))
                if not (0.0 < px < 1.0):
                    continue
                fee = np.ceil(fee_rate * stake * px * (1 - px) * 10_000) / 10_000
                if ev * stake - fee < gate / 100.0 * stake:
                    continue
                vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
                if not np.isfinite(vol) or vol < stake:
                    continue
                won = (r.y == 1) if side == 'yes' else (r.y == 0)
                rows.append({'window_open': w, 'pnl': (stake if won else 0.0) - px * stake - fee,
                             'won': won, 'price': px})
                break
        return pd.DataFrame(rows)

    print('\n=== TRADES PER FOLD (watching for silent folds) ===')
    money = {}
    for gate in (1.0, 1.5, 2.5):
        print(f'  gate {gate}pp:')
        per = []
        silent = 0
        for i in sorted(s._fold.unique()):
            g = s[s._fold == i]
            t = sim(g, gate)
            n = len(t)
            if n == 0:
                silent += 1
                print(f'    fold {i}: {g.window_open.nunique():>5,} windows -> '
                      f'**0 TRADES**')
                per.append({'fold': int(i), 'trades': 0})
                continue
            pnl = float(t.pnl.sum())
            print(f'    fold {i}: {g.window_open.nunique():>5,} windows -> '
                  f'{n:>5,} trades  ${pnl:>8.2f}  {pnl/n*100:>+7.3f}pp/ct  '
                  f'win {t.won.mean()*100:>5.1f}%')
            per.append({'fold': int(i), 'trades': n, 'pnl': pnl,
                        'pp_per_contract': pnl / n * 100,
                        'win_rate': float(t.won.mean())})
        allt = sim(s, gate)
        if len(allt):
            gw = allt.groupby('window_open').pnl.sum()
            tstat = float(gw.mean() / (gw.std(ddof=1) / np.sqrt(len(gw)))) if len(gw) > 1 else float('nan')
            lo, hi = _boot_ci(gw.to_numpy())
            print(f'    TOTAL: {len(allt):,} trades  ${allt.pnl.sum():.2f}  '
                  f'{allt.pnl.sum()/len(allt)*100:+.3f}pp/ct  t={tstat:.2f}  '
                  f'silent folds {silent}/{len(per)}')
            money[f'gate_{gate}'] = {'per_fold': per, 'silent_folds': silent,
                                     'trades': int(len(allt)),
                                     'pnl': float(allt.pnl.sum()),
                                     'pp_per_contract': float(allt.pnl.sum()/len(allt)*100),
                                     't_stat': tstat,
                                     'pnl_window_ci': [lo * len(gw), hi * len(gw)]}
        else:
            print('    TOTAL: no trades at any fold')
            money[f'gate_{gate}'] = {'per_fold': per, 'silent_folds': silent, 'trades': 0}
    res['money'] = money

    s[['symbol','window_open','offset_minutes','p_model','p_mkt','y',
       'yes_bid','yes_ask','ask_at_touch','bid_at_touch']].to_parquet(
        OUT / 'oof_stage9_full.parquet', index=False)
    (OUT / 'stage9_results.json').write_text(json.dumps(res, indent=2, default=str))
    log({'stage': 9, 'experiment': 'full training, all sources, Jul-Oct', 'result':
         {k2: v2 for k2, v2 in res.items() if k2 not in ('features','population','gain_share')},
         'runtime_s': round(time.time()-t0, 1)})
    print(f'\nstage 9 done in {time.time()-t0:.0f}s')
