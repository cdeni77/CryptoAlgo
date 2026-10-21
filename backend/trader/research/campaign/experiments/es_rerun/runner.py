"""The experiment engine. One place that knows how to fit and score a candidate.

Corrected architecture (the first attempt was wrong and is recorded as such in the
LEDGER): features in this repository are built INSIDE the fold by `fit_fold`,
because the volatility model, the intraday seasonality factor and the baseline's
scale/tail are all fitted and must not see test rows. A precomputed feature panel
cannot express that, so every candidate here fits per fold.

The market price is joined from the stream ladder AFTER the fold produces its
features, so no fit ever touches it except as an explicit input.
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2]))
import harness as H
from core.config import Config
from core.datastore import ResearchStore
from core.dataset import Dataset, load_minute_bars, fit_fold, apply_fold
from core.features import FEATURE_GROUPS

BARS_START = pd.Timestamp('2026-01-01', tz='UTC')   # enough history for seasonality
OFFSETS = tuple(range(1, 15))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_dataset(*, bars_end=pd.Timestamp('2026-09-18', tz='UTC'), depth=None, iv=None):
    cfg = Config(decision_offsets=OFFSETS, entry_offsets=OFFSETS,
                 max_quote_age_seconds=900.0, peer_max_quote_age_seconds=900.0)
    bars = load_minute_bars(cfg, store=ResearchStore(), start=BARS_START, end=bars_end)
    ds = Dataset.build(bars, cfg, depth=depth, ladder_fits=iv)
    return ds, cfg


def fold_tables(ds, cfg, folds, groups=None):
    """Yield (fold, train_table, test_table) with features and the baseline attached."""
    for f in folds:
        if len(f.train) < 200 or len(f.test) == 0:
            continue
        fit, tr = fit_fold(ds, f.train, cfg, groups=groups)
        te = apply_fold(ds, fit, f.test, cfg, groups=groups)
        yield f, tr, te


# Columns `core.quotes.attach_quotes` puts on the panel. They must be dropped before
# the campaign's own price is merged in, or pandas suffixes both copies `_x`/`_y` and
# every downstream reference breaks -- which is exactly how the book batch died.
# Dropping them is also correct on the merits: the panel's quote selector ranks
# observers and can return a price from the wrong minute (D5/D7), whereas the
# campaign's as-of join is availability-correct.
_PANEL_QUOTE_COLUMNS = (
    'ask_up', 'ask_down', 'market_probability', 'quote_age_seconds', 'quote_source',
    'quote_observer', 'depth_up', 'depth_down', 'bid_at_touch', 'ask_at_touch',
    'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_vol', 'ask_vol',
    'yes_bid', 'yes_ask', 'spread', 'p_mkt', 'book_age_s', 'usable',
)


def attach_market(table: pd.DataFrame, px: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    t = table.rename(columns={'offset': 'offset_minutes'}).copy()
    t = t.drop(columns=[c for c in _PANEL_QUOTE_COLUMNS if c in t.columns])
    t['window_open'] = pd.to_datetime(t.window_open, utc=True)
    t = (t.merge(px,  on=['symbol', 'window_open', 'offset_minutes'], how='inner')
          .merge(lab, on=['symbol', 'window_open'],                    how='inner'))
    # the stale-quote filter (D8/D9). Flagged upstream, applied here, counted.
    if 'usable' in t.columns:
        before = len(t)
        t = t[t.usable].copy()
        if before and len(t) < before:
            import logging
            logging.getLogger('runner').debug(
                'stale-quote filter kept %d of %d rows', len(t), before)
    t['y'] = t.settled_up.astype(int)
    t['split'] = np.where(t.window_open < H.DEV_END, 'dev', 'val')
    return t


def feature_cols(table, groups):
    cols = []
    for g in groups:
        cols += [c for c in FEATURE_GROUPS.get(g, ()) if c in table.columns]
    return sorted(set(cols))


# ------------------------------------------------------------- fitters -----
def _lgbm(Xtr, ytr, itr, Xte, ite, params, Xva=None, yva=None, iva=None):
    """LightGBM with a chronological inner validation set and early stopping.

    **The earlier version had neither**, and the difference is measured rather than
    assumed: E5 (no early stopping) scored model_minus_market -0.003791 against arm
    A's +0.000195 on the same architecture and features -- about 0.004 nats, larger
    than every effect in this project. Candidates E4-E11 were all fitted under the
    weaker protocol and are re-run here.
    """
    import lightgbm as lgb
    p = dict(objective='binary', metric='binary_logloss', verbosity=-1,
             learning_rate=0.03, num_leaves=15, max_depth=4,
             min_child_samples=500, bagging_fraction=0.8, bagging_freq=1,
             feature_fraction=0.8, lambda_l2=10.0, deterministic=True, seed=17)
    p.update({k: v for k, v in (params or {}).items()})
    dtr = lgb.Dataset(Xtr, label=ytr, init_score=itr, free_raw_data=False)
    if Xva is not None and len(Xva):
        dva = lgb.Dataset(Xva, label=yva, init_score=iva, reference=dtr,
                          free_raw_data=False)
        bst = lgb.train(p, dtr, num_boost_round=500, valid_sets=[dva],
                        valid_names=['inner'],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        bst = lgb.train(p, dtr, num_boost_round=300)
    return 1 / (1 + np.exp(-(np.asarray(bst.predict(Xte, raw_score=True), float) + ite))), bst


def _catboost(Xtr, ytr, itr, Xte, ite, params):
    from catboost import CatBoostClassifier, Pool
    p = dict(iterations=400, learning_rate=0.03, depth=4, l2_leaf_reg=6.0,
             loss_function='Logloss', verbose=0, allow_writing_files=False,
             thread_count=4)
    p.update(params or {})
    m = CatBoostClassifier(**p)
    m.fit(Pool(Xtr, ytr, baseline=itr))
    raw = m.predict(Pool(Xte, baseline=ite), prediction_type='RawFormulaVal')
    return 1 / (1 + np.exp(-raw)), m


def _logistic(Xtr, ytr, itr, Xte, ite, params):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    # the init score enters as a feature; a ridge penalty keeps it near 1
    A = np.column_stack([sc.transform(Xtr), itr])
    B = np.column_stack([sc.transform(Xte), ite])
    m = LogisticRegression(C=(params or {}).get('C', 0.05), max_iter=3000)
    m.fit(A, ytr)
    return m.predict_proba(B)[:, 1], m


FITTERS = {'lgbm': _lgbm, 'catboost': _catboost, 'logistic': _logistic}


def run(ds, cfg, folds, px, lab, *, name, hypothesis, groups, fitter='lgbm',
        init='market', params=None, symbols=None, trailing_days=None,
        budget_s=1200, store_groups=None):
    """Fit one candidate out-of-fold and score it under the campaign's rules."""
    t0 = time.time()
    from sklearn.impute import SimpleImputer
    preds, frames = [], []
    for f, tr, te in fold_tables(ds, cfg, folds, groups=store_groups or groups):
        if time.time() - t0 > budget_s:
            return {'name': name, 'status': 'FAILED',
                    'reason': f'compute budget {budget_s}s exceeded', 'groups': groups}, None
        tr = attach_market(tr, px, lab); te = attach_market(te, px, lab)
        if symbols:
            tr = tr[tr.symbol.isin(symbols)]; te = te[te.symbol.isin(symbols)]
        if trailing_days:
            cut = tr.window_open.max() - pd.Timedelta(days=trailing_days)
            tr = tr[tr.window_open >= cut]
        feats = feature_cols(tr, groups)
        if not feats or len(tr) < 500 or len(te) == 0:
            continue
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(tr[feats].to_numpy(float))
        Xte = imp.transform(te[feats].to_numpy(float))
        if init == 'market':
            itr, ite = logit(tr.p_mkt.to_numpy()), logit(te.p_mkt.to_numpy())
        elif init == 'baseline':
            itr, ite = (tr.baseline_probability_logit.to_numpy(),
                        te.baseline_probability_logit.to_numpy())
        else:
            itr, ite = np.zeros(len(tr)), np.zeros(len(te))
        if fitter == 'lgbm':
            # chronological inner split, purged by a day, for early stopping
            iw = np.array(sorted(pd.unique(tr.window_open)))
            cutw = iw[int(len(iw) * 0.75)]
            msk_i = (tr.window_open < cutw - pd.Timedelta(days=1)).to_numpy()
            msk_v = (tr.window_open >= cutw).to_numpy()
            if msk_i.sum() >= 500 and msk_v.sum() >= 200:
                p, _ = FITTERS[fitter](Xtr[msk_i], tr.y.to_numpy(int)[msk_i], itr[msk_i],
                                       Xte, ite, params,
                                       Xva=Xtr[msk_v], yva=tr.y.to_numpy(int)[msk_v],
                                       iva=itr[msk_v])
            else:
                p, _ = FITTERS[fitter](Xtr, tr.y.to_numpy(int), itr, Xte, ite, params)
        else:
            p, _ = FITTERS[fitter](Xtr, tr.y.to_numpy(int), itr, Xte, ite, params)
        te = te.copy(); te['p_model'] = p
        frames.append(te); preds.append(len(te))
    if not frames:
        return {'name': name, 'status': 'FAILED', 'reason': 'no fold produced predictions',
                'groups': groups}, None
    oof = pd.concat(frames, ignore_index=True)
    mean, lo, hi, pval, nw = H.model_minus_market(oof.p_model, oof.p_mkt, oof.y, oof.window_open)
    res = {
        'name': name, 'status': 'OK', 'hypothesis': hypothesis,
        'fitter': fitter, 'init': init, 'groups': groups,
        'features': feature_cols(oof, groups), 'n_features': len(feature_cols(oof, groups)),
        'params': params or {}, 'symbols': symbols or 'all',
        'trailing_days': trailing_days,
        'n_oof': int(len(oof)), 'windows': nw,
        'log_loss_model': H.logloss(oof.p_model, oof.y),
        'log_loss_market': H.logloss(oof.p_mkt, oof.y),
        'log_loss_baseline': (H.logloss(oof.baseline_probability, oof.y)
                              if 'baseline_probability' in oof.columns else None),
        'model_minus_market': mean, 'ci95': [lo, hi], 'p_one_sided': pval,
        'beats_market': bool(lo > 0),
        'calibration': H.calibration(oof.p_model, oof.y),
        'market_calibration': H.calibration(oof.p_mkt, oof.y),
        'runtime_s': round(time.time() - t0, 1),
    }
    return res, oof
