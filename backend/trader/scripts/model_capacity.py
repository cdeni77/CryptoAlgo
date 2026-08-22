"""Is the zero test IC the data, or the model's capacity?

    python -m scripts.model_capacity --horizon 4 --min-history-days 231
    python -m scripts.model_capacity --horizon 24 --recency-half-life-days 365

The production head fits its training folds at IC 0.26-0.53 and its test folds at
0.00. That gap is the definition of overfitting, and it has two causes that call
for opposite responses:

* the data carries no forecastable signal at this horizon, in which case every
  rung of the ladder lands at zero and the architecture is irrelevant;
* the model has far more capacity than the sample supports, in which case a
  drastically simpler one generalises better and the production head is
  mis-specified.

The arithmetic that motivates the question: at h=4h this panel carries roughly
1,700 *effective* observations after label overlap and the recency decay, against
61 features and a 300-tree depth-4 ensemble — on the order of 4,800 leaf
parameters. Three times more parameters than independent observations.

So the ladder runs from a constant through a 3-feature ridge to the production
ensemble, on identical purged folds, reporting train IC beside test IC. A model
with too little capacity to memorise that still scores zero is evidence about the
market; one scoring zero while its train IC is 0.5 is evidence about itself.

Feature selection happens **inside each training fold**. Ranking features on the
full panel and then cross-validating is the leak that makes a ladder like this
report a discovery.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from core.cv import purged_walk_forward, sample_weights
from core.model import HeadSpec, information_coefficient
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data

warnings.filterwarnings('ignore')


def _lgbm(**over):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**HeadSpec(**over).to_params())


def _ridge():
    return make_pipeline(SimpleImputer(strategy='median'),
                         StandardScaler(), Ridge(alpha=100.0))


def _ladder() -> dict:
    return {
        'constant':        dict(model=DummyRegressor(strategy='mean'), top_k=None),
        'ridge_top3':      dict(model=_ridge(), top_k=3),
        'ridge_top10':     dict(model=_ridge(), top_k=10),
        'ridge_all':       dict(model=_ridge(), top_k=None),
        'stump_depth2':    dict(model=DecisionTreeRegressor(
            max_depth=2, min_samples_leaf=2_000, random_state=7), top_k=None),
        'lgbm_tiny':       dict(model=_lgbm(n_estimators=40, max_depth=2,
                                            min_child_samples=2_000,
                                            learning_rate=0.05,
                                            colsample_bytree=1.0), top_k=None),
        'lgbm_small':      dict(model=_lgbm(n_estimators=120, max_depth=3,
                                            min_child_samples=500), top_k=None),
        'lgbm_production': dict(model=_lgbm(), top_k=None),
    }


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--folds', type=int, default=6)
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    horizon = int(dataset.horizon_bars)
    X = dataset.features
    y = dataset.targets['price'].reindex(X.index)
    keep = y.notna()
    X, y = X[keep], y[keep]

    times = pd.DatetimeIndex(X.index.get_level_values('event_time'))
    unique = times.unique().sort_values()
    folds = purged_walk_forward(unique, n_folds=args.folds,
                                min_train_bars=max(len(unique) // 4, 1),
                                purge_bars=horizon, embargo_bars=horizon)

    print(f'\n{len(X):,} rows | {X.shape[1]} features | horizon {horizon}h '
          f'| target sd {y.std() * 1e4:.1f}bp | {len(folds)} folds')

    rows: list[dict] = []
    for name, spec in _ladder().items():
        train_ics, test_ics = [], []
        for fold in folds:
            tr, te = times.isin(fold.train_idx), times.isin(fold.test_idx)
            xtr, ytr, xte, yte = X[tr], y[tr], X[te], y[te]
            if len(xtr) < 500 or len(xte) < 200:
                continue

            columns = list(xtr.columns)
            if spec['top_k']:
                scored = [(abs(information_coefficient(xtr[c].to_numpy(), ytr.to_numpy())), c)
                          for c in columns]
                scored = [(v, c) for v, c in scored if np.isfinite(v)]
                columns = [c for _, c in sorted(scored, reverse=True)[:spec['top_k']]]
            if not columns:
                continue

            weights = sample_weights(xtr.index.get_level_values('event_time'),
                                     horizon_bars=horizon,
                                     half_life_days=config.recency_half_life_days)
            model = spec['model']
            try:
                if name.startswith('lgbm'):
                    model.fit(xtr[columns], ytr, sample_weight=weights)
                else:
                    model.fit(xtr[columns], ytr)
            except Exception as exc:                                # noqa: BLE001
                print(f'  {name}: fold skipped ({exc})')
                continue
            train_ics.append(information_coefficient(
                model.predict(xtr[columns]), ytr.to_numpy()))
            test_ics.append(information_coefficient(
                model.predict(xte[columns]), yte.to_numpy()))

        finite = [v for v in test_ics if np.isfinite(v)]
        if not finite:
            rows.append({'model': name, 'train_ic': float('nan'),
                         'test_ic': float('nan'), 'agree': 0, 'folds': 0})
            continue
        median = float(np.median(finite))
        rows.append({
            'model': name,
            'train_ic': float(np.nanmedian(train_ics)) if train_ics else float('nan'),
            'test_ic': median,
            'agree': sum(1 for v in finite if (v > 0) == (median > 0)),
            'folds': len(finite),
        })

    frame = pd.DataFrame(rows)
    frame['gap'] = frame['train_ic'] - frame['test_ic']
    print('\n' + frame.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

    usable = frame.dropna(subset=['test_ic'])
    print()
    if usable.empty:
        print('nothing measurable')
        return 1
    best = usable.loc[usable['test_ic'].idxmax()]
    if float(best['test_ic']) <= 0.005:
        print('VERDICT: every rung lands at zero out of sample, including models '
              'with too little capacity to memorise. The production head is '
              'overfitting, and reducing capacity recovers nothing — there is no '
              'signal at this horizon to recover.')
    else:
        print(f"VERDICT: {best['model']} reaches test IC {float(best['test_ic']):+.4f} "
              f"({int(best['agree'])}/{int(best['folds'])} folds). Capacity was part "
              f"of the problem. Take this rung to a walk-forward backtest and the "
              f"cost gate before believing it.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
