"""Price the cross-sectional residual candidate: long the top, short the bottom.

    python -m scripts.cross_sectional_backtest --horizon 96 --min-history-days 231
    python -m scripts.cross_sectional_backtest --horizon 96 --model ridge_top3 --legs 3

`scripts.model_capacity --demeaned-target` found the first cell in this project
whose measured IC exceeds its own break-even requirement: ridge_top3 at h=96h,
test IC +0.0784 against a required 0.0736, on 6 of 6 folds. That is an IC, and an
IC is not a P&L. This turns it into one.

Why this is a separate execution path, stated rather than hidden
---------------------------------------------------------------
`core/signal.py:decide()` is deliberately the only place a trade is chosen, so
the backtest and the live writer cannot drift. This script does **not** call it,
because `decide()` chooses a direction for one instrument against an absolute
hurdle, and a market-neutral book chooses a *ranking* across the universe and is
flat in aggregate. Those are different decisions and forcing one through the other
would misprice both.

The cost of that is real: this is a second simulator, and it is a research tool
only. It is not wired into `core/promotion.py`, nothing here can promote, and if
the candidate survives it has to be folded into `decide()` before it goes
anywhere near live. Treat a number from this script as a reason to do that work,
never as a substitute for it.

What it does, and the choices that matter
-----------------------------------------
* **Forecasts are walk-forward.** Purged folds, the model refitted per fold,
  predictions taken only on held-out bars. Feature selection happens inside the
  training fold.
* **The target is the cross-sectional residual**, `r - mean(r)` at each bar.
  Demeaning uses only contemporaneous data, so it adds no lookahead.
* **Rebalance every `horizon` bars, not every bar.** Overlapping books at a
  four-day hold would be 96 simultaneous portfolios and the turnover would be
  fiction.
* **Trade only the change in holdings.** An instrument that stays in the long
  basket across a rebalance pays nothing. This is what a real implementation does
  and it roughly halves the cost against naive full-turnover.
* **Fills at the next open**, and every leg pays the measured schedule — 0.10% of
  notional plus $0.12/contract, plus the half-spread each way.
* **Equal weight** across legs. Weighting by forecast strength is the first thing
  to try and the first thing to overfit; equal weight has no free parameters.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.costs import get_contract_spec
from core.cv import purged_walk_forward, sample_weights
from core.metrics import drawdown_profile, sharpe_ratio
from core.model import HeadSpec, information_coefficient
from core.targets import round_trip_cost_series
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data

warnings.filterwarnings('ignore')


def _model(name: str):
    if name == 'ridge_top3':
        return make_pipeline(SimpleImputer(strategy='median'),
                            StandardScaler(), Ridge(alpha=100.0)), 3
    if name == 'ridge_all':
        return make_pipeline(SimpleImputer(strategy='median'),
                            StandardScaler(), Ridge(alpha=100.0)), None
    if name == 'lgbm_tiny':
        import lightgbm as lgb
        return lgb.LGBMRegressor(**HeadSpec(
            n_estimators=40, max_depth=2, min_child_samples=2_000,
            learning_rate=0.05, colsample_bytree=1.0).to_params()), None
    raise SystemExit(f'unknown model {name!r}')


def _walk_forward_residual_forecasts(X, y, times, folds, horizon, model_name, half_life):
    """Out-of-sample predictions of the cross-sectional residual."""
    model, top_k = _model(model_name)
    pieces, ics = [], []
    for fold in folds:
        tr, te = times.isin(fold.train_idx), times.isin(fold.test_idx)
        xtr, ytr, xte, yte = X[tr], y[tr], X[te], y[te]
        if len(xtr) < 500 or len(xte) < 200:
            continue
        columns = list(xtr.columns)
        if top_k:
            scored = [(abs(information_coefficient(xtr[c].to_numpy(), ytr.to_numpy())), c)
                      for c in columns]
            scored = [(v, c) for v, c in scored if np.isfinite(v)]
            columns = [c for _, c in sorted(scored, reverse=True)[:top_k]]
        weights = sample_weights(xtr.index.get_level_values('event_time'),
                                horizon_bars=horizon, half_life_days=half_life)
        if model_name.startswith('lgbm'):
            model.fit(xtr[columns], ytr, sample_weight=weights)
        else:
            model.fit(xtr[columns], ytr)
        pred = pd.Series(model.predict(xte[columns]), index=xte.index, name='forecast')
        pieces.append(pred)
        ics.append(information_coefficient(pred.to_numpy(), yte.to_numpy()))
    if not pieces:
        return pd.Series(dtype=float), []
    return pd.concat(pieces).sort_index(), ics


def _simulate(forecasts, bars, config, *, horizon, legs, equity):
    """Rebalance every `horizon` bars into an equal-weight long/short book."""
    stamps = forecasts.index.get_level_values('event_time').unique().sort_values()
    rebalances = stamps[::horizon]

    opens = pd.DataFrame({s: b.sort_index()['open'] for s, b in bars.items()})
    costs = pd.DataFrame({s: round_trip_cost_series(s, b.sort_index()['close'], config)
                          for s, b in bars.items()})
    # Per side: the round trip is symmetric, so half of it is one leg's toll.
    one_way = costs / 2.0

    held: dict[str, int] = {}
    curve, rows = {}, []
    cash = float(equity)

    for i, stamp in enumerate(rebalances):
        try:
            slice_ = forecasts.xs(stamp, level='event_time')
        except KeyError:
            continue
        slice_ = slice_.dropna()
        if len(slice_) < 2 * legs:
            continue

        # Fill at the next open, which is the first price this decision reaches.
        later = opens.index[opens.index > stamp]
        if later.empty:
            break
        fill_time = later[0]

        ranked = slice_.sort_values(ascending=False)
        longs = list(ranked.index[:legs])
        shorts = list(ranked.index[-legs:])
        target = {s: +1 for s in longs} | {s: -1 for s in shorts}

        # Only the change in holdings trades.
        traded = {s for s in set(held) | set(target) if held.get(s, 0) != target.get(s, 0)}
        per_leg = cash / max(2 * legs, 1)
        turn_cost = 0.0
        for symbol in traded:
            if symbol not in one_way.columns or fill_time not in one_way.index:
                continue
            toll = one_way.at[fill_time, symbol]
            if not np.isfinite(toll):
                continue
            # A flip pays twice: out of the old side and into the new one.
            sides = 2 if held.get(symbol, 0) * target.get(symbol, 0) < 0 else 1
            spread = (config.spread_bps / 10_000.0) if config.apply_slippage else 0.0
            turn_cost += per_leg * (toll + spread) * sides
        cash -= turn_cost

        # Hold to the next rebalance, marking the realised residual move.
        end = rebalances[i + 1] if i + 1 < len(rebalances) else stamps[-1]
        later_end = opens.index[opens.index > end]
        exit_time = later_end[0] if not later_end.empty else opens.index[-1]

        gross = 0.0
        legs_filled = 0
        for symbol, side in target.items():
            if symbol not in opens.columns:
                continue
            entry, exit_px = opens.at[fill_time, symbol], opens.at[exit_time, symbol]
            if not (np.isfinite(entry) and np.isfinite(exit_px)) or entry <= 0:
                continue
            gross += side * per_leg * (exit_px / entry - 1.0)
            legs_filled += 1
        cash += gross
        held = target
        curve[exit_time] = cash
        rows.append({'time': stamp, 'legs': legs_filled, 'traded': len(traded),
                     'cost': turn_cost, 'gross': gross, 'equity': cash})

    return pd.Series(curve).sort_index(), pd.DataFrame(rows)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--legs', type=int, default=3,
                        help='Instruments per side. 3 long and 3 short of 14.')
    parser.add_argument('--model', default='ridge_top3',
                        choices=('ridge_top3', 'ridge_all', 'lgbm_tiny'))
    parser.add_argument('--equity', type=float, default=100_000.0)
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
    residual = y - y.groupby(level='event_time').transform('mean')

    times = pd.DatetimeIndex(X.index.get_level_values('event_time'))
    unique = times.unique().sort_values()
    folds = purged_walk_forward(unique, n_folds=args.folds,
                                min_train_bars=max(len(unique) // 4, 1),
                                purge_bars=horizon, embargo_bars=horizon)

    forecasts, fold_ics = _walk_forward_residual_forecasts(
        X, residual, times, folds, horizon, args.model,
        config.recency_half_life_days)
    if forecasts.empty:
        print('no out-of-sample forecasts')
        return 1

    median_ic = float(np.median([v for v in fold_ics if np.isfinite(v)]))
    agree = sum(1 for v in fold_ics if np.isfinite(v) and (v > 0) == (median_ic > 0))
    print(f'\n{args.model} | horizon {horizon}h | {len(forecasts):,} OOS forecasts')
    print(f'residual IC {median_ic:+.4f} across {len(fold_ics)} folds '
          f'({agree}/{len(fold_ics)} agree)')

    curve, ledger = _simulate(forecasts, dataset.bars, config,
                              horizon=horizon, legs=args.legs, equity=args.equity)
    if curve.empty or ledger.empty:
        print('no rebalances executed')
        return 1

    returns = curve.pct_change().dropna()
    # Rebalances happen every `horizon` hours, so that is the period length.
    periods_per_year = int(round(24 * 365 / horizon))
    total = float(curve.iloc[-1] / args.equity - 1.0)
    dd = drawdown_profile(curve, periods_per_year=periods_per_year)

    print(f'\nrebalances       {len(ledger)}')
    print(f'legs per side    {args.legs} long / {args.legs} short')
    print(f'positions traded {int(ledger["traded"].sum())} '
          f'({ledger["traded"].mean():.1f} per rebalance of {2 * args.legs})')
    print(f'gross P&L        {ledger["gross"].sum():+,.0f}')
    print(f'costs            {-ledger["cost"].sum():+,.0f}')
    print(f'net P&L          {curve.iloc[-1] - args.equity:+,.0f}  ({total:+.2%})')
    print(f'Sharpe           {sharpe_ratio(returns, periods_per_year=periods_per_year):+.2f}')
    print(f'max drawdown     {dd.max_drawdown:.1%}')
    print(f'periods positive {(returns > 0).mean():.1%}')

    gross_sum, cost_sum = float(ledger['gross'].sum()), float(ledger['cost'].sum())
    print(f'\ncost as a share of gross  {cost_sum / abs(gross_sum):.1%}'
          if gross_sum else '\ngross is zero')
    if gross_sum > 0 and gross_sum > cost_sum:
        print('The residual forecast paid for its own trading over this sample.')
    elif gross_sum > 0:
        print('Gross is positive and costs exceed it: the edge is real and too '
              'small for the toll.')
    else:
        print('Gross is negative: the forecast did not translate into P&L, so the '
              'IC was not tradeable regardless of cost.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
