"""What `decide()` would have done at the venue's ACTUAL prices.

**Why this is different from the backtest.** `core/backtest.py` has no order book,
so `price_source` stands the calibrated baseline in for the market. Every money
number it produces — `realised_edge_pp`, `total_return`, `sharpe` — is therefore
computed against a price we invented, and "beat the price" and "beat the baseline"
collapse into one question answered twice with the same number. That is the single
largest known error in this repository's economics.

The backfilled quotes fix the price. For 19,341 symbol-windows we have the real
`yes_bid`/`yes_ask` at each decision offset, so `decide()` can be run against what
a trade would actually have cost.

**What this still cannot do**, and it is not a small caveat: there is no depth. The
orderbook endpoint returns empty for a settled market — no ladder, no resting size,
no queue position, ever. So this assumes every intended order fills at the touch,
and live measurement says 30% of them do not, with the failures carrying a HIGHER
claimed edge than the fills. So read this as "the price is now real, the fills are
still assumed", which is one of the two errors fixed rather than both.

Per DECISION_RULE.md the economic verdict belongs to the forward test on live
data. This does not substitute for it and does not touch it.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from core.dataset import (Dataset, FoldFit, apply_fold, apply_seasonality,
                          load_minute_bars)
from core.datastore import ResearchStore
from core.decide import Reason, WindowExposure, decide
from core.promotion import load_live

logger = logging.getLogger('retro_econ')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bankroll', type=float, default=100.0)
    parser.add_argument('--min-edge-pp', type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')

    config = DEFAULT_CONFIG
    if args.min_edge_pp is not None:
        config = config.replace(min_edge_pp=args.min_edge_pp) if hasattr(config, 'replace') else config
    store = ResearchStore(os.getenv('RESEARCH_STORE'))

    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    quotes = quotes.loc[quotes['usable'].astype(bool)
                        & quotes['offset_minutes'].isin(config.decision_offsets)]
    model = load_live(config=config)
    if model is None:
        print('no promoted artifact'); return 1

    lo = (quotes['window_open'].min() - pd.Timedelta(days=3)).tz_convert(None)
    hi = (quotes['window_open'].max() + pd.Timedelta(hours=1)).tz_convert(None)
    dataset = Dataset.build(load_minute_bars(config, store=store, start=lo, end=hi), config)
    bundle = model.scoring
    states = {s: apply_seasonality(dataset.states[s], bundle.seasonality[s])
              for s in dataset.states if s in bundle.seasonality}
    fit = FoldFit(seasonality=bundle.seasonality, vol_models=bundle.vol_models,
                  baseline=bundle.baseline, train_windows=model.n_train_windows,
                  states=states)
    scored = apply_fold(dataset, fit, dataset.window_index, config,
                        groups=model.groups or None)
    scored['model_probability'] = model.predict(scored)

    # The real book, in the columns `decide()` reads. `yes_ask` is the cost of the
    # UP side; the DOWN side costs `1 - yes_bid`, because buying NO is selling YES
    # at the bid.
    q = quotes.rename(columns={'offset_minutes': 'offset'})
    q['ask_up'] = q['yes_ask']
    q['ask_down'] = 1.0 - q['yes_bid']
    joined = scored.merge(q[['symbol', 'window_open', 'offset', 'ask_up', 'ask_down',
                             'market_probability']],
                          on=['symbol', 'window_open', 'offset'], how='inner')
    joined = joined.loc[np.isfinite(joined['model_probability'])].copy()
    print(f'{len(joined):,} scored rows carry a real quote, over '
          f'{joined.drop_duplicates(["symbol","window_open"]).shape[0]:,} symbol-windows\n')

    # Walk each window in offset order, one entry per (symbol, window), exactly as
    # `decide_window` does live. Exposure is seeded per window, not carried.
    decisions, reasons = [], {}
    for window, part in joined.groupby('window_open', sort=True):
        exposure = WindowExposure()
        for _, row in part.sort_values(['offset', 'symbol']).iterrows():
            d = decide(row, config, bankroll=args.bankroll, exposure=exposure,
                       require_quote=True)
            reasons[d.reason.value] = reasons.get(d.reason.value, 0) + 1
            if d.reason is not Reason.TRADED:
                continue
            won = bool(row['outcome']) if d.side.value == 'up' else not bool(row['outcome'])
            decisions.append({
                'symbol': d.symbol, 'window_open': window, 'offset': d.offset,
                'side': d.side.value, 'contracts': d.contracts, 'price': d.price,
                'stake': d.stake, 'fee': d.fee, 'edge': d.edge,
                'pnl': (d.contracts - d.stake) if won else -d.stake, 'won': won})
            exposure = WindowExposure(
                stake=exposure.stake + d.stake, positions=exposure.positions + 1,
                symbols_entered=exposure.symbols_entered | {d.symbol})

    print('why no trade')
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'  {reason:<26}{n:>8,}')
    if not decisions:
        print('\nnothing cleared the gates at real prices'); return 0

    trades = pd.DataFrame(decisions)
    staked = trades['stake'].sum()
    pnl = trades['pnl'].sum()
    print(f'\nat the venue\'s actual prices, {config.min_edge_pp}pp gate, '
          f'${args.bankroll:.0f} flat bankroll')
    print(f'  trades              {len(trades):,}')
    print(f'  coverage            {len(trades)/len(joined):.4f} of scored rows')
    print(f'  win rate            {trades["won"].mean():.4f}')
    print(f'  contract-weighted   {trades["stake"].sum()/trades["contracts"].sum():.4f} per contract')
    print(f'  staked              ${staked:,.2f}')
    print(f'  fees                ${trades["fee"].sum():,.2f} '
          f'({100*trades["fee"].sum()/staked:.2f}% of stake)')
    print(f'  P&L                 ${pnl:+,.2f}')
    print(f'  return on stake     {100*pnl/staked:+.2f}%')
    print(f'  realised edge       {100*pnl/trades["contracts"].sum():+.2f}pp per contract')
    print(f'  claimed edge        {100*trades["edge"].mean():+.2f}pp')
    print('\n  by symbol')
    for symbol, part in trades.groupby('symbol'):
        print(f'    {symbol}  n={len(part):>5,}  win {part["won"].mean():.3f}  '
              f'pnl ${part["pnl"].sum():+9,.2f}  '
              f'edge {100*part["pnl"].sum()/part["contracts"].sum():+.2f}pp')
    # --- the money gates, at real prices ------------------------------------
    #
    # `sharpe_implausible` fired on the promoted artifact at 5.61 and blocked it.
    # That gate's job is to notice a number that cannot be real, and it was right:
    # every money figure behind it came from a backtest where `price_source` is
    # the calibrated baseline, so the model was trading against its own null at a
    # price derived from the very thing it was fitted to correct. The Sharpe was
    # not measuring a forecast, it was measuring a self-referential trade.
    #
    # Recomputed here against the venue's actual bid and ask, using the identical
    # definition from `core/book.py`: daily PnL on calendar days with idle days
    # in the denominator, annualised on sqrt(365.25). Not per-trade, which can
    # carry the opposite sign from the account.
    trades = trades.sort_values('window_open')
    settle = pd.to_datetime(trades['window_open'], utc=True) + pd.Timedelta(
        minutes=config.window_minutes)
    daily = trades.assign(settle=settle).set_index('settle')['pnl'].resample('1D').sum()
    if len(daily) > 1:
        calendar = pd.date_range(daily.index.min(), daily.index.max(), freq='1D',
                                 tz=daily.index.tz)
        daily = daily.reindex(calendar, fill_value=0.0)
    values = daily.to_numpy(dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float('nan')
    sharpe = float(np.mean(values)) / sd * np.sqrt(365.25) if sd else float('nan')
    equity = args.bankroll + trades['pnl'].cumsum()
    drawdown = float(((equity.cummax() - equity) / equity.cummax()).max())
    total_return = float(pnl / args.bankroll)

    from core.metrics import DEFAULT_GATES, IMPLAUSIBLE_SHARPE
    print('\n  the money gates, recomputed at the venue\'s real prices')
    checks = [
        ('realised_edge_pp', 100 * pnl / trades['contracts'].sum(), 'min'),
        ('total_return', total_return, 'min'),
        ('sharpe', sharpe, 'min'),
        ('max_drawdown', drawdown, 'max'),
        ('trades', float(len(trades)), 'min'),
        ('coverage', len(trades) / len(joined), 'min'),
    ]
    for name, value, direction in checks:
        threshold = DEFAULT_GATES[name][0]
        ok = value >= threshold if direction == 'min' else value <= threshold
        print(f'    [{"pass" if ok else "FAIL"}] {name:<20}{value:>12.5f} '
              f'{">=" if direction == "min" else "<="} {threshold}')
    implausible = np.isfinite(sharpe) and sharpe > IMPLAUSIBLE_SHARPE
    print(f'    [{"FAIL" if implausible else "pass"}] {"sharpe_implausible":<20}'
          f'{sharpe:>12.5f} <= {IMPLAUSIBLE_SHARPE}'
          f'{"   <- still a bug signature" if implausible else "   <- plausible now"}')
    print(f'\n    for comparison, the backtest at its counterfactual price: '
          f'sharpe 5.61, return 71.51')

    print('\n  NOTE: fills are still assumed. There is no depth history, and live '
          'measurement\n  says ~30% of intended orders do not fill — with the '
          'failures carrying a HIGHER\n  claimed edge than the fills. The price is '
          'real now; the fill is not.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
