"""Rerun the economics against a SIMULATED book instead of an invented one.

`core/backtest.py` charges itself `F(x/sigma)` because it has no order book, so
`edge = model - price` becomes `model - baseline` — the model's own correction,
the quantity it was fitted to produce, positive by construction. That is the
self-referential trade behind the 5.61 Sharpe that failed `sharpe_implausible`.

`core/market_sim.py` fits what the venue actually does to a baseline number, on
77,349 rows carrying both: no bias worth modelling, 4.80pp of dispersion, 1.00c
median spread. Substituting that turns an assumption of ZERO dispersion into a
measured one, which is the difference that matters — at 4.8pp of scatter against a
model correction near 1pp, the sign of the edge is mostly the market's deviation.

Run over any span. The 69 days with real quotes are the control: the simulated
book should reproduce the real-price economics there, and if it does not the
simulator is wrong rather than the history interesting.
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
from core.market_sim import MarketSimulator
from core.metrics import DEFAULT_GATES, IMPLAUSIBLE_SHARPE
from core.promotion import load_live

logger = logging.getLogger('simbook')


def money(trades: pd.DataFrame, joined_rows: int, bankroll: float,
          window_minutes: int) -> dict:
    pnl = trades['pnl'].sum()
    settle = pd.to_datetime(trades['window_open'], utc=True) + pd.Timedelta(
        minutes=window_minutes)
    daily = trades.assign(s=settle).set_index('s')['pnl'].resample('1D').sum()
    if len(daily) > 1:
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(),
                                            freq='1D', tz=daily.index.tz), fill_value=0.0)
    values = daily.to_numpy(dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float('nan')
    equity = bankroll + trades['pnl'].cumsum()
    return {
        'trades': len(trades), 'coverage': len(trades) / max(joined_rows, 1),
        'win_rate': float(trades['won'].mean()),
        'realised_edge_pp': 100 * pnl / trades['contracts'].sum(),
        'pnl': float(pnl), 'staked': float(trades['stake'].sum()),
        'sharpe': (float(np.mean(values)) / sd * np.sqrt(365.25)) if sd else float('nan'),
        'max_drawdown': float(((equity.cummax() - equity) / equity.cummax()).max()),
    }


def replay(table: pd.DataFrame, config, bankroll: float) -> pd.DataFrame:
    out = []
    for window, part in table.groupby('window_open', sort=True):
        exposure = WindowExposure()
        for _, row in part.sort_values(['offset', 'symbol']).iterrows():
            d = decide(row, config, bankroll=bankroll, exposure=exposure,
                       require_quote=True)
            if d.reason is not Reason.TRADED:
                continue
            won = bool(row['outcome']) if d.side.value == 'up' else not bool(row['outcome'])
            out.append({'symbol': d.symbol, 'window_open': window, 'offset': d.offset,
                        'contracts': d.contracts, 'stake': d.stake, 'fee': d.fee,
                        'edge': d.edge, 'won': won,
                        'pnl': (d.contracts - d.stake) if won else -d.stake})
            exposure = WindowExposure(stake=exposure.stake + d.stake,
                                      positions=exposure.positions + 1,
                                      symbols_entered=exposure.symbols_entered | {d.symbol})
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', default=None, help='ISO date for the replay span')
    parser.add_argument('--end', default=None)
    parser.add_argument('--bankroll', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=20260825)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    model = load_live(config=config)

    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    quotes = quotes.loc[quotes['usable'].astype(bool)
                        & quotes['offset_minutes'].isin(config.decision_offsets)]

    # Score the span the simulator is FITTED on, to learn the deviation.
    lo = (quotes['window_open'].min() - pd.Timedelta(days=3)).tz_convert(None)
    hi = (quotes['window_open'].max() + pd.Timedelta(hours=1)).tz_convert(None)
    ds = Dataset.build(load_minute_bars(config, store=store, start=lo, end=hi), config)
    b = model.scoring
    states = {s: apply_seasonality(ds.states[s], b.seasonality[s]) for s in ds.states}
    fit = FoldFit(seasonality=b.seasonality, vol_models=b.vol_models,
                  baseline=b.baseline, train_windows=model.n_train_windows, states=states)
    scored = apply_fold(ds, fit, ds.window_index, config, groups=model.groups or None)
    scored['model_probability'] = model.predict(scored)
    fit_frame = scored.merge(
        quotes.rename(columns={'offset_minutes': 'offset'})[
            ['symbol', 'window_open', 'offset', 'market_probability', 'spread']],
        on=['symbol', 'window_open', 'offset'], how='inner')

    sim = MarketSimulator.fit(fit_frame)
    print(sim.summary())

    # Control: does the simulated book reproduce the real-price economics on the
    # same rows? If not, the simulator is wrong and nothing downstream is worth
    # reading.
    rng = np.random.default_rng(args.seed)
    real = fit_frame.assign(
        ask_up=fit_frame['market_probability'] + fit_frame['spread'] / 2,
        ask_down=1.0 - (fit_frame['market_probability'] - fit_frame['spread'] / 2))
    mid, half = sim.sample(fit_frame['baseline_probability'].to_numpy(),
                           fit_frame['offset'].to_numpy(), rng=rng)
    simulated = fit_frame.assign(ask_up=mid + half, ask_down=1.0 - (mid - half))
    invented = fit_frame.assign(
        ask_up=fit_frame['baseline_probability'] + config.half_spread_cents / 100,
        ask_down=1.0 - (fit_frame['baseline_probability'] - config.half_spread_cents / 100))

    print(f'\n{"book":>22}{"trades":>9}{"win":>8}{"edge_pp":>10}{"pnl":>11}'
          f'{"sharpe":>9}{"maxDD":>8}')
    results = {}
    for name, frame in (('real Kalshi quotes', real),
                        ('simulated book', simulated),
                        ("invented (price=baseline)", invented)):
        frame = frame.loc[np.isfinite(frame['model_probability'])]
        trades = replay(frame, config, args.bankroll)
        if trades.empty:
            print(f'{name:>22}   nothing cleared the gates'); continue
        stats = money(trades, len(frame), args.bankroll, config.window_minutes)
        results[name] = stats
        print(f'{name:>22}{stats["trades"]:>9,}{stats["win_rate"]:>8.3f}'
              f'{stats["realised_edge_pp"]:>+10.2f}{stats["pnl"]:>+11.2f}'
              f'{stats["sharpe"]:>9.2f}{stats["max_drawdown"]:>8.3f}')

    print(f'\n  sharpe_implausible fires above {IMPLAUSIBLE_SHARPE}')
    for name, stats in results.items():
        flag = 'FAIL' if stats['sharpe'] > IMPLAUSIBLE_SHARPE else 'pass'
        print(f'    [{flag}] {name}')
    print('\n  The invented book is the one the promotion gates actually read.')
    print('  If it flatters the strategy against the real one, every money gate')
    print('  in models/promotions/ has been scored on a fiction.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
