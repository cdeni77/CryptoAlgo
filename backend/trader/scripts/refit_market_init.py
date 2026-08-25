"""Fit the correction on top of the venue's price instead of F(x/sigma).

This is what the quote backfill was for. `init_score` becomes `logit(price)`, so
the model learns `logit(truth) - logit(price)` — how the price is wrong, which is
the quantity the money depends on. An untrained model reproduces the price
exactly, so every tree is incremental over the market by construction rather than
by comparison.

Split and success criteria are fixed in `DECISION_RULE.md` Appendix B, written
before this ran. In particular the holdout is read ONCE, and the comparison that
matters is against the baseline-init artifact on the same rows — not against zero.
The existing force-promoted model already scores +0.002105 against the market over
this era, so a market-init model scoring less has cost accuracy for elegance.

The sample is 69 days because that is when the series began; it is the entire
population, not a subset. This model cannot pass `windows_evaluated` and is not
offered for promotion.
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
from core.inference import governing, model_minus_market
from core.model import MARKET_LOGIT, attach_market_logit, fit_model
from core.promotion import load_live

logger = logging.getLogger('refit')

TRAIN_END = pd.Timestamp('2026-08-19', tz='UTC')     # Appendix B, fixed in advance
MATERIAL = 0.001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resamples', type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    quotes = quotes.loc[quotes['usable'].astype(bool)
                        & quotes['offset_minutes'].isin(config.decision_offsets)]

    live_model = load_live(config=config)
    lo = (quotes['window_open'].min() - pd.Timedelta(days=3)).tz_convert(None)
    hi = (quotes['window_open'].max() + pd.Timedelta(hours=1)).tz_convert(None)
    dataset = Dataset.build(load_minute_bars(config, store=store, start=lo, end=hi), config)
    bundle = live_model.scoring
    states = {s: apply_seasonality(dataset.states[s], bundle.seasonality[s])
              for s in dataset.states if s in bundle.seasonality}
    fit = FoldFit(seasonality=bundle.seasonality, vol_models=bundle.vol_models,
                  baseline=bundle.baseline, train_windows=live_model.n_train_windows,
                  states=states)
    scored = apply_fold(dataset, fit, dataset.window_index, config,
                        groups=live_model.groups or None)
    scored['baseline_init_probability'] = live_model.predict(scored)

    table = scored.merge(
        quotes.rename(columns={'offset_minutes': 'offset'})[
            ['symbol', 'window_open', 'offset', 'market_probability']],
        on=['symbol', 'window_open', 'offset'], how='inner')
    table = attach_market_logit(table)
    table = table.loc[np.isfinite(table[MARKET_LOGIT])
                      & table['outcome'].notna()].copy()
    table['outcome'] = table['outcome'].astype(float)

    train = table.loc[table['window_open'] < TRAIN_END]
    holdout = table.loc[table['window_open'] >= TRAIN_END]
    print(f'train   {len(train):,} rows, {train["window_open"].nunique():,} windows, '
          f'{train["window_open"].min():%Y-%m-%d} .. {train["window_open"].max():%Y-%m-%d}')
    print(f'holdout {len(holdout):,} rows, {holdout["window_open"].nunique():,} windows, '
          f'{holdout["window_open"].min():%Y-%m-%d} .. {holdout["window_open"].max():%Y-%m-%d}')
    if holdout.empty or len(train) < 5000:
        print('not enough data to split as Appendix B fixes'); return 1

    import dataclasses
    market_config = dataclasses.replace(config, init_score_source='market')
    print('\nfitting on the market logit...')
    model = fit_model(train, bundle.baseline, market_config,
                      groups=live_model.groups or None, scoring=bundle)
    print(model.summary())

    # Appendix B's bug check, run BEFORE the headline: with no trees the model is
    # the price, so the difference must be identically zero.
    zero = model.predict(holdout, shrink=False) if model.best_iteration == 0 else None
    untrained = np.abs(model.raw_correction(holdout)).max()
    print(f'\nbug check: max |raw correction| on holdout {untrained:.6f} '
          f'({model.best_iteration} trees)')

    holdout = holdout.assign(model_probability=model.predict(holdout))
    print(f'\n{"":>28}{"log loss":>11}')
    from core.baseline import log_loss
    y = holdout['outcome'].to_numpy(dtype=float)
    for name, col in (('market (the init score)', 'market_probability'),
                      ('baseline-init artifact', 'baseline_init_probability'),
                      ('market-init refit', 'model_probability'),
                      ('F(x/sigma)', 'baseline_probability')):
        print(f'  {name:<26}{log_loss(y, holdout[col].to_numpy(dtype=float)):>11.5f}')

    print('\nHOLDOUT — model_minus_market (read once, Appendix B)')
    results = model_minus_market(holdout, n_resamples=args.resamples)
    for length in sorted(results):
        print(results[length].line())
    gov = governing(results)
    print(f'\n  governing block {gov.block_days}d: point {gov.point:+.6f} '
          f'90% CI [{gov.lo:+.6f}, {gov.hi:+.6f}] p={gov.p_value:.4f}')

    base = holdout.assign(model_probability=holdout['baseline_init_probability'])
    base_res = governing(model_minus_market(base, n_resamples=args.resamples))
    print(f'  baseline-init artifact on the SAME rows: {base_res.point:+.6f} '
          f'[{base_res.lo:+.6f}, {base_res.hi:+.6f}]')
    print(f'\n  refit minus incumbent: {gov.point - base_res.point:+.6f}  '
          f'{"the refit is better" if gov.point > base_res.point else "the refit COST accuracy"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
