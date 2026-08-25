"""The retroactive forecast test, exactly as DECISION_RULE.md Appendix A/A.1 fixes it.

Scores the promoted artifact over every backfilled window and compares it against
the venue's own quote at the same instant. Runs nothing that is not pre-registered
and decides nothing that was not decided before the data existed.

The artifact was **force-promoted** (`forced: true, passed: false`, failing
`sharpe_implausible`, reason "live smoke test; edge not established"). So a
negative result confirms its own promotion record and is weak news, while a
positive one is surprising enough that the first hypothesis is a bug in here. The
checklist is run and printed before the headline number, not after it.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.config import DEFAULT_CONFIG
from core.dataset import (Dataset, FoldFit, apply_fold, apply_seasonality,
                          load_minute_bars)
from core.datastore import ResearchStore
from core.inference import governing, model_minus_market
from core.model import ForecastModel

logger = logging.getLogger('retro')

# The model's own demonstrated out-of-sample skill over F(x/sigma). An edge over
# the MARKET smaller than the edge it already showed over the BASELINE is not
# worth acting on. Fixed in Appendix A.1 before any number existed.
MATERIAL = 0.001
CONTROL_OFFSET = 14


def branch(result) -> str:
    """The pre-registered four-way outcome. Appendix A.1, decided in advance."""
    if result.lo > 0:
        return 'PASS — lower bound above 0. Apply the bug checklist before believing it.'
    if result.hi < 0:
        return 'FAIL — the market is the better forecaster, and it is established.'
    if result.hi < MATERIAL:
        return f'NO EDGE, resolved — upper bound {result.hi:+.6f} is under {MATERIAL}.'
    return ('CANNOT RESOLVE — the interval spans 0 and still admits an edge of at '
            f'least {MATERIAL}. The forward test in the body runs to term; nothing '
            'about the strategy changes on this.')


def score_artifact(model: ForecastModel, config, quotes: pd.DataFrame) -> pd.DataFrame:
    """Run the deployed artifact over the windows the backfill covers.

    The artifact carries `ScoringBundle` — seasonality, volatility models and the
    baseline, all fitted through ~2025-12-05. The series began 2026-06-17, so
    every window here is unseen by it. Rebuilding a `FoldFit` from the bundle is
    what makes this the *deployed* model rather than a refit one.
    """
    lo = quotes['window_open'].min() - pd.Timedelta(days=3)
    hi = quotes['window_open'].max() + pd.Timedelta(hours=1)
    bars = load_minute_bars(config, store=ResearchStore(), start=lo, end=hi)
    dataset = Dataset.build(bars, config)

    bundle = model.scoring
    states = {s: apply_seasonality(dataset.states[s], bundle.seasonality[s])
              for s in dataset.states if s in bundle.seasonality}
    fit = FoldFit(seasonality=bundle.seasonality, vol_models=bundle.vol_models,
                  baseline=bundle.baseline, train_windows=model.n_train_windows,
                  states=states)
    scored = apply_fold(dataset, fit, dataset.window_index, config,
                        groups=model.groups or None)
    scored['model_probability'] = model.predict(scored)
    return scored


def checklist(joined: pd.DataFrame) -> bool:
    """Appendix A: what a bug looks like versus what a result looks like."""
    print('\nBUG CHECKLIST (Appendix A) — absolute log loss by offset')
    print(f"{'offset':>7}{'n':>9}{'market_ll':>12}{'model_ll':>11}{'base_ll':>10}"
          f"{'diff':>11}{'base rate':>11}")
    ok = True
    for offset, part in joined.groupby('offset_minutes'):
        y = part['outcome'].to_numpy(dtype=float)
        mk = log_loss(y, part['market_probability'].to_numpy(dtype=float))
        md = log_loss(y, part['model_probability'].to_numpy(dtype=float))
        bs = log_loss(y, part['baseline_probability'].to_numpy(dtype=float))
        print(f'{int(offset):>7}{len(part):>9,}{mk:>12.5f}{md:>11.5f}{bs:>10.5f}'
              f'{mk-md:>+11.5f}{y.mean():>11.4f}')
        # (3) the market must beat a coin flip and beat F(x/sigma)
        if mk >= 0.69315:
            print(f'    FAIL: market_ll {mk:.5f} is no better than a coin flip at '
                  f'+{int(offset)}m — outcomes may be inverted'); ok = False
    for symbol, part in joined.groupby('symbol'):
        rate = part['outcome'].mean()
        if not 0.44 < rate < 0.56:
            print(f'    FAIL: {symbol} base rate {rate:.4f} is far from 0.50'); ok = False
    return ok


def control(model, config, quotes, joined) -> None:
    """(2) A deliberately wrong offset must lose badly, and market_ll at +14m must
    be low but BOUNDED AWAY FROM ZERO. Near 0.00 means the candle contains the
    settlement — the off-by-one this control is hunting, which the difference
    alone would hide because both forecasters share the leaked outcome."""
    ctrl = quotes.loc[quotes['offset_minutes'] == CONTROL_OFFSET]
    if ctrl.empty:
        print(f'\nCONTROL: no +{CONTROL_OFFSET}m quotes were backfilled; '
              f'run the backfill with that offset to exercise it')
        return
    merged = joined.merge(
        ctrl[['symbol', 'window_open', 'market_probability']].rename(
            columns={'market_probability': 'ctrl_market'}),
        on=['symbol', 'window_open'], how='inner')
    if merged.empty:
        print(f'\nCONTROL: no overlap at +{CONTROL_OFFSET}m'); return
    y = merged['outcome'].to_numpy(dtype=float)
    mk = log_loss(y, merged['ctrl_market'].to_numpy(dtype=float))
    md = log_loss(y, merged['model_probability'].to_numpy(dtype=float))
    print(f'\nCONTROL at +{CONTROL_OFFSET}m ({len(merged):,} rows): '
          f'market_ll {mk:.5f}, model_ll {md:.5f}, diff {mk-md:+.5f}')
    if mk < 0.01:
        print(f'    FAIL: market_ll {mk:.5f} is essentially zero — the candle at '
              f'+{CONTROL_OFFSET}m probably contains the settlement')
    elif mk - md > 0:
        print('    FAIL: the model beat the market on a nearly-settled window; '
              'the offset pairing is broken')
    else:
        print('    ok: the market wins decisively on a nearly-settled window')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=None)
    parser.add_argument('--resamples', type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')
    config = DEFAULT_CONFIG
    store = ResearchStore(args.store or os.getenv('RESEARCH_STORE'))

    quotes = store.read('venue_quotes')
    if quotes.empty:
        print('no venue_quotes; run scripts.backfill_quotes first'); return 1
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    quotes['usable'] = quotes['usable'].astype(bool)
    usable = quotes.loc[quotes['usable']].copy()
    print(f'{len(quotes):,} quote rows, {len(usable):,} usable '
          f'({100*len(usable)/len(quotes):.2f}%), '
          f'{quotes["window_open"].dt.floor("D").nunique()} UTC days')

    model = ForecastModel.load(None, config)
    print(f'artifact: {len(model.features)} features, alpha {model.residual_scale:.4f}, '
          f'trained on {model.n_train_windows:,} windows')
    scored = score_artifact(model, config, usable)

    joined = scored.merge(
        usable[['symbol', 'window_open', 'offset_minutes', 'market_probability']],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    joined = joined.loc[
        np.isfinite(joined['model_probability'])
        & np.isfinite(joined['market_probability'])
        & joined['outcome'].isin([0, 1, True, False])].copy()
    joined['outcome'] = joined['outcome'].astype(float)
    print(f'joined: {len(joined):,} rows over '
          f'{joined.drop_duplicates(["symbol","window_open"]).shape[0]:,} symbol-windows')

    clean = checklist(joined)
    control(model, config, usable, joined)
    if not clean:
        print('\nchecklist failed — not reporting a headline number')
        return 2

    print('\n' + '=' * 74)
    print('RETROACTIVE FORECAST TEST — model_minus_market')
    print('=' * 74)
    results = model_minus_market(joined, n_resamples=args.resamples)
    for length in sorted(results):
        print(results[length].line())
    gov = governing(results)
    print(f'\ngoverning (most conservative): block {gov.block_days}d')
    print(f'  point {gov.point:+.6f}  90% CI [{gov.lo:+.6f}, {gov.hi:+.6f}]  '
          f'p = {gov.p_value:.4f}')
    print(f'\n  {branch(gov)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
