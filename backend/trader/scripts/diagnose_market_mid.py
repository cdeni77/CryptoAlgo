"""Is the backfilled market probability noisier than the market's real forecast?

**Why this exists.** The retroactive forecast test came back PASS at +0.002105,
and with it a claim that should be disbelieved before it is believed: that the
venue's own price is a WORSE forecaster than `F(x/sigma)` at three of four
offsets. A liquid prediction market quoting 1c spreads, beaten by a closed-form
barrier formula.

The suspect is the estimator, not the market. `market_probability` is built as
`(yes_bid.close + yes_ask.close) / 2`, and those are the last bid and the last ask
*within* the minute — not necessarily the same instant. In a fast minute their
midpoint is a mid of two different moments, which is not a price anybody could
trade and is noisier than the real quote.

That matters because **log loss punishes noise**. Measuring the opponent with
error inflates its log loss even when its true forecast is unchanged, so a noisy
estimate of the market hands the model a win it did not earn. A +0.0021 result is
exactly the scale a little measurement noise would produce.

The test: compute several estimators of the same quantity, from the same candles,
and score them all against the same outcomes. If the smoother ones score
materially better, the mid is the problem and the retroactive test is void until
it is fixed. If they do not, the strangeness is real and needs harder scrutiny
than an afternoon.

Read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

import numpy as np
import pandas as pd

from core.baseline import log_loss
from core.config import DEFAULT_CONFIG
from core.dataset import Dataset, load_minute_bars
from core.datastore import ResearchStore
from scripts.backfill_quotes import OFFSETS, Throttle, _field

logger = logging.getLogger('mid')


def estimators(candle: dict) -> dict[str, float]:
    """Several readings of 'the market's probability' from one candle."""
    bid, ask = candle.get('yes_bid') or {}, candle.get('yes_ask') or {}
    price = candle.get('price') or {}

    def g(block, name):
        return _field(block, f'{name}_dollars', name)

    bc, ac = g(bid, 'close'), g(ask, 'close')
    bo, ao = g(bid, 'open'), g(ask, 'open')
    bh, bl = g(bid, 'high'), g(bid, 'low')
    ah, al = g(ask, 'high'), g(ask, 'low')
    out = {
        # what the backfill currently stores
        'mid_close': (bc + ac) / 2.0,
        # the same two sides at the START of the minute
        'mid_open': (bo + ao) / 2.0,
        # average of the two mids: halves any within-minute timing mismatch
        'mid_open_close': ((bc + ac) / 2.0 + (bo + ao) / 2.0) / 2.0,
        # the midpoint of each side's own range, which cannot mix instants
        'mid_of_ranges': ((bh + bl) / 2.0 + (ah + al) / 2.0) / 2.0,
        # traded price, last and volume-weighted mean
        'price_close': g(price, 'close'),
        'price_mean': g(price, 'mean'),
    }
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sample', type=int, default=1500)
    parser.add_argument('--rate', type=float, default=12.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')

    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    quotes = quotes.loc[quotes['offset_minutes'].isin(OFFSETS)]
    markets = (quotes[['symbol', 'window_open', 'market_ticker']]
               .drop_duplicates().sort_values('window_open').reset_index(drop=True))
    step = max(1, len(markets) // args.sample)
    markets = markets.iloc[::step].head(args.sample)
    logger.info('sampling %d markets, evenly spaced', len(markets))

    config = DEFAULT_CONFIG
    lo = (markets['window_open'].min() - pd.Timedelta(days=3)).tz_convert(None)
    hi = (markets['window_open'].max() + pd.Timedelta(hours=1)).tz_convert(None)
    bars = load_minute_bars(config, store=store, start=lo, end=hi)
    windows = Dataset.build(bars, config).windows
    truth = (windows[['symbol', 'window_open', 'outcome']]
             .drop_duplicates(['symbol', 'window_open']).dropna(subset=['outcome']))

    series_of = {'BTC-USD': 'KXBTC15M', 'ETH-USD': 'KXETH15M', 'SOL-USD': 'KXSOL15M'}
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    throttle = Throttle(args.rate)
    rows: list[dict] = []
    async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'], private_key_pem=pem) as c:
        for i, m in enumerate(markets.itertuples(), 1):
            ot = m.window_open.to_pydatetime()
            ots = int(ot.timestamp())
            await throttle.wait()
            try:
                payload = await c._request(  # noqa: SLF001
                    'GET', f'/series/{series_of[m.symbol]}/markets/{m.market_ticker}/candlesticks',
                    params={'start_ts': ots - 60, 'end_ts': ots + 16 * 60,
                            'period_interval': 1})
            except Exception:                      # noqa: BLE001
                continue
            by = {}
            for candle in payload.get('candlesticks', []):
                delta, rem = divmod(int(candle.get('end_period_ts', 0)) - ots, 60)
                if rem == 0:
                    by[delta] = candle
            for offset in OFFSETS:
                candle = by.get(offset)
                if candle is None:
                    continue
                rows.append({'symbol': m.symbol, 'window_open': m.window_open,
                             'offset': offset, **estimators(candle)})
            if i % 300 == 0:
                logger.info('%d/%d markets', i, len(markets))

    frame = pd.DataFrame(rows).merge(truth, on=['symbol', 'window_open'], how='inner')
    frame['outcome'] = frame['outcome'].astype(float)
    names = [k for k in estimators({}) if k in frame.columns]
    print(f'\n{len(frame):,} rows over {frame.drop_duplicates(["symbol","window_open"]).shape[0]:,} '
          f'symbol-windows\n')
    print(f"{'estimator':>16}" + ''.join(f'{f"+{o}m":>10}' for o in OFFSETS) + f"{'pooled':>10}{'usable':>9}")
    base = None
    for name in names:
        cells, pooled_ok = [], []
        for offset in OFFSETS:
            part = frame.loc[(frame['offset'] == offset) & np.isfinite(frame[name])
                             & (frame[name] > 0) & (frame[name] < 1)]
            cells.append(log_loss(part['outcome'].to_numpy(dtype=float),
                                  part[name].to_numpy(dtype=float)) if len(part) else float('nan'))
            pooled_ok.append(part)
        allrows = pd.concat(pooled_ok)
        pooled = log_loss(allrows['outcome'].to_numpy(dtype=float),
                          allrows[name].to_numpy(dtype=float)) if len(allrows) else float('nan')
        if base is None:
            base = pooled
        print(f'{name:>16}' + ''.join(f'{c:>10.5f}' for c in cells)
              + f'{pooled:>10.5f}{100*len(allrows)/len(frame):>8.1f}%'
              + ('' if name == 'mid_close' else f'   {base - pooled:+.5f} vs mid_close'))
    print('\n  A materially LOWER pooled log loss for a smoother estimator means the')
    print('  stored mid is noise, not the market — and the retroactive test is void')
    print('  until the backfill stores a better one.')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
