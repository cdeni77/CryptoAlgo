"""Synthetic `venue_depth` for the stacking and quote-filter tests.

The "market" is the barrier probability from each row's own displacement at a
fixed sigma, plus logit noise. It cannot see the other symbols, so when the bars
carry a planted BTC lead (`make_bars(lead=...)`) a spot model reading
`cross_asset` knows something the price does not — which is what stage 2's `b`
exists to measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.config import Config
from core.windows import build_window_panel

MARKET_SIGMA = 1.4e-4


def make_depth(bars: dict, config: Config, *, quoted_days: float,
               source: str = 'live_touch', noise: float = 0.15,
               age: float = np.nan, seed: int = 0) -> pd.DataFrame:
    """Kalshi rows for every window in the last `quoted_days` of `bars`."""
    windows, _ = build_window_panel(bars, config)
    cutoff = windows['window_open'].max() - pd.Timedelta(days=quoted_days)
    w = windows.loc[windows['window_open'] >= cutoff].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    remaining = np.array([config.remaining_variance_minutes(o) for o in w['offset']])
    z = w['displacement'].to_numpy(dtype=float) / (MARKET_SIGMA * np.sqrt(remaining))
    p = norm.cdf(z)
    lp = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    p = 1.0 / (1.0 + np.exp(-(lp + rng.normal(0, noise, len(w)))))
    p = np.clip(p, 0.02, 0.98)
    event = pd.to_datetime(w['window_open'], utc=True) + pd.to_timedelta(w['offset'], unit='m')
    return pd.DataFrame({
        'venue': 'kalshi', 'symbol': w['symbol'], 'window_open': w['window_open'],
        'offset_minutes': w['offset'], 'event_time': event,
        'available_time': event + pd.Timedelta(seconds=5),
        'yes_bid': p - 0.005, 'yes_ask': p + 0.005,
        'yes_bid_size': 100.0, 'yes_ask_size': 100.0,
        'quote_age_seconds': age, 'source': source,
    })
