"""Shared fixtures. The synthetic bar generator is the important one.

Almost every test here needs one-minute bars, and building them by hand invites
each test to invent a slightly different convention. `synthetic_bars` produces
one generator with known properties — a real intraday volatility shape, genuine
volatility clustering, and an optional lead-lag from Bitcoin to the other two —
so a test that plants a mechanism can assert the pipeline finds it, and a test
that plants nothing can assert it finds nothing.

That pairing is the point. A suite that only checks a model can find signal
cannot catch a pipeline that manufactures it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='session')
def repo_root() -> Path:
    """Where `configs/` lives.

    Inside the trader package, not the repository root: the trader's Docker build
    context is `backend/trader`, so anything above it is never copied into the
    image.
    """
    return ROOT


def make_bars(
    *,
    days: float = 20,
    seed: int = 5,
    lead: float = 0.0,
    symbols: tuple[str, ...] = ('BTC-USD', 'ETH-USD', 'SOL-USD'),
    close_noise: float = 0.0,
    start: str = '2025-01-01',
) -> dict[str, pd.DataFrame]:
    """One-minute bars with a known generating process.

    `lead` moves that fraction of Bitcoin's move into the *next* minute for the
    other symbols, which is the lead-lag the `cross_asset` group exists to find.
    `close_noise` adds mean-reverting noise to the close only, leaving the open
    clean — the bid-ask bounce the `microstructure` group exists to find, and the
    reason a close-anchored target manufactures edge.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 1440)
    times = pd.date_range(start, periods=n, freq='1min', tz='UTC')
    minute_of_day = np.arange(n) % 1440
    seasonal = 1.0 + 0.5 * np.sin(2 * np.pi * (minute_of_day + 300) / 1440)
    cluster = np.exp(
        pd.Series(rng.normal(0, 0.05, n)).ewm(halflife=720).mean().to_numpy() * 6)
    sigma = 1.4e-4 * seasonal * cluster
    lead_source = rng.normal(0, sigma)

    out: dict[str, pd.DataFrame] = {}
    spec = {'BTC-USD': (1.0, 0.0, 60_000.0), 'ETH-USD': (1.1, 0.6, 2_400.0),
            'SOL-USD': (1.3, 0.9, 150.0)}
    for symbol in symbols:
        beta, extra, price0 = spec.get(symbol, (1.0, 0.5, 100.0))
        if symbol == 'BTC-USD':
            returns = lead_source
        else:
            returns = beta * ((1 - lead) * lead_source + lead * np.roll(lead_source, 1))
            returns = returns + rng.normal(0, extra * sigma)
        price = price0 * np.exp(np.cumsum(returns))
        close = price * (1 + rng.normal(0, sigma * close_noise)) if close_noise else price
        high = np.maximum(price, close) * (1 + np.abs(rng.normal(0, sigma)))
        low = np.minimum(price, close) * (1 - np.abs(rng.normal(0, sigma)))
        out[symbol] = pd.DataFrame({
            'event_time': times, 'open': price, 'high': high, 'low': low,
            'close': close, 'volume': np.exp(rng.normal(0, 0.4, n)) * seasonal,
            # Both NULL, as the real store has them: Coinbase's candles endpoint
            # returns OHLCV and nothing else. This fixture used to synthesise a
            # plausible `trade_count`, which is how a feature that can never fire
            # on real data passed `test_every_declared_feature_is_produced`.
            'quote_volume': np.nan,
            'trade_count': np.nan,
        })
    return out


@pytest.fixture
def synthetic_bars():
    """The generator itself, so a test can choose its own mechanism."""
    return make_bars


@pytest.fixture(scope='module')
def clean_bars() -> dict[str, pd.DataFrame]:
    """Twenty days, three symbols, no planted mechanism."""
    return make_bars(days=20)
