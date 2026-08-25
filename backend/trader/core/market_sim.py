"""A synthetic market price, fitted to what the venue actually does.

**What this replaces.** `core/backtest.py` has no order book, so `price_source`
is `'baseline'` — it charges itself `F(x/sigma)`. That is not a small
simplification, it is the reason the backtest's money numbers cannot be believed:
`decide()` computes `edge = model_probability - effective_cost`, so when the cost
is derived from the baseline the edge IS the model's own correction, which is
exactly the quantity the model was fitted to produce. It is positive by
construction. The 5.61 Sharpe that failed `sharpe_implausible` is what a
self-referential trade looks like.

**What the venue actually does**, measured on 77,349 rows carrying both a baseline
and a real quote, over 2026-06-17 to 2026-08-25:

    mean |market - baseline|   3.26pp
    mean  (market - baseline)  -0.00pp     <- no bias worth modelling
    sd                          4.80pp     <- the whole story is dispersion
    correlation                 0.9882
    median spread               1.00c      <- the 0.5c half-spread was right

So the backtest is not mispricing on average. It is pricing with **zero
dispersion**, and a trade selected on `model - price` is enormously sensitive to
that: at 4.8pp of scatter against a model correction of ~1pp, the sign of the
edge is mostly the market's deviation, not the model's opinion.

The level effect is real but small — the market sits above the baseline in the
0.64-0.76 band by ~0.8pp and below it in the 0.23-0.44 band by ~0.7pp, an S-shape
worth carrying but not worth much on its own.

**DO NOT USE THIS. It fails its own control and it is kept as evidence.**

Replayed against the same rows it was fitted on, beside the real quotes:

    real Kalshi quotes   14,106 trades  win 0.465  +1.83pp  Sharpe  9.07
    simulated book       16,209 trades  win 0.506  +5.49pp  Sharpe 30.23
    invented (baseline)   2,975 trades  win 0.441  +1.66pp  Sharpe  3.25

It should reproduce the real book and instead makes the strategy three times
better. The reason is the whole lesson: this samples the deviation as
**independent noise** around the baseline, and a real market's disagreement with
`F(x/sigma)` is **informative** — correlated with the outcome. Noise is not. A
counterparty who deviates randomly is one you can always beat, because you select
the rows where the noise fell your way and call it edge. Simulating an opponent
with noise hands you a fake win in exactly the way that *measuring* one with noise
does.

Making it usable means matching the market's ACCURACY, not only its scatter: the
simulated quote's log loss has to land on the real one's (0.4489 pooled), which
needs the joint distribution of (market, outcome | baseline) rather than the
marginal of (market - baseline) that this fits. That is a project, and it wants
the 20,000-window sample first.

One thing the comparison did settle, against what I had assumed: the invented
`price = baseline` book does not FLATTER the strategy. It takes 2,975 trades
where the real book takes 14,106, because a price pinned to the baseline almost
never differs from the model by enough to clear the gate. The backtest was hiding
four fifths of the opportunity set, not inflating the edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_BUCKETS = 10
MIN_ROWS_PER_CELL = 200


@dataclass
class MarketSimulator:
    """Samples a plausible venue price given a baseline probability."""

    edges: np.ndarray                      # bucket boundaries on baseline_probability
    bias: dict[tuple[int, int], float] = field(default_factory=dict)
    scatter: dict[tuple[int, int], float] = field(default_factory=dict)
    spreads: np.ndarray = field(default_factory=lambda: np.array([0.01]))
    fallback_bias: float = 0.0
    fallback_scatter: float = 0.048
    n_rows: int = 0

    @classmethod
    def fit(cls, frame: pd.DataFrame, *, baseline_column: str = 'baseline_probability',
            market_column: str = 'market_probability',
            offset_column: str = 'offset', spread_column: str = 'spread'
            ) -> 'MarketSimulator':
        """Learn the deviation of the real quote from the baseline.

        Conditioned on (offset, baseline decile): the market's disagreement is
        wider early in a window than late, and has a mild level dependence.
        """
        rows = frame.dropna(subset=[baseline_column, market_column])
        gap = (rows[market_column] - rows[baseline_column]).to_numpy(dtype=float)
        base = rows[baseline_column].to_numpy(dtype=float)
        edges = np.quantile(base, np.linspace(0, 1, N_BUCKETS + 1))
        edges[0], edges[-1] = 0.0, 1.0
        bucket = np.clip(np.searchsorted(edges, base, side='right') - 1, 0, N_BUCKETS - 1)
        offsets = rows[offset_column].to_numpy(dtype=int)

        bias, scatter = {}, {}
        for off in np.unique(offsets):
            for b in range(N_BUCKETS):
                cell = (offsets == off) & (bucket == b)
                if cell.sum() < MIN_ROWS_PER_CELL:
                    continue
                bias[(int(off), b)] = float(np.mean(gap[cell]))
                scatter[(int(off), b)] = float(np.std(gap[cell], ddof=1))
        spreads = (rows[spread_column].to_numpy(dtype=float)
                   if spread_column in rows.columns else np.array([0.01]))
        spreads = spreads[np.isfinite(spreads) & (spreads > 0)]
        return cls(edges=edges, bias=bias, scatter=scatter,
                   spreads=spreads if len(spreads) else np.array([0.01]),
                   fallback_bias=float(np.mean(gap)),
                   fallback_scatter=float(np.std(gap, ddof=1)),
                   n_rows=len(rows))

    def sample(self, baseline: np.ndarray, offsets: np.ndarray, *,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """A plausible mid and half-spread for each row.

        Returns `(mid, half_spread)`. The mid is the venue's belief; what a trade
        costs is `mid + half_spread` on the side being bought, which is the
        caller's business.
        """
        base = np.asarray(baseline, dtype=float)
        offs = np.asarray(offsets, dtype=int)
        bucket = np.clip(np.searchsorted(self.edges, base, side='right') - 1,
                         0, N_BUCKETS - 1)
        mu = np.full(len(base), self.fallback_bias)
        sd = np.full(len(base), self.fallback_scatter)
        for i, (off, b) in enumerate(zip(offs, bucket)):
            key = (int(off), int(b))
            if key in self.bias:
                mu[i], sd[i] = self.bias[key], self.scatter[key]
        mid = np.clip(base + rng.normal(mu, np.maximum(sd, 1e-6)), 0.005, 0.995)
        half = rng.choice(self.spreads, size=len(base)) / 2.0
        return mid, half

    def summary(self) -> str:
        cells = sorted(self.scatter)
        widest = max(self.scatter.values()) if self.scatter else float('nan')
        return (f'market simulator: {self.n_rows:,} rows, {len(cells)} (offset, '
                f'decile) cells | mean bias {100*self.fallback_bias:+.2f}pp | '
                f'scatter {100*self.fallback_scatter:.2f}pp (widest cell '
                f'{100*widest:.2f}pp) | median spread '
                f'{100*float(np.median(self.spreads)):.2f}c')
