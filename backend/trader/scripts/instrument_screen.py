"""Can this instrument pay for its own trading? Answer before building anything.

    python -m scripts.instrument_screen
    python -m scripts.instrument_screen --venue coinbase --horizons 1,4,24
    python -m scripts.instrument_screen --max-required-ic 0.05

This is step one, and in this repo it was step forty. Every conclusion the system
produced reduces to one ratio:

    required IC = round_trip_cost / sigma_h

Cost is fixed per round trip; dispersion grows as sqrt(h). So the bar an instrument
sets is a property of the *venue and the contract*, settled before a model exists —
and it is cheap to measure. Eight months of feature engineering, a search
apparatus, a dashboard and an API were built on CDE nano perps before anyone
divided 27bp by 46bp and got 0.40, against a directional IC this data supports of
about 0.01.

Three columns, per instrument per horizon:

* `required_ic` — cost / sigma_h. What a forecast has to reach to break even.
* `ceiling_win_rate` — P(|move| > cost). The best win rate PERFECT direction can
  reach, which at h=1h is ~53%: on 47% of bars the price does not travel far
  enough to pay the toll, and those are losses whichever way you face. A low
  observed win rate is usually this, not a wrong sign.
* `gap_over_cost` — median close-to-next-open move against the round trip. Pure
  fill uncertainty: a bar's close is its last *trade*, the first fillable price is
  the next open, and nothing in a fee schedule or a signal removes the difference.

The verdict is a gate, not a ranking. An instrument needing an IC no honest
forecast reaches is not a hard problem, it is the wrong instrument — and no
architecture, feature set or horizon fixes a denominator.

Two gates, and they pull opposite ways
--------------------------------------
`required_ic` falls as `1/sqrt(h)` because cost is fixed per round trip. The
effective sample falls as `1/h` because a label spanning h bars overlaps its h-1
neighbours, and the recency decay caps it again at roughly `3 x half_life` of
history however deep the store goes. So a longer hold buys economics and spends
sample, and the only configurations worth modelling are the ones where both pass
at once. Reporting either alone recommends a horizon the other forbids: on this
store h=96h is the *only* horizon clearing the cost gate and it is also the one
CLAUDE.md opens by rejecting for having 18 effective observations. Both are true,
and the window between them is the design question.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import get_contract_spec
from core.cv import average_uniqueness
from core.datastore import ResearchStore
from core.targets import required_information_coefficient, round_trip_cost_series
from scripts._common import add_data_arguments, build_config, configure_logging

# Above this, a forecast has to be better than anything this class of model
# produces on hourly financial data. Not a law — a line drawn from the measured
# ceiling of the literature and of this repo, and stated so a run can argue with
# it rather than silently accept whatever it finds.
DEFAULT_MAX_REQUIRED_IC = 0.05

# Below this the promotion gates cannot distinguish a candidate from noise, and
# the deflated Sharpe discounts what is left by the trial count.
DEFAULT_MIN_EFFECTIVE_OBS = 200.0


def _effective_observations(index: pd.DatetimeIndex, horizon: int,
                           half_life_days: float) -> float:
    """Independent labels after overlap and the recency decay training applies.

    Overlap divides by the horizon; the decay then concentrates weight on recent
    rows, which is why a deeper store stops helping past about three half-lives.
    Both are applied here because reporting the raw row count is how a scrape
    gets recommended that cannot change the fit.
    """
    if len(index) < 2:
        return 0.0
    uniqueness = average_uniqueness(index, horizon)
    if half_life_days <= 0:
        return float(uniqueness.sum())
    age_days = (index.max() - index).total_seconds() / 86_400.0
    decay = np.power(0.5, np.asarray(age_days) / float(half_life_days))
    return float((uniqueness * decay).sum())


def screen(
    store: ResearchStore,
    config: Config,
    *,
    venue: str,
    horizons: list[int],
    min_bars: int = 500,
) -> pd.DataFrame:
    """One row per (instrument, horizon), with the economics and nothing else."""
    bars = store.read('bars', venue=venue)
    if bars.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for symbol, group in bars.groupby('symbol'):
        frame = group.set_index('event_time').sort_index()
        if len(frame) < min_bars or 'open' not in frame:
            continue

        opens, close = frame['open'], frame['close']
        cost = round_trip_cost_series(symbol, close, config)
        median_cost = float(cost.median())
        spec = get_contract_spec(symbol)

        # Fill uncertainty: the move between a bar's last trade and the first
        # price a decision from that bar could actually fill at.
        gap = (opens.shift(-1) / close - 1.0).abs()
        median_gap = float(gap.median())

        for horizon in horizons:
            # Anchored on the open, because that is the price a decision reaches.
            forward = (opens.shift(-(1 + horizon)) / opens.shift(-1) - 1.0)
            sigma = float(forward.std())
            paired = pd.concat([forward.abs(), cost], axis=1).dropna()
            ceiling = (float((paired.iloc[:, 0] > paired.iloc[:, 1]).mean())
                       if not paired.empty else float('nan'))
            rows.append({
                'symbol': symbol.split('-')[0],
                'horizon': horizon,
                'effective_obs': _effective_observations(
                    frame.index, horizon, config.recency_half_life_days),
                'days': round((frame.index[-1] - frame.index[0]).days),
                'notional_per_contract': float(spec.units * close.median()),
                'round_trip_bps': median_cost * 1e4,
                'sigma_bps': sigma * 1e4,
                'required_ic': median_cost / sigma if sigma > 0 else float('nan'),
                'ceiling_win_rate': ceiling,
                'gap_bps': median_gap * 1e4,
                'gap_over_cost': median_gap / median_cost if median_cost > 0 else float('nan'),
            })
    return pd.DataFrame(rows)


def _verdict(frame: pd.DataFrame, limit: float,
             min_obs: float) -> tuple[pd.DataFrame, bool]:
    """Per-horizon summary against both gates, and whether any horizon clears both.

    `effective_obs` is summed across the universe rather than averaged: the panel
    is pooled, so a fold sees every instrument's labels. That is generous —
    pairwise correlation across this book is 0.658, so 18 instruments are nearer
    1.5 independent ones — and it is stated so the number is not mistaken for
    independent breadth.
    """
    grouped = frame.groupby('horizon').agg(
        instruments=('symbol', 'nunique'),
        required_ic_median=('required_ic', 'median'),
        required_ic_best=('required_ic', 'min'),
        ceiling_win_rate=('ceiling_win_rate', 'median'),
        round_trip_bps=('round_trip_bps', 'median'),
        effective_obs=('effective_obs', 'sum'),
    ).reset_index()
    grouped['cost_ok'] = grouped['required_ic_best'] <= limit
    grouped['sample_ok'] = grouped['effective_obs'] >= min_obs
    grouped['passes'] = grouped['cost_ok'] & grouped['sample_ok']
    return grouped, bool(grouped['passes'].any())


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--horizons', default='1,4,24,96',
                        help='Comma-separated holding periods in hours')
    parser.add_argument('--max-required-ic', type=float, default=DEFAULT_MAX_REQUIRED_IC,
                        help='Reject an instrument needing more forecast skill than this')
    parser.add_argument('--min-effective-obs', type=float, default=DEFAULT_MIN_EFFECTIVE_OBS,
                        help='Reject a horizon with fewer effective observations than this')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = build_config(args)
    horizons = [int(h) for h in str(args.horizons).split(',') if h.strip()]
    frame = screen(ResearchStore(args.store), config, venue=args.venue, horizons=horizons)

    if frame.empty:
        print(f'no bars for venue {args.venue!r} — nothing to screen')
        return 1

    print(f'\nfee schedule: {config.cost_config_version} '
          f'({config.fee_pct_per_side * 100:.3f}%/side + '
          f'${config.per_contract_fee_usd:.2f}/contract, '
          f'spread {config.spread_bps:.1f}bp)')

    shown = frame.sort_values(['horizon', 'required_ic'])
    for horizon in horizons:
        block = shown[shown['horizon'] == horizon]
        if block.empty:
            continue
        print(f'\nh={horizon}h')
        print(block[['symbol', 'days', 'round_trip_bps', 'sigma_bps', 'required_ic',
                     'ceiling_win_rate', 'gap_over_cost']]
              .to_string(index=False, float_format=lambda x: f'{x:8.3f}'))

    summary, any_pass = _verdict(frame, args.max_required_ic, args.min_effective_obs)

    # Cross-check the per-instrument table against the single implementation the
    # `ic_covers_cost` promotion gate reads, so the screen and the gate can never
    # disagree about what an instrument requires.
    bars_by_symbol = {
        symbol: group.set_index('event_time').sort_index()
        for symbol, group in ResearchStore(args.store)
        .read('bars', venue=args.venue).groupby('symbol')
    }
    print('\ncross-check against core.targets.required_information_coefficient '
          '(what the promotion gate reads):')
    for horizon in horizons:
        shared = required_information_coefficient(
            bars_by_symbol, config, horizon_bars=horizon)
        mine = float(frame.loc[frame['horizon'] == horizon, 'required_ic'].median())
        flag = 'ok' if abs(shared - mine) < 5e-4 else 'DISAGREE'
        print(f'  h={horizon:>3}h  screen {mine:.4f}  gate {shared:.4f}  {flag}')
    print('\nby horizon')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))

    print(f'\ngates: required IC <= {args.max_required_ic:.3f} '
          f'AND effective observations >= {args.min_effective_obs:.0f} '
          f'(half-life {config.recency_half_life_days:.0f}d, '
          f'{int(frame["days"].max())}d of history)')
    best = frame.loc[frame['required_ic'].idxmin()]
    print(f"cheapest cell: {best['symbol']} at h={int(best['horizon'])}h needs "
          f"IC {best['required_ic']:.3f} ({best['round_trip_bps']:.1f}bp round trip "
          f"against {best['sigma_bps']:.0f}bp of dispersion)")

    if any_pass:
        horizons_ok = ', '.join(f"h={int(r.horizon)}h"
                                for r in summary.itertuples() if r.passes)
        print(f'\nPASS at {horizons_ok} — both gates clear. Those are the '
              f'horizons worth modelling.')
    else:
        cost_only = [int(r.horizon) for r in summary.itertuples()
                     if r.cost_ok and not r.sample_ok]
        sample_only = [int(r.horizon) for r in summary.itertuples()
                       if r.sample_ok and not r.cost_ok]
        print('\nREJECT: no horizon clears both gates.')
        if cost_only:
            need = max(3 * config.recency_half_life_days, 0)
            print(f'  h={cost_only} pay for themselves but have too few '
                  f'effective observations. Sample scales with history up to '
                  f'~3x the half-life ({need:.0f}d) and with the half-life '
                  f'itself — raise both, or accept a shorter hold.')
        if sample_only:
            print(f'  h={sample_only} have the sample but need a forecast this '
                  f'class of model does not produce. No feature set moves a '
                  f'denominator — change the instrument.')

    if args.json:
        print('\n' + json.dumps(frame.to_dict(orient='records'), indent=2, default=str))
    return 0 if any_pass else 2


if __name__ == '__main__':
    raise SystemExit(main())
