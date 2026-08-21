"""Emit live signals through the same decide() the backtest uses.

    python -m scripts.signals --model models/forecast.joblib
    python -m scripts.signals --dry-run

This is the live half of the guarantee that matters: it calls
`core.signal.decide` with forecasts from `core.model`, exactly as
`core.backtest` does. The previous system had three separate implementations of
this logic, and that is why its backtests and its paper trading disagreed.

Signals are written through `core.pg_writer.write_signal`, whose schema the API
and frontend already read, so nothing downstream changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtest import VOL_WINDOW_BARS
from core.model import ForecastModel
from core.signal import DecisionContext, GateCounter, decide
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data

logger = logging.getLogger(__name__)


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--model', default='models/forecast.joblib')
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print decisions without writing to the database')
    args = parser.parse_args()
    configure_logging(args.log_level)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error('no model at %s. Train one: python -m scripts.train', model_path)
        return 1
    model = ForecastModel.load(model_path)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    if model.feature_set_hash != '':
        try:
            model.assert_compatible(dataset.features)
        except ValueError as exc:
            logger.error('model is stale for this feature set: %s', exc)
            return 1
    if model.cost_config_version != config.cost_config_version:
        logger.warning(
            'model was trained under cost config %s but this run uses %s — '
            'the targets it learned were priced differently',
            model.cost_config_version, config.cost_config_version,
        )

    # Decide on the most recent complete bar only. Anything earlier is history,
    # and anything later does not exist yet.
    times = pd.DatetimeIndex(dataset.features.index.get_level_values('event_time'))
    latest = times.max()
    current = dataset.features.xs(latest, level='event_time', drop_level=False)
    if current.empty:
        logger.error('no features at the latest bar %s', latest)
        return 1

    cost = dataset.targets['cost'].reindex(current.index).ffill().fillna(0.0)
    forecasts = model.predict(current, cost=cost.to_numpy())

    counter = GateCounter()
    written = 0
    writer = None
    if not args.dry_run and os.environ.get('DATABASE_URL'):
        from core.pg_writer import PgWriter
        writer = PgWriter()

    # Which promoted version produced these, so a signal can still be attributed
    # after the next retrain — and so calibration is never measured across two
    # different models.
    from core.promotion import current_record

    live = current_record(model_path.parent)
    model_version = live.version if live else None
    if live is None:
        logger.warning(
            'no promotion record beside %s: this model was installed outside the '
            'gates, and its signals cannot be attributed to a version',
            model_path,
        )

    print(f'\nbar {latest} | model {model.feature_set_hash} '
          f'(version {model_version or "unrecorded"}) '
          f'trained through {model.train_end}')

    for (timestamp, symbol), row in forecasts.iterrows():
        bars = dataset.bars.get(symbol)
        if bars is None or timestamp not in bars.index:
            continue
        bar = bars.loc[timestamp]
        volatility = (
            bars['close'].pct_change().rolling(VOL_WINDOW_BARS).std().shift(1).get(timestamp, np.nan)
        )

        decision = decide(
            symbol=symbol, timestamp=timestamp, forecast=row,
            context=DecisionContext(
                equity=args.equity,
                volatility=float(volatility),
                bar_volume=float(bar.get('volume', 0.0)),
                price=float(bar['close']),
                max_positions=config.max_positions,
            ),
            config=config, profile=dataset.profiles.get(symbol), counter=counter,
        )

        marker = 'TRADE' if decision.tradeable else f'skip ({decision.gate.value})'
        print(
            f'  {symbol:5s} {marker:28s} '
            f'net {decision.expected_net * 10_000:+7.1f}bp '
            f'(price {decision.expected_price * 10_000:+7.1f}, '
            f'carry {decision.expected_carry * 10_000:+6.1f}, '
            f'cost {decision.cost * 10_000:5.1f}) '
            f'carry share {decision.carry_share:.0%}'
        )

        if writer is not None:
            profile = dataset.profiles.get(symbol)
            coin = profile.name if profile else symbol
            side = 'long' if decision.side >= 0 else 'short'
            # `confidence` is retained because the frontend and the paper engine
            # read it, and edge-to-risk is the closest honest analogue: how large
            # the expected edge is relative to its own uncertainty. Everything
            # else goes into the columns that mean what they say. The classifier
            # columns — raw_probability, model_auc, momentum_pass, ml_pass — are
            # left null rather than filled with a stand-in, because a fabricated
            # zero reads as a measurement.
            writer.write_signal(
                coin=coin,
                timestamp=datetime.now(timezone.utc),
                direction=side,
                confidence=float(min(max(decision.edge_to_risk, 0.0), 1.0)),
                price_at_signal=decision.price,
                regime_pass=decision.gate is None or decision.gate.value != 'volatility_regime',
                contracts_suggested=decision.contracts or None,
                notional_usd=decision.notional or None,
                passed_gates=decision.tradeable,
                gate_failure_reason=decision.gate.value if decision.gate else None,
                expected_net_bps=decision.expected_net * 10_000,
                expected_price_bps=decision.expected_price * 10_000,
                expected_carry_bps=decision.expected_carry * 10_000,
                cost_bps=decision.cost * 10_000,
                sigma_bps=decision.sigma * 10_000,
                edge_to_risk=decision.edge_to_risk,
                carry_share=decision.carry_share,
                participation=decision.participation,
                model_version=model_version,
                idempotency_key=(
                    f"{coin}_{timestamp.strftime('%Y%m%dT%H%M')}_{side}"
                ),
            )
            written += 1

    print(f'\n{counter}')
    if writer is not None:
        print(f'{written} signals written')
    elif not args.dry_run:
        print('DATABASE_URL not set: nothing written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
