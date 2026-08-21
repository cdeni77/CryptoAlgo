"""Evaluate a candidate model and install it only if the gates pass.

    python -m scripts.promote                  # evaluate, promote if it clears
    python -m scripts.promote --evaluate-only  # build the case, install nothing
    python -m scripts.promote --history        # what has been tried, and why not
    python -m scripts.promote --force 'reason' # override, recorded in the ledger

This is the only path a model takes to live. `scripts.train` writes an artifact
for inspection; nothing that has not been through here is scored against real
prices, because the gates and the artifact are written together and the live
signal writer reads the promoted one.

Every evaluation lands in `models/promotions/`, rejections included. The count of
attempts is what the deflated Sharpe discounts by, so a directory containing only
successes would flatter every survivor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.model import MODELS_DIR
from core.promotion import (
    current_record,
    evaluate_candidate,
    load_records,
    promote,
    report,
    trials_to_date,
)
from scripts._common import add_data_arguments, build_config, configure_logging, load, require_data


def _print_history(models_dir: Path, limit: int) -> int:
    records = load_records(models_dir, limit=limit)
    live = current_record(models_dir)

    if not records:
        print('no promotion history. Evaluate a candidate: python -m scripts.promote')
        return 0

    print(f'\n{len(records)} evaluation(s), newest first '
          f'(live: {live.version if live else "none"})\n')
    for record in records:
        marker = '*' if live and record.version == live.version else ' '
        verdict = 'promoted' if record.promoted else 'blocked'
        if record.forced:
            verdict += ' (forced)'
        sharpe = record.backtest.get('sharpe')
        trades = record.backtest.get('trades')
        detail = f'Sharpe {sharpe:+.2f} on {trades} trades' if sharpe is not None else 'no result'
        print(f'{marker} {record.version}  {verdict:18s} {detail}')
        if record.failed_gates:
            print(f'    failed: {", ".join(record.failed_gates)}')
        if record.error:
            print(f'    error: {record.error}')
    return 0


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--models-dir', default=str(MODELS_DIR))
    parser.add_argument('--periods', type=int, default=6, help='Walk-forward retrains')
    parser.add_argument('--equity', type=float, default=100_000.0)
    parser.add_argument('--spread-bps', type=float, default=4.0)
    parser.add_argument('--synthetic-paths', type=int, default=20)
    parser.add_argument('--quick', action='store_true',
                        help='Skip synthetic panels and cost stress. Cannot promote: '
                             'a skipped gate fails.')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Build the case and record it without installing')
    parser.add_argument('--force', default=None, metavar='REASON',
                        help='Install a blocked candidate, recording the reason')
    parser.add_argument('--history', action='store_true', help='Print past evaluations')
    parser.add_argument('--limit', type=int, default=20, help='History entries to show')
    parser.add_argument('--json', action='store_true', help='Emit the record as JSON')
    args = parser.parse_args()
    configure_logging(args.log_level)

    models_dir = Path(args.models_dir)
    if args.history:
        return _print_history(models_dir, args.limit)

    config = build_config(args)
    dataset = load(args, config)
    if not require_data(dataset, args.venue):
        return 1

    print(f'\ndataset: {dataset}')
    model, record = evaluate_candidate(
        dataset, config,
        n_periods=args.periods, initial_equity=args.equity,
        spread_bps=args.spread_bps, synthetic_paths=args.synthetic_paths,
        full=not args.quick, data_as_of=args.as_of,
        # Every candidate ever evaluated counts, including this one. The deflated
        # Sharpe discounts by this number, which is the whole reason rejections
        # stay in the ledger.
        trials=trials_to_date(models_dir) + 1,
    )

    if model is None:
        print(f'\n{record}')
        return 1

    print('\nprovenance:')
    for key, value in record.provenance.items():
        print(f'  {key}: {value}')
    print(f"\nwalk-forward: {json.dumps(record.backtest.get('trades'))} trades, "
          f"Sharpe {record.backtest.get('sharpe')}")
    print(f'\n{report(record)}')

    if args.evaluate_only:
        from core.promotion import write_record
        path = write_record(record, models_dir)
        print(f'\nrecorded {path} (--evaluate-only: nothing installed)')
        return 0 if record.promoted else 2

    installed, record = promote(
        model, record, models_dir=models_dir,
        force=args.force is not None, force_reason=args.force,
    )
    if args.json:
        print(json.dumps(record.as_dict(), indent=2, default=str))
    print(f'\n{record}')
    return 0 if installed else 2


if __name__ == '__main__':
    sys.exit(main())
