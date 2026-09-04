"""Evaluate a candidate and install it, gates permitting.

The only path to `models/forecast.joblib`. The live signal writer loads that
file by name and nothing else, so this is the single place a model becomes real.

Every attempt is recorded in `models/promotions/`, passed or blocked. That
ledger is the trial count, and the trial count is what any claim of skill has
to be discounted by — a project that deletes its failures cannot compute its own
multiple-testing correction.

    python -m scripts.promote
    python -m scripts.promote --history
    python -m scripts.promote --force --reason "skill is on the >0.9 tail; the
        average forecast is flat and the gates read averages"
"""

from __future__ import annotations

import argparse
import logging
import math
import os

import pandas as pd

from core.backtest import walk_forward
from core.metrics import evaluate_gates, gate_report
from core.promotion import history, load_live, promote, trial_count
from scripts._common import (
    add_data_arguments, config_from_args, groups_from_args, load_dataset, print_header,
    setup_logging,
)


logger = logging.getLogger('promote')


def market_measurement(scored=None, *, entry_offsets=None) -> dict[str, float]:
    """The market comparison, from recorded quotes.

    Returns values that FAIL when there is nothing to read. The gate has to be
    unmeasurable-and-failing rather than absent, because absent is how a
    benchmark quietly stops being applied — and this particular benchmark is the
    one the whole stack exists to satisfy.

    **Two sources now, in that order of preference.** Live-recorded predictions
    in the serving store are the original and remain first: they are the venue's
    quote at the instant a decision was actually made. Failing that, a backtest
    whose rows carry `market_probability` from the collected book — which for
    most of this project's life did not exist, and is why `market_gate_values`
    still says the comparison "cannot come from the backtest". Eight months of
    book have since been collected and validated to 0.70c against the live
    recording with the clock removed, so it can.

    They are not the same claim and the print says which was used. A backfilled
    quote is a reconstruction; a recorded one is an observation.
    """
    from core.metrics import market_gate_values, market_rows_from_scored

    url = os.getenv('DATABASE_URL')
    def _from_backtest(why: str) -> dict[str, float]:
        rows = (market_rows_from_scored(scored, entry_offsets=entry_offsets)
                if scored is not None else [])
        if not rows:
            print(f'  {why} and the backtest carries no recorded quotes, so the '
                  f'market gates fail as unmeasured. That is not the same as the '
                  f'model losing to the price, and the gate line says so.')
            return market_gate_values([])
        values = market_gate_values(rows)
        print(f'  {why}; using BACKFILLED quotes from the backtest: '
              f'{int(values["market_windows"]):,} windows, model_minus_market '
              f'{values["model_minus_market"]:+.6f}')
        print('  (a reconstruction, not an observation — validated to 0.70c '
              'against the live recording)')
        return values

    if not url:
        return _from_backtest('DATABASE_URL is unset')
    try:
        from core.pg_writer import PgWriter

        rows = PgWriter(database_url=url).scored_against_market()
    except Exception as exc:                      # noqa: BLE001 - report and fail closed
        return _from_backtest(f'the serving store could not be read ({exc})')
    rows = list(rows)
    # Same restriction as the backtest path: only offsets that can OPEN a
    # position. `scored_against_market` returns every scored row, and the loop
    # records all four offsets while trading one, so pooling here measured a
    # policy nobody runs — measured, +12m reads +0.00550 against a pooled
    # -0.00072. Row shape is (symbol, window_open, offset_minutes, ...).
    if entry_offsets is not None:
        wanted = {int(o) for o in entry_offsets}
        rows = [r for r in rows if int(r[2]) in wanted]
    if not rows:
        return _from_backtest('the serving store holds no scored quotes')
    values = market_gate_values(rows)
    print(f'  market comparison: {int(values["market_windows"]):,} windows of '
          f'recorded quotes, model_minus_market '
          f'{values["model_minus_market"]:+.6f}')
    return values


logger = logging.getLogger('promote')


def refit_on_all(dataset, config, *, groups=None):
    """Fit the same configuration on every window in the dataset.

    Returns None rather than raising if it cannot be built: a failed refit
    should fall back visibly to the last fold, not abort a promotion whose
    evidence is already computed.
    """
    try:
        from core.dataset import fit_fold
        from core.model import fit_model
        fit, table = fit_fold(dataset, dataset.window_index, config, groups=groups)
        # `scoring=` is not optional. It carries the volatility model, the
        # intraday seasonality factor and the baseline's scale/tail — the three
        # fitted things that turn a raw window into a feature row. Every fold
        # model gets one from `core/backtest.py`; the refit did not, so the
        # artifact this function returns is the one actually installed and was
        # the only one in the pipeline that could not score.
        return fit_model(table, fit.baseline, config, groups=groups,
                         scoring=fit.bundle(config))
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('refit on all data failed (%s); falling back',
                       str(exc)[:120])
        return None


def choose_candidate(fold_models, refit):
    """The artifact to deploy: the refit on all data, never the last fold's.

    A fold model's training ends where that fold's TEST block begins, so
    deploying it ships something one test block stale BY CONSTRUCTION. Measured
    on the artifact that traded live: trained through 2025-12-05, deployed in
    August, and it had therefore never seen a single Kalshi 15-minute market
    because they did not exist yet.

    The walk-forward is the EVIDENCE that a configuration works; what ships is
    that configuration refitted through the present. Conflating the two costs a
    test block of freshness every single time.

    A refit that is not strictly fresher than the last fold means something went
    wrong building it, and shipping it silently would hide that.
    """
    if not fold_models:
        raise ValueError('no models to promote: the walk-forward produced none')
    if refit is None:
        logger.warning(
            'no refit model; falling back to the LAST FOLD, which is stale by '
            'one test block. This is the artifact staleness that promotion is '
            'meant to avoid, so treat it as a defect rather than a default.')
        return fold_models[-1]
    last = getattr(fold_models[-1], 'n_train_windows', 0) or 0
    fresh = getattr(refit, 'n_train_windows', 0) or 0
    if fresh <= last:
        raise ValueError(
            f'the refit saw {fresh:,} training windows against the last fold\'s '
            f'{last:,}; it is not fresher, so it was not built on all the data')
    return refit


def main() -> int:
    parser = add_data_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__))
    parser.add_argument('--history', action='store_true',
                        help='What has been tried, and why not. Then exit.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Evaluate and score the gates, install nothing.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--reason', type=str, default=None,
                        help='Required with --force, and stored with the artifact.')
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.history:
        frame = history()
        if frame.empty:
            print('no promotion attempts recorded')
            return 0
        pd.set_option('display.width', 200, 'display.max_colwidth', 60)
        print(frame.to_string(index=False))
        print(f'\n{len(frame)} attempts, {int(frame["installed"].sum())} installed. '
              f'Any claim of skill discounts by the trial count.')
        return 0

    if args.force and not args.reason:
        raise SystemExit(
            '--force needs --reason. The one good argument for overriding these '
            'gates — skill on a high-conviction tail that the average forecast '
            'does not show — is also the argument that kept a losing system '
            'alive, so it has to be written down.')

    config = config_from_args(args)
    groups = groups_from_args(args)
    print_header('Promotion', config)
    print(f'  attempts so far: {trial_count()}')
    live = load_live()
    if live is not None:
        print(f'  currently live: alpha={live.residual_scale:.3f}, '
              f'{len(live.features)} features, trained on '
              f'{live.n_train_windows:,} windows')
    print()

    dataset = load_dataset(args, config)
    result = walk_forward(dataset, config, groups=groups, trade=True)
    print('\n' + result.report.summary())

    # Score the gate on the offsets that can actually OPEN a position.
    # Pooling all four rejected two candidates for losing to the market at
    # offsets they never trade, while +12m — the only one they do — was the
    # best of the four (+0.00550 against a pooled -0.00072).
    market = market_measurement(getattr(result, 'scored', None),
                                entry_offsets=config.entry_offsets)

    if args.dry_run:
        print('\n' + gate_report(evaluate_gates(result.report, extra=market)))
        print('\ndry run: nothing installed, nothing recorded')
        return 0

    # Refit on EVERYTHING through the present, and deploy that. The folds are
    # the evidence that this configuration works; they are not the artifact.
    # See `choose_candidate`.
    refit = refit_on_all(dataset, config, groups=groups)
    candidate = choose_candidate(result.models, refit)
    if refit is not None:
        print(f'  deploying a refit on {refit.n_train_windows:,} windows, '
              f'against the last fold\'s {result.models[-1].n_train_windows:,}')
    attempt = promote(candidate, result.report, force=args.force,
                      force_reason=args.reason, trades=result.trades(),
                      extra=market)
    print('\n' + attempt.summary())
    publish_to_serving(attempt, result.report)
    return 0 if attempt.installed else 1


def publish_to_serving(attempt, report) -> None:
    """Mirror the attempt and its reliability table into the serving store.

    `PgWriter.record_model_run` and `write_calibration` are the only writers of
    the `model_runs` and `calibration` tables and had **zero callers anywhere**,
    including tests — while `backend/api/controllers/serving.py` reads exclusively
    from them and the dashboard's Model and Calibration pages poll those routes.
    So those two tabs were empty forever regardless of real promotion activity,
    and reading the filesystem ledger instead is not a workaround: the `backend`
    compose service never mounts the `trader_models` volume.

    Best effort on purpose. Promotion must not depend on a database being
    reachable — the ledger under `models/promotions/` is the record of account,
    and this is the dashboard's copy of it. A failure here is logged and the exit
    status still reflects the gates.
    """
    if not os.getenv('DATABASE_URL'):
        logger.info('DATABASE_URL is unset, so the dashboard will not show this '
                    'attempt; the ledger under models/promotions/ still has it')
        return
    try:
        from core.pg_writer import PgWriter

        writer = PgWriter()
        gates = {g.name: g.value for g in attempt.gates}
        writer.record_model_run(
            version=attempt.version,
            installed=bool(attempt.installed),
            forced=bool(attempt.forced),
            force_reason=attempt.force_reason,
            folds=len(report.folds),
            windows_evaluated=int(report.total_windows),
            log_loss_skill=_finite(report.mean_skill),
            log_loss_skill_se=_finite(report.skill_standard_error),
            folds_positive=int(report.folds_positive),
            calibration_error=_finite(report.max_ece),
            residual_scale=_finite(report.mean_residual_scale),
            control_gain_share=_finite(report.max_control_gain_share),
            sharpe=_finite(gates.get('sharpe')),
            total_return=_finite(gates.get('total_return')),
            gates=[{'name': g.name, 'value': _finite(g.value),
                    'threshold': g.threshold, 'direction': g.direction,
                    'passed': bool(g.passed)} for g in attempt.gates],
            failed_gates=', '.join(g.name for g in attempt.gates if not g.passed),
            provenance=attempt.payload().get('config'),
        )
        # The reliability table of the fold that produced the candidate — the one
        # trained on the most history, and so the one deployed.
        table = report.folds[-1].reliability_table if report.folds else None
        if table is not None:
            frame = table.frame()
            writer.write_calibration(
                attempt.version, 'model',
                [{'bin_low': float(r.bin_low), 'bin_high': float(r.bin_high),
                  'predicted': _finite(r.predicted), 'observed': _finite(r.observed),
                  'count': int(r.count)}
                 for r in frame.itertuples() if r.count > 0])
        logger.info('recorded %s in the serving store', attempt.version)
    except Exception as exc:  # noqa: BLE001 - the ledger is authoritative, not this
        logger.warning('could not mirror %s into the serving store (%s); the '
                       'ledger under models/promotions/ is unaffected',
                       attempt.version, exc)


def _finite(value):
    """None rather than NaN: the serving columns are nullable and the API's
    contract is that a missing measurement is null with a reason, never a
    substitute."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


if __name__ == '__main__':
    raise SystemExit(main())
