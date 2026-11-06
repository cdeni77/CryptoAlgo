"""The bar series must be expressible, reach the config, and be checked.

BRTI was measured better on 2026-10-22, installed once, and erased three days
later by the weekly cron. Three things let that happen silently and each is
asserted here.
"""
import pytest

from core.config import DEFAULT_CONFIG
from scripts.live import build_parser, config_from_args


def test_the_default_bar_series_is_the_one_the_markets_settle_on():
    # Kalshi settles on CF Benchmarks BRTI. Coinbase is a different series and
    # its basis does not cancel between the strike and the decision.
    assert DEFAULT_CONFIG.venue == 'cfbenchmarks'


def test_live_can_be_told_which_bar_series_to_use():
    # The live loop had NO --venue at all, so it could only ever run the config
    # default -- which is why a measured improvement sat undeployed.
    args = build_parser().parse_args(['--venue', 'coinbase_spot'])
    assert args.venue == 'coinbase_spot'


def test_the_venue_flag_actually_reaches_the_config():
    # Declaring the flag without mapping it is the bug this repo has recorded
    # five times. `live.py` shadows `scripts._common.config_from_args`, so the
    # shared mapping does not help here.
    cfg = config_from_args(build_parser().parse_args(['--venue', 'coinbase_spot']))
    assert cfg.venue == 'coinbase_spot'
    assert config_from_args(build_parser().parse_args([])).venue == 'cfbenchmarks'


def test_a_venue_mismatch_is_refused_not_scored():
    """`venue` picks the series 35 of 55 features are built from.

    Absent from the provenance check, a BRTI-fitted artifact scored on Coinbase
    bars passed silently -- no error, no warning, wrong basis.
    """
    import dataclasses
    from core.model import ForecastModel

    fitted = dataclasses.replace(DEFAULT_CONFIG, venue='cfbenchmarks')
    running = dataclasses.replace(DEFAULT_CONFIG, venue='coinbase_spot')
    model = ForecastModel.__new__(ForecastModel)
    object.__setattr__(model, 'config_provenance', fitted.provenance())

    with pytest.raises(ValueError, match='venue'):
        model._verify_config_drift(running)

    # And the matching case must not raise, or the guard is just broken.
    model._verify_config_drift(fitted)


def test_sync_store_does_not_follow_the_model_default():
    # Legacy sqlite bars came from Coinbase. Labelling them with whatever series
    # the model currently reads would assert something untrue about their origin.
    import scripts.sync_store as ss
    import inspect
    src = inspect.getsource(ss.main)
    assert "default='coinbase_spot'" in src
    assert 'DEFAULT_CONFIG.venue' not in src.split('--venue')[1][:200]


def test_the_trader_adopts_the_basis_instead_of_freezing_it():
    """The compose must NOT set the adopted fields.

    `livetrade.artifact.config_for_artifact` adopts `venue`, `strike_source`
    and the price band from the installed artifact so the trader follows a
    migrated model. It skips any field the operator set explicitly -- so
    putting these in the compose disables adoption and pins the trader to
    today's basis. The next artifact fitted elsewhere would then be REFUSED by
    the provenance check rather than followed.
    """
    import shlex
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[3]
    override = yaml.safe_load((root / 'docker-compose.override.yml').read_text())
    cmd = override['services']['trader'].get('command') or []
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(x) for x in cmd]
    for flag in ('--venue', '--strike-source'):
        assert flag not in cmd, (
            f'{flag} is set on the trader, which marks it explicit and disables '
            f'adoption from the artifact provenance')


def test_the_adopted_fields_are_also_in_the_provenance_check():
    """Adoption and the guard must cover the same ground.

    Adoption makes the trader follow the artifact; the guard catches the case
    where it could not. A field in neither is how a basis mismatch runs
    silently for weeks.
    """
    import inspect
    from core.model import ForecastModel
    from livetrade import artifact as art

    guard = inspect.getsource(ForecastModel._verify_config_drift)
    adopt = inspect.getsource(art.config_for_artifact)
    for field in ('venue', 'strike_source'):
        assert f"'{field}'" in adopt, f'{field} must be adopted from the artifact'
        assert f"'{field}'" in guard, f'{field} must be checked by the guard'


def test_the_strike_backfill_refuses_without_credentials():
    """An auth failure must not be reported as an empty venue.

    Run in a service with no Kalshi key, every request fails instantly and the
    script reports "1185/1185 asked, 0 row(s) written" -- which reads as "the
    venue has no data for these windows". The same 1,185 requests returned
    1,395 rows from a service that had the key, seconds later.
    """
    import inspect
    import scripts.backfill_markets as bm

    src = inspect.getsource(bm)
    assert 'KALSHI_KEY_ID' in src, 'the credential check is gone'
    assert 'SystemExit' in src, 'a missing key must raise, not return zero rows'


def test_the_venue_strike_stays_current_with_the_rest_of_the_pipeline():
    """`venue_strikes` must advance with the loop, not by hand.

    `--strike-source venue` builds every window from it, so a stale table
    silently stops the TRAINING set while live keeps trading -- measured at 99
    hours stale on 2026-11-05.
    """
    import inspect
    import scripts.run_live as rl

    src = inspect.getsource(rl)
    assert "'scripts.backfill_markets'" in src, (
        'backfill_markets is not in the collection steps, so venue_strikes '
        'only advances when somebody remembers to run it')
