"""The deployed configs must agree, and this reads the REAL compose files.

**The repo's stated invariant is that an evaluation measures the system that
actually trades.** It has been violated repeatedly and silently, because nothing
compared the two ends. Measured 2026-10-18, on one evening:

* the weekly retrain was grading `--entry-offsets 12 --kelly-fraction 0.05` with
  no `--compound` while the account traded 3/6/9/12 at 0.10 Kelly compounding;
* `--exclude-price-band` was added to the trader's override to match the
  retrain, and `scripts/live.py` had no such argument -- the service would have
  exited `SystemExit 2` and placed nothing;
* `--maker-entry`, `--maker-time-in-force` and `--maker-cancel-seconds` parsed
  cleanly on live and were dropped, because `scripts/live.py` defines its OWN
  `config_from_args` that shadows the shared one. Live would have been a plain
  taker paying full fees while the gates were scored fee-free.

Every one of those is caught here. The test deliberately parses the compose
YAML rather than a copy of the flags, so it fails when the deployment changes
and nothing else does.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

ROOT = Path(__file__).resolve().parents[3]
OVERRIDE = ROOT / 'docker-compose.override.yml'
COMPOSE = ROOT / 'docker-compose.yml'

# Fields where a difference changes what the account does, or what the gates
# claim about it. `groups` is excluded: the retrain is allowed to propose a new
# feature set, which is the point of retraining.
DECISION_FIELDS = (
    'entry_offsets', 'kelly_fraction', 'compound', 'starting_bankroll',
    'min_edge_pp', 'min_traded_price', 'max_traded_price',
    'exclude_price_band', 'maker_entry', 'assume_maker',
    'maker_time_in_force', 'maker_cancel_seconds',
    'half_spread_cents', 'fee_rate', 'maker_fee_multiplier',
    'max_daily_loss_fraction', 'max_drawdown_fraction', 'ruin_floor_fraction',
    'max_positions_per_window', 'max_entries_per_symbol_window',
    'max_window_exposure_fraction',
    'max_stake_dollars',
)

# Policy the LIVE loop inherits from the installed artifact's provenance rather
# than from its own flags, via `livetrade/artifact.config_for_artifact`. The
# retrain must RECORD these or the next artifact writes the defaults and live
# adopts them -- silently dropping real refusal gates.
ADOPTED_FIELDS = ('ensemble_size', 'label_source', 'edge_lower_bound_z',
                  'require_member_agreement')


def _command(path: Path, service: str) -> list[str]:
    """The service's argv, split the way Docker Compose actually splits it.

    **`str.split()` is not how Compose reads a string `command`** — it uses
    shell-like splitting, so a quoted argument containing spaces is ONE token.
    Verified against `docker compose config`, which returns the 42-token list
    with `--reason "..."` intact. The naive split blew that one argument into
    dozens of tokens and failed a command the real parser accepts, which is a
    false alarm on the test whose entire job is to catch real ones.
    """
    doc = yaml.safe_load(path.read_text())
    cmd = doc['services'][service]['command']
    if isinstance(cmd, str):
        return shlex.split(cmd)
    return [str(c) for c in cmd]


def _live_argv() -> list[str]:
    argv = _command(OVERRIDE, 'trader')
    # run_live forwards everything after --trade-args to scripts/live.py
    return argv[argv.index('--trade-args') + 1:]


def _retrain_argv() -> list[str]:
    argv = _command(COMPOSE, 'retrain')
    return [a for a in argv if a not in ('-m', 'scripts.promote')]


@pytest.mark.skipif(not OVERRIDE.exists(), reason='no local override')
def test_the_live_override_actually_parses():
    """**The failure this would have shipped.** A flag the override passes and
    the parser does not know is not a misconfiguration, it is a dead service:
    `run_live` treats SystemExit from a component as deliberate and exits."""
    import scripts.live as live
    try:
        live.build_parser().parse_args(_live_argv())
    except SystemExit as exc:                                  # pragma: no cover
        pytest.fail(f'the trader service cannot start: SystemExit {exc.code}. '
                    f'A flag in docker-compose.override.yml is not in '
                    f'scripts/live.py\'s parser.')


@pytest.mark.skipif(not OVERRIDE.exists(), reason='no local override')
def test_every_flag_the_override_passes_actually_reaches_the_config():
    """**Parsed is not read.** `scripts/live.py` defines its own
    `config_from_args`, so a flag can parse and then be dropped on the floor --
    which is how maker mode came to be configured everywhere and active
    nowhere. Assert the specific values the override asks for."""
    import scripts.live as live
    cfg = live.config_from_args(live.build_parser().parse_args(_live_argv()))
    argv = _live_argv()
    if '--exclude-price-band' in argv:
        want = argv[argv.index('--exclude-price-band') + 1]
        lo, hi = (float(x) for x in want.split(','))
        assert cfg.exclude_price_band == pytest.approx((lo, hi)), \
            'the override asks for a dead band and the config does not have one'
    if '--maker-entry' in argv:
        assert cfg.maker_entry is True, \
            'the override asks to rest at the bid and the config says taker'
        assert cfg.assume_maker is True, 'a maker pays the maker fee'
    if '--maker-cancel-seconds' in argv:
        want = float(argv[argv.index('--maker-cancel-seconds') + 1])
        assert cfg.maker_cancel_seconds == pytest.approx(want)


@pytest.mark.skipif(not (OVERRIDE.exists() and COMPOSE.exists()),
                    reason='no local compose pair')
def test_the_retrain_grades_the_policy_the_account_runs():
    """The invariant, stated as a diff. Any field here that differs means the
    weekly gates are scoring a strategy nobody trades."""
    import argparse

    import scripts.live as live
    from scripts._common import config_from_args
    from scripts.promote import build_parser as promote_parser

    # `scripts.promote`'s OWN parser, not `add_data_arguments`. The shared one
    # does not know promote-only flags (`--force`, `--reason`), so approximating
    # with it made a valid compose command fail a test whose job is to say
    # whether the retrain can start.
    live_cfg = live.config_from_args(live.build_parser().parse_args(_live_argv()))
    retrain_ns = promote_parser().parse_args(_retrain_argv())
    retrain_cfg = config_from_args(retrain_ns)

    differ = {f: (getattr(live_cfg, f, None), getattr(retrain_cfg, f, None))
              for f in DECISION_FIELDS
              if getattr(live_cfg, f, None) != getattr(retrain_cfg, f, None)}
    assert not differ, (
        'the weekly retrain grades a different policy than the account trades:\n'
        + '\n'.join(f'  {k}: live={v[0]!r} retrain={v[1]!r}'
                    for k, v in sorted(differ.items())))


@pytest.mark.skipif(not COMPOSE.exists(), reason='no local compose')
def test_the_retrain_command_actually_parses():
    """Same failure class as the live one, on the other side: a flag the
    retrain passes that `scripts.promote` does not know exits 2, and the cron
    wrapper reports it as "promote did not complete" rather than a bad flag."""
    from scripts.promote import build_parser as promote_parser
    try:
        promote_parser().parse_args(_retrain_argv())
    except SystemExit as exc:                                  # pragma: no cover
        pytest.fail(f'the weekly retrain cannot start: SystemExit {exc.code}. '
                    f'A flag in docker-compose.yml is not in scripts.promote.')


@pytest.mark.skipif(not COMPOSE.exists(), reason='no local compose')
def test_the_retrain_preserves_the_deployed_ensemble_policy():
    """**The gates must not turn themselves off.** `config_for_artifact` adopts
    these from provenance and skips anything equal to the default, so a retrain
    that does not pass them writes 1 / proxy / 0.0 / False and live drops
    `EDGE_UNCERTAIN` and `MEMBERS_DISAGREE` at the next window boundary, with no
    log line and no restart."""
    import argparse
    import json
    from pathlib import Path

    from scripts._common import config_from_args
    from scripts.promote import build_parser as promote_parser
    prov = Path(__file__).resolve().parents[1] / 'models' / 'forecast.provenance.json'
    if not prov.exists():
        pytest.skip('no installed artifact to compare against')
    deployed = json.loads(prov.read_text()).get('config', {})
    cfg = config_from_args(promote_parser().parse_args(_retrain_argv()))
    differ = {f: (deployed.get(f), getattr(cfg, f, None))
              for f in ADOPTED_FIELDS
              if f in deployed and deployed.get(f) != getattr(cfg, f, None)}
    assert not differ, (
        'the retrain would install an artifact whose policy differs from the '
        'one live is running, and live adopts provenance silently:\n'
        + '\n'.join(f'  {k}: deployed={v[0]!r} retrain_would_write={v[1]!r}'
                     for k, v in sorted(differ.items())))
