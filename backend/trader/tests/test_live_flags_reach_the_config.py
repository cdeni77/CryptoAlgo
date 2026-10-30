"""Every flag `scripts.live` accepts must reach the Config it runs on.

`scripts.live.config_from_args` SHADOWS `scripts._common.config_from_args`,
so a flag added to either parser parses cleanly into `args` and is then
dropped unless it is mapped in live's own function. The repo has recorded
this six times: `--maker-entry`, `--maker-time-in-force` and
`--maker-cancel-seconds` on 2026-10-18, and `--no-post-only` on 2026-10-30,
where 373 of 373 orders kept sending `post_only: true` and the venue kept
refusing the crossing ones while the flag sat in the compose command.

Production makes it easy to miss: the trader runs `scripts.run_live`, which
forwards everything after `--trade-args` to `scripts.live` verbatim, so the
flag is three layers from the Config that decides anything.

This test is generic on purpose. A new flag is covered the day it is added,
without anyone remembering to extend a list.
"""

from __future__ import annotations

import pytest

from scripts.live import build_parser, config_from_args

# Flags that deliberately do not map to a Config field: they steer the script
# rather than the strategy. Anything NOT here must change the Config.
# Flags that steer the SCRIPT rather than the strategy, so they are not
# expected to appear on Config: reconciliation, gate enforcement, run mode.
NOT_CONFIG = {
    'help', 'verbose', 'mode', 'dry_run', 'place_orders', 'loop',
    'cycle_seconds', 'trade_args', 'require_model_version', 'reason',
    'force', 'clear_halt', 'once', 'reconcile', 'require_gates',
}


def _config(argv):
    return config_from_args(build_parser().parse_args(argv))


def test_no_post_only_reaches_the_config():
    """**The 2026-10-30 defect.** It parsed, and nothing read it."""
    assert _config(['--mode', 'live', '--maker-entry']).maker_post_only is True
    assert _config(
        ['--mode', 'live', '--maker-entry', '--no-post-only']
    ).maker_post_only is False


def test_maker_flags_reach_the_config():
    """The 2026-10-18 defects, kept so they cannot come back."""
    c = _config(['--mode', 'live', '--maker-entry',
                 '--maker-cancel-seconds', '45'])
    assert c.maker_entry is True
    assert c.maker_cancel_seconds == 45.0


def test_every_boolean_flag_reaches_the_field_it_names():
    """**The generic guard.** If a flag's `dest` names a Config field, the
    parsed value must arrive there. That is exactly what shadowing breaks,
    and it covers a new flag the day it is added."""
    from dataclasses import fields
    parser = build_parser()
    config_fields = {f.name for f in fields(_config(['--mode', 'live']))}
    dropped = []
    for action in parser._actions:
        dest = action.dest
        if dest in NOT_CONFIG or not action.option_strings:
            continue
        if action.__class__.__name__ not in (
                '_StoreTrueAction', '_StoreFalseAction'):
            continue
        if dest not in config_fields:
            continue
        flag = action.option_strings[0]
        args = parser.parse_args(['--mode', 'live', flag])
        want = getattr(args, dest)
        got = getattr(_config(['--mode', 'live', flag]), dest, '<missing>')
        if got != want:
            dropped.append(f'{flag} -> args.{dest}={want} but config.{dest}={got}')
    assert not dropped, (
        'these flags parse and never reach the Config: ' + '; '.join(dropped)
        + '. live.config_from_args shadows the shared one, so each must be '
          'mapped there explicitly.')
