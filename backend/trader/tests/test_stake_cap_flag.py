"""`--max-stake-dollars` must reach the loop and survive artifact adoption.

`config_for_artifact` adopts `kelly_fraction`, `min_edge_pp`,
`max_stake_dollars`, `max_stake_fraction`, `half_spread_cents` and `compound`
from the promoting run, so an artifact trades under the policy it was MEASURED
under. That means changing `Config.max_stake_dollars`' default would NOT reach
a running loop — the artifact's stored 25.0 would win.

Adoption explicitly yields to anything the command line set, via
`cli_overrides`, and its own comment says why: adoption exists so the artifact
trades under its measured policy, "not to overrule the operator — and silently
reverting a deliberate setting at the next promotion is exactly the failure
that would be hardest to notice."

So the cap needs a flag, and the flag needs to be recorded as explicit.
"""

from __future__ import annotations

import pytest

import scripts.live as live


def _config(argv):
    return live.config_from_args(live.build_parser().parse_args(argv))


def test_the_flag_sets_the_cap():
    assert _config(['--max-stake-dollars', '40']).max_stake_dollars == 40.0


def test_omitting_it_leaves_the_default():
    assert _config([]).max_stake_dollars == 25.0


def test_it_is_recorded_as_an_explicit_override():
    """Without this, `config_for_artifact` would silently restore the
    artifact's value at the next promotion."""
    cfg = _config(['--max-stake-dollars', '40'])
    assert 'max_stake_dollars' in (cfg.cli_overrides or ())


def test_adoption_does_not_overrule_the_flag():
    class _Model:
        config_provenance = {'max_stake_dollars': 25.0, 'kelly_fraction': 0.05}
        init_score_source = 'baseline'

    cfg = _config(['--max-stake-dollars', '40'])
    adopted = live.config_for_artifact(cfg, _Model(), mode='live')
    assert adopted.max_stake_dollars == 40.0, (
        'the operator set it deliberately; adoption must yield'
    )


def test_adoption_still_wins_when_the_flag_is_absent():
    """The cap is a claim about DEPTH, so an artifact measured under a
    different one should carry it when nobody said otherwise."""
    class _Model:
        config_provenance = {'max_stake_dollars': 12.0}
        init_score_source = 'baseline'

    adopted = live.config_for_artifact(_config([]), _Model(), mode='live')
    assert adopted.max_stake_dollars == 12.0
