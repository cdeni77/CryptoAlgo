"""The gate spec is frozen in a lock committed beside the code.

Gate thresholds and definitions were changed repeatedly while they were the
blocking failure. The lock does not stop a change; it makes one visible: this
test fails until `core/gate_spec.lock.json` is regenerated in the same commit,
and `scripts.promote` refuses a running spec that differs from it unless
`--accept-gate-change` records why.
"""
from __future__ import annotations

import json

from core.metrics import (
    DEFAULT_GATES, GATE_NOTES, GATE_SPEC, GATE_SPEC_LOCK, gate_spec, gate_spec_diff,
    gate_spec_hash, load_gate_lock,
)

REGENERATE = ('regenerate it in the same commit: python -c "from core.metrics '
              'import write_gate_lock; write_gate_lock()" — and bump '
              'definition_version for any gate whose computation changed')


def test_the_lock_matches_the_code():
    lock = load_gate_lock()
    assert lock, f'{GATE_SPEC_LOCK} is missing or unreadable'
    assert lock['gates'] == gate_spec(), f'the gate spec changed; {REGENERATE}'
    assert lock['gate_spec_hash'] == gate_spec_hash(), REGENERATE


def test_the_lock_lives_with_the_code_not_on_the_models_volume():
    assert GATE_SPEC_LOCK.parent.name == 'core'
    assert 'models' not in GATE_SPEC_LOCK.parts


def test_every_gate_has_a_version_a_direction_and_a_note():
    assert list(DEFAULT_GATES) == [g.name for g in GATE_SPEC]
    for gate in GATE_SPEC:
        assert gate.direction in ('min', 'max')
        assert isinstance(gate.definition_version, int) and gate.definition_version >= 1
        assert GATE_NOTES.get(gate.name), gate.name


def test_the_hash_moves_with_a_threshold_and_with_a_definition():
    base = gate_spec_hash()
    looser = dict(DEFAULT_GATES, calibration_error=(0.5, 'max'))
    assert gate_spec_hash(looser) != base

    rows = gate_spec()
    bumped = [dict(r, definition_version=r['definition_version'] + 1)
              if r['name'] == 'sharpe' else r for r in rows]
    canonical = lambda rs: json.dumps(rs, sort_keys=True)   # noqa: E731
    assert canonical(bumped) != canonical(rows)
    diff = gate_spec_diff(rows, bumped)
    assert list(diff['changed']) == ['sharpe']
    assert not diff['added'] and not diff['removed']


def test_the_skill_gate_records_that_its_aggregation_changed():
    versions = {g.name: g.definition_version for g in GATE_SPEC}
    assert versions['log_loss_skill'] >= 2, 'window weighting changed it'
    assert 'log_loss_skill_p' in versions
