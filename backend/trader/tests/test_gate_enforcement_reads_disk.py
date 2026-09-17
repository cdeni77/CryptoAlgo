"""`--require-gates` must ask about the artifact being TRADED.

It read `history().iloc[0]` — the newest promotion ATTEMPT. That diverges from
the artifact on disk in two ordinary cases, both of which happened on
2026-09-16:

  * a promotion that FAILS installs nothing and leaves the previous, passing
    model trading. Under the old check the next restart would refuse to start
    the loop — so one failed Sunday retrain could take the trader down at the
    next host reboot, silently, for as long as nobody looked.
  * a ROLLBACK deliberately writes no ledger entry at all, because the ledger is
    the multiple-testing denominator and a rollback is not a trial. The newest
    entry then describes an artifact that is no longer installed.

The question the gate asks is "did the model I am about to trade pass?", and
that is a property of the file. Identified by CONTENT: `promote` stages each
candidate under `models/.staging/<version>/` and renames it into place, so the
installed file is byte-identical to exactly one of them, and no new bookkeeping
is needed.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import scripts.live as live


def _artifact(root, version, blob, *, passed, installed=True, forced=False):
    staged = root / '.staging' / version
    staged.mkdir(parents=True, exist_ok=True)
    (staged / 'forecast.joblib').write_bytes(blob)
    (root / 'promotions').mkdir(parents=True, exist_ok=True)
    (root / 'promotions' / f'{version}.json').write_text(json.dumps({
        'version': version, 'passed': passed, 'installed': installed,
        'forced': forced, 'failed_gates': [] if passed else ['calibration_error'],
        'report': {},
    }))


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / 'models'
    root.mkdir()
    monkeypatch.setattr(live, 'MODELS_ROOT', root)
    monkeypatch.setattr('core.promotion.MODELS_ROOT', root)
    return root


def test_it_finds_the_installed_artifact_by_content(models):
    _artifact(models, '20260101T000000Z', b'old', passed=True)
    _artifact(models, '20260201T000000Z', b'new', passed=True)
    (models / 'forecast.joblib').write_bytes(b'old')
    assert live._installed_version() == '20260101T000000Z'


def test_a_failed_later_attempt_does_not_block_a_passing_artifact(models):
    """The safety property. Sunday's retrain fails, installs nothing, and the
    trader must still start on the model that IS deployed."""
    _artifact(models, '20260101T000000Z', b'good', passed=True)
    _artifact(models, '20260201T000000Z', b'bad', passed=False, installed=False)
    (models / 'forecast.joblib').write_bytes(b'good')
    live._refuse_if_blocked()          # must not raise


def test_a_forced_installed_artifact_is_still_refused(models):
    """The check must not become toothless: a model force-installed with
    failing gates is exactly what `--require-gates` exists to stop."""
    _artifact(models, '20260101T000000Z', b'forced', passed=False, forced=True)
    (models / 'forecast.joblib').write_bytes(b'forced')
    with pytest.raises(SystemExit):
        live._refuse_if_blocked()


def test_an_unrecognised_artifact_falls_back_to_the_newest_attempt(models):
    """A file matching nothing staged is not evidence of anything, so the old
    conservative behaviour stands rather than silently passing."""
    _artifact(models, '20260101T000000Z', b'known', passed=False, forced=True)
    (models / 'forecast.joblib').write_bytes(b'mystery')
    assert live._installed_version() is None
    with pytest.raises(SystemExit):
        live._refuse_if_blocked()


def test_no_artifact_at_all_is_not_a_pass(models):
    _artifact(models, '20260101T000000Z', b'x', passed=True)
    assert live._installed_version() is None
