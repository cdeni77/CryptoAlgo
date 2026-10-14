"""Tests that `--require-gates` checks the artifact actually installed.

The newest ledger entry can differ from the installed file: a failed promotion
installs nothing, and a rollback writes no entry. The installed file is
identified by content, since it is byte-identical to one staged candidate.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import livetrade.artifact as live_artifact


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
    monkeypatch.setattr(live_artifact, 'MODELS_ROOT', root)
    monkeypatch.setattr('core.promotion.MODELS_ROOT', root)
    return root


def test_it_finds_the_installed_artifact_by_content(models):
    _artifact(models, '20260101T000000Z', b'old', passed=True)
    _artifact(models, '20260201T000000Z', b'new', passed=True)
    (models / 'forecast.joblib').write_bytes(b'old')
    assert live_artifact._installed_version() == '20260101T000000Z'


def test_a_failed_later_attempt_does_not_block_a_passing_artifact(models):
    """A failed later promotion does not stop the trader starting on the passing installed
    model.
    """
    _artifact(models, '20260101T000000Z', b'good', passed=True)
    _artifact(models, '20260201T000000Z', b'bad', passed=False, installed=False)
    (models / 'forecast.joblib').write_bytes(b'good')
    live_artifact._refuse_if_blocked()  # must not raise


def test_a_forced_installed_artifact_is_still_refused(models):
    """A force-installed artifact that failed its gates is still refused."""
    _artifact(models, '20260101T000000Z', b'forced', passed=False, forced=True)
    (models / 'forecast.joblib').write_bytes(b'forced')
    with pytest.raises(SystemExit):
        live_artifact._refuse_if_blocked()


def test_an_unrecognised_artifact_falls_back_to_the_newest_attempt(models):
    """A file that matches no staged candidate falls back to the newest ledger entry."""
    _artifact(models, '20260101T000000Z', b'known', passed=False, forced=True)
    (models / 'forecast.joblib').write_bytes(b'mystery')
    assert live_artifact._installed_version() is None
    with pytest.raises(SystemExit):
        live_artifact._refuse_if_blocked()


def test_no_artifact_at_all_is_not_a_pass(models):
    _artifact(models, '20260101T000000Z', b'x', passed=True)
    assert live_artifact._installed_version() is None
