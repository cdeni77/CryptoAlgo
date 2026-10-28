"""A pin must keep the PINNED model running, not merely refuse its orders.

`--require-model-version` gated order placement only. `reload_if_changed`
swapped in whatever was installed, so the Sunday cron -- which installs by
standing policy -- would put an unpinned artifact into the loop at the next
window boundary and every decision would then be refused `model_not_pinned`.

The account stops trading and nothing says so: the startup warning has already
fired, and the swap itself logs success. A pin whose effect is "silently stop"
is worse than no pin.

So the swap is refused instead, and the loop keeps trading the pinned model.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from core.config import Config
from livetrade import artifact as artifact_module


class _Model:
    def __init__(self, version):
        self.version = version


def _reload(pinned, installed_version, monkeypatch, tmp_path):
    path = tmp_path / 'forecast.joblib'
    path.write_bytes(b'new-bytes')
    old = _Model('20261027T211836Z')
    config = Config(require_model_version=pinned)

    def _load(args, base_config):
        return _Model(installed_version), config

    monkeypatch.setattr(artifact_module, '_load_artifact', _load)
    return asyncio.run(artifact_module.reload_if_changed(
        types.SimpleNamespace(), config, path, old, config, signature=None))


def test_an_unpinned_artifact_is_not_swapped_in(monkeypatch, tmp_path):
    """The cron installs; the loop keeps trading what you pinned."""
    model, _, _ = _reload('20261027T211836Z', '20261101T050000Z',
                          monkeypatch, tmp_path)
    assert model.version == '20261027T211836Z', (
        'an unpinned artifact was swapped in; every decision would then be '
        'refused model_not_pinned and the account would stop silently')


def test_the_pinned_artifact_is_swapped_in(monkeypatch, tmp_path):
    """A reinstall of the pinned version must still load."""
    model, _, _ = _reload('20261101T050000Z', '20261101T050000Z',
                          monkeypatch, tmp_path)
    assert model.version == '20261101T050000Z'


def test_without_a_pin_any_artifact_is_swapped_in(monkeypatch, tmp_path):
    """Unpinned accounts keep the old behaviour exactly."""
    model, _, _ = _reload(None, '20261101T050000Z', monkeypatch, tmp_path)
    assert model.version == '20261101T050000Z'
