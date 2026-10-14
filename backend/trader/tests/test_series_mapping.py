"""Tests for `core.config.SERIES_BY_SYMBOL`, the one Kalshi series mapping.

The mapping reads `KALSHI_SERIES_*`, and `series_to_symbol()` derives the inverse from
it, so the trader and every recorder follow the same override.
"""

from __future__ import annotations

import importlib
import os

import pytest

from core import config as config_module


@pytest.fixture(autouse=True)
def _reload_config_module():
    """Reloads the config module after each test to undo the env-driven dict."""
    yield
    importlib.reload(config_module)


def test_the_default_mapping_is_the_three_traded_series():
    assert config_module.SERIES_BY_SYMBOL == {
        'BTC-USD': 'KXBTC15M', 'ETH-USD': 'KXETH15M', 'SOL-USD': 'KXSOL15M',
    }


def test_an_env_override_changes_the_mapping(monkeypatch):
    monkeypatch.setenv('KALSHI_SERIES_BTC', 'KXBTC15M-DEMO')
    reloaded = importlib.reload(config_module)
    assert reloaded.SERIES_BY_SYMBOL['BTC-USD'] == 'KXBTC15M-DEMO'
    assert reloaded.SERIES_BY_SYMBOL['ETH-USD'] == 'KXETH15M'


def test_series_to_symbol_is_the_exact_inverse():
    inverse = config_module.series_to_symbol()
    for symbol, series in config_module.SERIES_BY_SYMBOL.items():
        assert inverse[series] == symbol
    assert len(inverse) == len(config_module.SERIES_BY_SYMBOL)


def test_an_env_override_reaches_the_inverse_too(monkeypatch):
    """An env override also reaches the inverse mapping that recorders use."""
    monkeypatch.setenv('KALSHI_SERIES_BTC', 'KXBTC15M-DEMO')
    reloaded = importlib.reload(config_module)
    inverse = reloaded.series_to_symbol()
    assert inverse['KXBTC15M-DEMO'] == 'BTC-USD'
    assert 'KXBTC15M' not in inverse
