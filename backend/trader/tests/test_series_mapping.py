"""One Kalshi series mapping, not five copies that only one of them honours.

`scripts/live.py` built `SERIES_BY_SYMBOL` reading `KALSHI_SERIES_BTC/ETH/SOL`,
so the trader's series is configurable — e.g. pointing at a demo series without
touching code. Four other files (`record_ladder.py`, `record_depth.py`,
`collect_settlements.py`, `backfill_quotes.py`) each hardcoded the same three
tickers with no env read at all. Setting `KALSHI_SERIES_BTC` would move what the
trader trades while every recorder kept scraping production — a live account
disagreeing with its own data pipeline about which market is real, silently.

`core/config.SERIES_BY_SYMBOL` is now the one source, and `series_to_symbol()`
derives the inverse a recorder needs (series ticker -> our symbol) from it,
so the two directions cannot drift apart.
"""

from __future__ import annotations

import importlib
import os

import pytest

from core import config as config_module


@pytest.fixture(autouse=True)
def _reload_config_module():
    """Undo the env-driven module-level dict after each test."""
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
    """The whole point: a recorder built from the inverse must follow the env."""
    monkeypatch.setenv('KALSHI_SERIES_BTC', 'KXBTC15M-DEMO')
    reloaded = importlib.reload(config_module)
    inverse = reloaded.series_to_symbol()
    assert inverse['KXBTC15M-DEMO'] == 'BTC-USD'
    assert 'KXBTC15M' not in inverse
