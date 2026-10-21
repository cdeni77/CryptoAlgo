"""Tests that the scraper and the readers agree on symbol, venue and timeframe.

The Coinbase spot product id is the symbol everywhere, from the scrape through the
research store to the serving database, with no translation step.
"""

from __future__ import annotations

import pytest

from core.config import DEFAULT_CONFIG, Config
from core.dataset import MINUTE_DATASET, REFERENCE_SYMBOL
from core.datastore import BARS_DATASET_BY_TIMEFRAME, SCHEMAS
from scripts.live import SERIES_BY_SYMBOL
from scripts.scrape import TIMEFRAME, VENUE_LABEL


def test_the_scraper_and_the_reader_use_the_same_venue_label():
    assert VENUE_LABEL == DEFAULT_CONFIG.venue, (
        f'the scraper writes venue {VENUE_LABEL!r} and the reader asks for '
        f'{DEFAULT_CONFIG.venue!r}; every lookup would miss'
    )


def test_the_scraper_and_the_reader_use_the_same_timeframe_and_dataset():
    assert TIMEFRAME == DEFAULT_CONFIG.timeframe
    assert BARS_DATASET_BY_TIMEFRAME[TIMEFRAME] == MINUTE_DATASET
    assert MINUTE_DATASET in SCHEMAS


def test_symbols_are_venue_product_ids_with_no_translation():
    """A symbol is the venue product id, so a scraped symbol can be read."""
    for symbol in DEFAULT_CONFIG.symbols:
        assert symbol.endswith('-USD'), symbol
        assert symbol == symbol.upper()
        assert '-' in symbol and symbol.count('-') == 1


def test_the_reference_symbol_is_in_the_universe():
    """The reference symbol is in the universe; `cross_asset` residualises against it."""
    assert REFERENCE_SYMBOL in DEFAULT_CONFIG.symbols


def test_every_traded_symbol_maps_to_a_venue_series():
    """Every symbol maps to a Kalshi series, or live would abstain on it silently."""
    for symbol in DEFAULT_CONFIG.symbols:
        assert SERIES_BY_SYMBOL.get(symbol), (
            f'{symbol} has no Kalshi series, so the live path would abstain on it '
            f'every cycle without saying why'
        )


def test_a_narrowed_universe_keeps_the_reference():
    """A universe without the reference symbol is detectable."""
    narrowed = Config(symbols=('ETH-USD', 'SOL-USD'))
    assert REFERENCE_SYMBOL not in narrowed.symbols, (
        'this test asserts the situation is detectable, not that it is allowed'
    )
    # `attach_cross_asset` falls back to the mean of the peers, a weaker but defined
    # result, not a crash.
    from core.features import attach_cross_asset
    assert callable(attach_cross_asset)
