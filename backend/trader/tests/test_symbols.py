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


def test_the_reader_asks_for_a_series_something_actually_writes():
    """`minute_bars` now holds two series, so the labels no longer have to match.

    `scripts.scrape` writes spot under `VENUE_LABEL`; `scripts.backfill_index`
    writes the settlement index under its own `VENUE`. What must stay true is
    that whatever the reader asks for is written by SOMETHING -- a reader
    pointed at a series with no writer misses every lookup, which is how the
    old single-venue assertion earned its place.
    """
    from scripts.backfill_index import VENUE as INDEX_VENUE
    written = {VENUE_LABEL, INDEX_VENUE}
    assert DEFAULT_CONFIG.venue in written, (
        f'the reader asks for {DEFAULT_CONFIG.venue!r} and the writers are '
        f'{sorted(written)}; every lookup would miss'
    )


def test_the_index_series_is_the_default_because_the_markets_settle_on_it():
    from scripts.backfill_index import VENUE as INDEX_VENUE
    assert DEFAULT_CONFIG.venue == INDEX_VENUE
    assert VENUE_LABEL != INDEX_VENUE, 'spot and the index must stay distinguishable'


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


