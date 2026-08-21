"""A reference venue that returns nothing must say so.

`cross_venue_features` returns an empty DataFrame when there are no reference
bars, but `build_panel` then reindexes to the canonical column list — on purpose,
so a saved model always scores against the same matrix. The consequence is that
the seven cross-venue columns still exist, as **all-NaN**. The panel keeps its
full 76-column shape and looks healthy.

`feature_set_hash` hashes column *names*, so it is byte-identical whether or not
the reference venue was reachable: a model fit with basis and lead-lag and one
fit without them cannot be told apart from the artifact.

It is also the likely case for a US operator, which is what makes it worth a
test rather than a comment. Binance, OKX and Bybit all answer HTTP 451 to a US
IP, so the default `--reference-venue binance` yields nothing unless the scrape
went through a proxy — and if the scraper's fallback served a different exchange
instead, the bars are stamped with *that* venue's name and `binance` still
matches nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.dataset import load_dataset
from core.datastore import ResearchStore

SPELLINGS = ('BTC-PERP', 'ETH-PERP', 'SOL-PERP')

# The seven columns that exist only when a reference venue does.
CROSS_VENUE_COLUMNS = (
    'basis_bps', 'basis_z_168h', 'basis_change_1h',
    'ref_return_1h', 'ref_return_4h',
    'lead_lag_corr_72h', 'contemp_corr_72h',
)


def _write(store, symbols, venue, bars=400, seed=0):
    index = pd.date_range('2026-01-01', periods=bars, freq='h', tz='UTC')
    rng = np.random.default_rng(seed)
    for symbol in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        store.write('bars', pd.DataFrame({
            'symbol': symbol, 'venue': venue, 'event_time': index,
            'available_time': index, 'quality': 'valid',
            'open': close, 'high': close * 1.001, 'low': close * 0.999,
            'close': close, 'volume': 1000.0,
        }))


def test_a_reference_venue_with_no_bars_is_warned_about(tmp_path):
    """The regression guard. Silence here is what a geo-block looks like."""
    store = ResearchStore(tmp_path / 'research')
    _write(store, SPELLINGS, venue='coinbase')

    dataset = load_dataset(
        store, venue='coinbase', reference_venue='binance', min_quality='valid')

    assert not dataset.features.empty, 'the panel should still build'
    reference_warnings = [w for w in dataset.warnings if 'binance' in w]
    assert reference_warnings, (
        f'no warning that binance produced nothing. Warnings were: '
        f'{dataset.warnings}'
    )
    # And it has to name the consequence, not just the absence.
    assert any('cross-venue' in w for w in reference_warnings), reference_warnings


def test_the_cross_venue_features_are_present_but_empty(tmp_path):
    """Proof the warning is about something, and that the shape hides it.

    The columns are there — that is what makes this quiet — and they hold
    nothing. Both halves matter: if they were absent, `assert_compatible` would
    refuse the matrix and the failure would be loud.
    """
    store = ResearchStore(tmp_path / 'research')
    _write(store, SPELLINGS, venue='coinbase')

    without = load_dataset(
        store, venue='coinbase', reference_venue='binance', min_quality='valid')

    for column in CROSS_VENUE_COLUMNS:
        assert column in without.features.columns, (
            f'{column} is absent, not empty — build_panel no longer reindexes to '
            f'the canonical column list, which changes what this test guards'
        )
        assert without.features[column].isna().all(), (
            f'{column} has data with no reference venue in the store'
        )

    # Same store, plus reference bars under the venue name the reader asks for.
    _write(store, SPELLINGS, venue='binance', seed=1)
    with_reference = load_dataset(
        store, venue='coinbase', reference_venue='binance', min_quality='valid')

    populated = [
        column for column in CROSS_VENUE_COLUMNS
        if with_reference.features[column].notna().any()
    ]
    assert populated == list(CROSS_VENUE_COLUMNS), (
        f'reference bars are in the store but these are still empty: '
        f'{sorted(set(CROSS_VENUE_COLUMNS) - set(populated))}'
    )
    assert not [w for w in with_reference.warnings if 'cross-venue' in w], (
        'warned about a reference venue that is present'
    )


def test_the_model_records_which_features_carried_no_data(tmp_path):
    """The artifact has to be able to tell the two cases apart.

    `feature_set_hash` cannot: it hashes column names, and the names are
    identical either way. So a model trained behind a geo-block and one trained
    through a proxy have the same hash, and only `empty_features` distinguishes
    them.
    """
    from core.config import Config
    from core.model import feature_set_hash, train_forecast_model

    store = ResearchStore(tmp_path / 'research')
    _write(store, SPELLINGS, venue='coinbase', bars=900)

    config = Config()
    blocked = load_dataset(
        store, venue='coinbase', reference_venue='binance',
        min_quality='valid', config=config, horizon_bars=8,
    )
    model = train_forecast_model(
        blocked.features, blocked.targets, config=config, horizon_bars=8,
    )
    assert model is not None, 'the panel should still train'

    provenance = model.provenance()
    assert set(CROSS_VENUE_COLUMNS) <= set(provenance['empty_features']), (
        f"the cross-venue columns are all NaN but provenance does not say so: "
        f"{provenance['empty_features']}"
    )
    assert provenance['n_features_populated'] < provenance['n_features']

    # And the point of recording it: the hash alone cannot tell you.
    assert provenance['feature_set_hash'] == feature_set_hash(model.feature_columns), (
        'sanity check on the hash itself'
    )


def test_a_partial_reference_venue_names_the_symbols_it_lacks(tmp_path):
    """The mixed case is the confusing one: some symbols carry the features and
    some do not, so the panel is inconsistent across the universe."""
    store = ResearchStore(tmp_path / 'research')
    _write(store, SPELLINGS, venue='coinbase')
    _write(store, SPELLINGS[:1], venue='binance', seed=1)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue='binance', min_quality='valid')

    warning = next((w for w in dataset.warnings if 'cross-venue' in w), None)
    assert warning is not None, dataset.warnings
    # Names the ones that are missing, not the one that is present.
    assert 'ETH-PERP' in warning and 'SOL-PERP' in warning, warning
    assert '2 of 3' in warning, warning


def test_no_reference_venue_requested_is_not_a_warning(tmp_path):
    """`--reference-venue ''` is a deliberate choice, not a degradation."""
    store = ResearchStore(tmp_path / 'research')
    _write(store, SPELLINGS, venue='coinbase')

    dataset = load_dataset(
        store, venue='coinbase', reference_venue=None, min_quality='valid')

    assert not [w for w in dataset.warnings if 'cross-venue' in w], dataset.warnings
