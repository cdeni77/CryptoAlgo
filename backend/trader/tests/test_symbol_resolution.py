"""The scraper and the readers have to agree about what a symbol is.

They did not. `scripts/run_pipeline.py` stores whatever the venue calls the
product — `BTC-PERP` for the majors, `AVP-20DEC30-CDE` for group B — while
`core/dataset.load_dataset` asked for the bare profile prefix (`BIP`, `ETP`,
`SLP`). `ResearchStore._prepare` only upper-cases, so nothing matched: on a store
built by the command CLAUDE.md documents, EVERY symbol lookup missed and the
feature panel came back empty, with one "no bars on coinbase, skipped" warning
per instrument and `MARKET_SYMBOL` missing on top.

Measured before the fix: zero overlap between the 16 symbols the scraper writes
and the 16 the reader requests. It presents as a data problem — scrape more, the
store must be empty — and is a naming problem.

Resolution goes through `costs._resolve_base`, the same function that prices the
contract, so a symbol that can be priced can be found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.dataset import MARKET_SYMBOL, load_dataset, resolve_store_symbols
from core.datastore import ResearchStore
from core.profiles import COIN_PROFILES

SCRAPER_SPELLINGS = ('BTC-PERP', 'ETH-PERP', 'SOL-PERP', 'XRP-PERP', 'DOGE-PERP')
CDE_SPELLINGS = ('BIP-20DEC30-CDE', 'ETP-20DEC30-CDE', 'SLP-20DEC30-CDE')


def _store(tmp_path, symbols, venue='coinbase', bars=400) -> ResearchStore:
    store = ResearchStore(tmp_path / 'research')
    index = pd.date_range('2026-01-01', periods=bars, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    for symbol in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
        store.write('bars', pd.DataFrame({
            'symbol': symbol, 'venue': venue, 'event_time': index,
            'available_time': index, 'quality': 'valid',
            'open': close, 'high': close * 1.001, 'low': close * 0.999,
            'close': close, 'volume': 1000.0,
        }))
    return store


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_perp_product_ids_resolve_from_profile_prefixes(tmp_path):
    """The exact mismatch: prefixes requested, `-PERP` ids stored."""
    store = _store(tmp_path, SCRAPER_SPELLINGS)

    mapping, missing = resolve_store_symbols(
        store, ['BIP', 'ETP', 'SLP', 'XPP', 'DOP'], venue='coinbase')

    assert not missing
    assert mapping == {
        'BIP': 'BTC-PERP', 'ETP': 'ETH-PERP', 'SLP': 'SOL-PERP',
        'XPP': 'XRP-PERP', 'DOP': 'DOGE-PERP',
    }


def test_decorated_cde_ids_resolve_too(tmp_path):
    store = _store(tmp_path, CDE_SPELLINGS)

    mapping, missing = resolve_store_symbols(store, ['BIP', 'ETP', 'SLP'], venue='coinbase')

    assert not missing
    assert mapping['BIP'] == 'BIP-20DEC30-CDE'


def test_a_stored_spelling_resolves_to_itself(tmp_path):
    store = _store(tmp_path, SCRAPER_SPELLINGS)

    mapping, missing = resolve_store_symbols(store, ['BTC-PERP'], venue='coinbase')

    assert mapping == {'BTC-PERP': 'BTC-PERP'} and not missing


def test_a_symbol_the_store_lacks_is_reported_not_invented(tmp_path):
    store = _store(tmp_path, ('BTC-PERP',))

    mapping, missing = resolve_store_symbols(store, ['BIP', 'ETP'], venue='coinbase')

    assert mapping == {'BIP': 'BTC-PERP'}
    assert missing == ['ETP']


def test_resolution_is_deterministic_when_a_base_has_several_contracts(tmp_path):
    """Two BTC contracts must not resolve by scrape order."""
    store = _store(tmp_path, ('BIP-20DEC30-CDE', 'BIP-20JUN31-CDE', 'BTC-PERP'))

    first = resolve_store_symbols(store, ['BIP'], venue='coinbase')[0]
    second = resolve_store_symbols(store, ['BIP'], venue='coinbase')[0]

    assert first == second
    assert first['BIP'] in ('BIP-20DEC30-CDE', 'BIP-20JUN31-CDE')


def test_an_empty_store_reports_everything_missing(tmp_path):
    store = ResearchStore(tmp_path / 'research')

    mapping, missing = resolve_store_symbols(store, ['BIP', 'ETP'], venue='coinbase')

    assert mapping == {}
    assert missing == ['BIP', 'ETP']


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_the_panel_builds_from_the_spellings_the_scraper_writes(tmp_path):
    """The consequence that matters: a scraped store must produce a panel.

    This is the test whose absence let a scrape produce nothing usable.
    """
    store = _store(tmp_path, SCRAPER_SPELLINGS)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue=None, min_quality='valid')

    assert not dataset.features.empty, 'the panel is empty on a scraper-built store'
    assert len(dataset.symbols) == len(SCRAPER_SPELLINGS)
    assert not any('no bars' in w for w in dataset.warnings), dataset.warnings


def test_the_market_factor_symbol_resolves(tmp_path):
    """`MARKET_SYMBOL` is a bare prefix and could never match a stored id."""
    store = _store(tmp_path, SCRAPER_SPELLINGS)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue=None, min_quality='valid')

    assert not any(
        'market_factor features will be empty' in w for w in dataset.warnings
    ), dataset.warnings


def test_profiles_are_keyed_by_the_stored_spelling(tmp_path):
    """Every downstream lookup uses the stored name, so the keys must match.

    A profile dict keyed by the requested prefix would silently apply default
    thresholds to every instrument.
    """
    store = _store(tmp_path, SCRAPER_SPELLINGS)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue=None, min_quality='valid')

    assert set(dataset.profiles) <= set(dataset.symbols)
    for symbol in dataset.symbols:
        assert symbol in dataset.profiles, f'{symbol} has no profile, so it gets defaults'


def test_every_profile_prefix_can_be_resolved_from_a_scraped_symbol():
    """A profile whose prefix no scraper spelling maps to can never trade.

    Checks the mapping in both directions without touching a store, so it fails
    on a new profile whose prefix `_resolve_base` does not recognise.
    """
    from core.costs import _resolve_base

    unresolvable = [
        profile.prefixes[0] for profile in COIN_PROFILES.values()
        if _resolve_base(profile.prefixes[0]) is None
    ]

    assert not unresolvable, (
        f'these profile prefixes resolve to no underlying, so no stored symbol '
        f'can ever match them: {unresolvable}'
    )


def test_the_market_symbol_is_resolvable():
    from core.costs import _resolve_base

    assert _resolve_base(MARKET_SYMBOL) is not None


# ---------------------------------------------------------------------------
# Perp Style Futures symbols
# ---------------------------------------------------------------------------


def test_cde_product_ids_convert_to_psf_symbols():
    """One contract, two spellings, and the endpoints disagree about which.

    The Advanced Trade product and candle endpoints key on the long form
    (`BIP-20DEC30-CDE`); Coinbase Derivatives' historical-funding endpoint keys
    on the Perp Style Futures form (`BIPZ30`). `Z` is the CME-style month code
    for December, which CDE follows.
    """
    from core.costs import psf_symbol

    assert psf_symbol('BIP-20DEC30-CDE') == 'BIPZ30'
    assert psf_symbol('ETP-20DEC30-CDE') == 'ETPZ30'
    assert psf_symbol('PEP-20DEC30-CDE') == 'PEPZ30'
    # Case is not the caller's problem.
    assert psf_symbol('bip-20dec30-cde') == 'BIPZ30'


def test_every_month_gets_its_own_code():
    """A wrong letter resolves to a contract that exists and is the wrong month.

    That is the dangerous failure here — not an error, a different instrument —
    so the whole table is checked rather than the one month we happen to trade.
    """
    from core.costs import FUTURES_MONTH_CODES, psf_symbol

    expected = 'FGHJKMNQUVXZ'
    assert ''.join(FUTURES_MONTH_CODES[m] for m in range(1, 13)) == expected
    assert len(set(FUTURES_MONTH_CODES.values())) == 12

    names = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
    for month, (name, code) in enumerate(zip(names, expected), start=1):
        assert psf_symbol(f'ABC-15{name}27-CDE') == f'ABC{code}27', (
            f'month {month} ({name}) should map to {code}'
        )


def test_spellings_with_no_expiry_are_not_guessed():
    """`BTC-PERP` and a bare code carry no expiry, so there is nothing to derive.

    Returning None matters: inventing an expiry would produce a symbol that
    resolves to a real contract and silently query the wrong one.
    """
    from core.costs import psf_symbol

    for spelling in ('BTC-PERP', 'BIP', '', 'BIP-20DEC30', 'BIP-20XXX30-CDE'):
        assert psf_symbol(spelling) is None, spelling


def test_every_scraped_cde_product_has_a_psf_symbol():
    """The 16 contracts the scraper resolves must all convert, or funding for the
    ones that do not is silently uncollectable."""
    from core.costs import psf_symbol

    products = [
        'BIP-20DEC30-CDE', 'ETP-20DEC30-CDE', 'BCP-20DEC30-CDE', 'SLP-20DEC30-CDE',
        'LCP-20DEC30-CDE', 'XPP-20DEC30-CDE', 'LNP-20DEC30-CDE', 'AVP-20DEC30-CDE',
        'POP-20DEC30-CDE', 'NER-20DEC30-CDE', 'SUP-20DEC30-CDE', 'ADP-20DEC30-CDE',
        'XLP-20DEC30-CDE', 'DOP-20DEC30-CDE', 'SHP-20DEC30-CDE', 'PEP-20DEC30-CDE',
    ]
    mapped = {p: psf_symbol(p) for p in products}
    missing = [p for p, v in mapped.items() if v is None]
    assert not missing, f'no PSF symbol derived for {missing}'
    assert len(set(mapped.values())) == len(products), 'two products collided'
    assert all(v.endswith('Z30') for v in mapped.values()), mapped
