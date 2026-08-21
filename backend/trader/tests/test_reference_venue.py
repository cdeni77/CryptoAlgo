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
IP, so `--reference-venue binance` — which used to be the default — yields
nothing unless the scrape went through a proxy, and if the scraper's fallback
served a different exchange instead, the bars are stamped with *that* venue's
name and `binance` still matches nothing. The default is `coinbase_spot` now:
reachable, deeper than the nano perp, and the market the perp's index is built
from. `binance` stays in the tests below because it is what a stale config or an
explicit flag still asks for.
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


# ---------------------------------------------------------------------------
# Coinbase spot as the reference venue
# ---------------------------------------------------------------------------


def test_coinbase_spot_can_serve_as_the_reference_venue(tmp_path):
    """The fix for a geo-blocked reference venue, without leaving the exchange.

    Binance, OKX and Bybit all answer 451 to a US IP, which empties the
    cross-venue group. Coinbase's own spot book is deeper than the nano perp,
    quotes the same underlying, and is reachable — and it is the market the
    perp's index is built from, so its basis is the thing that actually drives
    funding.

    `resolve_base` already maps both spellings to the same base
    (`BTC-USD` -> BTC, `BIP-20DEC30-CDE` -> BTC), so the only requirement is that
    spot is stored under its own venue label.
    """
    store = ResearchStore(tmp_path / 'research')
    # Three instruments, not two: the relative groups are cross-sectionally
    # standardised with `min_universe=3`, so a smaller panel legitimately yields
    # NaN and would make this test fail for the wrong reason.
    perps = ('BIP-20DEC30-CDE', 'ETP-20DEC30-CDE', 'SLP-20DEC30-CDE')
    spot = ('BTC-USD', 'ETH-USD', 'SOL-USD')
    _write(store, perps, venue='coinbase')
    _write(store, spot, venue='coinbase_spot', seed=7)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue='coinbase_spot',
        min_quality='valid',
    )

    assert not dataset.features.empty
    populated = [c for c in CROSS_VENUE_COLUMNS if dataset.features[c].notna().any()]
    assert populated == list(CROSS_VENUE_COLUMNS), (
        f'spot is in the store but these stayed empty: '
        f'{sorted(set(CROSS_VENUE_COLUMNS) - set(populated))}'
    )
    assert not [w for w in dataset.warnings if 'cross-venue' in w], dataset.warnings

    # The basis has to be a real number, not a constant.
    basis = dataset.features['basis_bps'].dropna()
    assert len(basis) > 100
    assert basis.std() > 0, 'basis is constant, so it is not measuring two markets'


def test_one_venue_label_for_both_makes_the_basis_meaningless(tmp_path):
    """Why `--venue-label` exists. This is the trap it prevents.

    Store the perp and its spot index under the same venue and they resolve to
    the same base, so the reference lookup can return the instrument itself. A
    basis against itself is identically zero — a column full of a plausible
    number that measures nothing.
    """
    store = ResearchStore(tmp_path / 'research')
    _write(store, ('BIP-20DEC30-CDE',), venue='coinbase')
    _write(store, ('BTC-USD',), venue='coinbase', seed=7)

    dataset = load_dataset(
        store, venue='coinbase', reference_venue='coinbase',
        min_quality='valid',
    )

    basis = dataset.features['basis_bps'].dropna() if not dataset.features.empty else None
    if basis is not None and len(basis):
        assert (basis.abs() < 1e-9).all() or basis.std() == 0, (
            'expected a degenerate basis when both legs share a venue label; if '
            'this now measures something real, the resolution changed and the '
            '--venue-label guidance in CLAUDE.md should be revisited'
        )


# ---------------------------------------------------------------------------
# The live loop has to keep the reference venue current
# ---------------------------------------------------------------------------


def test_the_cycle_refreshes_the_reference_venue():
    """A frozen reference does not go missing — it gets forward-filled.

    `cross_venue_features` reindexes the reference series onto the panel index
    and `ffill()`s it, so a spot series that stops updating has its last close
    carried forward indefinitely. The basis then drifts with the perp against a
    frozen number: a live feature that looks alive and measures nothing.

    `_scrape` auto-resolves the perp contracts and never touches spot, so
    without a second step the reference would have gone stale from the first
    hour the loop ran.
    """
    import argparse
    import inspect
    from unittest import mock

    from scripts import live_orchestrator as orchestrator

    cycle = inspect.getsource(orchestrator._run_cycle)
    assert '_scrape_reference(' in cycle, (
        'the cycle no longer refreshes the reference venue, so the basis will '
        'be computed against a forward-filled stale price'
    )
    assert cycle.index('_scrape_reference(') < cycle.index('_sync_store('), (
        'the refresh must run before the store sync, or the new bars are not '
        'in the panel this cycle builds'
    )


def test_the_refresh_only_applies_to_coinbase_spot():
    """`--spot-universe` scrapes Coinbase products. Other venues come via CCXT
    on the perp scrape's own fallback path and must not trigger it."""
    import argparse
    from unittest import mock

    from scripts import live_orchestrator as orchestrator

    for reference, expected in (
        ('coinbase_spot', True), ('COINBASE_SPOT', True),
        ('binance', False), ('', False), (None, False),
    ):
        args = argparse.Namespace(reference_venue=reference, db_path='x.db')
        with mock.patch.object(orchestrator, '_run_step') as step:
            orchestrator._scrape_reference(args, 6)
        assert step.called is expected, f'reference_venue={reference!r}'
        if expected:
            command = step.call_args[0][1]
            assert '--spot-universe' in command
            assert '--backfill-hours' in command


# ---------------------------------------------------------------------------
# A spot-only scrape collects no funding, and that is not a failure
# ---------------------------------------------------------------------------


def test_spot_has_no_funding_to_collect():
    """One predicate decides both who gets a funding lookup and who is owed one.

    `resolve_coinbase_funding_product_map` skips spot spellings because
    `_extract_coin_code('BTC-USD')` resolves to 'BIP' and would otherwise file
    the perp's rate under a spot key. The exit code has to use the same rule or
    the two disagree.
    """
    from core.costs import spot_universe
    from core.profiles import COIN_PROFILES
    from scripts.run_pipeline import perpetual_symbols

    spot = spot_universe(sorted(COIN_PROFILES))
    assert spot, 'no spot spellings to test against'
    assert perpetual_symbols(spot) == [], (
        f'spot symbols treated as perpetuals: {perpetual_symbols(spot)}'
    )

    perps = ['BIP-20DEC30-CDE', 'ETP-26MAR26-CDE', 'DOP-26JUN26-CDE']
    assert perpetual_symbols(perps) == perps
    assert perpetual_symbols(perps + spot) == perps


def test_a_spot_only_scrape_exits_clean():
    """Zero funding is a failure only when something in the run should have had it.

    `run_pipeline` exits 1 when it collected no funding, which is right for a
    perp scrape — a price-only dataset cannot test the carry hypothesis at all.
    On the reference venue it was wrong twice over: spot has no funding by
    definition, so a completely correct run exited 1, and
    `live_orchestrator._run_step` raises on a non-zero exit — aborting the cycle
    before the store sync, every hour, forever.
    """
    import ast
    import inspect

    from scripts import run_pipeline

    tree = ast.parse(inspect.getsource(run_pipeline))

    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # The innermost guard only: `if not args.ohlcv_only:` encloses this one
        # and would match a subtree search.
        direct = [statement for statement in node.body
                  if isinstance(statement, ast.Expr)
                  and 'funding: 0 rates collected' in ast.dump(statement)]
        if not direct:
            continue
        guards.append({n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)})

    assert len(guards) == 1, f'expected one funding-failure guard, found {len(guards)}'
    assert 'perp_symbols' in guards[0], (
        'the funding failure is not conditioned on the run containing perpetuals, '
        'so a spot-only reference scrape exits 1 and aborts the live cycle'
    )
