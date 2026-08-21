"""CDE is not INTX, and the Coinbase SDK exposes both.

Two API families sit side by side on one client:

* `get_futures_balance_summary`, `list_futures_positions`, `get_futures_position`
  — FCM-margined futures. This is CDE, the US venue whose contracts this system
  trades.
* `get_perps_portfolio_balances`, `list_perps_positions`, `get_perps_position`
  — Coinbase International. A US account has no INTX portfolio at all:
  `GET /api/v3/brokerage/portfolios` returns a single DEFAULT entry.

Picking the wrong family has now cost twice. The scraper asked
`/api/v3/brokerage/intx/funding-rates` and got a 404 for every symbol, so a
scrape collected zero funding. The wallet controller called
`get_perps_portfolio_balances()` — which also requires a `portfolio_uuid` it was
not given — so the holdings panel reported `TypeError: missing 1 required
positional argument` as a portfolio status.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]

# INTX-only client methods. Reaching for one of these means reaching for a venue
# this account does not have.
INTX_METHODS = (
    'get_perps_portfolio_balances',
    'get_perps_portfolio_summary',
    'list_perps_positions',
    'get_perps_position',
)


def test_the_wallet_reads_futures_not_intx_perpetuals():
    source = (API_ROOT / 'controllers' / 'wallet.py').read_text()

    used = [name for name in INTX_METHODS if re.search(rf'client\.{name}\b', source)]
    assert not used, (
        f'wallet.py calls INTX-only methods {used}. CDE contracts are FCM '
        f'futures: use get_futures_balance_summary / list_futures_positions.'
    )

    assert 'client.get_futures_balance_summary(' in source, (
        'nothing reads the futures balance, so a CDE account shows no value'
    )
    assert 'client.list_futures_positions(' in source, (
        'nothing reads futures positions, so open CDE contracts never appear'
    )


def test_the_futures_calls_match_the_sdk_signatures():
    """The original failure was an arity error, not a wrong venue alone.

    `get_perps_portfolio_balances(portfolio_uuid)` was called with no arguments.
    Both futures calls take none, so this pins that they still do — if the SDK
    ever adds a required argument, this fails here rather than in the response
    body of a live request.
    """
    import inspect

    try:
        from coinbase.rest import RESTClient
    except ImportError:                                    # pragma: no cover
        pytest.skip('coinbase-advanced-py not installed')

    for name in ('get_futures_balance_summary', 'list_futures_positions'):
        signature = inspect.signature(getattr(RESTClient, name))
        required = [
            p for n, p in signature.parameters.items()
            if n != 'self'
            and p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        assert not required, f'{name} now requires {[p.name for p in required]}'
