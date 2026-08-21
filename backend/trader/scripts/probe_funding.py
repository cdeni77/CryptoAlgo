"""Ask Coinbase what funding data this account can actually see.

The scrape collected 768 bars and zero funding rates: every request to
`/api/v3/brokerage/intx/funding-rates` returned 404. INTX is Coinbase
*International* Exchange — a different venue from CDE, the US derivatives
entity whose contracts this account trades — so a US account has no INTX
portfolio and every `intx/` path is correctly a 404.

Rather than guess a replacement path, this dumps what the documented Advanced
Trade endpoints actually return for one CDE product. Run it and paste the
output; the shape of the response decides the fix.

    python -m scripts.probe_funding
    python -m scripts.probe_funding --product BIP-20DEC30-CDE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from data_collection.coinbase_connector import CoinbaseRESTClient

# Paths worth asking about, with why each is a candidate. Nothing here is
# invented: they are the Advanced Trade product endpoints plus the INTX ones
# the code currently uses, so the output shows the contrast directly.
PROBES = (
    ('/api/v3/brokerage/products/{product}', None,
     'the documented product endpoint — for a perp this carries '
     'future_product_details.perpetual_details, where a current funding_rate '
     'and funding_time live if they are exposed at all'),
    ('/api/v3/brokerage/market/products/{product}', None,
     'the public market variant of the same thing'),
    ('/api/v3/brokerage/intx/funding-rates', {'product_id': '{product}', 'limit': 5},
     'what the scraper uses today, and what returned 404'),
    ('/api/v3/brokerage/intx/products/{product}', None,
     'the INTX product endpoint the current-rate path falls back to'),
    ('/api/v3/brokerage/portfolios', None,
     'which portfolio types this key can see — INTX presence or absence is '
     'the whole explanation'),
)


def _trim(payload: Any, depth: int = 0) -> Any:
    """Keep the response readable: dict keys and small values, not every tick."""
    if depth > 3:
        return '...'
    if isinstance(payload, dict):
        return {k: _trim(v, depth + 1) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_trim(item, depth + 1) for item in payload[:2]] + (
            [f'... {len(payload) - 2} more'] if len(payload) > 2 else []
        )
    return payload


async def historical_funding(client, products: list[str], path: str,
                             symbol_param: str, extra: dict[str, str]) -> None:
    """Try the Coinbase Derivatives historical-funding endpoint with PSF symbols.

    The endpoint keys on the Perp Style Futures spelling (`BIPZ30`), not the
    Advanced Trade product id (`BIP-20DEC30-CDE`) — different naming for the same
    contract, and the long form is a standard futures id which has no funding at
    all. `core.costs.psf_symbol` does the conversion.

    `--funding-path` is a parameter rather than a constant because the path is
    not guessed here: paste it from the API reference and this reports what the
    account actually gets back, which is what decides the implementation.
    """
    from core.costs import psf_symbol

    print('=' * 72)
    print(f'HISTORICAL FUNDING — {path}')
    print('=' * 72)
    for product in products:
        psf = psf_symbol(product)
        if psf is None:
            print(f'{product:22} no PSF symbol derived, skipped')
            continue
        params = {symbol_param: psf, **extra}
        try:
            status, data = await client._request(
                'GET', path, params=params, authenticated=True
            )
        except Exception as exc:                          # noqa: BLE001
            print(f'{product:22} ({psf:8}) raised {type(exc).__name__}: {exc}')
            continue
        summary = json.dumps(_trim(data), default=str)
        print(f'{product:22} ({psf:8}) HTTP {status}  {summary[:280]}')
    print()


async def contract_sizes(client, products: list[str]) -> None:
    """Ask the venue for `contract_size`, which settles a 5x open question.

    `core/costs.py:CONTRACT_UNITS` and the shipped fee schedule disagree on
    AVAX (10 vs 5), LINK (50 vs 10) and LTC (5 vs 1). The product endpoint
    reports `future_product_details.contract_size` and `contract_root_unit`
    directly — the venue's own answer, which beats both of ours.
    """
    print('=' * 72)
    print('CONTRACT SIZES, as the venue reports them')
    print('=' * 72)
    print(f'{"product":22} {"code":6} {"unit":6} {"contract_size":>14}  vs CONTRACT_UNITS')
    try:
        from core.costs import CONTRACT_UNITS, resolve_base
    except Exception:                                    # noqa: BLE001
        CONTRACT_UNITS, resolve_base = {}, lambda _: None

    for product in products:
        status, data = await client._request(
            'GET', f'/api/v3/brokerage/products/{product}', authenticated=True
        )
        if status != 200:
            print(f'{product:22} HTTP {status}')
            continue
        d = data.get('future_product_details') or {}
        size = d.get('contract_size')
        base = resolve_base(product)
        ours = CONTRACT_UNITS.get(base) if base else None
        verdict = ''
        if size is not None and ours is not None:
            verdict = 'agree' if float(size) == float(ours) else f'DISAGREE (ours {ours:g})'
        print(f'{product:22} {str(d.get("contract_code","")):6} '
              f'{str(d.get("contract_root_unit","")):6} {str(size):>14}  {verdict}')
    print()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', default='BIP-20DEC30-CDE')
    parser.add_argument('--sizes-only', action='store_true',
                        help='Skip the endpoint probe, just dump contract sizes')
    parser.add_argument('--funding-path', default=None,
                        help='Path of the historical-funding endpoint, from the '
                             'API reference. Given this, the probe queries it '
                             'with PSF symbols (BIPZ30) for every product.')
    parser.add_argument('--funding-symbol-param', default='symbol',
                        help='Name of its symbol query parameter')
    parser.add_argument('--funding-params', default='',
                        help='Extra query params as k=v,k=v (start/end/limit)')
    parser.add_argument(
        '--products',
        default=('BIP-20DEC30-CDE,ETP-20DEC30-CDE,SLP-20DEC30-CDE,XPP-20DEC30-CDE,'
                 'DOP-20DEC30-CDE,AVP-20DEC30-CDE,ADP-20DEC30-CDE,LNP-20DEC30-CDE,'
                 'LCP-20DEC30-CDE,BCP-20DEC30-CDE,NER-20DEC30-CDE,SUP-20DEC30-CDE,'
                 'XLP-20DEC30-CDE,POP-20DEC30-CDE,SHP-20DEC30-CDE,PEP-20DEC30-CDE'),
        help='Comma-separated products for the contract-size table',
    )
    args = parser.parse_args()

    key = os.environ.get('COINBASE_API_KEY')
    secret = os.environ.get('COINBASE_API_SECRET')
    if not key or not secret:
        print('COINBASE_API_KEY / COINBASE_API_SECRET not set')
        return 1

    client = CoinbaseRESTClient(key, secret)
    try:
        products = [p.strip() for p in args.products.split(',') if p.strip()]

        if args.funding_path:
            extra = dict(
                pair.split('=', 1) for pair in args.funding_params.split(',') if '=' in pair
            )
            await historical_funding(
                client, products, args.funding_path,
                args.funding_symbol_param, extra,
            )
            return 0

        await contract_sizes(client, products)
        if args.sizes_only:
            return 0
        for path_template, params_template, why in PROBES:
            path = path_template.format(product=args.product)
            params = (
                {k: str(v).format(product=args.product) for k, v in params_template.items()}
                if params_template else None
            )
            print('=' * 72)
            print(f'GET {path}')
            if params:
                print(f'    params: {params}')
            print(f'    why: {why}')
            try:
                status, data = await client._request(
                    'GET', path, params=params, authenticated=True
                )
            except Exception as exc:                      # noqa: BLE001
                print(f'    -> raised {type(exc).__name__}: {exc}')
                continue
            print(f'    -> HTTP {status}')
            print(json.dumps(_trim(data), indent=2, default=str)[:4000])
            print()
    finally:
        await client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()) or 0)
