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


async def list_all_contracts(client) -> None:
    """Every futures product the venue lists, modelled or not.

    `run_pipeline` filters the venue's product list to `ASSET_TO_CODE_MAP`, which
    has exactly sixteen entries — so its "Found 16 contracts" was the count of
    hardcoded codes that matched, not the count Coinbase offers. This asks
    without a filter, which is the only way to answer "can I add ONDO?".
    """
    from core.costs import resolve_base

    print('=' * 72)
    print('EVERY CDE FUTURES PRODUCT THE VENUE LISTS')
    print('=' * 72)
    status, data = await client._request(
        'GET', '/api/v3/brokerage/products',
        params={'product_type': 'FUTURE'}, authenticated=True,
    )
    if status != 200:
        print(f'HTTP {status}: {data}')
        return

    products = data.get('products', [])
    print(f'{"product_id":24} {"code":6} {"unit":8} {"size":>10} {"funding":>10}  modelled')
    modelled = skipped = 0
    for raw in products:
        d = raw.get('future_product_details') or {}
        pid = raw.get('product_id', '')
        # Only the perpetual-style ones have a funding interval.
        funding = d.get('funding_interval') or ''
        known = resolve_base(pid) is not None
        modelled += known
        skipped += (not known) and bool(funding)
        print(f'{pid:24} {str(d.get("contract_code","")):6} '
              f'{str(d.get("contract_root_unit","")):8} '
              f'{str(d.get("contract_size","")):>10} {str(funding):>10}  '
              f'{"yes" if known else "NO - addable"}')
    print()
    print(f'{len(products)} products listed; {modelled} modelled; '
          f'{skipped} perpetual-style contract(s) not modelled')
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


async def probe_dated_futures(client, expired_probes: list[str]) -> None:
    """Do CDE's dated futures carry usable candle history?

    This decides whether the perp-vs-dated-future spread is testable now or has to
    wait. Unlike funding and open interest — snapshots with no history endpoint —
    candles come from `/products/{id}/candles`, which is a range query. So dated
    contracts might already hold months of bars.

    The trade it would enable matters because of one number: the basis trade's cost
    is dominated by the spot leg at 1.20 percent per side, 240bp round trip, which
    is 97 percent of the total and pushes breakeven to 57 days. Both legs of a
    calendar spread pay perp-style fees instead, roughly 20bp round trip each,
    which would drop breakeven to about six days. That attacks the term that
    dominates everything.

    Two questions, and the second is the one that matters:

    1. How much history does each *listed* dated contract have? They are listed a
       few months before expiry, so expect 60-180 days each — enough to measure a
       term structure, thin for gating.
    2. Does an *expired* contract still serve candles? A calendar spread needs a
       continuous series stitched across expiries. If delisted products still
       answer, years of history can be built today. If they 404, the idea joins
       funding in the accumulate-forward queue.

    Caveat worth carrying into any result: a dated contract on a nano venue is
    likely thinner than the perp, and six of fourteen perps already have median
    fill uncertainty exceeding their own fee. Cheap fees on a stale book is the
    trap that makes a backtest look good and a fill impossible, so run
    `scripts.preflight` on anything this turns up before believing a spread.
    """
    from datetime import datetime, timedelta, timezone

    status, data = await client._request(
        'GET', '/api/v3/brokerage/products',
        params={'product_type': 'FUTURE'}, authenticated=True,
    )
    if status != 200:
        print(f'product list: HTTP {status}')
        return

    perpetual, dated = [], []
    for product in data.get('products', []):
        details = product.get('future_product_details') or {}
        # Only the perpetual-style contracts carry a funding interval.
        target = perpetual if details.get('funding_interval') else dated
        target.append(product.get('product_id', ''))

    print('=' * 78)
    print(f'{len(perpetual)} perpetual, {len(dated)} dated')
    print('=' * 78)

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=400)

    print(f'\n{"listed dated contract":26}{"1h bars":>9}{"days":>7}{"first":>12}{"last":>12}')
    for product in sorted(dated):
        try:
            bars = await client.get_candles_range(
                product_id=product, granularity='1h', start=start, end=end)
        except Exception as exc:                                  # noqa: BLE001
            print(f'{product:26} {type(exc).__name__}: {str(exc)[:40]}')
            continue
        if not bars:
            print(f'{product:26}{0:>9}')
            continue
        stamps = sorted(b.event_time for b in bars)
        span = (stamps[-1] - stamps[0]).total_seconds() / 86_400
        print(f'{product:26}{len(bars):>9}{span:>7.0f}'
              f'{str(stamps[0].date()):>12}{str(stamps[-1].date()):>12}')

    print(f'\n{"expired probe":26}{"result":>9}   '
          f'(does a delisted product still serve candles?)')
    for product in expired_probes:
        try:
            bars = await client.get_candles_range(
                product_id=product, granularity='1h',
                start=end - timedelta(days=250), end=end)
            verdict = f'{len(bars)} bars' if bars else 'empty'
        except Exception as exc:                                  # noqa: BLE001
            verdict = f'{type(exc).__name__}'
        print(f'{product:26}{verdict:>9}')
    print('\nIf the expired probes return bars, a continuous stitched series can be '
          'built today.\nIf they are empty or error, dated futures accumulate '
          'forward only, like funding.')


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', default='BIP-20DEC30-CDE')
    parser.add_argument('--list-contracts', action='store_true',
                        help='Every futures product the venue lists, with which '
                             'ones this system models. Answers "can I add X?".')
    parser.add_argument('--sizes-only', action='store_true',
                        help='Skip the endpoint probe, just dump contract sizes')
    parser.add_argument('--dated-futures', action='store_true',
                        help="How much candle history CDE's dated futures carry, "
                             "and whether an expired contract still serves it. "
                             "Decides whether a perp-vs-dated-future spread is "
                             "testable now: both legs would pay perp-style fees "
                             "instead of 240bp for a spot leg, which is 97 percent "
                             "of the basis trade's cost.")
    parser.add_argument('--expired-probes',
                        default='BIT-26JUN26-CDE,SOL-26JUN26-CDE,ADA-27MAR26-CDE',
                        help='Comma-separated delisted product ids to test')
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

        if args.list_contracts:
            await list_all_contracts(client)
            return 0

        if args.dated_futures:
            await probe_dated_futures(
                client,
                [p.strip() for p in args.expired_probes.split(',') if p.strip()],
            )
            return 0

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
