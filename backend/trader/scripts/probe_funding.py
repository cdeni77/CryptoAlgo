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


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--product', default='BIP-20DEC30-CDE')
    args = parser.parse_args()

    key = os.environ.get('COINBASE_API_KEY')
    secret = os.environ.get('COINBASE_API_SECRET')
    if not key or not secret:
        print('COINBASE_API_KEY / COINBASE_API_SECRET not set')
        return 1

    client = CoinbaseRESTClient(key, secret)
    try:
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
