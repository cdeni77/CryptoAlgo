"""Prove the Kalshi credentials and the series tickers, before any money moves.

Four questions, in the order they can fail, and each one answered separately so
a failure names itself:

1. **Are the credentials loadable?** A key id and an RSA private key. This step
   needs no network, so a malformed PEM is caught here rather than as a 401.
2. **Does the venue accept the signature?** `GET /portfolio/balance` is the
   cheapest authenticated call, and it also tells you what you have to trade
   with.
3. **Do the series tickers exist?** The defaults in `.env.example` are
   **unverified**. This lists what the venue actually returns for each one.
4. **Can a market be resolved for the next window?** The real test. Markets are
   found by close time, not by pattern, so this is the step that proves the whole
   chain — series, clock alignment, and market status — and it prints the book so
   the assumed 1c half-spread can be checked against a real one.

Read-only throughout. It constructs the client without `live=True`, so it
*cannot* place an order even if something in it tried.

    python -m scripts.check_venue
    python -m scripts.check_venue --demo          # against the demo host
    python -m scripts.check_venue --series KXBTC  # try a different ticker
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone

from core.config import DEFAULT_CONFIG
from core.costs import fee_per_contract
from data_collection.kalshi_client import (
    DEMO_BASE_URL, KalshiClient, KalshiError, _parse_time,
)
from scripts.live import SERIES_BY_SYMBOL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--demo', action='store_true',
                        help='Use the demo host instead of production.')
    parser.add_argument('--series', type=str, default=None,
                        help='Comma-separated series tickers to try instead of '
                             'the configured ones.')
    parser.add_argument('--windows', type=int, default=2,
                        help='How many upcoming windows to try resolving.')
    return parser


def ok(text: str) -> None:
    print(f'  [ok]   {text}')


def bad(text: str) -> None:
    print(f'  [FAIL] {text}')


def note(text: str) -> None:
    print(f'         {text}')


async def main() -> int:
    args = build_parser().parse_args()
    base_url = DEMO_BASE_URL if args.demo else None
    failures = 0

    print('=' * 78)
    print('Kalshi preflight — read only, cannot place an order')
    print('=' * 78)

    # ---- 1. credentials --------------------------------------------------
    print('\n1. credentials')
    key_id = os.getenv('KALSHI_KEY_ID', '')
    pem = os.getenv('KALSHI_PRIVATE_KEY', '')
    path = os.getenv('KALSHI_PRIVATE_KEY_PATH', '')
    if not key_id:
        bad('KALSHI_KEY_ID is not set')
        note('set it in .env, or export it in this shell')
        return 1
    ok(f'KALSHI_KEY_ID present ({key_id[:8]}...)')
    if pem:
        ok('KALSHI_PRIVATE_KEY present (inline PEM)')
        note('a PEM in an env var appears in `docker inspect` and process '
             'listings; KALSHI_PRIVATE_KEY_PATH is the better form')
    elif path:
        ok(f'KALSHI_PRIVATE_KEY_PATH present ({path})')
    else:
        bad('neither KALSHI_PRIVATE_KEY nor KALSHI_PRIVATE_KEY_PATH is set')
        return 1

    try:
        client = KalshiClient(base_url=base_url, live=False)
    except Exception as exc:  # noqa: BLE001 - a bad PEM raises from cryptography
        bad(f'the private key could not be loaded: {exc}')
        note('it must be an unencrypted PKCS#8 or PKCS#1 RSA key in PEM form, '
             'starting with -----BEGIN PRIVATE KEY----- or -----BEGIN RSA '
             'PRIVATE KEY-----')
        return 1
    if not client.configured:
        bad('the client reports itself unconfigured')
        return 1
    ok('the private key loaded and the client can sign')
    note(f'host {client.base_url}')

    async with client:
        # ---- 2. the venue accepts the signature --------------------------
        print('\n2. authentication')
        try:
            balance = await client.balance()
            ok(f'the venue accepted the signature — balance ${balance:,.2f}')
            if balance <= 0:
                note('a zero balance still proves auth; it just cannot trade')
        except KalshiError as exc:
            bad(f'the request failed: {exc}')
            message = str(exc)
            # A network policy and a rejected signature both surface here as an
            # HTTP error, and the fix is completely different. Say which.
            if 'allowlist' in message or 'egress' in message or '407' in message:
                note('this is a NETWORK policy, not an authentication failure: '
                     'something between here and Kalshi refused the connection. '
                     'The credentials were never seen. Allow the host, or run '
                     'this from a machine that can reach it.')
            elif '401' in message or '403' in message:
                note('the venue rejected the signature. Usually the key id and '
                     'the private key are not a pair, or the key was created on '
                     'the other host — production keys do not work on demo, and '
                     'vice versa.')
            else:
                note('neither an auth rejection nor an obvious network block; '
                     'the message above is the venue\'s own.')
            return 1

        # ---- 3. the series exist ----------------------------------------
        print('\n3. series tickers')
        series_map = (
            {f'series-{i}': s for i, s in enumerate(args.series.split(','))}
            if args.series else dict(SERIES_BY_SYMBOL)
        )
        live_series: dict[str, str] = {}
        for symbol, series in series_map.items():
            try:
                markets = await client.markets(series_ticker=series, status='open',
                                               limit=200)
            except KalshiError as exc:
                bad(f'{symbol}: {series} -> {exc}')
                failures += 1
                continue
            if not markets:
                bad(f'{symbol}: series {series!r} returned no open markets')
                note('the series may be renamed, or closed right now. Try '
                     '--series with a candidate, or list what is open on the '
                     'venue and pass the prefix you see.')
                failures += 1
                continue
            ok(f'{symbol}: series {series!r} has {len(markets)} open markets')
            example = markets[0]
            note(f'e.g. {example.get("ticker")} closing '
                 f'{example.get("close_time")}')
            live_series[symbol] = series

        # ---- 4. resolve the next windows --------------------------------
        print('\n4. resolving the next windows')
        if not live_series:
            bad('no series resolved, so there is nothing to match a window '
                'against')
            return 1

        now = datetime.now(timezone.utc)
        window = now.replace(second=0, microsecond=0)
        window -= timedelta(minutes=window.minute % DEFAULT_CONFIG.window_minutes)
        for step in range(1, args.windows + 1):
            settle = window + timedelta(
                minutes=DEFAULT_CONFIG.window_minutes * step)
            print(f'\n  window settling {settle:%H:%M:%S} UTC')
            for symbol, series in live_series.items():
                market = await client.resolve_window_market(series, settle)
                if market is None:
                    bad(f'{symbol}: no {series} market closes within 90s of '
                        f'{settle:%H:%M:%S}')
                    note('this is the abstention path working. If it happens for '
                         'every window the series is wrong; if only for some, '
                         'the venue may not list that far ahead yet.')
                    failures += 1
                    continue
                ticker = str(market.get('ticker', ''))
                quote = await client.quote(ticker)
                if not quote.tradeable():
                    bad(f'{symbol}: {ticker} resolved but is not tradeable '
                        f'(status {quote.status}, bid {quote.yes_bid}, '
                        f'ask {quote.yes_ask})')
                    failures += 1
                    continue
                spread_cents = (quote.spread or 0) * 100
                fee = float(fee_per_contract(quote.yes_ask, DEFAULT_CONFIG))
                ok(f'{symbol}: {ticker}')
                note(f'book {quote.yes_bid:.2f} / {quote.yes_ask:.2f}  '
                     f'spread {spread_cents:.0f}c  volume {quote.volume:,}  '
                     f'OI {quote.open_interest:,}')
                note(f'buying "up" at {quote.yes_ask:.2f} costs '
                     f'{quote.yes_ask + fee:.4f} all-in; the backtest assumed '
                     f'{DEFAULT_CONFIG.half_spread_cents:.0f}c of half-spread, '
                     f'this book shows {spread_cents / 2:.1f}c')

    print('\n' + '=' * 78)
    if failures:
        print(f'{failures} check(s) failed. Nothing above placed an order.')
        return 1
    print('every check passed. The credentials work and the series resolve.')
    print()
    print('Next, and still placing nothing:')
    print('  python -m scripts.live --mode live --dry-run')
    print()
    print('That needs a promoted model, which needs data — so it will refuse')
    print('until `scripts.promote` has installed one. The credentials being')
    print('good is not the same as the edge being established.')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
