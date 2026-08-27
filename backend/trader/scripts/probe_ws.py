"""THROWAWAY. Phase 0 capture for the WebSocket design. Delete after Task 10.

Records every WS frame verbatim alongside periodic REST orderbook snapshots of
the same tickers, so the two can be compared at the same instant. That
comparison is the only evidence that folding deltas reproduces the real book.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from core.config import series_to_symbol
from data_collection.kalshi_client import KalshiClient

WS_URL = os.getenv('KALSHI_WS_URL') or 'wss://api.elections.kalshi.com/trade-api/ws/v2'
WS_PATH = '/trade-api/ws/v2'


async def open_tickers(client: KalshiClient) -> list[str]:
    out = []
    for series in series_to_symbol():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'status': 'open', 'limit': 5})
        out += [m['ticker'] for m in payload.get('markets', []) if m.get('ticker')]
    return out


async def rest_sampler(client, tickers, sink, every: float):
    while True:
        for ticker in tickers:
            try:
                book = await client._request(  # noqa: SLF001
                    'GET', f'/markets/{ticker}/orderbook')
            except Exception as exc:  # noqa: BLE001
                book = {'error': str(exc)[:200]}
            book.setdefault('ticker', ticker)
            sink({'t': time.time(), 'kind': 'rest', 'payload': book})
        await asyncio.sleep(every)


async def run(args) -> int:
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(
        os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open('w')

    def sink(record):
        handle.write(json.dumps(record) + '\n')
        handle.flush()

    async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                            private_key_pem=pem) as client:
        tickers = await open_tickers(client)
        print(f'subscribing to {len(tickers)}: {tickers}', flush=True)
        headers = client._headers('GET', WS_PATH)  # noqa: SLF001
        headers.pop('Content-Type', None)
        sampler = asyncio.create_task(
            rest_sampler(client, tickers, sink, args.rest_every))
        frames = 0
        try:
            async with client._session.ws_connect(  # noqa: SLF001
                    WS_URL, headers=headers, heartbeat=10) as ws:
                await ws.send_json({'id': 1, 'cmd': 'subscribe', 'params': {
                    'channels': ['orderbook_delta'], 'market_tickers': tickers}})
                deadline = time.time() + args.seconds
                async for msg in ws:
                    try:
                        payload = json.loads(msg.data)
                    except (ValueError, TypeError):
                        continue
                    sink({'t': time.time(), 'kind': 'ws', 'payload': payload})
                    frames += 1
                    if frames % 500 == 0:
                        print(f'{frames} frames', flush=True)
                    if time.time() > deadline:
                        break
        finally:
            sampler.cancel()
            handle.close()
    print(f'wrote {out} ({frames} ws frames)', flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seconds', type=float, default=1200.0)
    p.add_argument('--rest-every', type=float, default=30.0)
    p.add_argument('--out', default='tests/fixtures/ws/kalshi_capture.jsonl')
    return p


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run(build_parser().parse_args())))
