"""Set a standing target balance allocation across Kalshi's exchange shards.

Kalshi shards its exchange by category. The `KXBTC15M` / `KXETH15M` / `KXSOL15M`
series live on `exchange_index` 2, and **balances are local to a shard**: money
sitting on shard 0 cannot buy a contract on shard 2. It is refused
`insufficient_balance` against a reported total that includes it, which is the
most confusing possible way to be told.

A one-off transfer fixes today. A standing allocation fixes tomorrow too: the
venue rebalances every ~10 seconds, so a settlement that lands on the wrong
shard heals itself instead of bouncing orders in the middle of a session.

Read-only by default. `--apply` is a separate flag because this moves real
money, and the amount is derived from percentages rather than typed — the
transfer endpoint takes CENTICENTS, where a 100x slip moves a hundred times
the intended sum.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Mapping

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from data_collection.kalshi_client import KalshiClient  # noqa: E402

CRYPTO_SHARD = 2


def allocation_payload(percent_by_shard: Mapping[int, int]) -> dict:
    """The request body, refusing anything that does not total 100.

    The venue rejects a partial allocation. A silently-accepted 90 would leave
    the remaining 10% wherever it happened to sit, which is the failure this
    script exists to prevent.
    """
    total = sum(percent_by_shard.values())
    if total != 100:
        raise ValueError(
            f'allocations must total 100 percent, got {total}. A partial '
            f'allocation strands the remainder on whichever shard holds it.')
    return {'allocations': [{'exchange_index': int(index), 'percent': int(percent)}
                            for index, percent in sorted(percent_by_shard.items())]}


async def _run(shard: int, apply: bool) -> int:
    async with KalshiClient() as kalshi:
        before = await kalshi._request('GET', '/portfolio/balance')  # noqa: SLF001
        rows = before.get('balance_breakdown') or []
        print('  before')
        for row in rows:
            print(f"    shard {row.get('exchange_index')}: "
                  f"${float(row.get('balance') or 0):>10,.2f}")

        payload = allocation_payload({shard: 100})
        if not apply:
            print(f'\n  would POST /portfolio/target_balance_allocation {payload}')
            print('  --dry-run (default): nothing sent. Pass --apply to set it.')
            return 0

        await kalshi._request(  # noqa: SLF001
            'POST', '/portfolio/target_balance_allocation', body=payload)
        print(f'\n  set: 100% to shard {shard}')
        # The rebalance is ASYNCHRONOUS. Reading straight back shows the old
        # figures and looks like a failed no-op; the venue reconciles on roughly
        # a ten-second cycle.
        await asyncio.sleep(20)
        after = await kalshi._request('GET', '/portfolio/balance')  # noqa: SLF001
        print('  after (20s later; the rebalance is asynchronous)')
        for row in after.get('balance_breakdown') or []:
            print(f"    shard {row.get('exchange_index')}: "
                  f"${float(row.get('balance') or 0):>10,.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, default=CRYPTO_SHARD,
                        help=f'Which shard gets 100%% (default {CRYPTO_SHARD}, '
                             f'where the KX*15M crypto series live)')
    parser.add_argument('--apply', action='store_true',
                        help='Actually set it. Without this, nothing is sent.')
    args = parser.parse_args()
    return asyncio.run(_run(args.shard, args.apply))


if __name__ == '__main__':
    raise SystemExit(main())
