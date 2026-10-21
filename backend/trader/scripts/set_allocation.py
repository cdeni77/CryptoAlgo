"""Set a standing target balance allocation across Kalshi's exchange shards.

Balances are local to a shard: money on shard 0 cannot buy a contract on
shard 2, and the order is refused `insufficient_balance`. A standing
allocation keeps funds where the traded series live; the venue rebalances
about every ten seconds.

Read-only by default. `--apply` is separate because this moves real money.
Amounts are percentages, never typed sums, because the transfer endpoint
takes centicents.
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
    """Return the request body; refuse percentages that do not total 100."""
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
        # The rebalance is asynchronous, about every ten seconds, so an
        # immediate read-back still shows the old figures.
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
