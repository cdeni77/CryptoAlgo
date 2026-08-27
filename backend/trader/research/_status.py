"""Where the data pipeline actually stands. Run this first thing.

Reports what is collected, from where, over what span, and what is still
running — so the morning question ("is it all there?") is answered by
measurement rather than by reading the launch script and hoping.
"""

from __future__ import annotations

# This file moved into research// during a repo cleanup; `core`/`scripts`
# are packages rooted at backend/trader/, which Python does not add to
# sys.path automatically for a script run from a subdirectory (only the
# script's OWN directory is added). Without this, every `from core...`
# import below raises ModuleNotFoundError the moment the file is not sitting
# directly in backend/trader/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
import os
import sqlite3
import subprocess

import pandas as pd

from core.datastore import ResearchStore

pd.set_option('display.width', 210)


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return ''


def lines(path: str) -> int:
    try:
        with open(path) as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _collection_progress() -> None:
    """Where collection stands, from the ledger rather than from a log tail.

    `empty` is a RESULT (the venue answered, no book) and is never retried;
    `pending` is work not yet attempted. Keeping them apart is the whole point
    of the table — conflating them is what produced four wrong coverage claims
    before it existed.
    """
    path = os.path.join('data', 'collection', 'ledger.db')
    if not os.path.exists(path):
        return
    try:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=5)
        counts = dict(con.execute(
            'SELECT status, COUNT(*) FROM collection_ledger GROUP BY status'))
        con.close()
    except Exception as exc:                                  # noqa: BLE001
        print(f'  collection ledger unreadable: {str(exc)[:60]}')
        return
    total = sum(counts.values())
    if not total:
        return
    done = counts.get('ok', 0) + counts.get('empty', 0)
    print(f'\n  collection ledger: {done:,}/{total:,} attempted '
          f'({100 * done / total:.1f}%)')
    print(f'    ok {counts.get("ok", 0):,}   empty {counts.get("empty", 0):,}'
          f'   error {counts.get("error", 0):,}   pending '
          f'{counts.get("pending", 0):,}')


def main() -> int:
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    print('=' * 96)
    print('RESEARCH STORE')
    print('=' * 96)
    rows = []
    for name in ('minute_bars', 'venue_quotes', 'venue_ladder', 'pm_ladder',
                 'venue_depth', 'venue_settlements', 'venue_implied_vol'):
        try:
            d = store.read(name)
        except Exception as exc:                          # noqa: BLE001
            rows.append({'dataset': name, 'rows': 0, 'note': str(exc)[:40]})
            continue
        if not len(d):
            rows.append({'dataset': name, 'rows': 0, 'note': 'empty'})
            continue
        tcol = 'window_open' if 'window_open' in d else 'event_time'
        entry = {
            'dataset': name, 'rows': len(d),
            'symbols': d['symbol'].nunique(),
            'venues': ','.join(sorted(d['venue'].unique())),
            'from': str(d[tcol].min())[:16], 'to': str(d[tcol].max())[:16],
        }
        if 'offset_minutes' in d:
            off = sorted(int(x) for x in d['offset_minutes'].dropna().unique())
            entry['offsets'] = f'{off[0]}..{off[-1]} ({len(off)})' if off else '-'
        if name == 'venue_implied_vol' and 'implied_sigma_per_min' in d:
            entry['note'] = (
                f"median {1e4 * d['implied_sigma_per_min'].median():.2f}bp/min, "
                f"R2 {d['r2'].median():.3f}")
        if 'source' in d and d['source'].notna().any():
            entry['note'] = ' '.join(
                f'{k}={v:,}' for k, v in d['source'].value_counts().items())
        rows.append(entry)
    print(pd.DataFrame(rows).fillna('').to_string(index=False))

    print('\n' + '=' * 96)
    print('COLLECTION ARCHIVES (raw, outside the store)')
    print('=' * 96)
    # book_full.jsonl / pm_prices.jsonl / pm_markets.jsonl are gone: that
    # backfill was deleted and replaced by the ledger-driven collection, whose
    # catalogs and gzipped archive live under data/collection/.
    for path, what in (('data/collection/kalshi_catalog.jsonl', 'Kalshi market catalog'),
                       ('data/collection/pm_catalog.jsonl', 'Polymarket market catalog'),
                       ('data/iv_ladder.jsonl', 'KXBTCD implied-vol ladder')):
        n = lines(path)
        size = sh(f'du -h {path} 2>/dev/null | cut -f1') or '-'
        print(f'  {path:38s} {n:>8,} rows  {size:>7s}  {what}')
    arch = sh('du -sh data/collection/archive 2>/dev/null | cut -f1') or '-'
    files = sh('find data/collection/archive -name "*.gz" 2>/dev/null | wc -l') or '0'
    print(f'  {"data/collection/archive":38s} {files:>8s} parts  {arch:>7s}  '
          f'gzipped tick series, both venues')

    print('\n' + '=' * 96)
    print('RUNNING')
    print('=' * 96)
    ps = sh("docker compose -f ../../docker-compose.yml ps "
            "--format '{{.Service}}|{{.Status}}'")
    for line in ps.splitlines():
        if '|' in line:
            svc, status = line.split('|', 1)
            print(f'  {svc:24s} {status}')
    # The `_overnight.sh` / `_gap_fill*.sh` chain is gone: it existed to patch
    # holes in a collection that could not state its own coverage, and the
    # ledger replaces the whole category. See the collection design doc.
    for name, pattern in (('collection (ledger-driven)', 'run_collection'),
                          ('settlements', 'collect_settlements'),
                          ('depth refresher', '_depth_loop.sh'),
                          ('stream recorder', 'record_stream')):
        # `pgrep -f X` run through a shell matches the shell's OWN command
        # line, so every pattern reported "running". Bracketing the first
        # character makes the pattern not match itself, which is the same trick
        # `ps | grep [x]` uses.
        alive = bool(sh(f'pgrep -f "[{pattern[0]}]{pattern[1:]}"'))
        print(f'  {name:28s} {"running" if alive else "not running"}')

    _collection_progress()

    print('\n' + '=' * 96)
    print('LIVE TRADING')
    print('=' * 96)
    q = ("select status, count(*) n, sum(contracts) c from order_tickets "
         "group by 1 order by 2 desc")
    out = sh('docker compose -f ../../docker-compose.yml exec -T db '
             f'psql -U postgres -d trades_db -t -A -F"|" -c "{q}"')
    for line in out.splitlines():
        if '|' in line:
            status, n, c = (line.split('|') + ['', ''])[:3]
            print(f'  tickets {status:10s} {n:>4}  ({c} contracts)')
    out = sh('docker compose -f ../../docker-compose.yml exec -T db psql -U postgres '
             '-d trades_db -t -A -F"|" -c "select mode, bankroll, realized_pnl '
             'from account order by id limit 1"')
    if '|' in out:
        mode, bankroll, realised = out.split('|')[:3]
        print(f'  account {mode}  bankroll ${float(bankroll):.2f}  '
              f'realised ${float(realised):+.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
