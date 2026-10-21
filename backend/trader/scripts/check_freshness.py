"""Which datasets have stopped receiving rows? Exit 0 if all current.

Run it by hand, or on a schedule that NOTIFIES rather than restarts. See
`core/freshness.py` for why it is deliberately not the container healthcheck:
a dataset can be legitimately behind, and failing the healthcheck would
restart a process trading real money.

    python -m scripts.check_freshness
    python -m scripts.check_freshness --datasets venue_ladder,venue_depth
    python -m scripts.check_freshness --publish    # and write it for the API

`--publish` writes the same measurement to `core.collection_status.DEFAULT_PATH`,
which is what the dashboard's `/collection` route serves. The live loop already
does this hourly from `store_sync`; this is how to refresh it by hand without
restarting a process that is trading.
"""

from __future__ import annotations

import argparse
import os
import sys

from core import freshness


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store-root', type=str,
                        default=os.getenv('RESEARCH_STORE', 'data/research'))
    parser.add_argument('--datasets', type=str, default=None,
                        help='comma-separated; defaults to everything with an '
                             'expectation')
    parser.add_argument('--publish', action='store_true',
                        help='also write the status document the API serves')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    wanted = ([d.strip() for d in args.datasets.split(',') if d.strip()]
              if args.datasets else None)

    if args.publish:
        from core import collection_status
        document = collection_status.publish(args.store_root, datasets=wanted)
        late = collection_status.late(document)
        if not args.quiet:
            print('published %s' % collection_status.DEFAULT_PATH)
    else:
        late = freshness.stale(args.store_root, datasets=wanted)

    if not args.quiet:
        checked = wanted or [d for d, lim in freshness.EXPECTED_MAX_AGE.items()
                             if lim is not None]
        print('%-20s %14s  %s' % ('dataset', 'newest row', 'state'))
        for dataset in checked:
            seen = freshness.newest_event_time(args.store_root, dataset)
            limit = freshness.EXPECTED_MAX_AGE.get(dataset)
            if seen is None:
                print('%-20s %14s  NO ROWS' % (dataset, '-'))
                continue
            from datetime import datetime, timezone
            age = (datetime.now(timezone.utc) - seen).total_seconds()
            state = 'ok' if (limit is None or age <= limit) else 'STALE'
            print('%-20s %14s  %s' % (dataset, freshness.describe(age), state))

    if not late:
        return 0
    for dataset, age, limit in late:
        print('STALE %s: newest row %s, limit %s'
              % (dataset, freshness.describe(age), freshness.describe(limit)),
              file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
