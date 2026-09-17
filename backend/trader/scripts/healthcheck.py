"""Is the live process actually doing its job? Exit 0 if so, 1 if not.

**The check this replaces proved a TCP connect to Postgres.** It never touched
the loop, the venue, the stream or the model, so the process could be wedged,
abstaining on every window, or scoring all-NaN features while
`docker compose ps` read `Up (healthy)` — verified, five hours of it.

`supervise` restarts a component that raises or returns, and handles both
correctly. What it cannot distinguish is a coroutine legitimately awaiting from
one that will await forever — the documented websocket deadlock, and the
store-sync subprocess that had no timeout until today. Both presented as a
healthy container with one log line missing, and the absence of a log line is
not something anyone is required to notice.

So each component stamps `core.heartbeat` at the point its loop has
demonstrably done its job, and this asserts the stamps are recent. Only the
components this process was told to run are checked: `--disable stream` must
not fail the container for a stream nobody asked for.
"""

from __future__ import annotations

import argparse
import os
import sys

from core import heartbeat


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--components', type=str, default=None,
                        help='comma-separated; defaults to $LIVE_COMPONENTS, '
                             'then every known component')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    names = args.components or os.getenv('LIVE_COMPONENTS') or ''
    wanted = [n.strip() for n in names.split(',') if n.strip()] \
        or sorted(heartbeat.MAX_AGE)

    late = heartbeat.stale(wanted)
    if not late:
        if not args.quiet:
            print(f'ok: {len(wanted)} component(s) alive')
        return 0

    for name, seen, limit in late:
        where = 'never reported' if seen is None else f'{seen:.0f}s ago'
        print(f'STALE {name}: last cycle {where}, limit {limit:.0f}s',
              file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
