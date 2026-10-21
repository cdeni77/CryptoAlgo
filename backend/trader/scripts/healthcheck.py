"""Check that the live process is doing its job. Exit 0 if so, 1 if not.

`supervise` cannot tell a coroutine that is awaiting normally from one that
will await forever. So each component stamps `core.heartbeat` once its loop
has done its job, and this checks the stamps are recent. Only components this
process was told to run are checked.
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
