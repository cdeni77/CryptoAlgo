"""A query that cannot finish must FAIL, not fill the disk.

Unbounded, a killed DuckDB query leaves its spill behind. Measured
2026-10-22: 95 orphaned files totalling 352 GB took the host to 100% while
the live trader was running on it, and containers could no longer start. An
error is recoverable; a full disk during live trading is not.
"""

from __future__ import annotations

from pathlib import Path

from core import datastore


def test_the_spill_is_bounded():
    with datastore.connect() as con:
        got = con.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0]
    assert got, 'max_temp_directory_size is unset; a runaway query can fill the disk'
    assert 'GiB' in got or 'GB' in got or 'MiB' in got


def test_the_spill_goes_under_the_store_root_not_the_cwd(tmp_path):
    """Scattered spill is spill nobody finds. It also must not land in
    whatever directory a script happened to be run from."""
    with datastore.connect(tmp_path) as con:
        got = con.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    assert Path(got) == tmp_path / datastore.TEMP_SUBDIR
    assert (tmp_path / datastore.TEMP_SUBDIR).is_dir()


def test_nothing_connects_to_duckdb_unbounded():
    """Every read path must go through `connect`, or one of them reintroduces
    the unbounded spill while the others look fixed."""
    import re

    for name in ('core/datastore.py', 'core/index_features.py'):
        src = Path(name).read_text()
        bare = re.findall(r'duckdb\.connect\(\)', src)
        # datastore.connect itself holds the single legitimate call
        allowed = 1 if name.endswith('datastore.py') else 0
        assert len(bare) == allowed, f'{name} has {len(bare)} unbounded duckdb.connect()'


def test_no_script_or_research_module_connects_unbounded():
    """One unbounded caller reintroduces the whole failure while the rest
    look fixed. 352 GB of orphaned spill took the host to 100% on
    2026-10-22, with the live trader running on it.

    `core/datastore.connect` holds the single legitimate bare call.
    """
    import subprocess

    out = subprocess.run(
        ['grep', '-rn', 'duckdb.connect()', '--include=*.py',
         'research/', 'core/', 'scripts/'],
        capture_output=True, text=True).stdout.strip().splitlines()
    unbounded = [l for l in out if 'core/datastore.py' not in l]
    assert not unbounded, 'unbounded duckdb.connect():\n' + '\n'.join(unbounded)
