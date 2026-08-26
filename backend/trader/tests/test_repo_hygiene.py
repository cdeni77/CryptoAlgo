"""Every source file the build needs must be in the repository.

This exists because a blanket `lib/` in `.gitignore` — a Python venv rule —
matched at any depth and silently swallowed `frontend/src/lib/format.ts`. The
file was on the machine that wrote it, so every local typecheck, lint and test
passed. The Docker build copies from the repository, and failed with "Cannot find
module '../lib/format'" in seven files at once.

That is the second time a blanket rule here has hidden a real source file from a
clean clone (the first swallowed a `.gitkeep` that a test depended on). A comment
was not enough either time, so this is the mechanism: a file that the build reads
and git does not know about is a test failure, not a surprise on someone else's
machine.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

# Extensions that are source rather than artefact. Anything the build imports.
SOURCE_SUFFIXES = {'.py', '.ts', '.tsx', '.js', '.jsx', '.css', '.json',
                   '.html', '.yml', '.yaml', '.md', '.sql'}

# Directories that are genuinely generated or vendored.
SKIP_PARTS = {'node_modules', '__pycache__', 'dist', 'build', '.pytest_cache',
              '.git', '.venv', 'venv', 'data', 'models', '.mypy_cache',
              'coverage', '.ruff_cache'}

SEARCH_ROOTS = ('backend', 'frontend')


def _tracked() -> set[str]:
    result = subprocess.run(
        ['git', '-C', str(REPO), 'ls-files'],
        capture_output=True, text=True, check=True)
    return set(result.stdout.splitlines())


def _candidates() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if SKIP_PARTS & set(path.relative_to(REPO).parts):
                continue
            found.append(path)
    return found


def test_git_is_available():
    if not (REPO / '.git').exists():
        pytest.skip('not a git checkout')


def test_every_source_file_is_tracked():
    """A file the build reads and git does not know about is a broken clone."""
    if not (REPO / '.git').exists():
        pytest.skip('not a git checkout')

    tracked = _tracked()
    missing = [
        str(path.relative_to(REPO)) for path in _candidates()
        if str(path.relative_to(REPO)) not in tracked
    ]
    if missing:
        reasons = []
        for relative in missing:
            check = subprocess.run(
                ['git', '-C', str(REPO), 'check-ignore', '-v', relative],
                capture_output=True, text=True)
            why = check.stdout.strip() or 'not ignored, just never added'
            reasons.append(f'  {relative}\n      {why}')
        pytest.fail(
            'source files exist locally but are not in the repository, so a '
            'clean clone or a Docker build will not have them:\n'
            + '\n'.join(reasons))


def test_no_gitignore_rule_matches_a_source_directory_at_any_depth():
    """Anchor rules that name a directory a source tree might also use.

    `lib/`, `build/` and `bin/` are all venv or artefact names *and* plausible
    source directory names. Unanchored, they match everywhere.
    """
    if not (REPO / '.gitignore').exists():
        pytest.skip('no .gitignore')

    risky = {'lib', 'lib64', 'bin', 'build', 'src', 'app', 'api', 'core',
             'utils', 'types', 'hooks', 'components', 'pages', 'scripts'}
    offenders = []
    for number, line in enumerate(
            (REPO / '.gitignore').read_text().splitlines(), start=1):
        rule = line.strip()
        if not rule or rule.startswith('#') or rule.startswith('!'):
            continue
        if rule.startswith('/') or '/' in rule.rstrip('/'):
            continue          # already anchored, or a path rather than a name
        name = rule.rstrip('/')
        if name in risky:
            offenders.append(f'  .gitignore:{number}: {rule!r} — anchor it as /{rule}')
    if offenders:
        pytest.fail(
            'unanchored ignore rules that match a plausible source directory at '
            'any depth:\n' + '\n'.join(offenders))


def test_no_secret_bearing_file_is_tracked():
    """Nothing that carries a credential may be in the repository.

    `.gitignore` matched `.env` by basename only, so it covered
    `backend/api/.env` and missed `frontend/.env` and `frontend/.env.local`. Both
    misses were used: a live Coinbase key and an EC private key were committed in
    b70c78c (deleted the next commit, still reachable on origin/main), and an API
    token in 6097ed1. Deleting a file does not remove the blob, so the only
    control that works is never committing it — which is what this asserts.

    `*.example` files are the templates and are meant to be tracked.
    """
    offenders = []
    for path in sorted(_tracked()):
        name = Path(path).name
        if name.endswith('.example'):
            continue
        if name == '.env' or name.startswith('.env.'):
            offenders.append(f'{path} — an env file')
        elif Path(path).suffix in {'.pem', '.key', '.p8', '.pfx'}:
            offenders.append(f'{path} — a key file')
        elif name.startswith('id_rsa'):
            offenders.append(f'{path} — an SSH private key')

    assert not offenders, (
        'these tracked files carry credentials by convention:\n  '
        + '\n  '.join(offenders)
        + '\nUntrack them (`git rm --cached`), rotate whatever they held, and '
          'check the .gitignore pattern is `.env*` rather than `.env`.'
    )


def test_no_tracked_file_contains_a_private_key_block():
    """A PEM block in any tracked file, whatever the filename.

    The leak that happened was in a file named `.env`, but the pattern above only
    catches names. This catches content, so a key pasted into a config, a
    notebook or a test fixture is caught too.
    """
    # A header alone is not a key. `scripts/check_venue.py` names the PEM header
    # in a diagnostic message and this file quotes it to build the pattern, so
    # matching the header would report both forever and the test would be
    # switched off. Require a header line followed by an actual base64 body.
    header = re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')
    body = re.compile(r'^[A-Za-z0-9+/=]{40,}\s*$', re.MULTILINE)

    offenders = []
    for path in sorted(_tracked()):
        full = REPO / path
        if not full.is_file() or full.stat().st_size > 2_000_000:
            continue
        try:
            text = full.read_text(errors='ignore')
        except OSError:
            continue
        match = header.search(text)
        if match and body.search(text, match.end()):
            offenders.append(path)

    assert not offenders, (
        f'these tracked files contain a private-key block: {offenders}. '
        f'Rotate the key immediately — a commit is permanent even after deletion.'
    )



def _pins(path: Path) -> dict[str, str]:
    """Package -> exact version, from a requirements file."""
    pattern = re.compile(r'^([A-Za-z0-9_.\-]+)==([^\s#]+)')
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            found[match.group(1).lower().replace('_', '-')] = match.group(2)
    return found


def _requirement_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')]


REQUIREMENTS = ('backend/trader/requirements.txt', 'backend/api/requirements.txt')


@pytest.mark.parametrize('relative', REQUIREMENTS)
def test_every_dependency_is_pinned(relative):
    """A build that cannot be reproduced cannot be audited.

    `scikit-learn`, `lightgbm`, `scipy` and `joblib` carried no version at all and
    `coinbase-advanced-py` no constraint whatsoever, so `pip install` at
    image-build time pulled whatever was newest that day — in exactly the packages
    that deserialize the promoted model and sign live orders. A `.joblib` artifact
    is only reliably loadable by the library version that wrote it.
    """
    path = REPO / relative
    unpinned = [line for line in _requirement_lines(path) if '==' not in line]
    assert not unpinned, (
        f'{relative} has unpinned requirements: {unpinned}. Pin them to the '
        f'version the suite passes against, and move them one at a time.'
    )


def test_the_two_requirement_files_agree_on_shared_packages():
    """The API container runs trader scripts.

    `POST /jobs/{module}` executes `python -m scripts.<module>` with the API
    container's interpreter, so `joblib`, `lightgbm`, `scikit-learn`, `numpy` and
    `pandas` are load-bearing on both sides: an artifact written under one version
    and read under another either fails to load or, worse, loads and scores
    differently. `sqlalchemy` matters for a second reason — `backend/api/models/
    serving.py` is a hand-maintained mirror of `core/pg_writer.py`, and two ORM
    versions is a way for that mirror to drift while `test_orm_parity.py` still
    passes.
    """
    trader, api = (_pins(REPO / relative) for relative in REQUIREMENTS)
    shared = sorted(set(trader) & set(api))
    assert shared, 'the two files share no pinned package, which cannot be right'
    disagreements = {name: (trader[name], api[name])
                     for name in shared if trader[name] != api[name]}
    assert not disagreements, (
        'these packages are pinned to different versions in the two containers: '
        + '; '.join(f'{name}: trader {a}, api {b}'
                    for name, (a, b) in sorted(disagreements.items()))
    )
