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
