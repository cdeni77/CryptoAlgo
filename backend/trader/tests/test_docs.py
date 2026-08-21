"""Docs make checkable claims about the tree. Check them.

`AGENTS.md` listed `core/labels.py` as a live core module three lines above the
paragraph describing the net-return targets that replaced it, and documented six
endpoints under a router that had been deleted. A doc that names files which do
not exist is worse than no doc: it sends a reader — or an agent — looking for
code that was removed on purpose, and it lends authority to the design it
describes.

Only the mechanically checkable part is tested here: every path a doc mentions
either exists, or is named in that doc's own explicit "deleted" list.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = ('AGENTS.md', 'CLAUDE.md', 'README.md', 'docs/RESEARCH_PIPELINE.md')

# A backticked token that looks like a repo path: at least one slash, or a
# recognised source extension.
PATH_PATTERN = re.compile(r'`([A-Za-z0-9_./-]+\.(?:py|ts|tsx|md|json|yml|sh|cjs))`')

# Roots a bare path may be relative to, tried in order.
SEARCH_ROOTS = (
    REPO_ROOT,
    REPO_ROOT / 'backend' / 'trader',
    REPO_ROOT / 'backend' / 'api',
    REPO_ROOT / 'frontend',
)

# Paths that are deliberately named while absent. Each is either a file the docs
# record as deleted, or an example rather than a real path.
ALLOWED_ABSENT = {
    # Named in the "deleted in the rebuild" lists, and in the passages
    # explaining what replaced them. Naming them is the point.
    'core/labels.py', 'core/labeling.py', 'core/meta_labeling.py',
    'core/coin_profiles.py', 'core/paper_profile_overrides.py',
    'core/cv_splitters.py', 'core/metrics_significance.py',
    'core/study_significance.py', 'core/overfit_diagnostics.py',
    'core/preprocessing_cv.py', 'core/reason_codes.py', 'core/run_manifest.py',
    'core/trading_costs.py', 'core/execution_sim.py',
    # The same two, named by bare filename where the prose says what absorbed
    # them.
    'trading_costs.py', 'execution_sim.py',
    'scripts/train_model.py', 'scripts/compute_features.py',
    'scripts/validate_robustness.py', 'scripts/prune_features.py',
    'scripts/preflight_check.py', 'features/engineering.py',
    'backend/api/endpoints/trade.py', 'backend/api/controllers/trade.py',
    'frontend/src/api/tradesApi.ts', 'frontend/src/api/index.ts',
    'run_full_pipeline.sh',
    # Illustrative, not real.
    'configs/exchange/*.json',
    'package.json',  # exists, but under two roots; kept explicit for clarity
}


def _doc_paths(text: str) -> set[str]:
    return {m.group(1) for m in PATH_PATTERN.finditer(text)}


_BASENAMES: set[str] | None = None


def _basenames() -> set[str]:
    """Every filename in the tree, for bare references like `App.tsx`.

    Docs legitimately name a file without its directory when the surrounding
    prose has already established where it lives.
    """
    global _BASENAMES
    if _BASENAMES is None:
        skip = {'node_modules', '__pycache__', '.git', 'dist', '.pytest_cache'}
        _BASENAMES = {
            path.name
            for path in REPO_ROOT.rglob('*')
            if path.is_file() and not skip & set(path.parts)
        }
    return _BASENAMES


def _exists(candidate: str) -> bool:
    if '*' in candidate:
        return True
    if any((root / candidate).exists() for root in SEARCH_ROOTS):
        return True
    return '/' not in candidate and candidate in _basenames()


@pytest.mark.parametrize('doc', DOCS)
def test_every_path_a_doc_names_exists(doc):
    path = REPO_ROOT / doc
    if not path.exists():
        pytest.skip(f'{doc} not present')

    missing = sorted(
        candidate
        for candidate in _doc_paths(path.read_text())
        if candidate not in ALLOWED_ABSENT and not _exists(candidate)
    )

    assert not missing, (
        f'{doc} names paths that do not exist:\n  ' + '\n  '.join(missing)
        + '\n\nEither the path moved, or the doc is describing code that was '
          'deleted. If it is deliberately naming something deleted, add it to '
          'ALLOWED_ABSENT in this file with the reason.'
    )


def test_the_path_check_would_notice_a_dead_path():
    """The guard above is only worth having if it fails on a real regression."""
    assert not _exists('core/definitely_not_here.py')
    assert _exists('core/targets.py'), 'the search roots are wrong'
