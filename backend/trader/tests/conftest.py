from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(scope='session')
def repo_root() -> Path:
    """Repository root, for reading configs/ from tests."""
    return ROOT.parents[1]
