from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import pytest


@pytest.fixture(scope='session')
def repo_root() -> Path:
    """Where `configs/` lives.

    Inside the trader package, not the repository root: the trader's Docker build
    context is `backend/trader`, so anything above it is never copied into the
    image. `configs/` used to sit at the repository root, which meant the fee
    schedule did not exist in the container and every containerised run priced
    contracts at the hardcoded default.
    """
    return ROOT
