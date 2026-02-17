"""Script entrypoints for the trader package.

Use module-qualified execution, e.g.:
    python -m scripts.live_orchestrator
"""

from . import compute_features
from . import live_orchestrator
from . import optimize
from . import paper_engine
from . import parallel_launch
from . import run_pipeline
from . import train_model
from . import validate_robustness

__all__ = [
    "compute_features",
    "live_orchestrator",
    "optimize",
    "paper_engine",
    "parallel_launch",
    "run_pipeline",
    "train_model",
    "validate_robustness",
]
