"""Script entrypoints for the trader package.

Keep this module side-effect free so `python -m scripts.<name>` only loads
the target script instead of importing every script dependency.
"""

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
