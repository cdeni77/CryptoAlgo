# Phase 2 migration notes: significance metrics + gates

## What changed

- PSR and DSR calculations are now centralized in `core/metrics_significance.py`.
- `scripts/optimize.py` and `scripts/parallel_launch.py` consume the centralized helpers.
- Results now include:
  - `psr_cv`
  - `psr_holdout`
  - `deflated_sharpe` (unchanged key, now with standardized metadata)
  - `significance_gates`

## New optional gates (default behavior preserved)

All new gates are **disabled by default** when unset (`None`):

- `--min-psr-cv`
- `--min-psr-holdout`
- `--min-dsr`

Backwards compatibility:

- Existing `--min-psr` remains supported.
- If `--min-psr-cv` is not set, CV PSR gate falls back to `--min-psr`.
- Existing `deflated_sharpe` field remains in outputs.

## Assumptions

- Holdout PSR uses holdout Sharpe with effective observations approximated by holdout trade count.
- DSR `effective_test_count` uses completed trial count for that optimization run.
- Missing skew/kurtosis fall back to Gaussian assumptions (`skew=0`, `kurtosis=3`).
