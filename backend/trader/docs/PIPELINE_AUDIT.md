# Trading Pipeline Audit (Research-Aligned)

This audit focuses on moving from "works in backtest" to "paper-trade ready" by reducing overfitting and execution leakage.

## What was changed

1. **Optimization now includes Probabilistic Sharpe gating (PSR)**
   - Added a walk-forward fold-level PSR estimator in `scripts/optimize.py`.
   - Trials are rejected when PSR is too low (`< 0.55`) even if raw mean Sharpe is positive.
   - PSR is persisted into Optuna trial metadata (`psr`, `psr_z`) and contributes to objective scoring.

2. **Readiness scoring now explicitly validates PSR confidence**
   - Added a `psr_confident` readiness check in `scripts/validate_robustness.py`.
   - This avoids promoting strategies with unstable fold-level performance.

3. **Paper engine hardening for production-like behavior**
   - Added stale signal filter (`max_signal_age_minutes`) in `scripts/paper_engine.py`.
   - Added confidence gate (`signal.confidence >= Config.signal_threshold`) before fills.
   - Added no-pyramiding guard for same-side existing positions.
   - Added max-open-position guard and notional cap based on current cash + leverage.

## Why this helps

- PSR-style confidence checks are commonly used to reduce false discoveries from noisy Sharpe improvements.
- Enforcing confidence and freshness gates in paper execution reduces "queue lag" and stale-model execution risk.
- Pyramiding and unconstrained notional are common failure modes in first paper runs; these guards reduce blow-up risk.

## Suggested next steps (not yet implemented)

- Add execution-cost stress scenarios (2x and 3x fee/slippage) in robustness validation.
- Add rolling recalibration diagnostics (Brier score / calibration slope) on live paper signals.
- Add portfolio-level risk parity and correlated exposure caps across coins during paper mode.
