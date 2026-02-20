# Paper Trading Rescue Plan (Same-Day)

## What failed in the latest run

- Optimization completed for all five coins, but robustness validation timed out for every coin in Phase 2.
- Final report therefore marked validation as "not run", which forced a global `DO NOT TRADE` recommendation.

## Immediate objective

Get at least **two coins** to `READY` or `CAUTIOUS` with complete validation evidence by end of day.

## Execution plan for today

1. **Rerun validation with tuned runtime controls first**
   - Use higher validation timeouts and lower Monte Carlo simulation counts for first-pass screening.
   - Keep optimization artifacts fixed so we can compare apples-to-apples.
2. **Shortlist top 2–3 candidates from completed validation**
   - Prioritize coins with strongest holdout + CV stability combo (likely ETH/DOGE first from your run).
3. **Re-optimize only shortlisted coins with tighter search and stability bias**
   - Increase holdout days, reduce hyperparameter freedom, and enforce stronger minimum trade counts.
   - Use holdout-guided model selection (`--holdout-candidates`) so final params are chosen from top CV trials by real holdout behavior, not CV score alone.
4. **Apply risk overlay for paper rollout**
   - Use fractional Kelly sizing cap (e.g., 0.25 Kelly capped by vol target) on top of existing vol-based sizing.
   - Start with reduced size until live paper metrics confirm expected behavior.

## AI/ML tactics that typically improve crypto robustness

- **Model simplification beats over-parameterization** in non-stationary markets.
  - Prefer fewer, higher-signal features and stronger regularization.
- **Walk-forward + holdout consistency** is more predictive than in-sample Sharpe.
  - Track Sharpe decay from CV to holdout and reject fragile regimes.
- **Ensemble disagreement as a risk gate** can reduce tail losses.
  - If model variance spikes, reduce position size or block entries.
- **Fractional Kelly with drawdown cap** improves long-run growth while controlling blow-up risk.
  - Use edge/variance estimate from recent rolling window, clipped aggressively.
- **Regime-aware sizing** (trend vs chop) generally outperforms static size.
  - Keep signal generation constant; adapt exposure multiplier by realized volatility and spread conditions.

## Suggested commands (today)

```bash
# 1) Validate existing artifacts quickly but completely
cd backend/trader
python -m scripts.parallel_launch \
  --validate-only \
  --coins BTC,ETH,SOL,XRP,DOGE \
  --validation-jobs 2 \
  --validation-fast \
  --validation-timeout-scale 1.8 \
  --validation-timeout-cap 9000

# 2) Re-optimize likely survivors with stronger robustness assumptions
python -m scripts.parallel_launch \
  --coins ETH,DOGE,SOL \
  --trials 550 \
  --preset robust180 \
  --holdout-candidates 4 \
  --validation-jobs 2 \
  --validation-timeout-scale 2.0 \
  --validation-timeout-cap 9000
```

## Promotion criteria for paper trading tonight

- Readiness rating: `READY` (preferred) or `CAUTIOUS` with reduced size.
- Holdout must be positive return and positive Sharpe with adequate trades.
- Monte Carlo drawdown and ruin-probability checks pass.
- CV consistency check passes (low fold variance, non-catastrophic min fold).

If only one coin qualifies, deploy one and continue iterative tuning; do not force a second deployment without passing risk gates.
