# Phase 4 migration: versioned cost assumptions

## Default behavior (unchanged)

If you do nothing, optimizer/backtest keeps legacy defaults:
- `fee_pct_per_side=0.001`
- `min_fee_per_contract=0.20`
- `slippage_bps=2.0`
- funding applied
- impact disabled

## Enable versioned assumptions

Pass a config path into optimizer:

```bash
python -m scripts.optimize --coin BTC --cost-config-path configs/exchange/binance_perps_v202602.json
```

The optimizer result JSON now stores cost metadata under `cost_model` with:
- `version`
- `source_path`
- applied flags (`funding`, `slippage`, `impact`)

## Notes

- Cost config is optional and backward-compatible.
- Fees/slippage/impact are modularized in `core/costs.py`.
- Gross vs net metrics are both present in fold-level metrics (`avg_raw_pnl`/`avg_pnl`, `gross_total_return`/`net_total_return`).

## Coinbase US perps configs (retail parity vs raw CDE)

Use one of these two configs depending on your objective:

- **Retail app parity (recommended default for next run):**
  - `configs/exchange/coinbase_us_perps_retail_v202602.json`
  - Models an app-style execution fee path (`0.10%` bps) with hourly funding metadata.
  - Includes `observed_ui_fee_bps=10.0` and `observed_ui_fee_source=manual_app_ticket_observation` to document calibration rationale.

- **Raw CDE schedule modeling:**
  - `configs/exchange/coinbase_us_perps_cde_v202602.json`
  - Models per-contract USD exchange fees with symbol overrides (BIP/ETP vs SLP/XPP/DOP) and assumption metadata (`non_professional`, `electronic`).

Both configs encode Coinbase US perps hourly funding cadence and contract-size metadata.

### Recommended next pipeline run

```bash
python -m scripts.optimize --coin BTC --cost-config-path configs/exchange/coinbase_us_perps_retail_v202602.json
```

## Phase 5 migration: robustness diagnostics (optional)

New optional optimizer diagnostics can be enabled via CLI:

```bash
python -m scripts.optimize --coin BTC --enable-pbo-diagnostic --enable-cost-stress-diagnostics
```

Artifacts are persisted in `*_optimization.json` under `robustness_diagnostics`:

- `pbo`: CSCV-style leave-one-split-out estimate + methodology metadata
- `stress_costs`: finalist baseline vs stressed scenario metrics (`fees_plus_50pct`, `slippage_x2`, `adverse_funding`)

Assumptions:

- Diagnostics are **non-blocking** and disabled by default.
- Cost stress is run only for a limited finalist set (`--cost-stress-finalists`) to avoid expensive reruns.
- Adverse funding currently uses a conservative per-trade funding penalty proxy (`--cost-stress-funding-bps-per-trade`) when funding is enabled in the base config.

## Phase 6 migration: study-level significance diagnostic (optional)

A lightweight study-level multiple-testing diagnostic can now be enabled:

```bash
python -m scripts.optimize --coin BTC --enable-study-significance
```

Optional settings:

- `--study-significance-bootstrap-iterations` (default: `500`)
- `--study-significance-seed` (default: `42`)
- `--study-significance-score-source` (`fold_sharpe`, `fold_return`, `fold_expectancy`, `cv_sharpe`, `frequency_adjusted_score`)

Artifact output is persisted in `*_optimization.json` under `study_significance`:

- `p_value`: Reality-Check-like p-value for max candidate mean score
- `spa_like_p_value`: SPA-like p-value for the selected best candidate
- `bootstrap`: iterations, seed, and resampling style metadata
- `methodology`: score definition/source, candidate universe size, observation count

Assumptions and limitations:

- This is a **diagnostic-only** post-run check (non-blocking; default off).
- It uses a recentered bootstrap with i.i.d. index resampling for runtime control.
- With small fold counts and serial dependence, p-values should be interpreted cautiously as screening evidence rather than definitive significance proof.
