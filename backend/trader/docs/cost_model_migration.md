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
