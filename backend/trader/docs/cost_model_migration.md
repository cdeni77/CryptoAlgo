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
