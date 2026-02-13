# Crypto ML Trading Pipeline

A production-ready machine learning pipeline for trading cryptocurrency perpetual futures on Coinbase, based on academic research and best practices.

## Overview

This pipeline implements a momentum/trend-following strategy using:
- **Triple Barrier Labeling** (Lopez de Prado, 2018)
- **Purged K-Fold Cross-Validation** with embargo
- **Ensemble of XGBoost + LightGBM** with probability calibration
- **Proper position sizing** with volatility targeting

## Key Academic References

1. **Lopez de Prado, M. (2018)** - "Advances in Financial Machine Learning"
   - Triple Barrier Method for labeling
   - Purged K-Fold CV to prevent leakage
   - Sample uniqueness weighting

2. **Grądski et al. (2025)** - "Algorithmic crypto trading using information-driven bars"
   - CUSUM filtering effectiveness
   - Triple barrier with deep learning comparison

3. **Recent Meta-Studies (2024-2025)**:
   - Ensemble methods (XGBoost, LightGBM) consistently outperform deep learning for crypto
   - Funding rates and liquidation data provide significant alpha
   - Cross-exchange arbitrage signals remain effective

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   fetch_data.py │────▶│ engineer_       │────▶│  train_model.py │
│                 │     │ features.py     │     │                 │
│ - Binance API   │     │                 │     │ - PurgedKFold   │
│ - Coinbase API  │     │ - Volatility    │     │ - XGBoost       │
│ - Hyperliquid   │     │ - Momentum      │     │ - LightGBM      │
│ - Coinalyze     │     │ - Funding rates │     │ - Calibration   │
└─────────────────┘     │ - Triple Barrier│     │ - Backtest      │
                        └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ live_signals.py │
                                               │                 │
                                               │ - Real-time     │
                                               │   predictions   │
                                               │ - Trade signals │
                                               └─────────────────┘
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export COINBASE_API_KEY="your_key"
export COINBASE_API_SECRET="your_secret"
export COINALYZE_API_KEY="your_free_key"  # Get at coinalyze.net
```

## Usage

### Docker Compose Run Modes

Use the following exact commands from the repository root:

1. **Live mode** (continuous orchestration: initial backfill → features → signals, then hourly cycles):

```bash
docker compose up --build db backend frontend trader
```

2. **Backtest mode** (one-off historical evaluation, then exit):

```bash
docker compose run --rm trader \
  python train_model.py --backtest --threshold 0.74 --min-auc 0.54 --leverage 4 --exclude BIP,DOP
```

3. **Retrain-only mode** (no long-running loop, recompute features + fresh signals once):

```bash
docker compose run --rm trader \
  sh -lc "python compute_features.py && python train_model.py --signals --threshold 0.74 --min-auc 0.54 --leverage 4 --exclude BIP,DOP"
```

### 1. Fetch Data

```bash
python fetch_data.py
```

This fetches:
- **Price data**: Coinbase and Binance perpetuals (5-minute, resampled to 1H)
- **Spot prices**: For basis calculation
- **Funding rates**: Hyperliquid (hourly), Binance/OKX (8-hourly)
- **Open Interest**: Daily from Coinalyze
- **Liquidations**: Daily from Coinalyze

Output: `data/{COIN}_MASTER_1H.parquet`

### 2. Engineer Features

```bash
python engineer_features.py
```

Creates 60+ features including:
- **Volatility**: Parkinson, rolling std, volatility ratios
- **Momentum**: Multi-horizon returns, RSI, ADX
- **Mean reversion**: MA deviations, Bollinger bands, z-scores
- **Funding**: Cumulative funding, z-scores, cross-exchange spreads
- **Market structure**: Volume ratios, efficiency ratio
- **Labels**: Triple barrier with path-dependent outcomes

Output: `crypto_features_1h/{COIN}_ML_READY_1H.parquet`

### 3. Train Models

```bash
python train_model.py
```

Training process:
1. **Data split**: 70% train, 30% test (chronological)
2. **Cross-validation**: Purged K-Fold with 72-hour embargo
3. **Hyperparameter optimization**: Optuna with 50 trials per model
4. **Model training**: XGBoost + LightGBM ensemble
5. **Calibration**: Isotonic regression for probability calibration
6. **Backtesting**: Realistic with fees, slippage, funding costs

Output: `models/{COIN}.joblib`

### 4. Generate Signals

```bash
# Single coin
python live_signals.py --coin BTC-PERP-INTX

# All coins
python live_signals.py --all

# JSON output (for integration)
python live_signals.py --all --json
```

## Key Improvements Over Original Code

### 1. No Lookahead Bias
- All features computed using only past data
- Triple barrier labels use proper path-dependent logic
- Strict temporal ordering maintained throughout

### 2. Proper Cross-Validation
- **Before**: GroupKFold (doesn't prevent temporal leakage)
- **After**: PurgedKFold with 72-hour embargo matching max holding period

### 3. Simplified Model Ensemble
- **Before**: XGBoost + LightGBM + CatBoost + LSTM + DQN (complex, unstable)
- **After**: XGBoost + LightGBM with calibration (robust, interpretable)

### 4. Better Signal Generation
- **Before**: Very restrictive (41-48 trades in backtest)
- **After**: More trades with proper confidence scaling

### 5. Realistic Backtesting
- Transaction costs (4 bps per side)
- Slippage (volume-adjusted)
- Funding rate costs
- Weekend leverage reduction

## Feature Categories

### Volatility (Source of Risk Sizing)
- `vol_24h/72h/168h`: Rolling realized volatility
- `vol_parkinson_24h`: Range-based volatility (more efficient)
- `vol_ratio_24_72/168`: Volatility regime changes

### Momentum (Primary Alpha Source)
- `ret_2h/4h/6h/12h/24h/48h/72h`: Multi-horizon returns
- `rsi`, `rsi_deviation`: RSI and deviation from neutral
- `efficiency_ratio`: Trend vs. mean-reversion indicator
- `trend_direction`: Directional movement index

### Funding Rates (Alpha + Cost)
- `funding_sum_8h/24h/72h`: Cumulative funding
- `funding_zscore`: Extreme funding conditions
- `funding_spread`: Cross-exchange arbitrage signal

### Market Structure
- `volume_ratio_24h/72h`: Volume relative to average
- `oi_change_24h/72h`: Open interest momentum
- `liq_zscore`: Liquidation pressure indicator

## Configuration

Key parameters in `train_model.py`:

```python
TRAIN_RATIO = 0.70        # Train/test split
PURGE_HOURS = 72          # Embargo period
N_FOLDS = 5               # CV folds
N_TRIALS = 50             # Optuna trials

TAKER_FEE = 0.0004        # 4 bps per side
SLIPPAGE_BPS = 5.0        # Base slippage
```

Key parameters in `engineer_features.py`:

```python
ATR_PERIOD = 14           # ATR lookback
BARRIER_MULT_TP = 2.0     # Take profit multiplier
BARRIER_MULT_SL = 2.0     # Stop loss multiplier
MAX_HORIZON = 72          # Max holding period (hours)
MIN_HORIZON = 12          # Min holding period (hours)
```

## Expected Performance

Based on academic research, realistic expectations:
- **Sharpe Ratio**: 0.5 - 1.5 (after costs)
- **Win Rate**: 50-55%
- **Profit Factor**: 1.1 - 1.3
- **Max Drawdown**: 10-25%

**Important**: These are not guaranteed returns. Markets evolve, alpha decays, and past performance doesn't guarantee future results.

## Extending the Pipeline

### Adding New Features
1. Add computation in `engineer_features.py`
2. Ensure feature uses only past data (no lookahead!)
3. Add to the feature list (it's auto-detected)

### Adding New Data Sources
1. Add fetcher function in `fetch_data.py`
2. Merge into the MASTER file
3. Create derived features in `engineer_features.py`

### Different Coins
1. Add to `COINS` list
2. Add to `SYMBOL_MAP` with exchange symbols
3. Configure leverage in `LEVERAGE_CONFIG`

## Troubleshooting

### "Insufficient data"
- Ensure data fetching completed successfully
- Check that Coinalyze API key is valid
- Some coins may have limited history

### "No valid labels"
- Triple barrier needs enough forward-looking data
- Last 72 hours won't have valid labels (expected)

### "Model not found"
- Run `train_model.py` before `live_signals.py`
- Check `models/` directory

## License

MIT License - Use at your own risk.

## Disclaimer

This is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Never trade with money you can't afford to lose. Always do your own research and consult with financial professionals before making investment decisions.
