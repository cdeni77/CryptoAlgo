# AI-Driven Trading System for Coinbase US Perpetual Futures

## Executive Summary

This document presents a complete end-to-end architecture for an AI-driven trading model targeting the top 5 Coinbase US perpetual futures contracts by volume. The design prioritizes methodological rigor, realistic execution assumptions, and explicit safeguards against common backtesting failures.

**Critical Disclaimer**: This system is designed for research and development purposes. Crypto derivatives trading involves substantial risk of loss. Past performance in backtests—even rigorous ones—does not guarantee future results.

---

## 1. Target Instruments and Market Context

### 1.1 Coinbase US Perpetual-Style Futures

As of July 2025, Coinbase Financial Markets (CFM) offers CFTC-regulated perpetual-style futures to US customers. These contracts have the following characteristics:

| Feature | Specification |
|---------|---------------|
| Contract Structure | 5-year expiration (effectively perpetual) |
| Leverage | Up to 10x intraday |
| Settlement | USDC-margined |
| Funding Rate | Hourly accrual, settled twice daily |
| Trading Hours | 24/7 (with 1-hour weekly maintenance) |
| Minimum Notional | 10 USDC |

### 1.2 Top 5 Contracts by Volume (Expected)

Based on global crypto derivatives volume patterns:

1. **BTC-PERP** (nano Bitcoin perpetual, 1/100 BTC)
2. **ETH-PERP** (nano Ether perpetual, 1/10 ETH)
3. **SOL-PERP** (if/when launched)
4. **XRP-PERP** (if/when launched)
5. **DOGE-PERP** or **AVAX-PERP** (depending on listing availability)

**Note**: At launch, only BTC and ETH perpetuals are available. This design assumes expansion to 5 contracts. If fewer are available, the system should scale down accordingly.

### 1.3 Fee Structure

| Volume Tier | Maker Fee | Taker Fee |
|-------------|-----------|-----------|
| < $10K/month | 0.40% | 0.60% |
| $10K - $50K | 0.35% | 0.55% |
| > $400M | 0.00% | 0.05% |

**Critical Warning**: Coinbase's perpetual futures fees are among the highest globally. Any strategy must generate sufficient alpha to overcome ~0.5% round-trip costs at typical volume tiers.

---

## 2. Data Sources and Collection Architecture

### 2.1 Primary Data Sources (Free/Public APIs)

#### 2.1.1 Coinbase Advanced Trade API

| Endpoint | Data Type | Rate Limit | Granularity |
|----------|-----------|------------|-------------|
| `GET /products/{product_id}/candles` | OHLCV | 10 req/sec (public) | 1m, 5m, 15m, 1h, 6h, 1d |
| `GET /products/{product_id}/ticker` | Last trade, bid/ask | 10 req/sec | Real-time |
| `GET /products/{product_id}/trades` | Recent trades | 10 req/sec | Tick-level (paginated) |
| WebSocket `ticker` channel | Live price updates | 750 conn/sec, 8 msg/sec unauthenticated | Real-time |
| WebSocket `level2` channel | Order book snapshots | 10 subscriptions/product/channel | Real-time |
| `GET /intx/portfolio` | Perpetuals-specific data | 15 req/sec (authenticated) | Real-time |
| Funding rate endpoints | Funding rate history | 15 req/sec (authenticated) | Hourly |

**Authentication**: Cloud Developer Platform (CDP) API keys required for authenticated endpoints.

#### 2.1.2 CCXT Library (Supplementary/Historical)

For backtesting purposes, CCXT provides a unified interface to multiple exchanges:

```python
import ccxt

exchange = ccxt.coinbase({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}  # For perpetuals
})

# Fetch OHLCV with pagination
ohlcv = exchange.fetch_ohlcv('BTC/USD:USDC', '1h', since=start_timestamp, limit=1000)
```

**Rate Limits via CCXT**:
- Binance: 1200 requests/minute (for supplementary cross-exchange data)
- CCXT auto-handles rate limiting when `enableRateLimit=True`

#### 2.1.3 Free Alternative Data Sources

| Source | Data Available | Limitations |
|--------|----------------|-------------|
| CryptoDataDownload | Historical OHLCV, funding rates | Daily updates, may lag |
| Binance Public API | Cross-reference funding/OI | Not Coinbase-specific |
| Coinglass (free tier) | Aggregated funding/OI | Limited historical depth |

### 2.2 Data Collection Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  WebSocket   │    │   REST API   │    │   CCXT      │      │
│  │  Connector   │    │   Poller     │    │   Backfill  │      │
│  │  (Real-time) │    │  (Periodic)  │    │  (Historical)│      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│                   ┌─────────────────┐                           │
│                   │  Message Queue  │                           │
│                   │   (Redis/Kafka) │                           │
│                   └────────┬────────┘                           │
│                            ▼                                    │
│                   ┌─────────────────┐                           │
│                   │  Data Validator │                           │
│                   │  & Normalizer   │                           │
│                   └────────┬────────┘                           │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TIME-SERIES DATABASE                        │   │
│  │  (TimescaleDB / InfluxDB / QuestDB)                     │   │
│  │                                                          │   │
│  │  Tables:                                                 │   │
│  │  - ohlcv_1m (timestamp, symbol, o, h, l, c, v)          │   │
│  │  - trades (timestamp, symbol, price, size, side)        │   │
│  │  - funding_rates (timestamp, symbol, rate)              │   │
│  │  - orderbook_snapshots (timestamp, symbol, bids, asks)  │   │
│  │  - open_interest (timestamp, symbol, oi)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Quality Requirements

1. **Timestamp Precision**: All timestamps in UTC, millisecond precision
2. **Gap Detection**: Flag any gaps > 5 minutes in OHLCV data
3. **Outlier Detection**: Flag price moves > 10% in 1 minute as potential errors
4. **Funding Rate Validation**: Cross-reference against multiple sources when possible

---

## 3. No Data Leakage Protocol

### 3.1 Temporal Data Separation

The system enforces strict chronological separation:

```
Timeline:
├─────────────────────────────────────────────────────────────────┤
│                         FULL DATASET                            │
├─────────────────────────────────────────────────────────────────┤
│   TRAIN (60%)    │  VALIDATION (20%)  │    TEST (20%)          │
│   T₀ ────────── T₁ ──────────────── T₂ ─────────────── T₃      │
│                                                                 │
│   Model fitting   │  Hyperparameter    │  Final evaluation     │
│   Feature params  │  tuning            │  NEVER TOUCHED        │
│                   │  Walk-forward      │  until deployment     │
│                   │  optimization      │  decision             │
└─────────────────────────────────────────────────────────────────┘
```

**Critical Rule**: Test set is ONLY used once for final evaluation. Any iteration that touches test data invalidates the entire experiment.

### 3.2 Point-in-Time Data Reconstruction

All data must be stored with two timestamps:

```python
class DataPoint:
    event_time: datetime      # When the event occurred
    available_time: datetime  # When the data became available to the system
    
    # Example: Funding rate for 00:00 UTC
    # event_time = 2025-01-15 00:00:00 UTC
    # available_time = 2025-01-15 00:00:05 UTC (5 second API delay)
```

### 3.3 Feature Engineering Leak Prevention

**Prohibited Patterns**:

```python
# WRONG: Uses entire dataset for normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_data)  # LEAKAGE!

# CORRECT: Rolling/expanding window normalization
def normalize_point_in_time(df, lookback=252):
    """Normalize using only data available at each point in time."""
    normalized = pd.DataFrame(index=df.index, columns=df.columns)
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        mean = window.mean()
        std = window.std()
        normalized.iloc[i] = (df.iloc[i] - mean) / std
    return normalized.iloc[lookback:]
```

**Checklist for Every Feature**:
- [ ] Can this feature be computed using ONLY data available before time T?
- [ ] Does any aggregation window extend into the future?
- [ ] Are any global statistics (mean, std, min, max) computed across the full dataset?
- [ ] Are any lookups performed on future dates?

### 3.4 Data Split Implementation

```python
class ChronologicalSplitter:
    """Ensures strict temporal separation with buffer periods."""
    
    def __init__(self, train_frac=0.6, val_frac=0.2, buffer_days=7):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.buffer_days = buffer_days
    
    def split(self, df):
        n = len(df)
        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))
        buffer = self.buffer_days * 24  # Assuming hourly data
        
        train = df.iloc[:train_end]
        # Buffer period: discard to prevent information bleeding
        val = df.iloc[train_end + buffer : val_end]
        test = df.iloc[val_end + buffer:]
        
        return train, val, test
```

---

## 4. No Lookahead Bias Protocol

### 4.1 Event-Driven Signal Generation

Signals must be generated using only information available at decision time:

```python
class SignalGenerator:
    """Generate trading signals with strict temporal constraints."""
    
    def generate_signal(self, timestamp: datetime) -> Signal:
        """
        Generate signal for time T using only data available before T.
        
        Critical: The signal generated at T can only be ACTED UPON
        at T + execution_delay. The fill price will be at T + execution_delay
        plus any additional slippage.
        """
        # Get data up to but NOT including current timestamp
        available_data = self.data_store.get_data(end=timestamp - timedelta(seconds=1))
        
        # Compute features using only historical data
        features = self.compute_features(available_data)
        
        # Generate prediction
        signal = self.model.predict(features)
        
        return Signal(
            timestamp=timestamp,
            direction=signal,
            confidence=self.model.predict_proba(features)
        )
```

### 4.2 Realistic Execution Simulation

```python
class ExecutionSimulator:
    """Simulate realistic order execution."""
    
    def __init__(
        self,
        latency_ms: int = 100,          # Network + processing latency
        slippage_bps: float = 5.0,       # Base slippage in basis points
        market_impact_factor: float = 0.1  # Impact per 1% of volume
    ):
        self.latency_ms = latency_ms
        self.slippage_bps = slippage_bps
        self.market_impact_factor = market_impact_factor
    
    def simulate_fill(
        self,
        signal_time: datetime,
        direction: str,  # 'long' or 'short'
        size: float,
        orderbook: OrderBook
    ) -> Fill:
        """
        Simulate order fill with realistic assumptions.
        
        Critical assumptions:
        1. Signal at T cannot result in fill before T + latency
        2. Fill price is NOT the signal-time price
        3. Slippage increases with size relative to liquidity
        """
        # Execution time = signal time + latency
        execution_time = signal_time + timedelta(milliseconds=self.latency_ms)
        
        # Get orderbook at execution time (not signal time!)
        exec_orderbook = self.data_store.get_orderbook(execution_time)
        
        # Estimate market impact
        volume_pct = size / exec_orderbook.total_volume_1pct
        impact_bps = self.market_impact_factor * volume_pct * 10000
        
        # Total slippage
        total_slippage_bps = self.slippage_bps + impact_bps
        
        # Calculate fill price
        if direction == 'long':
            base_price = exec_orderbook.best_ask
            fill_price = base_price * (1 + total_slippage_bps / 10000)
        else:
            base_price = exec_orderbook.best_bid
            fill_price = base_price * (1 - total_slippage_bps / 10000)
        
        return Fill(
            signal_time=signal_time,
            execution_time=execution_time,
            fill_price=fill_price,
            slippage_bps=total_slippage_bps
        )
```

### 4.3 Funding Rate Handling

Funding rates require careful temporal handling:

```python
def apply_funding_rate(position, funding_time, funding_rate, mark_price):
    """
    Apply funding rate to position.
    
    Critical: Funding rate is known AFTER the funding period.
    For predictive purposes, only use PAST funding rates.
    
    Coinbase funding:
    - Accrues hourly
    - Settled twice daily
    - Payment = Position_Size * Mark_Price * Funding_Rate
    """
    # WRONG: Using funding rate before it's known
    # current_funding = get_funding_rate(current_time)  # LOOKAHEAD!
    
    # CORRECT: Funding at T affects position at T
    # But funding rate for T is only known at T (or slightly after)
    payment = position.size * mark_price * funding_rate
    
    if position.direction == 'long' and funding_rate > 0:
        position.pnl -= payment  # Longs pay shorts
    elif position.direction == 'short' and funding_rate > 0:
        position.pnl += payment  # Shorts receive from longs
    # Reverse for negative funding
    
    return position
```

---

## 5. Market Edge Discovery

### 5.1 Plausible Sources of Edge

#### 5.1.1 Funding Rate Dynamics

**Hypothesis**: Funding rates exhibit predictable mean-reversion patterns that can be exploited.

**Measurable Signal**:
```python
def funding_rate_zscore(funding_history, lookback=168):  # 1 week of hourly data
    """
    Calculate z-score of current funding rate vs historical.
    
    Edge hypothesis: Extreme funding rates (|z| > 2) tend to revert,
    creating profitable opportunities for contrarian positions.
    """
    current = funding_history[-1]
    mean = funding_history[-lookback:].mean()
    std = funding_history[-lookback:].std()
    return (current - mean) / std
```

**Why Edge Might Persist**:
1. Retail traders systematically overpay for leverage during euphoric/panic periods
2. Funding rate arbitrage requires capital commitment and execution infrastructure
3. Cross-exchange funding arbitrage is capital-intensive and operationally complex

**Edge Decay Risk**: HIGH. This is well-known and heavily arbitraged by professional firms.

#### 5.1.2 Liquidation Cascade Dynamics

**Hypothesis**: Large liquidation events create temporary price dislocations that revert.

**Measurable Signal**:
```python
def liquidation_cascade_detector(
    price_series,
    volume_series,
    oi_series,
    lookback=24
):
    """
    Detect potential liquidation cascades.
    
    Signals:
    1. Sharp price move (> 2 std in 1 hour)
    2. Elevated volume (> 3x average)
    3. Open interest decline (positions being liquidated)
    """
    price_return = price_series.pct_change()
    price_std = price_return.rolling(lookback).std()
    price_zscore = price_return / price_std
    
    volume_ratio = volume_series / volume_series.rolling(lookback).mean()
    
    oi_change = oi_series.pct_change()
    
    cascade_score = (
        (abs(price_zscore) > 2).astype(int) +
        (volume_ratio > 3).astype(int) +
        (oi_change < -0.05).astype(int)  # > 5% OI decline
    )
    
    return cascade_score >= 2  # At least 2 of 3 conditions
```

**Why Edge Might Persist**:
1. Liquidations are forced selling, not informed trading
2. Market makers widen spreads during volatility, limiting competition
3. Recovery is gradual due to risk aversion after large moves

**Edge Decay Risk**: MEDIUM. Requires fast execution and risk tolerance.

#### 5.1.3 Cross-Asset Momentum/Mean-Reversion

**Hypothesis**: Price movements in BTC lead price movements in altcoin perps.

**Measurable Signal**:
```python
def cross_asset_momentum(btc_returns, alt_returns, lag_hours=1):
    """
    Measure lead-lag relationship between BTC and altcoins.
    
    Hypothesis: BTC moves first, alts follow with delay.
    """
    # Lagged correlation
    btc_lagged = btc_returns.shift(lag_hours)
    correlation = btc_lagged.corr(alt_returns)
    
    # Tradable signal: Strong BTC move predicts alt move
    btc_zscore = btc_returns / btc_returns.rolling(24).std()
    signal = btc_zscore.shift(lag_hours)  # Use lagged BTC z-score
    
    return signal, correlation
```

**Why Edge Might Persist**:
1. Altcoins have lower liquidity, slower price discovery
2. Institutional flow concentrates in BTC first
3. Retail trades altcoins with lag

**Edge Decay Risk**: MEDIUM-HIGH. Well-documented in academic literature.

#### 5.1.4 Volatility Regime Shifts

**Hypothesis**: Volatility clustering creates predictable regime transitions.

**Measurable Signal**:
```python
def volatility_regime(returns, short_window=24, long_window=168):
    """
    Classify current volatility regime.
    
    Regimes:
    - Low vol (short_vol < 0.7 * long_vol)
    - Normal vol
    - High vol (short_vol > 1.5 * long_vol)
    """
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    
    ratio = short_vol / long_vol
    
    if ratio < 0.7:
        return 'low_vol'
    elif ratio > 1.5:
        return 'high_vol'
    else:
        return 'normal_vol'
```

**Why Edge Might Persist**:
1. Regime shifts often occur at predictable points (funding settlements, option expiries)
2. Many strategies are not regime-aware
3. Position sizing based on regime improves risk-adjusted returns

**Edge Decay Risk**: LOW. More of a risk management tool than pure alpha source.

### 5.2 Edge Quantification Framework

For each potential edge, compute:

```python
@dataclass
class EdgeMetrics:
    """Metrics to quantify and track potential edge."""
    
    signal_name: str
    
    # Predictive power
    ic: float  # Information coefficient (correlation with forward returns)
    ic_ir: float  # IC information ratio (IC mean / IC std)
    
    # Tradability
    turnover: float  # Average daily turnover of signal
    capacity: float  # Estimated capacity in $ before impact
    
    # Stability
    ic_rolling_std: float  # Stability of IC over time
    regime_sensitivity: Dict[str, float]  # IC by volatility regime
    
    # Decay
    ic_trend: float  # Slope of IC over time (negative = decaying)
    
    def is_viable(self) -> bool:
        """
        Minimum thresholds for a tradable signal.
        
        Conservative thresholds:
        - IC > 0.02 (weak but statistically meaningful)
        - IC IR > 0.5 (signal is somewhat stable)
        - IC trend > -0.001 (not rapidly decaying)
        """
        return (
            self.ic > 0.02 and
            self.ic_ir > 0.5 and
            self.ic_trend > -0.001
        )
```

---

## 6. Modeling Approach Comparison

### 6.1 Candidate Approaches

#### Approach A: Gradient Boosted Trees (XGBoost/LightGBM)

**Architecture**:
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.01,
    num_leaves=31,
    min_child_samples=100,  # Prevent overfitting
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)

# Target: Sign of next-hour return
# Classes: -1 (short), 0 (neutral/flat), +1 (long)
```

**Pros**:
- Handles non-linear feature interactions
- Robust to outliers and missing data
- Fast training and inference
- Built-in feature importance

**Cons**:
- Prone to overfitting on noisy financial data
- No native handling of time series structure
- Requires careful feature engineering

**Appropriate When**: Feature engineering is strong, data quality is high, regime is relatively stable.

#### Approach B: Rule-Based Regime System

**Architecture**:
```python
class RegimeBasedStrategy:
    """
    Multi-regime trading system with explicit rules.
    """
    
    def __init__(self):
        self.regimes = {
            'trending_up': TrendFollowingRules(),
            'trending_down': TrendFollowingRules(short=True),
            'mean_reverting': MeanReversionRules(),
            'high_volatility': VolatilityReducedRules(),
            'low_volatility': VolatilityExpansionRules()
        }
        self.regime_classifier = RegimeClassifier()
    
    def generate_signal(self, data):
        current_regime = self.regime_classifier.classify(data)
        return self.regimes[current_regime].generate_signal(data)

class RegimeClassifier:
    """Classify market regime using measurable indicators."""
    
    def classify(self, data):
        # Trend detection
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        trend = 'up' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'down'
        
        # Volatility regime
        current_vol = data['returns'].rolling(24).std().iloc[-1]
        historical_vol = data['returns'].rolling(168).std().iloc[-1]
        vol_regime = 'high' if current_vol > 1.5 * historical_vol else 'normal'
        
        # Mean reversion detection
        rsi = compute_rsi(data['close'], 14)
        mean_revert = rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70
        
        # Combine
        if vol_regime == 'high':
            return 'high_volatility'
        elif mean_revert:
            return 'mean_reverting'
        elif trend == 'up':
            return 'trending_up'
        else:
            return 'trending_down'
```

**Pros**:
- Fully interpretable and auditable
- No overfitting to historical data
- Easier to debug and modify
- Stable performance across regimes

**Cons**:
- Cannot capture complex non-linear patterns
- Rule design requires domain expertise
- May underperform in novel regimes

**Appropriate When**: Regulatory requirements demand interpretability, capital is limited, execution speed is not critical.

#### Approach C: Reinforcement Learning (PPO/SAC)

**Architecture**:
```python
import gym
from stable_baselines3 import PPO

class TradingEnvironment(gym.Env):
    """
    RL environment for perpetual futures trading.
    
    State: [price_features, position, pnl, margin]
    Action: [-1, 0, +1] (short, flat, long) or continuous
    Reward: Risk-adjusted PnL (Sharpe-like)
    """
    
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,)
        )
    
    def step(self, action):
        # Execute action
        self._execute_trade(action)
        
        # Compute reward (risk-adjusted)
        reward = self._compute_reward()
        
        # Check for liquidation/margin call
        done = self._check_termination()
        
        return self._get_observation(), reward, done, {}
    
    def _compute_reward(self):
        """
        Reward function is CRITICAL for RL performance.
        
        Bad: Simple PnL (leads to excessive risk-taking)
        Better: Sharpe-like reward with drawdown penalty
        """
        pnl = self.current_pnl - self.previous_pnl
        vol = self.pnl_history[-20:].std() if len(self.pnl_history) > 20 else 1
        sharpe_reward = pnl / (vol + 1e-8)
        
        # Drawdown penalty
        drawdown = self._compute_drawdown()
        drawdown_penalty = -0.1 * max(0, drawdown - 0.1)  # Penalize > 10% DD
        
        return sharpe_reward + drawdown_penalty

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
```

**Pros**:
- Learns optimal action policy directly
- Can handle sequential decision making
- Naturally incorporates transaction costs and position constraints

**Cons**:
- Extremely prone to overfitting
- Reward function design is fragile
- Training is unstable and sample-inefficient
- Difficult to interpret decisions

**Appropriate When**: Large amounts of quality data available, significant compute resources, team has deep RL expertise.

### 6.2 Recommended Approach: Hybrid System

Given the constraints (free data sources, crypto market characteristics, need for robustness), the recommended approach is:

**Primary**: Rule-Based Regime System with ML-Enhanced Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID TRADING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  REGIME CLASSIFIER                        │  │
│  │           (Rule-Based + ML Confidence)                    │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            ▼                                    │
│  ┌────────────┬────────────┬────────────┬────────────┐         │
│  │  Trending  │  Mean Rev  │  High Vol  │  Low Vol   │         │
│  │   Rules    │   Rules    │   Rules    │   Rules    │         │
│  └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘         │
│        │            │            │            │                 │
│        └────────────┴─────┬──────┴────────────┘                 │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ML SIGNAL FILTER (LightGBM)                  │  │
│  │         (Filters low-confidence rule signals)             │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              POSITION SIZING & RISK MGMT                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Justification**:
1. Rule-based core ensures interpretability and prevents black-box failures
2. ML filter improves signal quality without introducing complex dependencies
3. Regime awareness handles crypto's non-stationarity
4. Modular design allows component-by-component testing

---

## 7. Feature Engineering

### 7.1 Price-Based Features

```python
class PriceFeatures:
    """Price-derived features for perpetual futures trading."""
    
    @staticmethod
    def compute(df, lookbacks=[1, 4, 12, 24, 48, 168]):
        """
        Compute price features with multiple lookback periods.
        
        All features are computed using ONLY historical data.
        """
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            # Returns
            features[f'return_{lb}h'] = df['close'].pct_change(lb)
            
            # Momentum (rate of change)
            features[f'roc_{lb}h'] = (
                df['close'] - df['close'].shift(lb)
            ) / df['close'].shift(lb)
            
            # Volatility (realized)
            features[f'volatility_{lb}h'] = df['close'].pct_change().rolling(lb).std()
            
            # Range (high-low as % of close)
            features[f'range_{lb}h'] = (
                df['high'].rolling(lb).max() - df['low'].rolling(lb).min()
            ) / df['close']
            
            # Distance from moving average
            ma = df['close'].rolling(lb).mean()
            features[f'ma_distance_{lb}h'] = (df['close'] - ma) / ma
        
        # RSI (standard)
        features['rsi_14'] = compute_rsi(df['close'], 14)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands position
        ma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - ma_20) / (2 * std_20)
        
        return features
```

### 7.2 Volume and Volatility Features

```python
class VolumeVolatilityFeatures:
    """Volume and volatility features."""
    
    @staticmethod
    def compute(df, lookbacks=[1, 4, 12, 24, 48]):
        features = pd.DataFrame(index=df.index)
        
        for lb in lookbacks:
            # Volume relative to average
            avg_volume = df['volume'].rolling(lb).mean()
            features[f'volume_ratio_{lb}h'] = df['volume'] / avg_volume
            
            # Volume-weighted price change
            vwap = (df['close'] * df['volume']).rolling(lb).sum() / df['volume'].rolling(lb).sum()
            features[f'vwap_distance_{lb}h'] = (df['close'] - vwap) / vwap
            
            # Volatility ratio (short vs long)
            if lb > 1:
                short_vol = df['close'].pct_change().rolling(lb).std()
                long_vol = df['close'].pct_change().rolling(lb * 4).std()
                features[f'vol_ratio_{lb}h'] = short_vol / long_vol
        
        # Parkinson volatility (uses high/low, more efficient estimator)
        features['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df['high'] / df['low']) ** 2).rolling(24).mean()
        )
        
        # Volume trend
        features['volume_trend'] = df['volume'].rolling(24).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        
        return features
```

### 7.3 Derivatives-Specific Features

```python
class DerivativesFeatures:
    """Features specific to perpetual futures."""
    
    @staticmethod
    def compute(df):
        """
        Compute derivatives-specific features.
        
        Required columns: funding_rate, open_interest, mark_price, index_price
        """
        features = pd.DataFrame(index=df.index)
        
        # Funding rate features
        features['funding_rate'] = df['funding_rate']
        features['funding_rate_ma_24h'] = df['funding_rate'].rolling(24).mean()
        features['funding_rate_zscore'] = (
            df['funding_rate'] - df['funding_rate'].rolling(168).mean()
        ) / df['funding_rate'].rolling(168).std()
        
        # Cumulative funding (carry cost/benefit)
        features['cumulative_funding_24h'] = df['funding_rate'].rolling(24).sum()
        features['cumulative_funding_168h'] = df['funding_rate'].rolling(168).sum()
        
        # Open interest features
        features['oi_change_1h'] = df['open_interest'].pct_change()
        features['oi_change_24h'] = df['open_interest'].pct_change(24)
        features['oi_ma_distance'] = (
            df['open_interest'] - df['open_interest'].rolling(168).mean()
        ) / df['open_interest'].rolling(168).mean()
        
        # Price vs OI divergence (potential liquidation signal)
        price_direction = np.sign(df['close'].pct_change(24))
        oi_direction = np.sign(df['open_interest'].pct_change(24))
        features['price_oi_divergence'] = price_direction != oi_direction
        
        # Basis (perp vs spot)
        features['basis'] = (df['mark_price'] - df['index_price']) / df['index_price']
        features['basis_ma_24h'] = features['basis'].rolling(24).mean()
        features['basis_zscore'] = (
            features['basis'] - features['basis'].rolling(168).mean()
        ) / features['basis'].rolling(168).std()
        
        # Long/short imbalance proxy (via funding rate persistence)
        features['funding_persistence'] = df['funding_rate'].rolling(48).apply(
            lambda x: (x > 0).sum() / len(x)
        )
        
        return features
```

### 7.4 Cross-Asset Features

```python
class CrossAssetFeatures:
    """Features capturing cross-asset relationships."""
    
    @staticmethod
    def compute(dfs: Dict[str, pd.DataFrame], reference='BTC-PERP'):
        """
        Compute cross-asset features.
        
        dfs: Dictionary of {symbol: dataframe}
        reference: Reference asset (typically BTC)
        """
        features = {}
        ref_df = dfs[reference]
        
        for symbol, df in dfs.items():
            if symbol == reference:
                continue
            
            symbol_features = pd.DataFrame(index=df.index)
            
            # Beta to BTC
            ref_returns = ref_df['close'].pct_change()
            asset_returns = df['close'].pct_change()
            rolling_cov = ref_returns.rolling(168).cov(asset_returns)
            rolling_var = ref_returns.rolling(168).var()
            symbol_features[f'beta_to_{reference}'] = rolling_cov / rolling_var
            
            # Correlation to BTC
            symbol_features[f'corr_to_{reference}'] = (
                ref_returns.rolling(168).corr(asset_returns)
            )
            
            # Lead-lag relationship
            for lag in [1, 2, 4]:
                lagged_corr = ref_returns.shift(lag).rolling(168).corr(asset_returns)
                symbol_features[f'lag_{lag}h_corr_to_{reference}'] = lagged_corr
            
            # Relative strength
            ref_perf = ref_df['close'].pct_change(24)
            asset_perf = df['close'].pct_change(24)
            symbol_features['relative_strength_24h'] = asset_perf - ref_perf
            
            # Funding rate differential
            symbol_features['funding_diff_vs_btc'] = (
                df['funding_rate'] - ref_df['funding_rate']
            )
            
            features[symbol] = symbol_features
        
        return features
```

### 7.5 Feature Summary Table

| Category | Feature | Lookbacks | Rationale |
|----------|---------|-----------|-----------|
| Price | Returns | 1, 4, 12, 24, 48, 168h | Momentum at multiple scales |
| Price | MA Distance | 24, 168h | Mean reversion signal |
| Price | RSI | 14 periods | Overbought/oversold |
| Price | Bollinger Position | 20 periods | Volatility-adjusted extremes |
| Volume | Volume Ratio | 1, 4, 12, 24, 48h | Unusual activity detection |
| Volume | VWAP Distance | 24h | Institutional fair value |
| Volatility | Realized Vol | 24, 168h | Risk regime detection |
| Volatility | Parkinson Vol | 24h | Efficient vol estimator |
| Derivatives | Funding Rate Z-Score | 168h baseline | Funding rate extremes |
| Derivatives | Cumulative Funding | 24, 168h | Carry cost/benefit |
| Derivatives | OI Change | 1, 24h | Position buildup/unwind |
| Derivatives | Basis Z-Score | 168h baseline | Perp vs spot divergence |
| Cross-Asset | BTC Beta | 168h rolling | Systematic exposure |
| Cross-Asset | Lead-Lag Corr | 1, 2, 4h lags | Cross-asset momentum |
| Cross-Asset | Relative Strength | 24h | Outperformance/underperformance |

---

## 8. Backtesting Framework

### 8.1 Event-Driven Architecture

```python
class EventDrivenBacktester:
    """
    Event-driven backtesting engine with strict temporal ordering.
    """
    
    def __init__(
        self,
        strategy: Strategy,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        risk_manager: RiskManager,
        start_date: datetime,
        end_date: datetime
    ):
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.risk_manager = risk_manager
        self.start_date = start_date
        self.end_date = end_date
        
        self.event_queue = []
        self.portfolio = Portfolio()
        self.trade_log = []
    
    def run(self):
        """
        Main event loop.
        
        Event types:
        - MarketEvent: New market data available
        - SignalEvent: Strategy generates signal
        - OrderEvent: Order submitted
        - FillEvent: Order filled
        - FundingEvent: Funding rate settlement
        """
        current_time = self.start_date
        
        while current_time <= self.end_date:
            # 1. Process market data
            market_event = self.data_handler.get_next_event(current_time)
            if market_event:
                self._handle_market_event(market_event)
            
            # 2. Generate signals
            signal_event = self.strategy.generate_signal(
                self.data_handler.get_data(end=current_time)
            )
            if signal_event:
                self._handle_signal_event(signal_event)
            
            # 3. Risk check
            risk_approved = self.risk_manager.check_signal(
                signal_event, self.portfolio
            )
            
            # 4. Generate order
            if signal_event and risk_approved:
                order_event = self.execution_handler.create_order(
                    signal_event, self.portfolio
                )
                self._handle_order_event(order_event)
            
            # 5. Simulate fill (with latency)
            fill_event = self.execution_handler.simulate_fill(
                order_event, current_time
            )
            if fill_event:
                self._handle_fill_event(fill_event)
            
            # 6. Process funding (if applicable)
            if self._is_funding_time(current_time):
                funding_event = self._generate_funding_event(current_time)
                self._handle_funding_event(funding_event)
            
            # 7. Update portfolio mark-to-market
            self.portfolio.update_mtm(current_time, self.data_handler)
            
            # Move to next event
            current_time = self._get_next_timestamp(current_time)
    
    def _handle_fill_event(self, fill_event):
        """Process fill and update portfolio."""
        self.portfolio.process_fill(fill_event)
        self.trade_log.append({
            'timestamp': fill_event.timestamp,
            'symbol': fill_event.symbol,
            'direction': fill_event.direction,
            'quantity': fill_event.quantity,
            'fill_price': fill_event.fill_price,
            'commission': fill_event.commission,
            'slippage': fill_event.slippage_bps
        })
    
    def _handle_funding_event(self, funding_event):
        """Apply funding rate to open positions."""
        for symbol, position in self.portfolio.positions.items():
            if position.quantity != 0:
                funding_payment = (
                    position.quantity *
                    funding_event.mark_prices[symbol] *
                    funding_event.funding_rates[symbol]
                )
                # Longs pay when positive, shorts receive
                if position.quantity > 0:
                    position.pnl -= funding_payment
                else:
                    position.pnl += funding_payment
```

### 8.2 Realistic Cost Model

```python
@dataclass
class CostModel:
    """
    Comprehensive cost model for perpetual futures trading.
    """
    
    # Trading fees (Coinbase tiers)
    maker_fee_bps: float = 40.0  # 0.40% for < $10K volume
    taker_fee_bps: float = 60.0  # 0.60% for < $10K volume
    
    # Slippage model
    base_slippage_bps: float = 5.0  # Base slippage
    volatility_slippage_multiplier: float = 2.0  # Additional slippage in high vol
    size_impact_coefficient: float = 0.1  # Market impact per % of volume
    
    # Funding (for carry calculations)
    avg_positive_funding_bps: float = 1.0  # Average hourly funding when positive
    avg_negative_funding_bps: float = 0.5  # Average hourly funding when negative
    
    def calculate_trade_cost(
        self,
        order_type: str,  # 'market' or 'limit'
        size_usd: float,
        daily_volume_usd: float,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Calculate total cost of a trade in basis points.
        """
        # Trading fee
        fee_bps = self.taker_fee_bps if order_type == 'market' else self.maker_fee_bps
        
        # Slippage (market orders only)
        if order_type == 'market':
            vol_multiplier = max(1.0, current_volatility / avg_volatility)
            slippage_bps = self.base_slippage_bps * vol_multiplier
            
            # Size impact
            size_pct = size_usd / daily_volume_usd
            impact_bps = self.size_impact_coefficient * size_pct * 10000
            
            total_slippage_bps = slippage_bps + impact_bps
        else:
            total_slippage_bps = 0.0
        
        return fee_bps + total_slippage_bps
    
    def calculate_funding_cost(
        self,
        position_direction: str,  # 'long' or 'short'
        holding_period_hours: int,
        avg_funding_rate_bps: float
    ) -> float:
        """
        Estimate funding cost over holding period.
        
        Note: This is an estimate. Actual funding varies.
        """
        return holding_period_hours * avg_funding_rate_bps
```

### 8.3 Walk-Forward Optimization

```python
class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter selection.
    """
    
    def __init__(
        self,
        train_period_days: int = 180,
        test_period_days: int = 30,
        step_days: int = 30
    ):
        self.train_period = timedelta(days=train_period_days)
        self.test_period = timedelta(days=test_period_days)
        self.step = timedelta(days=step_days)
    
    def run(self, data, strategy_factory, param_grid):
        """
        Run walk-forward optimization.
        
        Returns:
        - Optimized parameters for each window
        - Out-of-sample performance for each window
        - Aggregate out-of-sample performance
        """
        results = []
        
        # Generate windows
        windows = self._generate_windows(data)
        
        for window in windows:
            train_data = data.loc[window['train_start']:window['train_end']]
            test_data = data.loc[window['test_start']:window['test_end']]
            
            # Grid search on training data
            best_params = None
            best_train_sharpe = -np.inf
            
            for params in param_grid:
                strategy = strategy_factory(**params)
                train_results = self._backtest(strategy, train_data)
                
                if train_results['sharpe'] > best_train_sharpe:
                    best_train_sharpe = train_results['sharpe']
                    best_params = params
            
            # Evaluate on test data (out-of-sample)
            strategy = strategy_factory(**best_params)
            test_results = self._backtest(strategy, test_data)
            
            results.append({
                'window': window,
                'best_params': best_params,
                'train_sharpe': best_train_sharpe,
                'test_sharpe': test_results['sharpe'],
                'test_returns': test_results['returns'],
                'test_drawdown': test_results['max_drawdown']
            })
        
        # Aggregate OOS performance
        all_oos_returns = pd.concat([r['test_returns'] for r in results])
        aggregate_sharpe = self._calculate_sharpe(all_oos_returns)
        
        return {
            'windows': results,
            'aggregate_oos_sharpe': aggregate_sharpe,
            'param_stability': self._analyze_param_stability(results)
        }
    
    def _generate_windows(self, data):
        """Generate train/test windows."""
        windows = []
        current_start = data.index[0]
        
        while current_start + self.train_period + self.test_period <= data.index[-1]:
            windows.append({
                'train_start': current_start,
                'train_end': current_start + self.train_period,
                'test_start': current_start + self.train_period,
                'test_end': current_start + self.train_period + self.test_period
            })
            current_start += self.step
        
        return windows
```

---

## 9. Risk Management

### 9.1 Position Sizing

```python
class PositionSizer:
    """
    Risk-based position sizing.
    """
    
    def __init__(
        self,
        max_position_pct: float = 0.2,  # Max 20% of portfolio in single position
        target_vol_pct: float = 0.15,   # Target annualized volatility
        max_leverage: float = 3.0,       # Max gross leverage
        min_position_usd: float = 100.0  # Minimum position size
    ):
        self.max_position_pct = max_position_pct
        self.target_vol_pct = target_vol_pct
        self.max_leverage = max_leverage
        self.min_position_usd = min_position_usd
    
    def calculate_position_size(
        self,
        signal: Signal,
        portfolio_value: float,
        asset_volatility: float,
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate position size using volatility targeting.
        
        Size = (Target Vol * Portfolio Value) / (Asset Vol * sqrt(252 * 24))
        
        Constraints:
        1. Max position as % of portfolio
        2. Max leverage
        3. Signal confidence scaling
        """
        # Annualize hourly volatility
        annual_vol = asset_volatility * np.sqrt(252 * 24)
        
        # Base size from vol targeting
        base_size = (self.target_vol_pct * portfolio_value) / annual_vol
        
        # Scale by signal confidence
        confidence_scaled_size = base_size * signal.confidence
        
        # Apply constraints
        max_by_position_limit = portfolio_value * self.max_position_pct
        
        current_gross_exposure = sum(abs(p) for p in current_positions.values())
        max_by_leverage = (self.max_leverage * portfolio_value) - current_gross_exposure
        
        final_size = min(
            confidence_scaled_size,
            max_by_position_limit,
            max(0, max_by_leverage)
        )
        
        # Apply minimum
        if final_size < self.min_position_usd:
            return 0.0  # Too small to trade
        
        return final_size
```

### 9.2 Drawdown Controls

```python
class DrawdownController:
    """
    Drawdown-based risk controls.
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 0.15,      # Liquidate at 15% drawdown
        warning_drawdown_pct: float = 0.10,   # Reduce size at 10%
        cooldown_hours: int = 24,             # Hours before re-entry after breach
        size_reduction_factor: float = 0.5    # Reduce size by 50% in warning zone
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.warning_drawdown_pct = warning_drawdown_pct
        self.cooldown_hours = cooldown_hours
        self.size_reduction_factor = size_reduction_factor
        
        self.peak_value = 0.0
        self.last_breach_time = None
    
    def update(self, current_value: float, current_time: datetime):
        """Update peak and check drawdown."""
        self.peak_value = max(self.peak_value, current_value)
        current_drawdown = (self.peak_value - current_value) / self.peak_value
        
        return {
            'current_drawdown': current_drawdown,
            'peak_value': self.peak_value,
            'status': self._get_status(current_drawdown, current_time)
        }
    
    def _get_status(self, drawdown: float, current_time: datetime) -> str:
        """
        Determine trading status.
        
        Returns:
        - 'normal': Trade at full size
        - 'reduced': Trade at reduced size
        - 'halted': No new positions
        - 'cooldown': Waiting after max drawdown breach
        """
        # Check cooldown
        if self.last_breach_time:
            hours_since_breach = (current_time - self.last_breach_time).total_seconds() / 3600
            if hours_since_breach < self.cooldown_hours:
                return 'cooldown'
            else:
                self.last_breach_time = None
        
        if drawdown >= self.max_drawdown_pct:
            self.last_breach_time = current_time
            return 'halted'
        elif drawdown >= self.warning_drawdown_pct:
            return 'reduced'
        else:
            return 'normal'
    
    def get_size_multiplier(self, status: str) -> float:
        """Get position size multiplier based on status."""
        return {
            'normal': 1.0,
            'reduced': self.size_reduction_factor,
            'halted': 0.0,
            'cooldown': 0.0
        }[status]
```

### 9.3 Regime Filters

```python
class RegimeFilter:
    """
    Filter trades based on market regime.
    """
    
    def __init__(
        self,
        vol_shutdown_threshold: float = 3.0,   # Shutdown if vol > 3x normal
        correlation_breakdown_threshold: float = 0.3,  # Min correlation for cross-asset
        max_funding_rate_bps: float = 50.0     # Avoid extreme funding environments
    ):
        self.vol_shutdown_threshold = vol_shutdown_threshold
        self.correlation_breakdown_threshold = correlation_breakdown_threshold
        self.max_funding_rate_bps = max_funding_rate_bps
    
    def should_trade(
        self,
        current_vol: float,
        baseline_vol: float,
        btc_correlation: float,
        funding_rate_bps: float,
        strategy_type: str
    ) -> Tuple[bool, str]:
        """
        Determine if conditions are suitable for trading.
        
        Returns:
        - (allow_trade, reason)
        """
        # Volatility check
        vol_ratio = current_vol / baseline_vol
        if vol_ratio > self.vol_shutdown_threshold:
            return False, f"High volatility: {vol_ratio:.1f}x normal"
        
        # Correlation check (for cross-asset strategies)
        if strategy_type in ['cross_asset', 'beta_neutral']:
            if abs(btc_correlation) < self.correlation_breakdown_threshold:
                return False, f"Correlation breakdown: {btc_correlation:.2f}"
        
        # Funding rate check (for funding-sensitive strategies)
        if strategy_type in ['funding_arb', 'carry']:
            if abs(funding_rate_bps) > self.max_funding_rate_bps:
                return False, f"Extreme funding: {funding_rate_bps:.1f} bps"
        
        return True, "OK"
```

---

## 10. Evaluation Metrics

### 10.1 Core Metrics

```python
class PerformanceMetrics:
    """
    Comprehensive performance evaluation.
    """
    
    @staticmethod
    def calculate_all(returns: pd.Series, positions: pd.Series) -> Dict:
        """
        Calculate full suite of performance metrics.
        
        returns: Series of period returns
        positions: Series of position sizes (for turnover)
        """
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 * 24 / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(252 * 24)
        
        # Risk-adjusted returns
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Downside risk
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252 * 24)
        sortino = ann_return / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Win/loss ratio
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        # Tail risk
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Turnover
        position_changes = positions.diff().abs()
        daily_turnover = position_changes.resample('D').sum().mean()
        
        # Skewness and kurtosis
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        return {
            # Returns
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            
            # Risk-adjusted
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Drawdown
            'max_drawdown': max_drawdown,
            'avg_drawdown': drawdowns.mean(),
            'drawdown_duration': PerformanceMetrics._max_drawdown_duration(drawdowns),
            
            # Win/loss
            'hit_rate': hit_rate,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            
            # Tail risk
            'var_95': var_95,
            'cvar_95': cvar_95,
            
            # Higher moments
            'skewness': skew,
            'kurtosis': kurt,
            
            # Activity
            'daily_turnover': daily_turnover,
            'num_trades': (positions.diff() != 0).sum()
        }
```

### 10.2 Regime-Conditional Performance

```python
def regime_conditional_analysis(
    returns: pd.Series,
    regime_labels: pd.Series
) -> pd.DataFrame:
    """
    Analyze performance conditional on market regime.
    """
    results = []
    
    for regime in regime_labels.unique():
        regime_mask = regime_labels == regime
        regime_returns = returns[regime_mask]
        
        if len(regime_returns) < 20:  # Minimum sample
            continue
        
        results.append({
            'regime': regime,
            'num_periods': len(regime_returns),
            'mean_return': regime_returns.mean(),
            'volatility': regime_returns.std(),
            'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252 * 24),
            'hit_rate': (regime_returns > 0).mean(),
            'max_drawdown': calculate_max_drawdown(regime_returns)
        })
    
    return pd.DataFrame(results)
```

### 10.3 Overfitting Detection

```python
class OverfitDetector:
    """
    Detect potential overfitting in backtested strategies.
    """
    
    @staticmethod
    def deflated_sharpe_ratio(
        observed_sharpe: float,
        num_trials: int,
        returns_skew: float,
        returns_kurtosis: float,
        sample_length: int
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio (DSR).
        
        Accounts for multiple testing and non-normal returns.
        Reference: Bailey & Lopez de Prado (2014)
        """
        from scipy.stats import norm
        
        # Expected max Sharpe under null hypothesis
        euler_mascheroni = 0.5772156649
        expected_max_sharpe = (
            (1 - euler_mascheroni) * norm.ppf(1 - 1/num_trials) +
            euler_mascheroni * norm.ppf(1 - 1/(num_trials * np.e))
        )
        
        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt(
            (1 + 0.5 * observed_sharpe**2 - 
             returns_skew * observed_sharpe + 
             (returns_kurtosis - 3) / 4 * observed_sharpe**2) / sample_length
        )
        
        # DSR test statistic
        dsr = norm.cdf((observed_sharpe - expected_max_sharpe) / se_sharpe)
        
        return dsr
    
    @staticmethod
    def probability_of_backtest_overfitting(
        is_sharpes: List[float],
        oos_sharpes: List[float],
        num_trials: int
    ) -> float:
        """
        Calculate Probability of Backtest Overfitting (PBO).
        
        Uses combinatorial analysis of IS vs OOS performance.
        Reference: Bailey et al. (2015)
        """
        from itertools import combinations
        
        num_windows = len(is_sharpes)
        overfit_count = 0
        total_count = 0
        
        # For each possible split
        for comb in combinations(range(num_windows), num_windows // 2):
            is_subset = [is_sharpes[i] for i in comb]
            oos_subset = [oos_sharpes[i] for i in range(num_windows) if i not in comb]
            
            # Best IS performer
            best_is_idx = np.argmax(is_subset)
            best_is_oos = oos_subset[best_is_idx] if best_is_idx < len(oos_subset) else 0
            
            # Median OOS
            median_oos = np.median(oos_subset)
            
            # Is overfitting detected?
            if best_is_oos < median_oos:
                overfit_count += 1
            total_count += 1
        
        return overfit_count / total_count if total_count > 0 else 0.5
    
    @staticmethod
    def warning_signs() -> List[str]:
        """
        Common warning signs of overfitting.
        """
        return [
            "Sharpe ratio > 2.5 (exceptionally rare in crypto)",
            "Max drawdown < 5% over multi-year backtest",
            "Hit rate > 60% for trend-following strategy",
            "Performance degrades significantly out-of-sample",
            "Optimal parameters are at grid boundaries",
            "Parameters change dramatically across walk-forward windows",
            "Strategy requires complex, many-parameter model",
            "Performance depends on specific time periods",
            "Curve fitting to specific events (e.g., COVID crash)"
        ]
```

---

## 11. Deployment Considerations

### 11.1 Live Data Ingestion

```python
class LiveDataIngestion:
    """
    Production-grade live data ingestion.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.websocket_client = None
        self.data_buffer = {}
        self.last_heartbeat = {}
    
    async def start(self):
        """Start WebSocket connections for live data."""
        from coinbase.websocket import WSClient
        
        def on_message(msg):
            self._process_message(msg)
        
        self.websocket_client = WSClient(
            api_key=os.environ['CDP_API_KEY'],
            api_secret=os.environ['CDP_API_SECRET'],
            on_message=on_message
        )
        
        # Subscribe to channels
        await self.websocket_client.subscribe(
            product_ids=self.symbols,
            channels=['ticker', 'level2', 'user']
        )
        
        # Start heartbeat monitor
        asyncio.create_task(self._heartbeat_monitor())
    
    def _process_message(self, msg):
        """Process incoming WebSocket message."""
        channel = msg.get('channel')
        product_id = msg.get('product_id')
        
        if channel == 'ticker':
            self.data_buffer[product_id] = {
                'timestamp': datetime.utcnow(),
                'price': float(msg['price']),
                'bid': float(msg['best_bid']),
                'ask': float(msg['best_ask']),
                'volume_24h': float(msg.get('volume_24_h', 0))
            }
            self.last_heartbeat[product_id] = datetime.utcnow()
    
    async def _heartbeat_monitor(self):
        """Monitor for stale data."""
        while True:
            await asyncio.sleep(10)
            now = datetime.utcnow()
            
            for symbol, last_time in self.last_heartbeat.items():
                if (now - last_time).seconds > 30:
                    logger.warning(f"Stale data for {symbol}: {(now - last_time).seconds}s")
                    # Trigger reconnection
                    await self._reconnect(symbol)
```

### 11.2 Model Retraining Pipeline

```python
class ModelRetrainingPipeline:
    """
    Automated model retraining with monitoring.
    """
    
    def __init__(
        self,
        retrain_frequency_days: int = 7,
        min_samples_for_retrain: int = 168 * 7,  # 1 week of hourly data
        performance_threshold: float = 0.8  # Retrain if Sharpe < 80% of historical
    ):
        self.retrain_frequency_days = retrain_frequency_days
        self.min_samples = min_samples_for_retrain
        self.performance_threshold = performance_threshold
        self.last_retrain_date = None
        self.model_version = 0
    
    def check_retrain_needed(
        self,
        current_date: datetime,
        recent_performance: float,
        historical_performance: float
    ) -> Tuple[bool, str]:
        """
        Determine if model retraining is needed.
        
        Triggers:
        1. Scheduled retraining interval
        2. Performance degradation
        """
        reasons = []
        
        # Check scheduled interval
        if self.last_retrain_date is None:
            reasons.append("Initial training")
        elif (current_date - self.last_retrain_date).days >= self.retrain_frequency_days:
            reasons.append(f"Scheduled ({self.retrain_frequency_days} days)")
        
        # Check performance degradation
        perf_ratio = recent_performance / historical_performance if historical_performance > 0 else 0
        if perf_ratio < self.performance_threshold:
            reasons.append(f"Performance degradation ({perf_ratio:.1%} of historical)")
        
        return len(reasons) > 0, "; ".join(reasons)
    
    def retrain(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        model_factory: Callable
    ) -> Dict:
        """
        Execute model retraining.
        """
        # Train new model
        new_model = model_factory()
        new_model.fit(training_data)
        
        # Validate
        val_metrics = self._evaluate(new_model, validation_data)
        
        # Compare to current model
        current_metrics = self._evaluate(self.current_model, validation_data)
        
        # Only deploy if improvement
        if val_metrics['sharpe'] > current_metrics['sharpe']:
            self.current_model = new_model
            self.model_version += 1
            self.last_retrain_date = datetime.utcnow()
            
            return {
                'status': 'deployed',
                'old_sharpe': current_metrics['sharpe'],
                'new_sharpe': val_metrics['sharpe'],
                'version': self.model_version
            }
        else:
            return {
                'status': 'rejected',
                'old_sharpe': current_metrics['sharpe'],
                'new_sharpe': val_metrics['sharpe'],
                'reason': 'No improvement'
            }
```

### 11.3 Edge Decay Monitoring

```python
class EdgeDecayMonitor:
    """
    Monitor for alpha/edge decay in production.
    """
    
    def __init__(
        self,
        rolling_window_days: int = 30,
        decay_threshold: float = -0.5,  # Alert if IC drops 50%
        minimum_ic: float = 0.01  # Minimum viable IC
    ):
        self.rolling_window = rolling_window_days
        self.decay_threshold = decay_threshold
        self.minimum_ic = minimum_ic
        self.ic_history = []
    
    def update(
        self,
        predictions: pd.Series,
        realized_returns: pd.Series
    ) -> Dict:
        """
        Update IC tracking and check for decay.
        """
        # Calculate rolling IC
        current_ic = predictions.corr(realized_returns)
        self.ic_history.append({
            'timestamp': datetime.utcnow(),
            'ic': current_ic
        })
        
        # Calculate IC statistics
        ic_df = pd.DataFrame(self.ic_history)
        recent_ic = ic_df.tail(self.rolling_window)['ic'].mean()
        historical_ic = ic_df['ic'].mean()
        
        # Decay detection
        ic_change = (recent_ic - historical_ic) / historical_ic if historical_ic != 0 else 0
        
        status = 'healthy'
        alerts = []
        
        if recent_ic < self.minimum_ic:
            status = 'critical'
            alerts.append(f"IC below minimum: {recent_ic:.4f}")
        elif ic_change < self.decay_threshold:
            status = 'warning'
            alerts.append(f"IC decay detected: {ic_change:.1%}")
        
        # Calculate trend
        if len(ic_df) >= 30:
            ic_trend = np.polyfit(range(30), ic_df.tail(30)['ic'], 1)[0]
        else:
            ic_trend = 0
        
        return {
            'current_ic': current_ic,
            'rolling_ic': recent_ic,
            'historical_ic': historical_ic,
            'ic_change_pct': ic_change,
            'ic_trend': ic_trend,
            'status': status,
            'alerts': alerts
        }
```

---

## 12. Common Crypto Backtesting Traps

### 12.1 Critical Warnings

| Trap | Description | Mitigation |
|------|-------------|------------|
| **Survivorship Bias** | Only testing on assets that still exist. Many tokens have delisted/failed. | Include delisted assets; use point-in-time universe construction |
| **Exchange Risk** | FTX, Mt. Gox collapses not reflected in backtests | Include exchange-level risk scenarios; diversify across exchanges |
| **Funding Rate Data** | Historical funding may not be available; using wrong timestamps | Validate funding data availability; use bi-temporal storage |
| **24/7 Market Assumption** | Assuming constant liquidity; ignoring weekends, holidays | Model liquidity variation; increase slippage during low-volume periods |
| **Leverage/Margin** | Ignoring margin requirements and liquidation risk | Simulate full margin lifecycle; include liquidation events |
| **Flash Crashes** | May 2021, March 2020 crashes create unrealistic fills | Use realistic slippage models; cap maximum fills per bar |
| **Stablecoin Depegs** | USDT, USDC depegs affect settlement | Include stablecoin risk scenarios |
| **Regulatory Events** | China bans, US enforcement not predictable | Stress test against historical regulatory shocks |
| **Data Quality** | Missing data, bad prints common in crypto | Validate all data; use multiple sources |
| **Market Microstructure** | Order book depth, queue position matter | Simulate realistic order placement; avoid unrealistic fills |

### 12.2 Realistic Expectation Setting

Based on academic research and practitioner experience:

| Metric | Unrealistic | Suspicious | Reasonable |
|--------|-------------|------------|------------|
| Sharpe Ratio | > 3.0 | 2.0 - 3.0 | 0.5 - 2.0 |
| Max Drawdown | < 5% | 5% - 10% | 15% - 30% |
| Hit Rate | > 65% | 55% - 65% | 45% - 55% |
| Annual Return | > 100% | 50% - 100% | 15% - 50% |
| Trade Frequency | >> 100/day | 20-100/day | 1-20/day |

**Key Principle**: If your backtest results look too good to be true, they almost certainly are. Assume there's a bug or data issue until proven otherwise.

---

## 13. Implementation Checklist

### 13.1 Pre-Development

- [ ] Verify API access and rate limits for all data sources
- [ ] Confirm perpetual futures contract availability and specifications
- [ ] Establish data quality validation procedures
- [ ] Define clear success criteria (target Sharpe, max drawdown)
- [ ] Set budget for trading fees and maximum capital at risk

### 13.2 Development

- [ ] Implement point-in-time data storage with bi-temporal timestamps
- [ ] Build feature engineering pipeline with strict temporal constraints
- [ ] Create event-driven backtesting engine
- [ ] Implement realistic cost model (fees + slippage + funding)
- [ ] Build walk-forward optimization framework
- [ ] Implement risk management modules (position sizing, drawdown controls)
- [ ] Create comprehensive performance metrics suite

### 13.3 Validation

- [ ] Run overfitting detection (DSR, PBO)
- [ ] Perform regime-conditional analysis
- [ ] Compare walk-forward OOS performance to in-sample
- [ ] Stress test against historical crisis periods
- [ ] Validate with paper trading (minimum 30 days)

### 13.4 Deployment

- [ ] Set up production data pipeline with monitoring
- [ ] Implement model retraining automation
- [ ] Create edge decay monitoring dashboard
- [ ] Establish kill switch and emergency procedures
- [ ] Document all systems and procedures

---

## 14. Conclusion

This document provides a comprehensive framework for building an AI-driven trading system for Coinbase US perpetual futures. Key principles:

1. **Rigor over results**: A robust methodology that produces modest backtested returns is worth far more than a flawed approach showing spectacular performance.

2. **Temporal discipline**: Strict separation of train/validation/test data and elimination of lookahead bias are non-negotiable requirements.

3. **Realistic expectations**: Crypto markets are competitive. Sustainable edges are small and require excellent execution to capture.

4. **Risk first**: Position sizing, drawdown controls, and regime filters are essential for survival.

5. **Continuous monitoring**: Edges decay. Successful deployment requires constant vigilance and adaptation.

**Final Warning**: This system is designed for research and educational purposes. Trading perpetual futures involves substantial risk of loss, including the potential loss of your entire investment. Never trade with money you cannot afford to lose.

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Author: AI Trading System Design Framework*