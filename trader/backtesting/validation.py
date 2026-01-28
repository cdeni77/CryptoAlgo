"""
Walk-Forward Validation for robust strategy evaluation.

Prevents overfitting by training on past data and testing on future data,
rolling forward through time.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .engine import Backtester, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from one walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    best_params: Dict


class WalkForwardValidator:
    """
    Walk-forward optimization and validation.
    
    Key principle: Never use future data to make decisions.
    
    Process:
    1. Train on window [T0, T1]
    2. Test on window [T1+buffer, T2]
    3. Roll forward and repeat
    """
    
    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 14,
        step_days: int = 14,
        buffer_days: int = 1,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.buffer_days = buffer_days
    
    def run(
        self,
        strategy_factory: Callable[..., Any],
        param_grid: List[Dict],
        ohlcv_data: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        backtester: Backtester,
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward validation.
        
        Args:
            strategy_factory: Function(params) -> Strategy
            param_grid: List of parameter dicts to try
            ohlcv_data: OHLCV data
            features: Feature data
            backtester: Backtester instance
        """
        # Get date range
        all_dates = set()
        for df in ohlcv_data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)
        
        if not dates:
            return []
        
        start_date = dates[0]
        end_date = dates[-1]
        
        windows = self._generate_windows(start_date, end_date)
        
        if not windows:
            logger.warning("Not enough data for walk-forward validation")
            return []
        
        logger.info(f"Walk-forward: {len(windows)} windows, {len(param_grid)} param sets")
        
        results = []
        
        for i, window in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}")
            
            best_params = None
            best_sharpe = -np.inf
            best_train_metrics = None
            
            # Find best params on training data
            for params in param_grid:
                strategy = strategy_factory(**params)
                
                _, train_metrics = backtester.run(
                    strategy=strategy,
                    ohlcv_data=ohlcv_data,
                    features=features,
                    start_date=window['train_start'],
                    end_date=window['train_end'],
                )
                
                if train_metrics.sharpe_ratio > best_sharpe:
                    best_sharpe = train_metrics.sharpe_ratio
                    best_params = params
                    best_train_metrics = train_metrics
            
            # Test with best params
            strategy = strategy_factory(**best_params)
            _, test_metrics = backtester.run(
                strategy=strategy,
                ohlcv_data=ohlcv_data,
                features=features,
                start_date=window['test_start'],
                end_date=window['test_end'],
            )
            
            results.append(WalkForwardResult(
                train_start=window['train_start'],
                train_end=window['train_end'],
                test_start=window['test_start'],
                test_end=window['test_end'],
                train_metrics=best_train_metrics,
                test_metrics=test_metrics,
                best_params=best_params,
            ))
            
            logger.info(f"  Train Sharpe: {best_train_metrics.sharpe_ratio:.2f}, "
                       f"Test Sharpe: {test_metrics.sharpe_ratio:.2f}, "
                       f"Params: {best_params}")
        
        return results
    
    def _generate_windows(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate train/test windows."""
        windows = []
        current = start_date
        
        while True:
            train_end = current + timedelta(days=self.train_days)
            test_start = train_end + timedelta(days=self.buffer_days)
            test_end = test_start + timedelta(days=self.test_days)
            
            if test_end > end_date:
                break
            
            windows.append({
                'train_start': current,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })
            
            current += timedelta(days=self.step_days)
        
        return windows
    
    def summarize(self, results: List[WalkForwardResult]) -> Dict:
        """Summarize walk-forward results."""
        if not results:
            return {'error': 'No results'}
        
        train_sharpes = [r.train_metrics.sharpe_ratio for r in results]
        test_sharpes = [r.test_metrics.sharpe_ratio for r in results]
        train_returns = [r.train_metrics.total_return for r in results]
        test_returns = [r.test_metrics.total_return for r in results]
        
        return {
            'num_windows': len(results),
            'avg_train_sharpe': np.mean(train_sharpes),
            'avg_test_sharpe': np.mean(test_sharpes),
            'sharpe_decay': np.mean(train_sharpes) - np.mean(test_sharpes),
            'test_sharpe_std': np.std(test_sharpes),
            'pct_profitable_windows': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'pct_positive_sharpe': sum(1 for s in test_sharpes if s > 0) / len(test_sharpes),
            'avg_train_return': np.mean(train_returns),
            'avg_test_return': np.mean(test_returns),
            'total_test_return': np.prod([1 + r for r in test_returns]) - 1,
        }
    
    def print_summary(self, results: List[WalkForwardResult]):
        """Print formatted summary."""
        summary = self.summarize(results)
        
        print("\n" + "=" * 60)
        print("WALK-FORWARD VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Windows: {summary['num_windows']}")
        print(f"\nSharpe Ratios:")
        print(f"  Avg Train: {summary['avg_train_sharpe']:.2f}")
        print(f"  Avg Test:  {summary['avg_test_sharpe']:.2f}")
        print(f"  Decay:     {summary['sharpe_decay']:.2f}")
        print(f"  Test Std:  {summary['test_sharpe_std']:.2f}")
        print(f"\nReturns:")
        print(f"  Avg Train: {summary['avg_train_return']*100:.2f}%")
        print(f"  Avg Test:  {summary['avg_test_return']*100:.2f}%")
        print(f"  Total OOS: {summary['total_test_return']*100:.2f}%")
        print(f"\nRobustness:")
        print(f"  % Profitable Windows: {summary['pct_profitable_windows']*100:.0f}%")
        print(f"  % Positive Sharpe:    {summary['pct_positive_sharpe']*100:.0f}%")
        print("=" * 60)