"""
Feature Engineering Module for Crypto Trading System.

This module provides:
- Price-based features (momentum, mean-reversion indicators)
- Volume and volatility features
- Derivatives-specific features (funding rates, basis, OI)
- Cross-asset features (correlations, beta, relative strength)
- Regime detection features

All features are computed with strict point-in-time constraints
to prevent lookahead bias in backtesting.
"""

from .engineering import (
    # Main pipeline
    FeaturePipeline,
    FeatureConfig,
    
    # Individual feature classes
    PriceFeatures,
    VolumeVolatilityFeatures,
    DerivativesFeatures,
    CrossAssetFeatures,
    RegimeFeatures,
    
    # Utility functions
    normalize_point_in_time,
    winsorize_point_in_time,
    get_feature_importance_names,
)

__all__ = [
    'FeaturePipeline',
    'FeatureConfig',
    'PriceFeatures',
    'VolumeVolatilityFeatures',
    'DerivativesFeatures',
    'CrossAssetFeatures',
    'RegimeFeatures',
    'normalize_point_in_time',
    'winsorize_point_in_time',
    'get_feature_importance_names',
]