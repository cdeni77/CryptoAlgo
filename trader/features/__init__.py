"""
Feature Engineering Module for Crypto Trading System.
"""

from .engineering import (
    FeaturePipeline,
    FeatureConfig,
    PriceFeatures,
    VolumeVolatilityFeatures,
    FundingFeatures,
    CrossAssetFeatures,
    RegimeFeatures,
    normalize_point_in_time,
    winsorize_point_in_time,
    get_feature_importance_names,
)

__all__ = [
    'FeaturePipeline',
    'FeatureConfig',
    'PriceFeatures',
    'VolumeVolatilityFeatures',
    'FundingFeatures',
    'CrossAssetFeatures',
    'RegimeFeatures',
    'normalize_point_in_time',
    'winsorize_point_in_time',
    'get_feature_importance_names',
]