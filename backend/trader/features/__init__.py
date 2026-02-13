"""
Feature Engineering Module for Crypto Trading System.
"""

from .engineering import (
    FeaturePipeline,
    FeatureConfig,
    PriceFeatures,
    VolumeVolatilityFeatures,
    FundingFeatures,
    OpenInterestFeatures,
    RegimeFeatures,
    BTCMeanReversionFeatures,
    SOLEcosystemFeatures,
    DOGESentimentFeatures,
    get_coin_feature_class,
    normalize_point_in_time,
    winsorize_point_in_time,
)

__all__ = [
    'FeaturePipeline',
    'FeatureConfig',
    'PriceFeatures',
    'VolumeVolatilityFeatures',
    'FundingFeatures',
    'OpenInterestFeatures',
    'RegimeFeatures',
    'BTCMeanReversionFeatures',
    'SOLEcosystemFeatures',
    'DOGESentimentFeatures',
    'get_coin_feature_class',
    'normalize_point_in_time',
    'winsorize_point_in_time',
]