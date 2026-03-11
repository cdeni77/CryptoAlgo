"""Utilities to load paper-candidate profile overrides from optimizer artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Optional

from core.coin_profiles import COIN_PROFILES, CoinProfile

logger = logging.getLogger(__name__)


def _resolve_coin_profile_name(name_or_symbol: str) -> Optional[str]:
    token = str(name_or_symbol or "").upper()
    if not token:
        return None

    if token in COIN_PROFILES:
        return token

    for profile_name, profile in COIN_PROFILES.items():
        prefixes = [str(p).upper() for p in profile.prefixes]
        if token in prefixes:
            return profile_name
    return None


def _coerce_coin_name(payload: dict, source_path: Path) -> Optional[str]:
    raw_coin = payload.get('coin')
    if raw_coin is None:
        raw_coin = source_path.stem
    return _resolve_coin_profile_name(str(raw_coin))


def _iter_candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() == '.json':
            yield path
        return

    if path.is_dir():
        for candidate in sorted(path.glob('*.json')):
            if candidate.is_file():
                yield candidate


def _build_override_profile(base_profile: CoinProfile, params: Dict[str, object]) -> CoinProfile:
    allowed_fields = {
        'signal_threshold',
        'label_forward_hours',
        'label_vol_target',
        'min_momentum_magnitude',
        'vol_mult_tp',
        'vol_mult_sl',
        'max_hold_hours',
        'cooldown_hours',
        'min_vol_24h',
        'max_vol_24h',
        'max_ensemble_std',
        'min_directional_agreement',
        'meta_probability_threshold',
        'strategy_family',
        'trade_freq_bucket',
        'position_size',
        'vol_sizing_target',
        'min_val_auc',
        'n_estimators',
        'max_depth',
        'learning_rate',
        'min_child_samples',
    }
    updates = {k: v for k, v in params.items() if k in allowed_fields}
    return replace(base_profile, **updates)


def load_paper_profile_overrides(path: Optional[str]) -> Dict[str, CoinProfile]:
    """Load paper profile override artifacts into CoinProfile overrides keyed by coin name."""
    if not path:
        return {}

    root = Path(path)
    if not root.exists():
        logger.warning("PAPER_PROFILE_OVERRIDES_PATH not found: %s", root)
        return {}

    overrides: Dict[str, CoinProfile] = {}
    for artifact_path in _iter_candidate_files(root):
        try:
            payload = json.loads(artifact_path.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning("Skipping unreadable paper candidate artifact %s: %s", artifact_path, exc)
            continue

        if not isinstance(payload, dict):
            logger.warning("Skipping malformed paper candidate artifact %s", artifact_path)
            continue

        coin_name = _coerce_coin_name(payload, artifact_path)
        if coin_name is None:
            logger.warning("Skipping artifact without recognized coin: %s", artifact_path)
            continue

        if not bool(payload.get('holdout_passed', False)):
            logger.info("Skipping non-passing holdout artifact for %s (%s)", coin_name, artifact_path)
            continue

        evaluated_params = payload.get('evaluated_params')
        if not isinstance(evaluated_params, dict):
            logger.warning("Skipping artifact missing evaluated_params: %s", artifact_path)
            continue

        base_profile = COIN_PROFILES.get(coin_name)
        if base_profile is None:
            logger.warning("Skipping artifact with no base profile for coin %s (%s)", coin_name, artifact_path)
            continue

        overrides[coin_name] = _build_override_profile(base_profile, evaluated_params)

    if overrides:
        logger.info("Loaded %s paper profile override(s) from %s", len(overrides), root)
    return overrides
