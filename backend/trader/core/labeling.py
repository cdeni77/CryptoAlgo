from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TripleBarrierSpec:
    """Execution-aligned barrier configuration.

    Labels use 3-class encoding:
      1.0  => TP first touch
      -1.0 => SL first touch
      0.0  => timeout / neutral within horizon

    Neutral-direction entries (no momentum consensus) are left as NaN and are excluded
    from model training before SL/TP are binary-mapped to 0/1.
    """

    horizon_hours: int
    tp_mult: float
    sl_mult: float


def resolve_profile_label_horizon(max_hold_hours: int, label_forward_hours: int) -> int:
    """Prefer execution hold horizon; fallback to explicit label horizon."""
    if max_hold_hours and max_hold_hours > 0:
        return int(max_hold_hours)
    return int(label_forward_hours)


def momentum_direction_series(ohlcv: pd.DataFrame) -> pd.Series:
    """Return directional intent per bar: 1 long, -1 short, 0 neutral."""
    close = ohlcv['close']
    ret_24h = close.pct_change(24).ffill()
    ret_72h = close.pct_change(72).ffill()
    sma_50 = close.rolling(50).mean()

    direction = pd.Series(index=ohlcv.index, dtype=float)
    for ts in ohlcv.index:
        entry_px = close.loc[ts]
        r24 = ret_24h.get(ts, 0.0)
        r72 = ret_72h.get(ts, 0.0)
        sma = sma_50.get(ts, entry_px)
        if pd.isna(r24):
            r24 = 0.0
        if pd.isna(r72):
            r72 = 0.0
        if pd.isna(sma):
            sma = entry_px

        score = (1 if r24 > 0 else -1) + (1 if r72 > 0 else -1) + (1 if entry_px > sma else -1)
        if score >= 2:
            direction.loc[ts] = 1.0
        elif score <= -2:
            direction.loc[ts] = -1.0
        else:
            direction.loc[ts] = 0.0
    return direction


def _label_for_barrier_path(future: pd.DataFrame, direction: int, tp_px: float, sl_px: float) -> float:
    label = 0.0
    for _, bar in future.iterrows():
        tp_hit = (direction == 1 and bar['high'] >= tp_px) or (direction == -1 and bar['low'] <= tp_px)
        sl_hit = (direction == 1 and bar['low'] <= sl_px) or (direction == -1 and bar['high'] >= sl_px)

        if tp_hit and not sl_hit:
            label = 1.0
            break
        if sl_hit and not tp_hit:
            label = -1.0
            break
        if tp_hit and sl_hit:
            label = -1.0
            break
    return label


def compute_labels_from_ohlcv_iteration(
    ohlcv: pd.DataFrame,
    spec: TripleBarrierSpec,
    direction: pd.Series,
) -> pd.Series:
    target = pd.Series(index=ohlcv.index, dtype=float)
    vol = ohlcv['close'].pct_change().rolling(24).std().ffill()

    for idx in range(len(ohlcv) - spec.horizon_hours):
        ts = ohlcv.index[idx]
        row_vol = vol.iloc[idx]
        if pd.isna(row_vol) or row_vol <= 0:
            continue

        side = int(direction.get(ts, 0.0))
        if side == 0:
            continue

        entry_px = ohlcv['close'].iloc[idx]
        tp_move = spec.tp_mult * row_vol
        sl_move = spec.sl_mult * row_vol

        if side == 1:
            tp_px = entry_px * (1 + tp_move)
            sl_px = entry_px * (1 - sl_move)
        else:
            tp_px = entry_px * (1 - tp_move)
            sl_px = entry_px * (1 + sl_move)

        future = ohlcv.iloc[idx + 1: idx + 1 + spec.horizon_hours]
        target.iloc[idx] = _label_for_barrier_path(future, side, tp_px, sl_px)

    return target


def compute_labels_from_feature_index(
    ohlcv: pd.DataFrame,
    feature_index: pd.DatetimeIndex,
    spec: TripleBarrierSpec,
    direction: pd.Series,
) -> pd.Series:
    target = pd.Series(index=feature_index, dtype=float)
    vol = ohlcv['close'].pct_change().rolling(24).std().ffill()

    for ts in feature_index:
        if ts not in ohlcv.index or ts not in vol.index:
            continue

        row_vol = vol.loc[ts]
        if pd.isna(row_vol) or row_vol <= 0:
            continue

        side = int(direction.get(ts, 0.0))
        if side == 0:
            continue

        pos = ohlcv.index.get_loc(ts)
        if pos + spec.horizon_hours >= len(ohlcv):
            continue

        entry_px = ohlcv.loc[ts, 'close']
        tp_move = spec.tp_mult * row_vol
        sl_move = spec.sl_mult * row_vol

        if side == 1:
            tp_px = entry_px * (1 + tp_move)
            sl_px = entry_px * (1 - sl_move)
        else:
            tp_px = entry_px * (1 - tp_move)
            sl_px = entry_px * (1 + sl_move)

        future = ohlcv.iloc[pos + 1: pos + 1 + spec.horizon_hours]
        target.loc[ts] = _label_for_barrier_path(future, side, tp_px, sl_px)

    return target


def assert_label_path_consistency(
    ohlcv: pd.DataFrame,
    feature_index: pd.DatetimeIndex,
    spec: TripleBarrierSpec,
    direction: pd.Series,
    sample_size: int = 200,
) -> None:
    from_ohlcv = compute_labels_from_ohlcv_iteration(ohlcv, spec, direction)
    from_feature = compute_labels_from_feature_index(ohlcv, feature_index, spec, direction)

    aligned = pd.concat([from_ohlcv.rename('ohlcv_path'), from_feature.rename('feature_path')], axis=1)
    aligned = aligned.dropna(subset=['ohlcv_path', 'feature_path'])
    if aligned.empty:
        return

    sampled = aligned.iloc[:sample_size]
    mismatch = sampled[sampled['ohlcv_path'] != sampled['feature_path']]
    if not mismatch.empty:
        preview = mismatch.head(5).to_dict('index')
        raise ValueError(
            f"Label path mismatch detected for {len(mismatch)} rows in first {len(sampled)} samples: {preview}"
        )
