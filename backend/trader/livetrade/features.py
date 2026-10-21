"""The features `score_live` cannot build from bars: book, cross-venue, implied vol.

Each uses the backtest's definition, on the snapshot the order is priced
against. A missing feature scores NaN without error, so this module warns.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from livetrade import state
from livetrade.quotes import (
    _cumulative, _ladder_touch, _resting_total, depth_dollars,
    ladder_from_cache, quote_ladder,
)

logger = logging.getLogger('live')

# The gap at each decision offset of the current window, per symbol, so
# `venue_gap_change_5` matches the backtest's `shift(1)` within one
# (symbol, window_open).
_GAP_HISTORY: dict = {}

# The store-backed implied-vol lookup, refreshed at most this often, because
# `ResearchStore.read` is a synchronous Parquet read on the decision path.
_FIT_CACHE: dict = {}
_FIT_REFRESH_SECONDS = 120.0


def _venue_caches() -> tuple[dict, dict]:
    """Return the recorders' in-process caches, empty when they are not running."""
    # Prefer the Polymarket stream, because training assumes a peer book
    # contemporaneous with the Kalshi book. Each entry is tagged with its
    # `transport`.
    pm = {}
    try:
        from scripts.record_pm_stream import CACHE as pm_ws
        pm = {sym: dict(entry, transport='stream')
              for sym, entry in dict(pm_ws).items()}
    except Exception:                                         # noqa: BLE001
        pm = {}
    if not pm:
        try:
            from scripts.record_pm_ladder import CACHE as pm_rest
            pm = {sym: dict(entry, transport='rest')
                  for sym, entry in dict(pm_rest).items()}
        except Exception:                                     # noqa: BLE001
            pm = {}
    try:
        from scripts.record_implied_vol import CACHE as iv
    except Exception:                                         # noqa: BLE001
        iv = {}
    return pm, iv


def config_offsets() -> tuple:
    """Return the running config's decision offsets, for the gap-change lookback."""
    return tuple((state._RUNNING_CONFIG or DEFAULT_CONFIG).decision_offsets)


def reset_gap_history() -> None:
    """Clear the gap history."""
    _GAP_HISTORY.clear()


def gap_change(symbol: str, gap, *, window_open, offset: int) -> float:
    """Return the change in the cross-venue gap since the previous decision offset.

    NaN at the first offset of a window: the backtest never differences across
    a window boundary. A missing gap returns NaN and is not stored.
    """
    key = (str(symbol), pd.Timestamp(window_open))
    # Evict by window, never by arrival, so symbols of one window do not evict
    # each other.
    if key not in _GAP_HISTORY:
        for stale in [k for k in _GAP_HISTORY if k[1] != key[1]]:
            del _GAP_HISTORY[stale]
        _GAP_HISTORY[key] = {}
    previous = _GAP_HISTORY[key]
    try:
        value = float(gap)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(value):
        return float('nan')
    # Only the immediately previous offset, as in training's `shift(1)`. A hole
    # there gives NaN; skipping it would return a longer change.
    offsets = sorted(config_offsets())
    try:
        position = offsets.index(int(offset))
    except ValueError:
        previous[int(offset)] = value
        return float('nan')
    prior = offsets[position - 1] if position > 0 else None
    before = previous.get(prior) if prior is not None else None
    previous[int(offset)] = value
    if before is None:
        return float('nan')
    return float(value - before)


# The book at every MINUTE of the window in progress, per symbol, so
# `book_flow` has a history to difference. Same shape and same eviction rule as
# `_GAP_HISTORY` above.
#
# **Why this has to exist at all.** `core/book_flow.book_flow_features` is
# called from exactly one place, `core/dataset.py` -- the research path. The
# live path never built it, so all six features were NaN on every real-money
# decision while the deployed artifact was fitted on them. The loop said so
# every cycle ("scoring with features the row cannot supply") and nothing
# escalated it.
#
# **Why a MINUTE sampler and not the decision cycle.** `book_flow` differences
# over one and three minutes, and decisions happen only at offsets 3/6/9/12, so
# the decision cycle alone can never supply offset-1 or offset-3. The history
# has to be recorded every minute.
_BOOK_HISTORY: dict = {}

#: How fresh a recorded minute must be to be worth keeping, in seconds.
#: `core/quotes.decision_age` ages a minute-grid source FORWARD --
#: `latency - raw` -- so a row observed more than `decision_latency_seconds`
#: after its minute goes NEGATIVE and `book_flow_features` drops it. Measured
#: 2026-10-21: at age 0s a stream row yields 6/6 features, at 7s it yields 0/6.
#: The live decision cycle starts ~7s into the minute, so sampling there would
#: have reproduced the all-NaN bug while looking like a fix.
BOOK_SAMPLE_MAX_AGE_SECONDS = 3.0


def reset_book_history() -> None:
    """Drop every recorded minute. For tests, and for a clean restart."""
    _BOOK_HISTORY.clear()


def _touch_size(levels, best: float, *, invert: bool) -> float:
    """Resting size at exactly the touch.

    `build_depth.py` defines `yes_bid_size` as the size at the best price and
    nothing behind it, so this matches it rather than inventing a second
    convention for the same column name.
    """
    import numpy as _np
    total, seen = 0.0, False
    for entry in levels or []:
        try:
            price, size = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        effective = (1.0 - price) if invert else price
        if _np.isfinite(best) and abs(effective - best) < 1e-9:
            total += size
            seen = True
    return total if seen else float('nan')


def record_book_minute(symbol: str, *, window_open, offset: int,
                       yes_levels, no_levels,
                       age_seconds: float = 0.0) -> bool:
    """Remember this minute's book. Returns whether it was kept.

    Refused, rather than stored with a flattering stamp, when the sample is
    older than `BOOK_SAMPLE_MAX_AGE_SECONDS`: `book_flow_features` would drop
    it anyway, and storing it would hide a sampler that has drifted off the
    minute boundary behind features that merely look absent.
    """
    import numpy as _np
    try:
        off = int(offset)
        age = float(age_seconds)
    except (TypeError, ValueError):
        return False
    if not _np.isfinite(age) or age < 0.0 or age > BOOK_SAMPLE_MAX_AGE_SECONDS:
        return False
    if not yes_levels and not no_levels:
        return False

    bid = _ladder_touch(yes_levels, invert=False, fallback=_np.nan)
    ask = _ladder_touch(no_levels, invert=True, fallback=_np.nan)
    key = (str(symbol), pd.Timestamp(window_open))
    if key not in _BOOK_HISTORY:
        # Evict by WINDOW, never by arrival: three symbols are sampled in the
        # same minute of the same window and must not evict each other.
        for stale in [k for k in _BOOK_HISTORY if k[1] != key[1]]:
            del _BOOK_HISTORY[stale]
        _BOOK_HISTORY[key] = {}
    _BOOK_HISTORY[key][off] = {
        'venue': 'kalshi', 'symbol': str(symbol),
        'window_open': pd.Timestamp(window_open), 'offset_minutes': off,
        'yes_bid': bid, 'yes_ask': ask,
        'yes_bid_size': _touch_size(yes_levels, bid, invert=False),
        'yes_ask_size': _touch_size(no_levels, ask, invert=True),
        # The stream is the same observer `live_ws` names in `venue_depth`, and
        # the age is the real one -- see the constant above for why it must be
        # small rather than merely recorded.
        'source': 'live_ws', 'quote_age_seconds': age,
    }
    return True


def book_history_depth(window_open=None) -> pd.DataFrame:
    """Everything remembered, in `venue_depth` shape."""
    rows = [r for (sym, open_at), minutes in _BOOK_HISTORY.items()
            if window_open is None or open_at == pd.Timestamp(window_open)
            for r in minutes.values()]
    return pd.DataFrame(rows)


def book_flow_row(scored: pd.DataFrame) -> pd.DataFrame:
    """`book_flow` for the rows being scored, from the recorded minutes.

    Calls `core.book_flow.book_flow_features` -- the function the backtest
    calls, not a live re-implementation of it -- so the two cannot drift. The
    only live-specific part is assembling the depth frame it reads.
    """
    from core.book_flow import BOOK_FLOW, book_flow_features
    import numpy as _np

    out = pd.DataFrame(index=scored.index)
    for col in BOOK_FLOW:
        out[col] = _np.nan
    if not len(scored):
        return out
    depth = book_history_depth()
    if not len(depth):
        return out
    windows = pd.DataFrame({
        'symbol': scored['symbol'].to_numpy(),
        'window_open': pd.to_datetime(scored['window_open'], utc=True),
        'offset': pd.to_numeric(scored.get('offset'), errors='coerce'),
    }, index=scored.index)
    got = book_flow_features(windows, depth,
                             decision_latency_seconds=float(
                                 getattr(DEFAULT_CONFIG,
                                         'decision_latency_seconds', 3.0)))
    for col in BOOK_FLOW:
        if col in got.columns:
            out[col] = got[col].to_numpy()
    return out


# A peer cap under this refuses most peer books on transport latency alone.
TIGHT_PEER_CAP_SECONDS = 5.0


def peer_max_age_seconds(config=None) -> float:
    """Return the peer book's age cap, as `core.dataset.peer_quote_limits` does."""
    config = config or state._RUNNING_CONFIG or DEFAULT_CONFIG
    peer = getattr(config, 'peer_max_quote_age_seconds', None)
    return float(config.max_quote_age_seconds if peer is None else peer)


def warn_if_peer_cap_tight(config) -> None:
    """Warn when the peer cap is under `TIGHT_PEER_CAP_SECONDS`."""
    cap = peer_max_age_seconds(config)
    if cap < TIGHT_PEER_CAP_SECONDS:
        logger.warning('peer book age cap is %.1fs, under %.0fs: this would void '
                       'most peer books', cap, TIGHT_PEER_CAP_SECONDS)


# Why a peer book is not used. `future` is a book stamped after the decision.
PEER_REFUSALS = ('no_book', 'wrong_window', 'too_old', 'future', 'one_sided')
PEER_USED = 'used'

# window_open -> {verdict: count}, one verdict per symbol per decision cycle.
_PEER_COUNTS: dict = {}
# The refusals a stuck stream cache produces. `future` and `one_sided` describe
# a book that is current, so they do not count toward a resubscribe.
STUCK_REFUSALS = ('no_book', 'wrong_window', 'too_old')
# symbol -> consecutive decision cycles refused for a STUCK_REFUSALS reason.
_PEER_STREAK: dict = {}


def peer_refusal(pm: Optional[dict], *, now=None, window_open=None,
                 config=None) -> Optional[str]:
    """Return the reason `pm` cannot be used for this decision, or None.

    The window check comes first, because a rollover leaves a fresh book for
    the previous window. A reading with no window or no stamp is kept, as in
    `attach_quotes`. REST names the window `window_open`; the stream names it
    `window`.
    """
    if not pm:
        return 'no_book'
    peer_window = pm.get('window') or pm.get('window_open')
    if (window_open is not None and peer_window is not None
            and pd.Timestamp(peer_window) != pd.Timestamp(window_open)):
        return 'wrong_window'
    if pm.get('at') is not None and now is not None:
        age = (pd.Timestamp(now) - pd.Timestamp(pm['at'])).total_seconds()
        if age < 0:
            return 'future'
        if age > peer_max_age_seconds(config):
            return 'too_old'
    if np.isnan(_two_sided_mid(pm.get('best_bid'), pm.get('best_ask'))):
        return 'one_sided'
    return None


def _note_peer(window_open, verdict: str, *, symbol=None, config=None) -> None:
    if window_open is not None:
        key = pd.Timestamp(window_open)
        counts = _PEER_COUNTS.setdefault(key, {})
        counts[verdict] = counts.get(verdict, 0) + 1
    if symbol is not None:
        _track_streak(str(symbol), verdict, config)


def _track_streak(symbol: str, verdict: str, config=None) -> None:
    """Ask the stream to resubscribe `symbol` after N stuck refusals in a row.

    N is `peer_resubscribe_after_cycles`. The streak restarts after each request.
    """
    if verdict not in STUCK_REFUSALS:
        _PEER_STREAK.pop(symbol, None)
        return
    streak = _PEER_STREAK.get(symbol, 0) + 1
    config = config or state._RUNNING_CONFIG or DEFAULT_CONFIG
    if streak < int(config.peer_resubscribe_after_cycles):
        _PEER_STREAK[symbol] = streak
        return
    _PEER_STREAK.pop(symbol, None)
    logger.warning('%s: peer book refused %d decision cycles running (last: %s); '
                   'asking the Polymarket stream to resubscribe',
                   symbol, streak, verdict)
    try:
        from scripts.record_pm_stream import request_resubscribe
        request_resubscribe(symbol, verdict)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('%s: could not signal the Polymarket stream (%s)',
                       symbol, str(exc)[:110])


def reset_peer_tracking() -> None:
    """Clear the refusal counts and streaks."""
    _PEER_COUNTS.clear()
    _PEER_STREAK.clear()


def take_peer_counts(window_open) -> dict:
    """Pop {verdict: count} for `window_open` and drop older windows."""
    key = pd.Timestamp(window_open)
    counts = _PEER_COUNTS.pop(key, {})
    for stale in [k for k in _PEER_COUNTS if k < key]:
        del _PEER_COUNTS[stale]
    return counts


def cross_venue_row(pm: Optional[dict], quote, *, now=None, window_open=None,
                    config=None, symbol: Optional[str] = None) -> dict:
    """Return the `cross_venue` features from the in-process Polymarket cache.

    `best_bid` and `best_ask` are in cents on both sides. The peer must be for
    this window and within `peer_max_age_seconds`. Each verdict is counted per
    window (`take_peer_counts`).
    """
    from core.book_features import CROSS_VENUE

    out = {name: float('nan') for name in CROSS_VENUE}
    refusal = peer_refusal(pm, now=now, window_open=window_open, config=config)
    _note_peer(window_open, refusal or PEER_USED, symbol=symbol, config=config)
    if refusal is not None:
        if pm:
            logger.debug('%s: peer book refused (%s): window %s, at %s, %s/%s',
                         symbol or '?', refusal,
                         pm.get('window') or pm.get('window_open'), pm.get('at'),
                         pm.get('best_bid'), pm.get('best_ask'))
        else:
            logger.debug('%s: peer book refused (%s)', symbol or '?', refusal)
        pm = None
    k_mid = _two_sided_mid(getattr(quote, 'yes_bid', np.nan) * 100.0,
                           getattr(quote, 'yes_ask', np.nan) * 100.0)
    p_mid = _two_sided_mid(*(
        (pm.get('best_bid'), pm.get('best_ask')) if pm else (np.nan, np.nan)))
    # Absence is not agreement: a missing peer gives NaN, never a zero gap.
    out['pm_available'] = 0.0 if np.isnan(p_mid) else 1.0
    out['venue_prob_gap'] = k_mid - p_mid
    k_spread = _spread_cents(getattr(quote, 'yes_bid', np.nan) * 100.0,
                             getattr(quote, 'yes_ask', np.nan) * 100.0)
    p_spread = _spread_cents(*(
        (pm.get('best_bid'), pm.get('best_ask')) if pm else (np.nan, np.nan)))
    if k_spread > 0 and p_spread > 0:
        out['venue_spread_ratio'] = float(np.log(k_spread / p_spread))
    # The caller adds `venue_gap_change_5` from `gap_change`.
    return out


def _two_sided_mid(bid, ask) -> float:
    """Return the mid as a probability, or NaN unless both sides exist."""
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(bid) or np.isnan(ask):
        return float('nan')
    return (bid + ask) / 2.0 / 100.0


def _spread_cents(bid, ask) -> float:
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return float('nan')
    if np.isnan(bid) or np.isnan(ask):
        return float('nan')
    return ask - bid


def reset_fit_lookup() -> None:
    """Clear the memoised store read."""
    _FIT_CACHE.clear()


def _read_recent_fits():
    """Return recent `venue_implied_vol` rows, or an empty frame."""
    try:
        from core.datastore import ResearchStore
        frame = ResearchStore().read(
            'venue_implied_vol',
            columns=['symbol', 'event_time', 'implied_sigma_per_min',
                     'r2', 'n_strikes'])
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('implied-vol store read: %s', str(exc)[:110])
        return pd.DataFrame()
    if frame is None or not len(frame):
        return pd.DataFrame()
    stamp = pd.to_datetime(frame['event_time'], utc=True, errors='coerce')
    from core.book_features import MAX_FIT_AGE_MINUTES
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=MAX_FIT_AGE_MINUTES)
    return frame.assign(_at=stamp)[stamp >= cutoff]


def latest_fit(symbol: str, *, now, cache: dict, read=None):
    """Return the freshest usable ladder fit for `symbol`, or None.

    Tries the in-process recorder cache, then the store, which the backtest
    also reads. Both are limited to `MAX_FIT_AGE_MINUTES`.
    """
    from core.book_features import MAX_FIT_AGE_MINUTES

    now = pd.Timestamp(now)
    live = cache.get(symbol) if cache else None
    if live is not None:
        age = (now - pd.Timestamp(live['at'])).total_seconds() / 60.0
        if 0 <= age <= MAX_FIT_AGE_MINUTES:
            return dict(live, source='recorder')

    reader = read or _read_recent_fits
    stamp = _FIT_CACHE.get('at')
    if stamp is None or (now - stamp).total_seconds() >= _FIT_REFRESH_SECONDS:
        _FIT_CACHE['at'] = now
        _FIT_CACHE['frame'] = reader()
    frame = _FIT_CACHE.get('frame')
    if frame is None or not len(frame):
        return None
    if '_at' not in frame.columns:
        frame = frame.assign(
            _at=pd.to_datetime(frame['event_time'], utc=True, errors='coerce'))
    rows = frame[(frame['symbol'] == symbol) & frame['_at'].notna()]
    if not len(rows):
        return None
    row = rows.loc[rows['_at'].idxmax()]
    age = (now - row['_at']).total_seconds() / 60.0
    if age < 0 or age > MAX_FIT_AGE_MINUTES:
        return None
    return {'implied_sigma_per_min': float(row['implied_sigma_per_min']),
            'r2': float(row['r2']), 'n_strikes': float(row['n_strikes']),
            'at': row['_at'], 'source': 'store'}


async def prewarm_fit_lookup(*, horizon: float = 75.0) -> None:
    """Refresh the store-backed implied-vol frame in a thread, before it expires.

    This keeps the synchronous Parquet read in `latest_fit` off the decision path.
    """
    now = pd.Timestamp.now(tz='UTC')
    stamp = _FIT_CACHE.get('at')
    if stamp is not None and (now - stamp).total_seconds() < _FIT_REFRESH_SECONDS - horizon:
        return
    frame = await asyncio.to_thread(_read_recent_fits)
    _FIT_CACHE['frame'] = frame
    _FIT_CACHE['at'] = now


def implied_vol_row(fit: Optional[dict], *, sigma_per_min, now) -> dict:
    """Return the `implied_vol` features from a ladder fit.

    Staleness is a feature up to `MAX_FIT_AGE_MINUTES`, so the model can tell
    an old fit from a fresh one. Past the cap the fit is not used.
    """
    from core.book_features import IMPLIED_VOL, MAX_FIT_AGE_MINUTES

    out = {name: float('nan') for name in IMPLIED_VOL}
    if not fit:
        return out
    age = (pd.Timestamp(now) - pd.Timestamp(fit['at'])).total_seconds() / 60.0
    out['iv_staleness_minutes'] = float(age)
    if age > MAX_FIT_AGE_MINUTES or age < 0:
        return out
    implied = float(fit.get('implied_sigma_per_min', np.nan))
    out['implied_sigma_per_min'] = implied
    out['iv_r2'] = float(fit.get('r2', np.nan))
    out['iv_n_strikes'] = float(fit.get('n_strikes', np.nan))
    try:
        realised = float(sigma_per_min)
    except (TypeError, ValueError):
        return out
    if implied > 0 and realised > 0:
        out['iv_minus_realised'] = float(np.log(implied / realised))
    return out


def book_feature_row(quote, yes_levels, no_levels, *,
                     baseline_probability: float) -> dict:
    """Return the model's book features from the book the order prices against.

    Uses `core.book_features.market_state_features`, as the backtest does.
    Prices go in as cents.
    """
    import numpy as _np
    import pandas as _pd
    from core.book_features import market_state_features

    bid = quote.yes_bid if getattr(quote, 'yes_bid', None) is not None else _np.nan
    ask = quote.yes_ask if getattr(quote, 'yes_ask', None) is not None else _np.nan
    best_bid = float(bid) * 100.0 if _np.isfinite(_np.float64(bid)) else _np.nan
    best_ask = float(ask) * 100.0 if _np.isfinite(_np.float64(ask)) else _np.nan
    _bid_touch = _ladder_touch(yes_levels, invert=False, fallback=bid)
    _ask_touch = _ladder_touch(no_levels, invert=True, fallback=ask)

    frame = _pd.DataFrame([{
        'best_bid': best_bid, 'best_ask': best_ask,
        'baseline_probability': baseline_probability,
        'bid_at_touch': float(getattr(quote, 'yes_bid_size', 0.0) or 0.0),
        'ask_at_touch': float(getattr(quote, 'yes_ask_size', 0.0) or 0.0),
        # `best` is the ladder's own touch, not the quote's (see `_ladder_touch`).
        'bid_1c': _cumulative(yes_levels, best=_bid_touch, within=0.01,
                              invert=False) if yes_levels else _np.nan,
        'ask_1c': _cumulative(no_levels, best=_ask_touch, within=0.01,
                              invert=True) if no_levels else _np.nan,
        'bid_5c': _cumulative(yes_levels, best=_bid_touch, within=0.05,
                              invert=False) if yes_levels else _np.nan,
        'ask_5c': _cumulative(no_levels, best=_ask_touch, within=0.05,
                              invert=True) if no_levels else _np.nan,
        'bid_vol': _resting_total(yes_levels),
        'ask_vol': _resting_total(no_levels),
    }])
    return market_state_features(frame).iloc[0].to_dict()


def _index_cache():
    """Return the index recorder's in-process cache, or None."""
    try:
        from scripts.record_index import CACHE
        return CACHE
    except Exception:                                         # noqa: BLE001
        return None


def _attach_index_proxy(scored: pd.DataFrame, config=None, cache=None) -> None:
    """Attach the `index_proxy` group to the scoring rows, in place.

    Uses `index_features_row` at the row's `decision_time`, as the backtest
    does. With no recorder the columns stay NaN.
    """
    from core.index_features import INDEX_PROXY, index_features_row

    for column in INDEX_PROXY:
        if column not in scored.columns:
            scored[column] = np.nan
    cache = _index_cache() if cache is None else cache
    if cache is None or not len(scored):
        return
    config = config or state._RUNNING_CONFIG or DEFAULT_CONFIG
    for i, symbol in enumerate(scored['symbol']):
        window_open = pd.Timestamp(scored['window_open'].iloc[i])
        if 'decision_time' in scored.columns:
            decided = pd.Timestamp(scored['decision_time'].iloc[i])
        else:
            decided = window_open + pd.Timedelta(minutes=int(scored['offset'].iloc[i]))
        try:
            row = index_features_row(
                cache.snapshot(symbol), window_open, decided,
                float(scored['sigma_remaining'].iloc[i]),
                last_price=float(scored['last_price'].iloc[i]),
                symbol=symbol,
                max_age_seconds=float(config.index_proxy_max_age_seconds))
        except Exception as exc:                              # noqa: BLE001
            logger.warning('%s: index proxy features unavailable (%s)',
                           symbol, str(exc)[:80])
            continue
        for column, value in row.items():
            scored.iloc[i, scored.columns.get_loc(column)] = value


def _attach_book_features(scored: pd.DataFrame, quotes: dict, cache=None, *,
                          config=None, groups=None) -> None:
    """Attach book, cross-venue and implied-vol features to the scoring rows.

    Works in place, from the quote the order prices against. Also attaches
    `index_proxy` when `groups` names it.
    """
    from core.book_features import (
        CROSS_VENUE, IMPLIED_VOL, MARKET_PRICE, MARKET_STATE)

    if groups and 'index_proxy' in groups:
        _attach_index_proxy(scored, config)

    pm_cache, iv_cache = _venue_caches()
    now = pd.Timestamp.now(tz='UTC')
    columns = (tuple(MARKET_STATE) + tuple(MARKET_PRICE)
               + tuple(CROSS_VENUE) + tuple(IMPLIED_VOL)
               # Not features: `decide()` reads these to cap the stake.
               + ('depth_up', 'depth_down'))
    for column in columns:
        if column not in scored.columns:
            scored[column] = np.nan
    if not len(scored):
        return
    base = pd.to_numeric(scored.get('baseline_probability'), errors='coerce')
    for i, symbol in enumerate(scored['symbol']):
        entry = quotes.get(symbol)
        if entry is None:
            continue
        try:
            # Use the ladder the touch came from, so depth features describe
            # the priced snapshot.
            yes_levels, no_levels = quote_ladder(entry[0])
            if not yes_levels and not no_levels:
                yes_levels, no_levels = ladder_from_cache(cache, entry[1])
            row = book_feature_row(
                entry[0], yes_levels, no_levels,
                baseline_probability=float(base.iloc[i]) if len(base) > i else np.nan)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('%s: book features unavailable (%s)', symbol, str(exc)[:80])
            continue
        # The stake cap `decide()` reads: dollars at the crossed touch.
        up, down = depth_dollars(entry[0], yes_levels, no_levels)
        if up is not None:
            row['depth_up'] = up
        if down is not None:
            row['depth_down'] = down
        window_open = scored['window_open'].iloc[i]
        offset = (int(scored['offset'].iloc[i]) if 'offset' in scored.columns
                  else int(scored['offset_minutes'].iloc[i]))
        peer = pm_cache.get(symbol)
        fit = latest_fit(symbol, now=now, cache=iv_cache)
        DECISION_INPUTS[symbol] = {
            'window_open': window_open, 'offset': offset, 'peer': peer,
            'refusal': peer_refusal(peer, now=now, window_open=window_open,
                                    config=config),
            'fit': fit}
        row.update(cross_venue_row(
            peer, entry[0], now=now, window_open=window_open, config=config,
            symbol=symbol))
        row.update(implied_vol_row(
            fit, now=now,
            sigma_per_min=scored['sigma_per_min'].iloc[i]
            if 'sigma_per_min' in scored.columns else np.nan))
        row['venue_gap_change_5'] = gap_change(
            symbol, row.get('venue_prob_gap'), window_open=window_open,
            offset=offset)
        for column in columns:
            if column in row:
                scored.iloc[i, scored.columns.get_loc(column)] = row[column]


# symbol -> the peer book, refusal reason and implied-vol fit the last decision
# read. `_record_touch` writes them beside the Kalshi row.
DECISION_INPUTS: dict = {}
# symbol -> fit times this process has already written to `venue_implied_vol`.
_FITS_WRITTEN: dict = {}
_FITS_KEPT = 32


def _cents_to_probability(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return value / 100.0 if np.isfinite(value) else float('nan')


def peer_depth_row(symbol: str, inputs: dict, *, decided_at,
                   written_at=None) -> dict:
    """Return the `venue_depth` row for the peer book one decision used or refused.

    A used book is 'valid'. A refused book is 'invalid' when it does not
    describe this window, else 'suspicious'. Both stay out of the default read.
    """
    pm = inputs.get('peer') or {}
    refusal = inputs.get('refusal')
    window_open = pd.Timestamp(inputs['window_open'])
    offset = int(inputs['offset'])
    bid = _cents_to_probability(pm.get('best_bid'))
    ask = _cents_to_probability(pm.get('best_ask'))
    at = pm.get('at')
    age = (max(0.0, (pd.Timestamp(decided_at) - pd.Timestamp(at)).total_seconds())
           if at is not None and decided_at is not None else float('nan'))
    if refusal in ('no_book', 'wrong_window'):
        quality = 'invalid'
    elif (refusal is not None
          or not (0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0 and ask >= bid)):
        quality = 'suspicious'
    else:
        quality = 'valid'
    return {
        'venue': 'polymarket', 'symbol': symbol,
        'event_time': window_open + pd.Timedelta(minutes=offset),
        'available_time': (pd.Timestamp.now(tz='UTC') if written_at is None
                           else pd.Timestamp(written_at)),
        'quality': quality, 'market_ticker': pm.get('slug'),
        'window_open': window_open, 'offset_minutes': offset,
        'yes_bid': bid, 'yes_ask': ask,
        'yes_bid_size': float('nan'), 'yes_ask_size': float('nan'),
        'depth_bid_1c': float('nan'), 'depth_bid_5c': float('nan'),
        'depth_ask_1c': float('nan'), 'depth_ask_5c': float('nan'),
        'depth_bid_total': float('nan'), 'depth_ask_total': float('nan'),
        'levels_bid': float('nan'), 'levels_ask': float('nan'),
        'seq': float('nan'), 'gaps': 0.0, 'source': 'live_touch',
        'quote_age_seconds': age, 'transport': pm.get('transport'),
    }


def fit_row(symbol: str, fit: Optional[dict]) -> Optional[dict]:
    """Return the `venue_implied_vol` row for the fit one decision used, or None.

    None when there is no fit, the fit came from the store, or it was already
    written. `event_time` and `available_time` are both the capture time, so the
    point-in-time join sees the same age live used.
    """
    if not fit or fit.get('source') == 'store' or fit.get('at') is None:
        return None
    at = pd.Timestamp(fit['at'])
    if at in _FITS_WRITTEN.get(symbol, ()):
        return None
    return {
        'venue': 'kalshi', 'symbol': symbol, 'event_time': at,
        'available_time': at, 'quality': 'valid',
        'event_ticker': fit.get('event_ticker'),
        'close_time': fit.get('close_time'),
        'minutes_to_close': float(fit.get('minutes_to_close', np.nan)),
        'implied_sigma_per_min': float(fit.get('implied_sigma_per_min', np.nan)),
        'implied_spot': float(fit.get('implied_spot', np.nan)),
        'atm_strike': float(fit.get('atm_strike', np.nan)),
        'n_strikes': float(fit.get('n_strikes', np.nan)),
        'r2': float(fit.get('r2', np.nan)),
    }


def mark_fits_written(rows) -> None:
    """Remember written fits, so a fit used by several decisions is written once."""
    for row in rows:
        seen = _FITS_WRITTEN.setdefault(row['symbol'], [])
        seen.append(pd.Timestamp(row['event_time']))
        del seen[:-_FITS_KEPT]


def decision_input_rows(symbols, window_open, offset, *, decided_at):
    """Return (peer depth rows, fit rows) for the decision at (window_open, offset)."""
    depth, fits = [], []
    for symbol in symbols:
        inputs = DECISION_INPUTS.get(symbol)
        if (not inputs or inputs.get('window_open') is None
                or pd.Timestamp(inputs['window_open']) != pd.Timestamp(window_open)
                or int(inputs['offset']) != int(offset)):
            continue
        depth.append(peer_depth_row(symbol, inputs, decided_at=decided_at))
        row = fit_row(symbol, inputs.get('fit'))
        if row is not None:
            fits.append(row)
    return depth, fits


# Features that are legitimately absent at a known offset.
EXPECTED_NAN = {
    # A first difference has no prior value at the first offset of a window.
    'venue_gap_change_5': lambda offsets: min(offsets) if offsets else None,
}


def _warn_unscoreable_features(scored: pd.DataFrame, model,
                               *, offsets=()) -> None:
    """Warn when the model wants a feature this row cannot supply.

    A NaN feature does not raise, so this warning is the only signal. Absences
    in `EXPECTED_NAN` at their expected offset log at debug instead.
    """
    wanted = list(getattr(model, 'features', ()) or ())
    if not wanted or not len(scored):
        return
    offsets = tuple(int(o) for o in (offsets or ()))
    here = pd.to_numeric(scored.get('offset'), errors='coerce').dropna().unique() \
        if 'offset' in scored.columns else ()
    missing = [c for c in wanted if c not in scored.columns]
    empty = [c for c in wanted
             if c in scored.columns and not pd.to_numeric(
                 scored[c], errors='coerce').notna().any()]
    expected = [c for c in empty
                if c in EXPECTED_NAN and len(here) == 1
                and EXPECTED_NAN[c](offsets) == int(here[0])]
    if expected:
        logger.debug('all-NaN as expected at offset %s: %s',
                     int(here[0]) if len(here) == 1 else '?', expected)
        empty = [c for c in empty if c not in expected]
    if missing or empty:
        logger.warning(
            'scoring with features the row cannot supply — missing %s, all-NaN %s. '
            'LightGBM will substitute its learned default, so this is a different '
            'model from the one whose gates were measured.',
            missing or 'none', empty or 'none')


def prepare_init_score(scored: pd.DataFrame, model) -> pd.DataFrame:
    """Attach the init score this artifact was fitted on.

    `score_live` already attaches the baseline logit. A market-init artifact
    uses the quote mid, not the ask. A one-sided book gives NaN, and `decide()`
    abstains; it never falls back to the baseline.
    """
    source = getattr(model, 'init_score_source', 'baseline')
    if source != 'market':
        return scored
    from core.model import attach_market_logit

    out = scored.copy()
    out['market_probability'] = out['market_mid']
    return attach_market_logit(out)
