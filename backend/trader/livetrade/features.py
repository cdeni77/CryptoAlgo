"""The features `score_live` cannot build from bars: book, cross-venue, implied vol.

Each is computed through the same definition the backtest uses, from the same
snapshot the order is priced against. A feature the artifact was fitted on and
this cannot supply is scored NaN, which LightGBM does not reject — so the
warnings here are the only sign live is not the model that was measured.
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

# The gap at each decision offset of the window in progress, per symbol, so
# `venue_gap_change_5` reproduces the backtest's `shift(1)` over the previous
# offset of the SAME (symbol, window_open).
_GAP_HISTORY: dict = {}

# The store-backed implied-vol lookup, refreshed at most this often. The hot
# path is a decision, and `ResearchStore.read` is a synchronous Parquet read; a
# value that changes every few minutes does not belong on the event loop every
# cycle.
_FIT_CACHE: dict = {}
_FIT_REFRESH_SECONDS = 120.0


def _venue_caches() -> tuple[dict, dict]:
    """The recorders' in-process caches, empty when they are not running.

    Imported lazily and defensively: `scripts.live` also runs standalone, where
    neither recorder exists and every feature they feed is honestly unknown.
    """
    # The Polymarket SOCKET first: it is contemporaneous with the Kalshi book,
    # as training assumes. The REST recorder writes ~32s past the minute, and a
    # gap against a minute-stale peer correlates only 0.672 with the
    # contemporaneous one. `cross_venue_row` refuses a stale reading either way.
    pm = {}
    try:
        from scripts.record_pm_stream import CACHE as pm_ws
        pm = dict(pm_ws)
    except Exception:                                         # noqa: BLE001
        pm = {}
    if not pm:
        try:
            from scripts.record_pm_ladder import CACHE as pm_rest
            pm = pm_rest
        except Exception:                                     # noqa: BLE001
            pm = {}
    try:
        from scripts.record_implied_vol import CACHE as iv
    except Exception:                                         # noqa: BLE001
        iv = {}
    return pm, iv


def config_offsets() -> tuple:
    """The decision offsets in force, for the gap-change lookback.

    The RUNNING config's, not the library default's: a loop started with other
    offsets must difference against its own previous offset.
    """
    return tuple((state._RUNNING_CONFIG or DEFAULT_CONFIG).decision_offsets)


def reset_gap_history() -> None:
    """Drop everything remembered. For tests, and for a clean restart."""
    _GAP_HISTORY.clear()


def gap_change(symbol: str, gap, *, window_open, offset: int) -> float:
    """The change in the cross-venue gap since the previous decision offset.

    NaN at the first offset of a window, by construction and not by accident:
    consecutive windows CHAIN — a window's strike is the previous window's
    settlement value — so differencing across a boundary produces a number that
    looks entirely correct and answers a different question. The backtest
    refuses it with `groupby(['symbol', 'window_open'])`, and so does this.

    A missing gap is a hole, not an observation: it neither differences nor
    displaces the last real reading, so the following offset still measures
    against something true.
    """
    key = (str(symbol), pd.Timestamp(window_open))
    # Evict by WINDOW, never by arrival: three symbols are scored in every
    # cycle of the same window, and must not evict each other.
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
    # The IMMEDIATELY previous decision offset, or nothing. Training is
    # `shift(1)` over a panel carrying every offset, so a hole at the previous
    # offset is NaN there; skipping it would return a six- or nine-minute change
    # under a column named for three.
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


# A peer cap under this refuses most peer books on transport latency alone.
TIGHT_PEER_CAP_SECONDS = 5.0


def peer_max_age_seconds(config=None) -> float:
    """The peer book's age cap: `peer_max_quote_age_seconds`, else
    `max_quote_age_seconds`. The same rule `core.dataset.peer_quote_limits`
    gives the backtest."""
    config = config or state._RUNNING_CONFIG or DEFAULT_CONFIG
    peer = getattr(config, 'peer_max_quote_age_seconds', None)
    return float(config.max_quote_age_seconds if peer is None else peer)


def warn_if_peer_cap_tight(config) -> None:
    """Log a WARNING when the effective peer cap is under
    `TIGHT_PEER_CAP_SECONDS`."""
    cap = peer_max_age_seconds(config)
    if cap < TIGHT_PEER_CAP_SECONDS:
        logger.warning('peer book age cap is %.1fs, under %.0fs: this would void '
                       'most peer books', cap, TIGHT_PEER_CAP_SECONDS)


# Why a peer book is not used. `future` is a book stamped after the decision.
PEER_REFUSALS = ('no_book', 'wrong_window', 'too_old', 'future', 'one_sided')
PEER_USED = 'used'

# window_open -> {verdict: count}, one verdict per symbol per decision cycle.
_PEER_COUNTS: dict = {}


def peer_refusal(pm: Optional[dict], *, now=None, window_open=None,
                 config=None) -> Optional[str]:
    """The reason `pm` cannot be used for this decision, or None.

    The window check comes first: a rollover leaves a fresh-looking book for
    the previous window. A reading with no window or no stamp is kept: the REST
    fallback publishes the window as `window_open`, the socket as `window`, and
    absence of a stamp is not evidence of staleness (matching `attach_quotes`).
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


def _note_peer(window_open, verdict: str) -> None:
    if window_open is None:
        return
    key = pd.Timestamp(window_open)
    counts = _PEER_COUNTS.setdefault(key, {})
    counts[verdict] = counts.get(verdict, 0) + 1


def take_peer_counts(window_open) -> dict:
    """{verdict: count} for `window_open`, removed from the tally. Older
    windows are dropped with it."""
    key = pd.Timestamp(window_open)
    counts = _PEER_COUNTS.pop(key, {})
    for stale in [k for k in _PEER_COUNTS if k < key]:
        del _PEER_COUNTS[stale]
    return counts


def cross_venue_row(pm: Optional[dict], quote, *, now=None, window_open=None,
                    config=None, symbol: Optional[str] = None) -> dict:
    """`cross_venue` from the in-process Polymarket cache.

    Two independent books on the same fifteen minutes, settling on different
    oracles — CF Benchmarks BRTI against Chainlink's BTC-USD TWAP-60s — that
    nonetheless agree on 99.52% of shared windows. So a price gap is information
    or liquidity, not a different question.

    `best_bid`/`best_ask` are CENTS on both sides, matching `core.book_features`.

    The peer must be for THIS window and no staler than the peer cap the
    backtest applies (`peer_max_age_seconds`); otherwise the gap partly
    measures Kalshi's own move since the peer was read. Each verdict is counted
    per window for the heartbeat line (`take_peer_counts`).
    """
    from core.book_features import CROSS_VENUE

    out = {name: float('nan') for name in CROSS_VENUE}
    refusal = peer_refusal(pm, now=now, window_open=window_open, config=config)
    _note_peer(window_open, refusal or PEER_USED)
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
    # Absence is not agreement: a zeroed gap would read as two venues concurring.
    out['pm_available'] = 0.0 if np.isnan(p_mid) else 1.0
    out['venue_prob_gap'] = k_mid - p_mid
    k_spread = _spread_cents(getattr(quote, 'yes_bid', np.nan) * 100.0,
                             getattr(quote, 'yes_ask', np.nan) * 100.0)
    p_spread = _spread_cents(*(
        (pm.get('best_bid'), pm.get('best_ask')) if pm else (np.nan, np.nan)))
    if k_spread > 0 and p_spread > 0:
        out['venue_spread_ratio'] = float(np.log(k_spread / p_spread))
    # `venue_gap_change_5` needs the gap five minutes ago, which the backtest
    # takes from the previous row of the same window. Live carries it per
    # symbol below; absent a prior observation it is honestly unknown.
    return out


def _two_sided_mid(bid, ask) -> float:
    """The mid as a probability, or NaN. A lone bid says the probability is at
    LEAST something, which is not a probability."""
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
    """Drop the memoised store read. For tests, and after a store rebuild."""
    _FIT_CACHE.clear()


def _read_recent_fits():
    """The last few hours of `venue_implied_vol`, or an empty frame."""
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
    """The freshest usable ladder fit for `symbol`, or None.

    The in-process recorder's cache first — it is the newest reading. Then the
    STORE, which is what the backtest reads (`attach_implied_vol`, an as-of
    join tolerating `MAX_FIT_AGE_MINUTES`), so a restart does not blank the
    iv_* features until the recorder next succeeds. The artifact was fitted on
    rows that all carry a fit, so a NaN here is out of distribution.
    """
    from core.book_features import MAX_FIT_AGE_MINUTES

    now = pd.Timestamp(now)
    live = cache.get(symbol) if cache else None
    if live is not None:
        age = (now - pd.Timestamp(live['at'])).total_seconds() / 60.0
        if 0 <= age <= MAX_FIT_AGE_MINUTES:
            return live

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
            'at': row['_at']}


async def prewarm_fit_lookup(*, horizon: float = 75.0) -> None:
    """Refresh the store-backed implied-vol frame in a thread, ahead of need.

    `latest_fit` would otherwise do a synchronous whole-table Parquet read on
    the event loop between the book read and the order whenever its memo
    expired. Refreshing early, off the loop, keeps it from ever expiring there.
    """
    now = pd.Timestamp.now(tz='UTC')
    stamp = _FIT_CACHE.get('at')
    if stamp is not None and (now - stamp).total_seconds() < _FIT_REFRESH_SECONDS - horizon:
        return
    frame = await asyncio.to_thread(_read_recent_fits)
    _FIT_CACHE['frame'] = frame
    _FIT_CACHE['at'] = now


def implied_vol_row(fit: Optional[dict], *, sigma_per_min, now) -> dict:
    """`implied_vol` from the in-process ladder-fit cache.

    The baseline scales a BACKWARD-looking realised vol. The strike ladder
    inverts to a FORWARD-looking one, and where they disagree the baseline's
    `sigma_remaining` is wrong in a knowable direction — the only quantity the
    barrier framing says needs forecasting at all.

    Staleness is a FEATURE, not a filter, up to `MAX_FIT_AGE_MINUTES`: coverage
    is ~15% of the timeline with a five-hour mean gap, so a sigma forward-filled
    from three hours ago is a different claim from a fresh one and the model has
    to be able to tell them apart. Beyond the cap it describes a different
    session and carrying it would be a fabrication.
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
    """The model's book features, from the book the fill will price against.

    Built through `core.book_features.market_state_features`, the same function
    the backtest uses, off the same quote the order is priced against, so the
    two cannot drift. Prices go in as CENTS because that is what the snapshot
    shape uses; a one-sided book yields no probability, exactly as in the
    backtest.
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
        # ONE sampler. `best` is the ladder's own touch, not the quote's — see
        # `_ladder_touch`; mixing them empties the 1c bucket while 5c is full.
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


def _attach_book_features(scored: pd.DataFrame, quotes: dict, cache=None, *,
                          config=None) -> None:
    """Book features onto the scoring rows, in place, from the touch.

    One row per symbol: the quote is the same for every offset in the cycle,
    and it is the book the order will actually be priced against.
    """
    from core.book_features import (
        CROSS_VENUE, IMPLIED_VOL, MARKET_PRICE, MARKET_STATE)

    pm_cache, iv_cache = _venue_caches()
    now = pd.Timestamp.now(tz='UTC')
    columns = (tuple(MARKET_STATE) + tuple(MARKET_PRICE)
               + tuple(CROSS_VENUE) + tuple(IMPLIED_VOL)
               # Not features — `decide()` reads these to cap the stake.
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
            # The ladder the touch was read from, so every depth feature
            # describes the same snapshot the order is priced against.
            yes_levels, no_levels = quote_ladder(entry[0])
            if not yes_levels and not no_levels:
                yes_levels, no_levels = ladder_from_cache(cache, entry[1])
            row = book_feature_row(
                entry[0], yes_levels, no_levels,
                baseline_probability=float(base.iloc[i]) if len(base) > i else np.nan)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('%s: book features unavailable (%s)', symbol, str(exc)[:80])
            continue
        # The stake cap `decide()` reads: dollars at the touch on the side
        # actually crossed.
        up, down = depth_dollars(entry[0], yes_levels, no_levels)
        if up is not None:
            row['depth_up'] = up
        if down is not None:
            row['depth_down'] = down
        row.update(cross_venue_row(
            pm_cache.get(symbol), entry[0], now=now,
            window_open=scored['window_open'].iloc[i], config=config,
            symbol=symbol))
        row.update(implied_vol_row(
            latest_fit(symbol, now=now, cache=iv_cache), now=now,
            sigma_per_min=scored['sigma_per_min'].iloc[i]
            if 'sigma_per_min' in scored.columns else np.nan))
        row['venue_gap_change_5'] = gap_change(
            symbol, row.get('venue_prob_gap'),
            window_open=scored['window_open'].iloc[i],
            offset=int(scored['offset'].iloc[i])
            if 'offset' in scored.columns
            else int(scored['offset_minutes'].iloc[i]))
        for column in columns:
            if column in row:
                scored.iloc[i, scored.columns.get_loc(column)] = row[column]


# Features that are legitimately absent at a known offset, with the reason. A
# warning that fires every window is not a warning.
EXPECTED_NAN = {
    # `gap_change` is a first difference against the previous offset, so it has
    # nothing to difference against at the FIRST offset of a window. That is
    # arithmetic, not a fault.
    'venue_gap_change_5': lambda offsets: min(offsets) if offsets else None,
}


def _warn_unscoreable_features(scored: pd.DataFrame, model,
                               *, offsets=()) -> None:
    """Say loudly when the model wants a feature this row cannot supply.

    A NaN feature does not raise; the booster substitutes the direction it
    learned in training. So the only signal that a live model is not the model
    that was measured is this line.

    It must not cry wolf, so absences in `EXPECTED_NAN` at their expected
    offset are reported at debug instead — still reported, because the day one
    of them stops being expected, silence would be worse than noise.
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
    """Attach whatever init score this artifact was fitted on.

    The baseline logit is already on the table — `score_live` attaches it from the
    fold's own baseline. A market-initialised artifact needs the recorded quote's
    implied probability instead, taken from the de-spread mid rather than the ask:
    the init score is the market's estimate of the probability, not what we would
    pay for it.

    `Quote.mid` is None on a one-sided book, so those rows carry NaN and
    `ForecastModel.predict` returns NaN for them. That is the correct outcome — a
    correction to a price we could not read is not a forecast — and `decide()`
    abstains on it rather than falling back to the baseline under the wrong
    provenance.
    """
    source = getattr(model, 'init_score_source', 'baseline')
    if source != 'market':
        return scored
    from core.model import attach_market_logit

    out = scored.copy()
    out['market_probability'] = out['market_mid']
    return attach_market_logit(out)
