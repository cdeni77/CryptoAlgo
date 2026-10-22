"""Price a window row against the venue's recorded book.

**This closes the circularity the audit named.** `core/decide.py` falls back to
`p_market = baseline_probability` when a row carries no `ask_up`/`ask_down` —
which is every backtest row — so the simulated counterparty is the model's own
training target. Two of the eighteen gates, `market_windows` and
`model_minus_market`, read NaN and fail for want of a price rather than for want
of skill. `AUDIT_REPORT.md` calls that the sharpest complaint against the whole
result, and eight months of collected book is the answer to it.

**Denomination is the thing that kills you quietly.** A Kalshi YES book quotes
what YES costs. The price of the DOWN side is `1 - yes_bid`, because buying NO
is taking the other side of what YES bidders will pay — not `yes_bid`, and not
the mid, which is a price nobody will sell you. Getting it wrong is an error of
the spread with the sign inverted on one side, and nothing raises. It is the
same class of mistake as the `no_levels` trap that put a 0.51 YES ask in the
column holding a 0.51 NO bid.

A useful property falls out and is asserted in the tests: the two costs must sum
to `1 + spread`. Summing to exactly 1 means the mid was used; summing to less
than 1 means the book is crossed and the row is not a book at all.

**A row with no quote keeps NaN on purpose.** Most of five years predates the
venue, and those rows must go on pricing against the calibrated baseline exactly
as before. `price_source` then still reads `baseline`, which is the distinction
the schema has always insisted on: a backtest and a fill are different claims.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# The store's depth columns, renamed to what `core.book_features` expects.
# Declared here so a rename cannot leave FEATURE_GROUPS pointing at a column
# nothing joins — which is exactly how a run reported "46 features (12 empty),
# 2 trees": a quarter of the matrix was all-NaN and only a warning said so.
DEPTH_MAP = {
    'yes_bid_size': 'bid_at_touch', 'yes_ask_size': 'ask_at_touch',
    'depth_bid_1c': 'bid_1c', 'depth_ask_1c': 'ask_1c',
    'depth_bid_5c': 'bid_5c', 'depth_ask_5c': 'ask_5c',
    'depth_bid_total': 'bid_vol', 'depth_ask_total': 'ask_vol',
}
QUOTE_COLUMNS = (('ask_up', 'ask_down', 'market_probability',
                  'quote_age_seconds', 'quote_source', 'quote_observer',
                  # Not features — `decide()` reads these to cap the stake.
                  'depth_up', 'depth_down')
                 + tuple(DEPTH_MAP.values()))
# Which observer to believe when several recorded the same minute.
#
# **`quote_age_seconds` is NOT comparable across sources, so it cannot be the
# ranking key.** It means a different thing in each producer: for `backfill` it
# is time since the last book CHANGE (Predexon serves changes, so a quiet book
# reads ~0.5s), while for the live recorders it is the sampler's own lag, and
# `scripts/record_ladder.py` samples at +25s past the minute. Measured at
# offset 12: backfill 0.52s median, live/live_ws 32.7s, `live_touch` NaN. Rank
# on that and backfill wins essentially every contested row — not because it is
# fresher, but because its clock starts later.
#
# That is the same defect already recorded for `levels_bid`/`levels_ask` across
# sources (ratio 0.579), except used as a SELECTOR rather than a feature, so it
# silently chose a provenance while appearing to choose a freshness.
#
# It matters because the model is fitted to correct `market_probability`
# (`init_score_source=market`) and then trades against whatever `scripts/live.py`
# reads at the decision instant. Those must be the same object. `live_touch` IS
# that object — `live._record_touch` writes the quote the loop actually decided
# on — so it ranks first, ahead of the minute-grid recorders, ahead of the
# after-the-fact reconstruction. The old ordering put it LAST of the four.
# **An unlisted source ranks LAST**, below the Predexon backfill -- the map
# below fills a miss with `len(rank)`. So a new producer that is not named here
# is fetched, stored, and then silently never selected. Both sources added on
# 2026-10-21 hit exactly that: `candles` and `live_ws_replay` would have been
# written and ignored.
#
# `candles` is Kalshi's own per-minute `yes_bid`/`yes_ask`, on the venue's true
# `tapered_deci_cent` grid. It outranks `backfill` because the Predexon rows
# round every tail quote to a whole cent -- measured 2026-10-21, 100% of
# backfill quotes sit on a whole cent in every band against ~10% of live ones
# in the tails, which is a 0.5c error where a 4c quote makes it a 12.5%
# relative error in the probability. It ranks BELOW every live observer,
# because a candle close is a 60-second bar boundary and not a decision-instant
# quote.
#
# `live_ws_replay` is the stream archive re-folded through the same `BookCache`
# the live path uses; it validated at 100.0000% exact whole-ladder agreement
# against the recorded ladder over 8,169 overlapping minutes. It is a
# RECONSTRUCTION, so it sits below the three live observers and above the
# candles, which are a summary.
QUOTE_SOURCE_PRIORITY = tuple(
    s.strip() for s in os.getenv(
        'QUOTE_SOURCE_PRIORITY',
        'live_touch,live_ws,live,live_ws_replay,candles,backfill').split(',')
    if s.strip())

# A book quoted twenty minutes ago is not a price this decision could have
# taken. Measured on the real store, 88.1% of backfilled minute rows are fresher
# than 30s and 70.8% fresher than 5s, so tightening this costs little coverage.
#
# Read from `Config.max_quote_age_seconds`, the one limit `core.metrics` and the
# live order check also read: trades were once priced against quotes up to 900s
# old while `model_minus_market` counted only those under 30s, and stale quotes
# are exactly where fake edge comes from (9x higher at 900s than at 5s).
DEFAULT_MAX_AGE = float(DEFAULT_CONFIG.max_quote_age_seconds)

# The one observer whose `available_time` bounds its quote's age. `live_touch`
# rows are written by the trading cycle itself: the cycle starts at or after
# `event_time` (the decision instant), reads the book, and stamps
# `available_time` when it writes the row afterwards. So the book it priced was
# read inside [event_time, available_time] and is at most that far from the
# decision instant. No other source has that property — the backfill's
# `available_time` is when the backfill RAN, weeks later.
AGE_BOUNDED_SOURCES = ('live_touch',)

# Buckets for the source/age coverage report, in seconds.
AGE_BUCKETS = ((2.0, '<=2s'), (5.0, '2-5s'), (30.0, '5-30s'),
               (float('inf'), '>30s'))


def derived_quote_age(book: pd.DataFrame) -> pd.Series:
    """`available_time - event_time` in seconds on age-bounded rows, else NaN.

    An UPPER bound on how far the priced book can be from the decision instant,
    for rows that recorded no age of their own (every `live_touch` row before
    2026-10-14). It includes the cycle's post-decision latency, so it
    overstates the true age — the conservative direction for a staleness filter.
    """
    out = pd.Series(np.nan, index=book.index, dtype=float)
    if not {'source', 'event_time', 'available_time'} <= set(book.columns):
        return out
    bounded = book['source'].isin(AGE_BOUNDED_SOURCES)
    if not bounded.any():
        return out
    event = pd.to_datetime(book.loc[bounded, 'event_time'], utc=True, errors='coerce')
    available = pd.to_datetime(book.loc[bounded, 'available_time'], utc=True,
                               errors='coerce')
    seconds = (available - event).dt.total_seconds()
    out.loc[bounded] = seconds.clip(lower=0.0).to_numpy(dtype=float)
    return out


def filter_sources(book: pd.DataFrame, sources: Sequence[str]) -> pd.DataFrame:
    if not sources:
        return book
    if 'source' not in book.columns:
        logger.warning('quote_sources=%s asked for, but the depth table records '
                       'no observer; nothing can be shown to qualify',
                       list(sources))
        return book.iloc[0:0]
    return book[book['source'].isin(tuple(sources))]


# How each producer's `quote_age_seconds` is measured. The backfill counts back
# from the row's minute. `live_touch` counts back from the decision instant
# itself. The minute-grid recorders count FORWARD: they sample after the minute,
# so their age is how long after it the book was seen. Read with one rule, a
# forward row is a book from after the decision.
MINUTE_BACKWARD_SOURCES = ('backfill',)


def decision_age(book: pd.DataFrame, null_age: str,
                 latency_seconds: float) -> pd.Series:
    """Seconds from when each row's book was seen to the decision instant.

    The decision instant is the row's minute plus `latency_seconds`. A negative
    value is a book seen after the decision, which a decision cannot use.
    """
    raw = _effective_age(book, null_age)
    source = (book['source'] if 'source' in book.columns
              else pd.Series(None, index=book.index, dtype=object))
    from_minute = source.isin(MINUTE_BACKWARD_SOURCES)
    from_decision = source.isin(AGE_BOUNDED_SOURCES)
    forward = ~(from_minute | from_decision)
    age = raw.copy()
    age[from_minute] = raw[from_minute] + latency_seconds
    age[forward] = latency_seconds - raw[forward]
    return age


def _effective_age(book: pd.DataFrame, null_age: str) -> pd.Series:
    """The age a row is judged on. Under 'stale', a missing age is first bounded
    where the row's own timestamps allow, and stays NaN (dropped) otherwise."""
    age = (pd.to_numeric(book['quote_age_seconds'], errors='coerce')
           if 'quote_age_seconds' in book.columns
           else pd.Series(np.nan, index=book.index, dtype=float))
    if null_age == 'stale':
        bound = derived_quote_age(book)
        age = age.where(age.notna(), bound)
    return age


def attach_quotes(windows: pd.DataFrame, depth: pd.DataFrame, *,
                  venue: str = 'kalshi',
                  other_venue: Optional[str] = 'polymarket',
                  max_age_seconds: float = DEFAULT_MAX_AGE,
                  sources: Sequence[str] = (),
                  null_age: str = 'stale',
                  peer_max_age_seconds: Optional[float] = None,
                  peer_null_age: str = 'fresh',
                  peer_sources: Optional[Sequence[str]] = None,
                  decision_latency_seconds: float = 3.0) -> pd.DataFrame:
    """`ask_up` / `ask_down` on each window row, from `venue_depth`.

    Joined on (symbol, window_open, offset) EXACTLY. Not as-of across offsets:
    a decision at +3m priced with the book at +12m would be nine minutes of a
    fifteen-minute window ahead of itself, which is most of the question and
    reads exactly like skill. The at-or-before rule already lives one layer
    down, in how `venue_depth`'s minute rows are built.

    Only `venue` is used. Polymarket is a cross-venue FEATURE, never the book we
    execute against — pricing a Kalshi trade off a Polymarket quote would book a
    fill at a price the venue never showed.

    `sources` (`Config.quote_sources`), when non-empty, admits only those
    observers on BOTH venues — an excluded row does not price, rank or supply
    depth. `null_age='stale'` (`Config.null_quote_age`) drops a quote with no
    age instead of treating it as fresh, after bounding `live_touch` ages from
    their own timestamps (`derived_quote_age`).

    The PEER has its own cap, `peer_max_age_seconds` (None follows
    `max_age_seconds`), its own source list, `peer_sources` (None follows
    `sources`; an empty list frees it entirely), and its own null-age rule,
    `peer_null_age`, which stays 'fresh' unless the caller asks: only ~5% of live Polymarket rows are 5s
    fresh, so a tight Kalshi cap applied to the peer voids `venue_prob_gap`
    almost everywhere.

    Both venues are point-in-time: a row is usable only if its book was seen at
    or before the decision instant, the minute plus `decision_latency_seconds`
    (see `decision_age`). `quote_age_seconds` on the output is that
    decision-relative age.
    """
    out = windows.copy()
    for column in QUOTE_COLUMNS:
        out[column] = np.nan
    out['quote_source'] = None
    out['quote_observer'] = None
    if depth is None or not len(depth) or not len(out):
        return out

    book = filter_sources(depth[depth['venue'] == venue], sources).copy()
    if not len(book):
        return out

    bid = pd.to_numeric(book['yes_bid'], errors='coerce')
    ask = pd.to_numeric(book['yes_ask'], errors='coerce')
    age = decision_age(book, null_age, decision_latency_seconds)

    # A price outside [0, 1] is not a probability, and an ask below the bid is
    # not a book — kept, it would show a guaranteed profit and the backtest
    # would take it every time.
    sane = ((bid.isna() | ((bid >= 0.0) & (bid <= 1.0)))
            & (ask.isna() | ((ask >= 0.0) & (ask <= 1.0)))
            & (bid.isna() | ask.isna() | (ask >= bid)))
    fresh = (age >= 0.0) & (age <= max_age_seconds)
    if null_age == 'fresh':
        fresh = fresh | age.isna()
    keep = sane & fresh

    book['ask_up'] = ask.where(keep)
    book['ask_down'] = (1.0 - bid).where(keep)
    # The market's FORECAST, which is the mid — not the ask, which is what a
    # trade costs. `model_minus_market` compares log losses, so scoring the
    # market at its ask would credit the model with half the spread as skill on
    # every row, in its own favour.
    book['market_probability'] = ((bid + ask) / 2.0).where(
        keep & bid.notna() & ask.notna())
    # The stake cap `decide()` reads, in DOLLARS on the side actually crossed.
    # `scripts/live.py` supplies these from the live ladder; without them here
    # the backtest would size against `max_stake_dollars` alone while live sized
    # against the book, and "one decide()" is the invariant that stops the two
    # from drifting. NaN, never zero, when the venue reported no size: zero
    # refuses every trade where the honest answer is "unmeasured".
    # The RAW column names: `bid_at_touch`/`ask_at_touch` are the DEPTH_MAP
    # renames and that loop has not run yet at this point in the function.
    def _size(name):
        return (pd.to_numeric(book[name], errors='coerce')
                if name in book.columns
                else pd.Series(np.nan, index=book.index))

    bid_size, ask_size = _size('yes_bid_size'), _size('yes_ask_size')
    book['depth_up'] = (ask_size * book['ask_up']).where(keep)
    book['depth_down'] = (bid_size * book['ask_down']).where(keep)
    book['quote_age_seconds'] = age.where(keep)
    book['quote_source'] = np.where(keep, venue, None)
    # WHICH observer priced the row, as opposed to which venue. `quote_source`
    # has always held the venue despite its name, so nothing downstream could
    # tell a reconstruction from an observation — which is precisely why the
    # ranking defect above survived: the mixture was invisible in every frame
    # that carried it. `research/validate` and the market gates can now split
    # on it, and a shift in the mix is a thing that can be seen rather than
    # inferred from a disagreeing number.
    book['quote_observer'] = (np.where(keep, book['source'], None)
                              if 'source' in book.columns else None)

    # One row per (symbol, window, offset). `venue_depth` can hold the same
    # minute from more than one observer — that is what `source` is for — so
    # collapse before joining rather than fanning the window table out.
    #
    # Ranked on three keys, in this order:
    #   1. `_unusable` — a row that failed `sane`/`fresh` must never displace a
    #      row that passed. The previous version got this for free by NaN-ing
    #      such a row's age and sorting nulls last; now that age is no longer
    #      the leading key, it has to be said out loud.
    #   2. `_source_rank` — see QUOTE_SOURCE_PRIORITY. This is the fix.
    #   3. age bucketed to WHOLE SECONDS — a sub-second difference between two
    #      rows of the same source is noise, not information.
    #   4. whether the row carries SIZES. `live_touch` writes two rows per
    #      decision, one with depth and one without, and the unsized one is
    #      fresher by a median 140ms. Ranking on raw age therefore handed the
    #      unsized row the win in 8,013 of 8,013 groups, leaving only 53.2% of
    #      live-era decision rows with `yes_bid_size`, `depth_*` and
    #      `bid_at_touch` -- `market_state`'s complete-case column among them.
    #      Half the live era lost its depth for a seventh of a second.
    #   5. exact age, as the final tie-break.
    #
    # Bucketing before the size preference is what keeps this honest: inside
    # one second, take the row that says more; across seconds, the fresher row
    # still wins, because four seconds of staleness is real.
    book['_unusable'] = (~keep).astype(int)
    if 'source' in book.columns:
        rank = {name: i for i, name in enumerate(QUOTE_SOURCE_PRIORITY)}
        book['_source_rank'] = (book['source'].map(rank)
                                .fillna(len(rank)).astype(int))
    else:
        book['_source_rank'] = 0
    _age = pd.to_numeric(book.get('quote_age_seconds'), errors='coerce')
    book['_age_bucket'] = np.floor(_age.fillna(np.inf))
    _sizes = [c for c in ('yes_bid_size', 'yes_ask_size') if c in book.columns]
    book['_no_sizes'] = (
        (~book[_sizes].notna().any(axis=1)).astype(int) if _sizes
        else 0)
    book = (book.sort_values(['_unusable', '_source_rank', '_age_bucket',
                              '_no_sizes', 'quote_age_seconds'],
                             na_position='last')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='first')
                .drop(columns=['_unusable', '_source_rank', '_age_bucket',
                               '_no_sizes']))

    for src, dst in DEPTH_MAP.items():
        book[dst] = pd.to_numeric(book.get(src), errors='coerce').where(keep) \
            if src in book.columns else np.nan

    take = ['symbol', 'window_open', 'offset_minutes', 'ask_up', 'ask_down',
            'market_probability', 'quote_age_seconds', 'quote_source',
            'quote_observer',
            # The stake cap. Listed here and not derived from QUOTE_COLUMNS
            # because that tuple is also the drop list above; keeping them in
            # step is the reason this line exists rather than a comprehension.
            'depth_up', 'depth_down']
    take += list(DEPTH_MAP.values())
    merged = out.drop(columns=list(QUOTE_COLUMNS)).merge(
        book[take],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'],
        how='left')
    merged = merged.drop(columns=['offset_minutes'])

    # The other venue, under its own prefix. `cross_venue` needs both books on
    # one row and they must never share a column: a Polymarket bid sitting in a
    # column holding a Kalshi one is the `no_levels` denomination trap wearing a
    # different name.
    if other_venue:
        # **The peer gets its OWN source list, and this is not a detail.**
        # `sources` exists to exclude a producer whose KALSHI prices are not
        # executable -- Predexon, which quantises a deci-cent grid to whole
        # cents. Polymarket's backfill is a different pipeline (the gamma API)
        # and has nothing to do with that defect, but `filter_sources` matches
        # on the bare name, so naming Kalshi's live observers discarded
        # Polymarket's `backfill` too: 77,759 windows, and with
        # `--complete-cases` every row needing a `cross_venue` feature went
        # with them. Measured 2026-10-21, 6,465 candidate windows became 956
        # and the walk-forward could not form a single fold.
        #
        # `None` follows `sources`, which is what every caller got before and
        # keeps the old behaviour exactly; an explicit list -- including an
        # empty one, meaning no restriction -- frees the peer. Same shape as
        # `peer_max_age_seconds` and `peer_null_age` for the same reason: the
        # peer is a cross-venue FEATURE, never the book we execute against.
        peer = filter_sources(depth[depth['venue'] == other_venue],
                              sources if peer_sources is None
                              else peer_sources).copy()
        merged['pm_market_probability'] = np.nan
        merged['pm_spread'] = np.nan
        if len(peer):
            pbid = pd.to_numeric(peer['yes_bid'], errors='coerce')
            pask = pd.to_numeric(peer['yes_ask'], errors='coerce')
            # **The SAME staleness bar as the traded side.** This block had none,
            # and Polymarket rows reach 899 seconds in the store. `venue_prob_gap`
            # is a DIFFERENCE of the two venues, and a difference is exactly where
            # a timing asymmetry turns into apparent skill.
            #
            # It matters here more than anywhere: `cross_venue` is the only
            # load-bearing group — dropping it takes skill +0.00282 -> -0.00015 —
            # while scoring +0.000030 ALONE, below the clock control. Useless
            # alone and essential combined is what an interaction looks like AND
            # what a leak looks like, and the pattern cannot separate them.
            # Filtering both sides to one tolerance is what makes the group's
            # contribution interpretable either way.
            #
            # Measured on the 431,198 rows the model sees, Polymarket is 4.0s
            # STALER on average, so the leak is not the mean case — but it is
            # fresher by >10s on 6% of rows and nothing bounded it.
            #
            # A missing stamp is kept: absence is not evidence of staleness, and
            # the backfilled fixtures carry rows without one.
            peer_cap = (max_age_seconds if peer_max_age_seconds is None
                        else float(peer_max_age_seconds))
            page = decision_age(peer, peer_null_age, decision_latency_seconds)
            peer['quote_age_seconds'] = page
            pfresh = (page >= 0.0) & (page <= peer_cap)
            if peer_null_age == 'fresh':
                pfresh = pfresh | page.isna()
            two_sided = pbid.notna() & pask.notna() & (pask >= pbid) & pfresh
            peer['pm_market_probability'] = ((pbid + pask) / 2.0).where(two_sided)
            peer['pm_spread'] = (pask - pbid).where(two_sided)
            # Ranked, not taken in store order. `pm_ladder` carries a
            # `transport` key, so duplicates per (symbol, window, offset) are
            # real -- and an arbitrary winner that fails `two_sided` sends
            # `venue_prob_gap` to NaN while a usable duplicate sits beside it.
            # That changes `pm_available` and, under `--complete-cases`, which
            # ROWS survive at all. The traded side has been ranked since the
            # source-priority fix; this is the same rule for the other half of
            # the same feature.
            peer['_unusable'] = (~two_sided).astype(int)
            peer = (peer.sort_values(['_unusable', 'quote_age_seconds'],
                                     na_position='last')
                        .drop_duplicates(['symbol', 'window_open',
                                          'offset_minutes'], keep='first')
                        .drop(columns=['_unusable']))
            merged = merged.drop(columns=['pm_market_probability', 'pm_spread']).merge(
                peer[['symbol', 'window_open', 'offset_minutes',
                      'pm_market_probability', 'pm_spread']],
                left_on=['symbol', 'window_open', 'offset'],
                right_on=['symbol', 'window_open', 'offset_minutes'],
                how='left').drop(columns=['offset_minutes'])
    # `how='left'` preserves order and count, but the table is fed to decide()
    # row by row and losing or reordering rows silently changes which windows
    # were traded. Cheap to assert, expensive to discover.
    if len(merged) != len(out):
        raise ValueError(f'quote join changed the row count: '
                         f'{len(out)} -> {len(merged)}')
    return merged


def age_bucket(age: pd.Series) -> pd.Series:
    """Label each age with its AGE_BUCKETS bucket; a missing age is 'null'."""
    age = pd.to_numeric(age, errors='coerce').abs()
    out = pd.Series('null', index=age.index, dtype=object)
    lower = -np.inf
    for upper, name in AGE_BUCKETS:
        out[(age > lower) & (age <= upper)] = name
        lower = upper
    return out


def source_age_table(frame: pd.DataFrame, *, source_column: str = 'source',
                     age_column: str = 'quote_age_seconds') -> pd.DataFrame:
    """Row counts by observer x age bucket, one row per observer.

    Run on the raw depth table it shows what a source/age filter is about to
    discard; on the joined window table, what actually priced the rows.
    """
    if frame is None or not len(frame):
        return pd.DataFrame()
    source = (frame[source_column].fillna('none').astype(str)
              if source_column in frame.columns
              else pd.Series('none', index=frame.index))
    age = (frame[age_column] if age_column in frame.columns
           else pd.Series(np.nan, index=frame.index))
    table = pd.crosstab(source.rename('source'), age_bucket(age).rename('age'))
    order = [name for _, name in AGE_BUCKETS] + ['null']
    table = table.reindex(columns=[c for c in order if c in table.columns],
                          fill_value=0)
    table['total'] = table.sum(axis=1)
    return table.reset_index()


def quote_coverage(table: pd.DataFrame) -> dict:
    """How many rows a real price actually reached.

    `market_windows` gates on exactly this, so it is worth reporting rather than
    inferring from a gate that failed.
    """
    if not len(table) or 'ask_up' not in table.columns:
        return {'rows': 0, 'quoted': 0, 'share': 0.0, 'by_source_age': []}
    quoted = table['ask_up'].notna()
    # Which observer priced the quoted rows, and how old its quote was: the
    # mix is the thing a source or age filter changes, and a headline share
    # cannot show it.
    by_source = source_age_table(table.loc[quoted], source_column='quote_observer')
    return {'rows': int(len(table)), 'quoted': int(quoted.sum()),
            'share': int(quoted.sum()) / max(len(table), 1),
            'by_source_age': by_source.to_dict('records')}
