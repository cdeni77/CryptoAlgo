"""Shared evaluation harness. Every experiment uses these functions unchanged.

The rules in PREREGISTRATION.md are implemented here once so that no candidate can
be scored under different rules from another. In particular:

* the market benchmark is the STREAM LADDER only (`live`, `live_ws`), joined
  explicitly from venue_depth rather than taken from the panel's attached quote --
  the panel's quote selector ranks observers and can pick a backfill row, which D5
  showed is a price from the wrong minute;
* the label is the VENUE's settlement, never the Coinbase proxy;
* folds split on the window with a one-day embargo;
* significance is a paired bootstrap over WINDOWS, never rows.
"""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence
import logging
import duckdb, numpy as np, pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
STORE = ROOT.parents[1] / 'data' / 'research'
LEDGER = ROOT / 'experiments'

# ---------------------------------------------------------------- universe --
# Widened after D7. The earlier 23-day restriction came from a diagnostic that was
# itself wrong: `live@m` is sampled at m+32s and is 32 seconds of LOOKAHEAD, while
# the Predexon backfill is stamped and available at m exactly. Against a genuinely
# contemporaneous price the backfill is statistically indistinguishable
# (-0.000040, CI [-0.00286, +0.00276]), so it is a legitimate point-in-time price
# and the usable span is the whole backfill era.
BENCH_START = pd.Timestamp('2026-07-01', tz='UTC')
BENCH_END   = pd.Timestamp('2026-10-15', tz='UTC')
DEV_END     = pd.Timestamp('2026-09-16', tz='UTC')   # dev < DEV_END <= validation
DECISION_LATENCY_S = 3.0

# **The second freshness filter, and it is not optional.** `available_time` answers
# "could we have known this row"; it does NOT answer "is this price real". Predexon
# serves book CHANGES, so a gap in its feed is carried forward and still stamped as
# available. D9 measured what a large `quote_age_seconds` actually means: the LIVE
# Kalshi book kept moving 8.4-9.2pp per minute regardless of the backfill's claimed
# book age, and 72-77% of rows showed it moving >2pp/min even in the >120s band.
# So a stale backfill quote is a FEED HOLE, not a resting market.
#
# D8 measured the damage: the barrier baseline alone "beat" the market by +0.0080
# nats overall, but on rows with book age <= 5s it LOSES by -0.0062, and the
# apparent edge rises monotonically to +0.173 above 120s. The whole result was the
# artifact CLAUDE.md already documented once.
MAX_BOOK_AGE_S = 5.0

STREAM_SOURCES = ('live', 'live_ws')


def market_prices(*, latency_s: float = DECISION_LATENCY_S,
                  start: pd.Timestamp = BENCH_START,
                  end: pd.Timestamp = BENCH_END,
                  max_book_age_s: float = MAX_BOOK_AGE_S) -> pd.DataFrame:
    """The point-in-time market price at each decision instant.

    **The rule is availability, not the stamp.** For a decision at
    `window_open + m + latency`, the usable quote is the most recent Kalshi row whose
    `available_time` is at or before that instant. Measured availability lags:
    backfill +0s, live/live_ws +32s, live_touch +3s -- so a `live` row labelled
    offset m was not knowable at m, and joining it (as the first version of this
    harness did) is 32 seconds of lookahead in a 15-minute market.

    Implemented as an ASOF join so the selection is the same rule for every source
    and every offset, rather than a per-source special case.
    """
    con = duckdb.connect()
    con.execute(f"""create or replace view q as
    with x as (select *, row_number() over (
        partition by venue,symbol,event_time,source,offset_minutes
        order by available_time desc) rn
      from read_parquet('{STORE}/venue_depth/**/*.parquet', union_by_name=true))
    select symbol, window_open, offset_minutes, event_time, available_time, source,
           yes_bid, yes_ask, quote_age_seconds as book_age_s,
           depth_ask_total, depth_bid_total, yes_ask_size, yes_bid_size
    from x
    where rn=1 and venue='kalshi'
      and yes_ask is not null and yes_bid is not null and yes_ask > yes_bid
      and window_open >= TIMESTAMPTZ '{start}' and window_open < TIMESTAMPTZ '{end}'""")
    # the decision grid: every (symbol, window, offset) we might score
    con.execute(f"""create or replace view grid as
    select distinct symbol, window_open, offset_minutes,
           window_open + to_minutes(offset_minutes)
             + to_seconds({int(latency_s)}) as decision_instant
    from q where offset_minutes between 1 and 14""")
    df = con.execute("""
    select g.symbol, g.window_open, g.offset_minutes, g.decision_instant,
           p.event_time as quote_event_time, p.available_time as quote_available_time,
           p.source as quote_source,
           p.yes_bid, p.yes_ask, p.book_age_s,
           p.depth_ask_total as ask_vol, p.depth_bid_total as bid_vol,
           p.yes_ask_size as ask_at_touch, p.yes_bid_size as bid_at_touch
    from grid g asof join q p
      on g.symbol = p.symbol and g.window_open = p.window_open
     and p.available_time <= g.decision_instant
    """).df()
    df['window_open'] = pd.to_datetime(df.window_open, utc=True)
    df['p_mkt'] = (df.yes_bid + df.yes_ask) / 2.0
    df['spread'] = df.yes_ask - df.yes_bid
    # how old the chosen quote was at the decision instant -- an honest column,
    # unlike each source's own `quote_age_seconds`, which means a different thing
    # in each producer.
    df['quote_age_s'] = (pd.to_datetime(df.decision_instant, utc=True)
                         - pd.to_datetime(df.quote_available_time, utc=True)
                         ).dt.total_seconds()
    # Flagged, never silently dropped. `usable` is what an experiment filters on and
    # the counts of both are reported.
    df['book_age_s'] = pd.to_numeric(df.book_age_s, errors='coerce')
    # **Applied to BACKFILL ONLY.** D10: `quote_age_seconds` measures a Predexon
    # sampling gap for `backfill` but the sampler's own ~32s polling lag for the
    # live recorders, where the as-of join on `available_time` already handles it.
    # One flat cut across both deleted 98.9% of live rows for a reason unrelated to
    # staleness.
    is_backfill = df.quote_source.eq('backfill')
    df['usable'] = (~is_backfill) | (df.book_age_s.notna()
                                     & (df.book_age_s <= float(max_book_age_s)))
    n_tot, n_ok = len(df), int(df.usable.sum())
    logger.info('market_prices: %s rows, %s usable at book_age <= %.0fs (%.1f%%)',
                f'{n_tot:,}', f'{n_ok:,}', max_book_age_s, n_ok / max(n_tot, 1) * 100)
    return df


def stream_prices(**kw) -> pd.DataFrame:
    """Retained name, corrected behaviour. See `market_prices`."""
    return market_prices(**kw)


def venue_labels() -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(f"""
    with s as (select *, row_number() over (
        partition by venue,symbol,event_time order by available_time desc) rn
      from read_parquet('{STORE}/venue_settlements/**/*.parquet', union_by_name=true))
    select symbol, window_open, settled_up from s
    where rn=1 and venue='kalshi' and settled_up is not null""").df()
    df['window_open'] = pd.to_datetime(df.window_open, utc=True)
    return df.drop_duplicates(['symbol','window_open'])


def benchmark_frame(panel_path: Path, offsets: Sequence[int] = (1,2,3,4,5,6,7,8,9,10,11,12,13,14)) -> pd.DataFrame:
    """Panel features x stream price x venue label, over the benchmark era."""
    panel = pd.read_parquet(panel_path)
    # core/windows.py names it `offset`; venue_depth names it `offset_minutes`.
    # Normalised here, once, so no experiment joins on the wrong key and silently
    # gets an empty frame.
    if 'offset' in panel.columns and 'offset_minutes' not in panel.columns:
        panel = panel.rename(columns={'offset': 'offset_minutes'})
    panel['window_open'] = pd.to_datetime(panel.window_open, utc=True)
    panel = panel[(panel.window_open >= BENCH_START) & (panel.window_open < BENCH_END)]
    px = stream_prices(); lab = venue_labels()
    # the panel's own attached quote columns are DROPPED: D5 showed the selector
    # can return a backfill price stamped at the wrong minute.
    drop = [c for c in ('market_probability','ask_up','ask_down','quote_age_seconds',
                        'quote_source','quote_observer','depth_up','depth_down',
                        'bid_at_touch','ask_at_touch','bid_vol','ask_vol',
                        'bid_1c','ask_1c','bid_5c','ask_5c') if c in panel.columns]
    panel = panel.drop(columns=drop)
    df = (panel.merge(px,  on=['symbol','window_open','offset_minutes'], how='inner')
               .merge(lab, on=['symbol','window_open'],                  how='inner'))
    df = df[df.offset_minutes.isin(list(offsets))].copy()
    if 'usable' in df.columns:
        df = df[df.usable].copy()
    df['y'] = df.settled_up.astype(int)
    df['split'] = np.where(df.window_open < DEV_END, 'dev', 'val')
    return df.sort_values(['window_open','symbol','offset_minutes']).reset_index(drop=True)

# ------------------------------------------------------------- metrics ------

def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    y = np.asarray(y, float)
    return -(y*np.log(p) + (1-y)*np.log(1-p))

def logloss(p, y) -> float:
    return float(np.mean(_ll(p, y)))

def window_bootstrap(delta: np.ndarray, windows: np.ndarray, B: int = 10000, seed: int = 0):
    """Paired bootstrap over WINDOWS. `delta` is per-row; it is averaged within a
    window first, because four offsets of one window share a settlement."""
    g = pd.DataFrame({'d': delta, 'w': windows}).groupby('w')['d'].mean()
    v = g.to_numpy(); n = len(v)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, n, size=(B, n))].mean(axis=1)
    p_one_sided = float((bs <= 0).mean())          # P(effect <= 0) under resampling
    return float(v.mean()), float(np.quantile(bs,.025)), float(np.quantile(bs,.975)), p_one_sided, n

def model_minus_market(p_model, p_mkt, y, windows):
    """Positive == the model beat the price. The campaign's primary metric."""
    d = _ll(p_mkt, y) - _ll(p_model, y)
    return window_bootstrap(d, windows)

def calibration(p, y, bins: int = 10) -> dict:
    df = pd.DataFrame({'p': np.asarray(p,float), 'y': np.asarray(y,float)})
    try: df['b'] = pd.qcut(df.p, bins, duplicates='drop')
    except ValueError: return {'error': float('nan'), 'max_dev': float('nan'), 'table': []}
    g = df.groupby('b', observed=True).agg(n=('y','size'), p=('p','mean'), y=('y','mean'))
    g['dev'] = (g.p - g.y).abs()
    return {'error': float((g.dev*g.n).sum()/g.n.sum()),
            'max_dev': float(g.dev[g.n >= 50].max()) if (g.n>=50).any() else float('nan'),
            'table': [{'n':int(r.n),'p_mean':float(r.p),'y_mean':float(r.y)} for r in g.itertuples()]}

# ----------------------------------------------------------------- folds ----

def window_folds(windows: pd.DatetimeIndex, n_folds: int = 5, embargo_minutes: int = 1440):
    """Expanding, purged, split on the window. Imported from core so the campaign
    and the repository cannot disagree about what a fold is."""
    import sys; sys.path.insert(0, str(ROOT.parents[1]))
    from core.cv import purged_walk_forward
    return purged_walk_forward(pd.DatetimeIndex(sorted(set(windows))),
                               n_folds=n_folds, embargo_minutes=embargo_minutes,
                               min_train_windows=200)

# --------------------------------------------------------------- money ------

def simulate(df: pd.DataFrame, p_col: str, *, min_edge_pp: float, offsets: Sequence[int],
             stake_contracts: int = 1, fee_rate: float = 0.07,
             require_depth: bool = True) -> dict:
    """One entry per (symbol, window). Walk offsets in order, take the FIRST that
    clears the gate -- at offset 3 you cannot know what offset 12 looks like, so
    best-of-offsets is not a strategy that can be run.

    Buys at the live ASK (YES) or 1-ask_down equivalent (NO via the bid side).
    Fee is the venue schedule. Settlement is free. Held to expiry.
    """
    d = df[df.offset_minutes.isin(list(offsets))].sort_values(
        ['symbol','window_open','offset_minutes'])
    trades = []
    for (sym, w), g in d.groupby(['symbol','window_open'], sort=False):
        for r in g.itertuples():
            p = getattr(r, p_col)
            if not np.isfinite(p): continue
            # YES: buy at the ask. NO: buy at (1 - bid), which is the NO ask.
            yes_px, no_px = r.yes_ask, 1.0 - r.yes_bid
            ev_yes = p - yes_px
            ev_no  = (1.0 - p) - no_px
            side, px, ev = ('yes', yes_px, ev_yes) if ev_yes >= ev_no else ('no', no_px, ev_no)
            if not (0.0 < px < 1.0): continue
            fee = np.ceil(fee_rate * stake_contracts * px * (1-px) * 10_000)/10_000
            net_ev = ev * stake_contracts - fee
            if net_ev < min_edge_pp/100.0 * stake_contracts: continue
            if require_depth:
                vol = r.ask_at_touch if side == 'yes' else r.bid_at_touch
                if not np.isfinite(vol) or vol < stake_contracts: continue
            won = (r.y == 1) if side == 'yes' else (r.y == 0)
            pnl = (stake_contracts if won else 0.0) - px*stake_contracts - fee
            trades.append({'symbol': sym, 'window_open': w, 'offset': r.offset_minutes,
                           'side': side, 'price': px, 'fee': fee, 'p_model': p,
                           'p_mkt': r.p_mkt, 'net_ev': net_ev, 'won': won, 'pnl': pnl,
                           'contracts': stake_contracts,
                           'split': r.split})
            break
    t = pd.DataFrame(trades)
    if not len(t):
        return {'trades': 0, 'pnl': 0.0, 'pp_per_contract': float('nan'),
                'eligible_windows': int(d.groupby(['symbol','window_open']).ngroups)}
    gw = t.groupby('window_open')['pnl'].sum()
    rng = np.random.default_rng(0); v = gw.to_numpy(); n = len(v)
    bs = v[rng.integers(0,n,size=(10000,n))].sum(axis=1)
    return {
        'trades': int(len(t)), 'contracts': int(t.contracts.sum()),
        'pnl': float(t.pnl.sum()),
        'pp_per_contract': float(t.pnl.sum()/t.contracts.sum()*100),
        'win_rate': float(t.won.mean()),
        'mean_price': float(t.price.mean()),
        'fees': float(t.fee.sum()),
        'independent_windows': int(n),
        'pnl_ci95': [float(np.quantile(bs,.025)), float(np.quantile(bs,.975))],
        't_stat': float(gw.mean()/(gw.std(ddof=1)/np.sqrt(n))) if n > 1 else float('nan'),
        'eligible_windows': int(d.groupby(['symbol','window_open']).ngroups),
        'by_offset': t.groupby('offset')['pnl'].agg(['size','sum']).to_dict('index'),
        'by_symbol': t.groupby('symbol')['pnl'].agg(['size','sum']).to_dict('index'),
        'max_drawdown': float((gw.cumsum().cummax() - gw.cumsum()).max()),
        '_trades': t,
    }

# --------------------------------------------------------------- ledger -----

def record(exp_id: str, payload: dict) -> Path:
    LEDGER.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload['exp_id'] = exp_id
    payload['recorded_at'] = pd.Timestamp.utcnow().isoformat()
    path = LEDGER / f'{exp_id}.json'
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
