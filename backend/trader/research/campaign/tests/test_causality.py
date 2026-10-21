"""Automated temporal-causality tests. Training is blocked until these pass.

The decisive test is a REPLAY: rebuild a window's features from bars truncated at
the decision instant and compare to the panel's values. If a feature uses a bar it
could not have seen, the truncated rebuild differs. This is stronger than inspecting
the feature code, because it tests the code that actually ran.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, pytest
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

PANEL = HERE.parent / 'datasets' / 'panel_bench.parquet'
needs_panel = pytest.mark.skipif(not PANEL.exists(), reason='panel not built yet')


@pytest.fixture(scope='module')
def panel():
    return pd.read_parquet(PANEL)


@needs_panel
def test_decision_instant_is_inside_its_window(panel):
    off = panel['offset'] if 'offset' in panel.columns else panel['offset_minutes']
    assert (off >= 1).all()
    assert (off <= 14).all()


@needs_panel
def test_features_vary_with_the_offset(panel):
    """A feature identical at offset 1 and offset 14 for nearly every window is
    either a window-level constant (legitimate) or computed from the WHOLE window
    (lookahead). Flags candidates for inspection."""
    from core.features import FEATURE_GROUPS
    cols = [c for g in ('geometry', 'vol_state', 'microstructure') for c in FEATURE_GROUPS[g]
            if c in panel.columns]
    sample = panel[panel.symbol == 'BTC-USD'].head(40000)
    suspicious = [c for c in cols
                  if len(sample) and (sample.groupby('window_open')[c].nunique(dropna=True) <= 1).mean() > 0.95]
    assert not suspicious, (
        'these never vary with the offset, so they may read the whole window: '
        f'{suspicious}')


def test_market_price_is_not_from_the_future():
    """The stream quote joined at offset m must describe minute m. A barrier price
    must become MORE decided as the window resolves; if the join were shifted
    forward the early offsets would look too confident."""
    import harness as H
    px = H.stream_prices()
    px = px[px.offset_minutes.between(1, 14)]
    spread = px.groupby('offset_minutes').p_mkt.apply(lambda s: (s - 0.5).abs().mean())
    early, late = spread.loc[1:4].mean(), spread.loc[11:14].mean()
    assert late > early, (
        f'the market does not become more decided as the window runs out '
        f'(|p-0.5| early {early:.4f}, late {late:.4f}) -- the offset join is suspect')


@needs_panel
def test_label_is_the_venue_not_the_proxy(panel):
    assert 'venue_settled_up' in panel.columns
    assert panel.venue_settled_up.notna().mean() > 0.5


def test_replay_features_match_truncated_rebuild():
    """THE test. Rebuild windows from bars truncated at each decision instant and
    compare against the full-history panel. Any difference is lookahead."""
    from core.config import Config
    from core.datastore import ResearchStore
    from core.dataset import Dataset, load_minute_bars
    from core.features import FEATURE_GROUPS

    cfg = Config(decision_offsets=(3, 9), entry_offsets=(3, 9))
    store = ResearchStore()
    bars = load_minute_bars(cfg, store=store,
                            start=pd.Timestamp('2026-09-01', tz='UTC'),
                            end=pd.Timestamp('2026-09-04', tz='UTC'))
    full = Dataset.build(bars, cfg).windows
    full = full[(full.window_open >= pd.Timestamp('2026-09-03', tz='UTC')) &
                (full.window_open < pd.Timestamp('2026-09-03 04:00', tz='UTC'))]
    assert len(full), 'no windows to replay'

    cols = [c for g in ('geometry', 'vol_state', 'microstructure', 'cross_asset')
            for c in FEATURE_GROUPS[g] if c in full.columns]
    bad: dict[str, list] = {}
    for w in sorted(full.window_open.unique())[:4]:
        for off in (3, 9):
            cut = pd.Timestamp(w) + pd.Timedelta(minutes=off)
            trunc = {s: b.loc[pd.to_datetime(b.event_time, utc=True) < cut].reset_index(drop=True)
                     for s, b in bars.items()}
            if min(len(b) for b in trunc.values()) < 1500:
                continue
            rebuilt = Dataset.build(trunc, cfg).windows
            r = rebuilt[(rebuilt.window_open == w) & (rebuilt.offset == off)]
            f = full[(full.window_open == w) & (full.offset == off)]
            if not len(r) or not len(f):
                continue
            for sym in sorted(set(r.symbol) & set(f.symbol)):
                rr = r[r.symbol == sym].iloc[0]
                ff = f[f.symbol == sym].iloc[0]
                for c in cols:
                    a, b_ = rr.get(c, np.nan), ff.get(c, np.nan)
                    if pd.isna(a) and pd.isna(b_):
                        continue
                    if pd.isna(a) != pd.isna(b_) or not np.isclose(
                            float(a), float(b_), rtol=1e-9, atol=1e-12):
                        bad.setdefault(c, []).append((str(w), off, sym, float(a), float(b_)))
    assert not bad, (
        'these features CHANGE when bars after the decision instant are removed, so '
        'they read the future: ' +
        '; '.join(f'{k} ({len(v)} rows, e.g. {v[0]})' for k, v in list(bad.items())[:5]))
