"""Discovery must not restart from page 1 every time it is resumed.

Measured tonight: the discovery walk stopped on a bare Predexon 500 at page
909, then again at page 844 on the very next attempt — both past `get()`'s own
retry budget. Every resume re-walked from page 1, re-fetching every page it
already knew (skipped by slug, but still a full request each) before making
any forward progress. Going from page 0 to page 844 took over two hours; the
walk needs roughly 4,000 pages to reach the 2026-03-02 coverage boundary, so
every resume was paying most of an hour's worth of requests just to relearn
what it already had.

A second, independent inefficiency, verified live before writing this: one
unfiltered page of `/v2/polymarket/markets?tags=15M` mixes BTC, ETH, SOL,
BNB, DOGE, ZEC, XRP and HYPE. Filtering discovery to one asset via
`PM_ASSET` (as the three separate BTC/ETH/SOL backfill runs did tonight) does
not reduce the page count at all — it walks the identical page stream three
times over, keeping a different eighth of each page each time. Discovery now
defaults to keeping every `*-updown-15m-*` slug regardless of asset;
`PM_ASSET` still narrows it when that is genuinely wanted.
"""

from __future__ import annotations

import asyncio
import json

import research.collect._collect_pm as pm


def test_is_short_accepts_any_asset_by_default():
    assert pm.is_short('btc-updown-15m-123', '')
    assert pm.is_short('eth-updown-15m-123', '')
    assert pm.is_short('bnb-updown-15m-123', '')


def test_is_short_can_still_be_narrowed_to_one_asset():
    assert pm.is_short('btc-updown-15m-123', 'btc-')
    assert not pm.is_short('eth-updown-15m-123', 'btc-')


def test_is_short_rejects_a_non_15m_market():
    assert not pm.is_short('btc-daily-market-123', '')


# -- the two settlement eras -------------------------------------------------
#
# Polymarket has run 15-minute crypto up/down under two DIFFERENT settlement
# rules, and the slug is the only thing that distinguishes them. Both were read
# live from the venue's own `description`/`resolutionSource`:
#
#   `{asset}-up-or-down-15m-{ts}`  (from 2025-09-12)
#       "resolve to Up if the Ethereum price AT THE END of the time range ...
#        is greater than or equal to the price at the beginning"
#       source: https://data.chain.link/streams/eth-usd        (spot stream)
#       first market measured: $23.66 volume, $0.00 liquidity
#
#   `{asset}-updown-15m-{ts}`      (the current instrument)
#       "resolve to Up if the TIME-WEIGHTED AVERAGE PRICE (TWAP) ... of the
#        time range ... is >= the price at the beginning of that range"
#       source: https://data.chain.link/streams/btc-usd-twap-60s-streams
#       measured liquidity on a live market: $21,873
#
# An endpoint reading and a 60-second TWAP are not the same random variable —
# per CLAUDE.md's own invariant a time-average over an interval carries a
# THIRD of its endpoint's variance — so pooling the two eras into one training
# set would silently mix two instruments. `is_short` used to match only
# 'updown-15m', which excluded the old era by accident rather than on purpose;
# that is the right outcome reached for the wrong reason, and it also meant
# nothing could ever measure where the boundary is. Discovery now keeps both
# and labels each row, so the choice to use one era is made deliberately
# downstream instead of by a substring that happens not to match.

def test_the_current_twap_era_is_recognised():
    assert pm.era_of('btc-updown-15m-1787873400') == pm.TWAP_ERA


def test_the_old_endpoint_era_is_recognised():
    assert pm.era_of('eth-up-or-down-15m-1757724300') == pm.ENDPOINT_ERA


def test_the_two_eras_are_not_confused_for_one_another():
    """The substrings must not overlap — this is the whole distinction."""
    assert pm.era_of('btc-updown-15m-1') != pm.era_of('btc-up-or-down-15m-1')


def test_a_market_that_is_neither_era_has_no_era():
    assert pm.era_of('btc-daily-market-123') is None
    assert pm.era_of('') is None
    assert pm.era_of(None) is None


def test_discovery_no_longer_silently_drops_the_old_era():
    """The bug: 'updown-15m' does not appear in 'up-or-down-15m', so an entire
    era of real, settled markets was skipped without anything reporting it."""
    assert pm.is_short('eth-up-or-down-15m-1757724300', '')


def test_the_old_era_still_honours_an_asset_filter():
    assert pm.is_short('eth-up-or-down-15m-1', 'eth-')
    assert not pm.is_short('eth-up-or-down-15m-1', 'btc-')


def test_discover_records_which_era_each_market_belongs_to(tmp_path, monkeypatch):
    """Era must be stored, not re-derived later by whoever happens to remember
    the two spellings."""
    async def fake_get(session, path, params):
        return {'markets': [
            {'market_slug': 'btc-updown-15m-9999999999', 'condition_id': 'a'},
            {'market_slug': 'btc-up-or-down-15m-9999999998', 'condition_id': 'b'},
        ], 'pagination': {'pagination_key': None}}, None

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get)
    asyncio.run(pm.discover(session=None))

    rows = [json.loads(line) for line in
            open(tmp_path / 'markets.jsonl').read().splitlines() if line.strip()]
    by_slug = {r['market_slug']: r for r in rows}
    assert by_slug['btc-updown-15m-9999999999']['era'] == pm.TWAP_ERA
    assert by_slug['btc-up-or-down-15m-9999999998']['era'] == pm.ENDPOINT_ERA


def test_cursor_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, 'CURSOR_FILE', str(tmp_path / 'cursor.json'))
    pm._save_cursor('abc123', 42)
    assert pm._load_cursor() == {'cursor': 'abc123', 'pages': 42}


def test_load_cursor_returns_none_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, 'CURSOR_FILE', str(tmp_path / 'nope.json'))
    assert pm._load_cursor() is None


def test_clear_cursor_removes_the_file(tmp_path, monkeypatch):
    cursor_file = tmp_path / 'cursor.json'
    cursor_file.write_text('{"cursor": "x", "pages": 1}')
    monkeypatch.setattr(pm, 'CURSOR_FILE', str(cursor_file))
    pm._clear_cursor()
    assert not cursor_file.exists()


def test_clear_cursor_is_a_noop_when_no_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, 'CURSOR_FILE', str(tmp_path / 'nope.json'))
    pm._clear_cursor()  # must not raise


def _wire_discover(tmp_path, monkeypatch, *, fake_get, max_pages=5,
                    asset_prefix='', coverage_start=0):
    monkeypatch.setattr(pm, 'MARKETS_OUT', str(tmp_path / 'markets.jsonl'))
    monkeypatch.setattr(pm, 'CURSOR_FILE', str(tmp_path / 'cursor.json'))
    monkeypatch.setattr(pm, 'MAX_PAGES', max_pages)
    monkeypatch.setattr(pm, 'ASSET_PREFIX', asset_prefix)
    monkeypatch.setattr(pm, 'COVERAGE_START', coverage_start)
    monkeypatch.setattr(pm, 'get', fake_get)


def test_discover_resumes_from_a_saved_cursor_instead_of_restarting(tmp_path, monkeypatch):
    seen_params = []

    async def fake_get(session, path, params):
        seen_params.append(dict(params))
        return {'markets': [{'market_slug': 'btc-updown-15m-9999999999',
                              'condition_id': 'c'}],
                'pagination': {'pagination_key': None}}, None

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get)
    pm._save_cursor('resume-here', 3)

    asyncio.run(pm.discover(session=None))

    assert seen_params[0].get('pagination_key') == 'resume-here', \
        'must resume from the saved cursor, not restart at page 1'


def test_discover_starts_at_page_1_when_no_cursor_is_saved(tmp_path, monkeypatch):
    seen_params = []

    async def fake_get(session, path, params):
        seen_params.append(dict(params))
        return {'markets': [{'market_slug': 'btc-updown-15m-9999999999',
                              'condition_id': 'c'}],
                'pagination': {'pagination_key': None}}, None

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get)

    asyncio.run(pm.discover(session=None))

    assert 'pagination_key' not in seen_params[0]


def test_discover_saves_the_cursor_that_led_to_a_mid_walk_error(tmp_path, monkeypatch):
    calls = {'n': 0}

    async def fake_get(session, path, params):
        calls['n'] += 1
        if calls['n'] == 1:
            return {'markets': [{'market_slug': 'btc-updown-15m-9999999999',
                                  'condition_id': 'c'}],
                    'pagination': {'pagination_key': 'page-2-cursor'}}, None
        return None, '500:boom'

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get)

    asyncio.run(pm.discover(session=None))

    assert pm._load_cursor() == {'cursor': 'page-2-cursor', 'pages': 1}, \
        'must save the cursor for the failing page, so a retry lands there, not page 1'


def test_discover_clears_a_saved_cursor_once_it_reaches_the_boundary(tmp_path, monkeypatch):
    async def fake_get(session, path, params):
        return {'markets': [{'market_slug': 'btc-updown-15m-1', 'condition_id': 'c'}],
                'pagination': {'pagination_key': 'next'}}, None

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get, coverage_start=99999999999)
    pm._save_cursor('stale', 2)  # left behind by an earlier interrupted run

    asyncio.run(pm.discover(session=None))

    assert pm._load_cursor() is None, 'a clean finish must not leave a stale cursor behind'


def test_discover_saves_the_cursor_when_max_pages_is_exhausted(tmp_path, monkeypatch):
    async def fake_get(session, path, params):
        return {'markets': [{'market_slug': 'btc-updown-15m-9999999999',
                              'condition_id': 'c'}],
                'pagination': {'pagination_key': 'still-going'}}, None

    _wire_discover(tmp_path, monkeypatch, fake_get=fake_get, max_pages=2)

    asyncio.run(pm.discover(session=None))

    assert pm._load_cursor() == {'cursor': 'still-going', 'pages': 2}, \
        'running out of MAX_PAGES must save the next cursor, not discard progress'
