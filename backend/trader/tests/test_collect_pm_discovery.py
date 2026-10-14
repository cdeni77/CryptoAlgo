"""Tests for Polymarket market discovery in `research.collect._collect_pm`.

Discovery saves a cursor so a resumed walk continues from the failing page, not
page 1. It keeps every asset by default, because one page mixes all assets and
filtering by asset does not reduce the page count. `PM_ASSET` still narrows it.
"""

from __future__ import annotations

import datetime as dt
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
# Polymarket has run 15-minute up/down markets under two settlement rules, and
# the slug is the only way to tell them apart:
#
#   `{asset}-up-or-down-15m-{ts}`  price at the end of the range (spot stream)
#   `{asset}-updown-15m-{ts}`      60-second TWAP over the range (current)
#
# An endpoint price and a 60-second average are different random variables, so
# the two eras must not be pooled. Discovery keeps both and labels each row.

def test_the_current_twap_era_is_recognised():
    assert pm.era_of('btc-updown-15m-1787873400') == pm.TWAP_ERA


def test_the_old_endpoint_era_is_recognised():
    assert pm.era_of('eth-up-or-down-15m-1757724300') == pm.ENDPOINT_ERA


def test_the_two_eras_are_not_confused_for_one_another():
    """The era substrings do not overlap."""
    assert pm.era_of('btc-updown-15m-1') != pm.era_of('btc-up-or-down-15m-1')


def test_a_market_that_is_neither_era_has_no_era():
    assert pm.era_of('btc-daily-market-123') is None
    assert pm.era_of('') is None
    assert pm.era_of(None) is None


def test_discovery_no_longer_silently_drops_the_old_era():
    """`is_short` accepts the old era; 'updown-15m' is not a substring of 'up-or-down-15m'.
    """
    assert pm.is_short('eth-up-or-down-15m-1757724300', '')


def test_the_old_era_still_honours_an_asset_filter():
    assert pm.is_short('eth-up-or-down-15m-1', 'eth-')
    assert not pm.is_short('eth-up-or-down-15m-1', 'btc-')


def test_a_market_before_book_coverage_is_not_worth_a_request():
    """Markets before the first retrievable Polymarket book are not priced, which saves a
    request each.
    """
    before = int(dt.datetime(2025, 11, 1, tzinfo=dt.timezone.utc).timestamp())
    after = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp())
    assert not pm.worth_pricing({'market_slug': f'btc-updown-15m-{before}'})
    assert pm.worth_pricing({'market_slug': f'btc-updown-15m-{after}'})


def test_the_old_endpoint_era_is_never_worth_pricing():
    """The old era is never priced: it is a different instrument with no retrievable book.
    """
    after = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp())
    assert not pm.worth_pricing({'market_slug': f'btc-up-or-down-15m-{after}'})


def test_a_malformed_slug_is_skipped_rather_than_crashing_the_run():
    assert not pm.worth_pricing({'market_slug': 'btc-updown-15m-notanumber'})
    assert not pm.worth_pricing({'market_slug': ''})
    assert not pm.worth_pricing({})


def test_discover_records_which_era_each_market_belongs_to(tmp_path, monkeypatch):
    """Discovery stores the era on each row."""
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
