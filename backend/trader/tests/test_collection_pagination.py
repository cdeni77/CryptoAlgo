"""Both venues paginate. Assuming one of them does not truncated 44% of it.

`fetch_pm` was written as a single request because the August trial windows
averaged 358 snapshots and never approached the 2,000-per-page cap. On busy
months they do: 3,097 collected Polymarket windows had EXACTLY 2,000 snapshots
and none had more, while Kalshi — which paginates — had one at exactly 2,000
and thousands above it. That distribution is what a silent truncation looks
like.

Verified against the live endpoint: a January BTC window returns 2,000
snapshots with `has_more: true` and a pagination_key, and the second page
returns 2,000 more with `has_more` still true.
"""

from __future__ import annotations

import datetime as dt

from research.collect.run_collection import fetch_kalshi, fetch_pm

UTC = dt.timezone.utc


class FakeApi:
    """Replays paged responses and records the params it was asked for."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get(self, path, params):
        self.calls.append(dict(params) if isinstance(params, dict) else params)
        if not self.pages:
            return {'snapshots': [], 'pagination': {}}, True
        return self.pages.pop(0), True


def _page(n, *, more, key='next'):
    return {'snapshots': [{'timestamp': i} for i in range(n)],
            'pagination': {'has_more': more, 'pagination_key': key if more else None}}


class Item:
    venue = 'polymarket'
    symbol = 'BTC-USD'
    market_id = 'btc-updown-15m-1767837600'
    window_open = dt.datetime(2026, 1, 8, 2, 0, tzinfo=UTC)


TOKENS = {Item.market_id: 'tok'}


def test_polymarket_follows_pagination_instead_of_keeping_the_first_page():
    api = FakeApi([_page(2000, more=True), _page(2000, more=True),
                   _page(431, more=False)])
    snaps, err = fetch_pm(api, Item(), TOKENS)
    assert err is None
    assert len(snaps) == 4431, 'must keep every page, not just the first 2000'
    assert len(api.calls) == 3


def test_polymarket_passes_the_cursor_on_later_pages():
    api = FakeApi([_page(2000, more=True), _page(10, more=False)])
    fetch_pm(api, Item(), TOKENS)
    assert 'pagination_key' not in api.calls[0]
    assert api.calls[1]['pagination_key'] == 'next'


def test_polymarket_stops_when_the_venue_says_there_is_no_more():
    api = FakeApi([_page(500, more=False), _page(2000, more=True)])
    snaps, _ = fetch_pm(api, Item(), TOKENS)
    assert len(snaps) == 500, 'must not request a page the venue did not offer'


def test_polymarket_stops_on_an_empty_page_even_if_more_is_claimed():
    """A venue that says has_more but returns nothing would loop forever."""
    api = FakeApi([_page(2000, more=True), _page(0, more=True)])
    snaps, _ = fetch_pm(api, Item(), TOKENS)
    assert len(snaps) == 2000


def test_polymarket_without_a_token_is_an_error_not_an_empty_window():
    snaps, err = fetch_pm(FakeApi([]), Item(), {})
    assert snaps is None and 'token' in err


class KItem(Item):
    venue = 'kalshi'
    market_id = 'KXBTC15M-26JAN080215-15'


def test_kalshi_still_paginates_as_it_did():
    api = FakeApi([_page(2000, more=True), _page(700, more=False)])
    snaps, err = fetch_kalshi(api, KItem(), {})
    assert err is None and len(snaps) == 2700
