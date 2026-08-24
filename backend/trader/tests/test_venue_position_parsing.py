"""A position the venue holds must be visible to us. Measured payload, verbatim.

The live loop read `row['position']`. V2 does not send that field — it sends
`position_fp`, a fixed-point string, negative for the short-YES leg that a NO
position is held as. So `int(row.get('position') or 0)` was `int(0)` for every
row, and `venue_open` was the empty set on every cycle of the first live night.

That broke the position cross-check in both directions simultaneously:

* **forward** — "we hold N contracts the venue does not report. Most likely the
  order never filled" fired on every genuinely open position, once a minute. The
  warning for a killed fill_or_kill was therefore always on, which is the same as
  it being off.
* **reverse** — "the venue reports an open position we have no record of" could
  never fire. That is the discrepancy the audit singled out as the one that costs
  money silently: what a POST that times out after the venue accepted it leaves
  behind.

The payload below is the real one, copied from `/portfolio/positions` while a
BTC-USD NO position we had just watched fill was open.
"""

from __future__ import annotations

import pytest

from data_collection.kalshi_client import KalshiClient

# Verbatim from the venue, 2026-08-24T13:49:13Z, 5 NO contracts on BTC.
REAL_ROW = {
    'exchange_index': 0,
    'fees_paid_dollars': '0.085800',
    'last_updated_ts': '2026-08-24T13:49:13.549069Z',
    'market_exposure_dollars': '2.150000',
    'position_fp': '-5.00',
    'realized_pnl_dollars': '0.000000',
    'ticker': 'KXBTC15M-26AUG241000-00',
    'total_traded_dollars': '2.150000',
}


def test_a_real_no_position_is_seen_as_open():
    """The regression, in one line. This returned 0 for the whole first night."""
    assert KalshiClient.position_size(REAL_ROW) == pytest.approx(-5.0)
    assert KalshiClient.position_size(REAL_ROW) != 0, (
        'a position we watched fill read as flat'
    )


def test_the_field_the_old_documentation_names_is_simply_absent():
    """Which is why the bug was silent rather than loud."""
    assert 'position' not in REAL_ROW
    assert int(REAL_ROW.get('position') or 0) == 0, (
        'the old expression really did evaluate to zero on a real open position'
    )


def test_a_short_and_a_long_are_both_open():
    """Sign carries the side. `!= 0` is the open test; `> 0` would drop every NO
    position, which is half the strategy — the band is symmetric on purpose."""
    long_yes = dict(REAL_ROW, position_fp='7.00')
    assert KalshiClient.position_size(long_yes) == pytest.approx(7.0)
    assert KalshiClient.position_size(REAL_ROW) < 0 < KalshiClient.position_size(long_yes)


def test_a_flat_market_is_flat():
    """Kalshi keeps the row after a position closes, so this must read zero or
    the forward check warns forever about markets we left."""
    assert KalshiClient.position_size(dict(REAL_ROW, position_fp='0.00')) == 0


@pytest.mark.parametrize('row', [
    {},                                       # nothing at all
    {'position': 5},                          # the legacy integer encoding
    {'position': '5'},                        # ... as a string
    {'position_fp': '5.00', 'position': 5},   # both, agreeing
])
def test_the_legacy_encoding_still_parses(row):
    """Accept both, the way the quote parsing had to. A venue that restores the
    integer field must not silently read as flat again."""
    expected = 0.0 if not row else 5.0
    assert KalshiClient.position_size(row) == pytest.approx(expected)


def test_garbage_reads_as_flat_rather_than_raising():
    """`int('-5.00')` raises ValueError, and this runs inside a set comprehension
    in `reconcile_with_venue` where nothing would catch it — the reconciliation
    would abort mid-cycle and the balance would never be adopted."""
    assert KalshiClient.position_size({'position_fp': 'nonsense'}) == 0
    assert KalshiClient.position_size({'position_fp': None}) == 0
