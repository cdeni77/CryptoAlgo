"""The live recorder discarded the fits the model was trained on.

`research/collect/implied_vol_backfill.py` applies NO R-squared gate: it keeps
any fit with at least MIN_STRIKES rungs. Measured on what it actually wrote,
accepted fits run down to R2 = 0.0367 (BTC), 0.0134 (ETH), 0.0002 (SOL), and
the 5th percentile is 0.85 / 0.72 / 0.56.

`scripts/record_implied_vol.py` defaulted `--min-r2` to 0.90, so live kept only
the top of that distribution. Over 67 hours it produced 2,805 BTC fits, 13 ETH
and ZERO SOL — against training coverage of 22,258 / 5,286 / 3,110.

Two reasons that is the wrong direction, not a conservative one:

  * `--complete-cases` fitted the artifact on rows that all carry a fit. A
    symbol with no fits has no complete rows, so live scores ETH and SOL out of
    distribution — or abstains on them entirely.
  * `iv_r2` IS ONE OF THE FIVE FEATURES. The model saw R2 from 0.0002 to 1.0
    and learned what a weak fit is worth. Filtering upstream removes the
    information it was given to make that judgement.
"""
from __future__ import annotations

import datetime as dt

from scripts.record_implied_vol import build_parser, fits_for


def _market(strike, price, close):
    return {'event_ticker': 'KXSOLD-X', 'close_time': close,
            'floor_strike': strike,
            'yes_bid_dollars': f'{max(price - 0.01, 0.0):.4f}',
            'yes_ask_dollars': f'{min(price + 0.01, 1.0):.4f}'}


def test_the_live_default_matches_the_backfill_which_gates_on_nothing():
    assert build_parser().parse_args([]).min_r2 == 0.0


def test_a_noisy_ladder_is_kept_and_its_r2_reported():
    """SOL's near-term ladder fitted at R2=0.54 when this was measured — the
    backfill would have kept it, live threw it away, and the model has iv_r2
    precisely so it can decide what a 0.54 fit is worth."""
    now = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    close = (now + dt.timedelta(minutes=40)).isoformat()
    # Deliberately non-monotone: a real thin book, not a clean smile.
    noisy = ((95.0, 0.88), (100.0, 0.42), (105.0, 0.55), (110.0, 0.11),
             (115.0, 0.19))
    rows, latest = fits_for([_market(s, p, close) for s, p in noisy],
                            now=now, symbol='SOL-USD',
                            min_minutes=2.0, max_minutes=180.0, min_r2=0.0)
    assert rows, 'a noisy but well-formed ladder produced no fit'
    assert 0.0 <= rows[0]['r2'] <= 1.0
    assert latest is not None


def test_a_threshold_can_still_be_asked_for_explicitly():
    """The flag remains, for diagnostics. It is the DEFAULT that was wrong."""
    assert build_parser().parse_args(['--min-r2', '0.9']).min_r2 == 0.9
