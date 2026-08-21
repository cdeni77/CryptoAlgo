"""Calibration compares a forecast to an outcome. Both over the same period.

`/research/summary` and the per-coin health rows report `delta_bps` — realised
net minus expected net, in basis points of notional — and the health grade
("healthy" / "watch" / "at_risk") is derived from it. The forecast mean came
from the most recent `SIGNAL_WINDOW` signals; the outcome mean came from every
position the account had ever closed. A retrained model was therefore graded on
returns earned by the model before it, and the further back the account's
history went, the less the two numbers had to do with each other.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def clean_tables():
    from database import SessionLocal
    from models.signals import Signal
    from models.trade import PaperPosition

    db = SessionLocal()
    db.query(Signal).delete()
    db.query(PaperPosition).delete()
    db.commit()
    db.close()
    yield
    db = SessionLocal()
    db.query(Signal).delete()
    db.query(PaperPosition).delete()
    db.commit()
    db.close()


def _signal(coin: str, when: datetime, expected_bps: float):
    from models.signals import Signal

    return Signal(
        coin=coin,
        timestamp=when,
        direction='long',
        confidence=0.6,
        price_at_signal=100.0,
        passed_gates=True,
        expected_net_bps=expected_bps,
        cost_bps=10.0,
        carry_share=0.5,
    )


def _closed(coin: str, opened: datetime, realized: float, notional: float):
    from models.trade import PaperPosition

    return PaperPosition(
        coin=coin,
        side='long',
        contracts=1,
        entry_price=100.0,
        mark_price=100.0,
        notional=notional,
        realized_pnl=realized,
        unrealized_pnl=0.0,
        fees_paid=0.0,
        opened_at=opened,
        is_open=False,
    )


def test_a_position_opened_before_the_signal_window_is_not_calibration_evidence(
    client, clean_tables
):
    """The old disaster must not move a delta measured over the recent window."""
    from controllers.research import MIN_CALIBRATION_SAMPLE, _calibration, _closed_paper_positions, _window_start
    from database import SessionLocal
    from models.signals import Signal
    from sqlalchemy import desc

    now = datetime.now(timezone.utc)
    db = SessionLocal()

    # Recent: forecasts of +20bp that realised +20bp. Perfectly calibrated.
    for i in range(MIN_CALIBRATION_SAMPLE + 2):
        when = now - timedelta(hours=i)
        db.add(_signal('BTC', when, 20.0))
        db.add(_closed('BTC', when, realized=2.0, notional=1_000.0))  # +20bp

    # Ancient: one position from a year ago that lost 90% of its notional. No
    # signal in the window corresponds to it.
    db.add(_closed('BTC', now - timedelta(days=365), realized=-900.0, notional=1_000.0))
    db.commit()

    signals = (
        db.query(Signal).filter(Signal.coin == 'BTC').order_by(desc(Signal.timestamp)).all()
    )
    windowed = _calibration(signals, _closed_paper_positions(db, 'BTC', since=_window_start(signals)))
    unwindowed = _calibration(signals, _closed_paper_positions(db, 'BTC'))
    db.close()

    assert windowed.delta_bps is not None, 'enough matched observations to measure'
    assert abs(windowed.delta_bps) < 1.0, (
        f'forecast and outcome agree, so the delta should be ~0, got '
        f'{windowed.delta_bps:.1f}bp'
    )
    # And prove the guard is load-bearing: without it, the year-old loss drags
    # the same measurement into "at_risk" territory.
    assert unwindowed.delta_bps is not None
    assert unwindowed.delta_bps < -50.0, (
        'the unwindowed comparison should be badly distorted, otherwise this '
        'test is not exercising the bug'
    )


def test_coin_health_windows_its_own_outcomes(client, clean_tables):
    """The endpoint, not just the helper: a fresh model is not graded on old losses."""
    from controllers.research import MIN_CALIBRATION_SAMPLE
    from database import SessionLocal

    now = datetime.now(timezone.utc)
    db = SessionLocal()
    for i in range(MIN_CALIBRATION_SAMPLE + 2):
        when = now - timedelta(hours=i)
        db.add(_signal('ETH', when, 20.0))
        db.add(_closed('ETH', when, realized=2.0, notional=1_000.0))
    db.add(_closed('ETH', now - timedelta(days=365), realized=-900.0, notional=1_000.0))
    db.commit()
    db.close()

    response = client.get('/research/summary')
    assert response.status_code == 200, response.text
    eth = [r for r in response.json()['coins'] if r['coin'] == 'ETH']
    assert eth, 'ETH should appear once it has signals'
    row = eth[0]
    assert row['calibration']['delta_bps'] is not None
    assert abs(row['calibration']['delta_bps']) < 1.0, row['calibration']
    assert row['health'] == 'healthy', (row['health'], row['health_reason'])
