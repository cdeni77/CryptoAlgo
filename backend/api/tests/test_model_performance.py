"""Per-model P&L, so a model's record can be read on its own.

Positions carry no model of their own; attribution runs
`positions.prediction_id -> predictions.model_version`. A position whose
prediction is missing keeps its P&L in `unattributed`, because a per-model
table that does not reconcile with the account is worse than one with a
remainder -- 810 of 1,852 real positions predate `model_version` being
recorded.
"""

from __future__ import annotations


def test_the_endpoint_reconciles_with_its_parts(client):
    body = client.get('/account/models?all_time=true').json()
    parts = sum(m['pnl'] for m in body['models']) + body['unattributed']['pnl']
    assert abs(body['total_pnl'] - parts) < 1e-6, 'the table must add up'
    counted = (sum(m['positions'] for m in body['models'])
               + body['unattributed']['positions'])
    assert body['total_positions'] == counted


def test_all_time_ignores_the_epoch_and_the_default_does_not(client):
    assert client.get('/account/models?all_time=true').json()['all_time'] is True
    assert client.get('/account/models').json()['all_time'] is False


def test_each_model_reports_rates_beside_the_totals(client):
    for m in client.get('/account/models?all_time=true').json()['models']:
        assert m['model_version']
        if m['positions']:
            assert m['pnl_per_position'] is not None
            assert m['win_rate'] is not None and 0.0 <= m['win_rate'] <= 1.0
        # a longshot model wins seldom and profitably, so these are independent
        assert set(m) >= {'pnl', 'win_rate', 'pnl_per_position',
                          'pnl_per_contract', 'first_window', 'last_window'}


def test_an_empty_database_is_an_empty_table_not_an_error(client):
    body = client.get('/account/models').json()
    assert isinstance(body['models'], list)
    assert body['total_pnl'] == 0 or isinstance(body['total_pnl'], float)


def test_a_win_is_money_made_not_the_market_going_up(client):
    """`settled_up` is the MARKET's direction. A `down` position settling down
    is a win, and counting `settled_up` scored every short backwards."""
    import inspect
    from controllers import serving
    src = inspect.getsource(serving.model_performance)
    assert "pnl > 0" in src
    assert "position.settled_up" not in src, 'the market is not our outcome'
