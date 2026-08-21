"""A missing measurement must read as missing.

The failure these guard against was not a crash. The research surface reported
`pr_auc` as `holdout_auc - 0.06` and `precision_at_threshold` as
`holdout_auc - 0.04` — one number three times with different constants — and,
when the artifact it wanted was absent, substituted a hardcoded table of six
plausible feature names with plausible weights. Every one of those rendered
identically to a real measurement, which is what made it dangerous.

So the property under test is the boring one: with no model and no ledger, the
API returns nulls, empty lists, and a reason — never a value.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _record(version: str, *, promoted: bool, forced: bool = False) -> dict:
    return {
        'version': version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'promoted': promoted,
        'forced': forced,
        'force_reason': 'operator override' if forced else None,
        'gates': [
            {'name': 'pbo', 'value': 0.12, 'threshold': 0.30,
             'comparison': 'max', 'passed': True},
            {'name': 'cpcv_median_sharpe', 'value': 0.9 if promoted else -0.3,
             'threshold': 0.5, 'comparison': 'min', 'passed': promoted},
        ],
        'provenance': {
            'feature_set_hash': 'abc123',
            'n_features': 76,
            'heads': ['price', 'carry', 'dispersion'],
            'uses_symbol_identity': False,
            'horizon_bars': 24,
            'cost_config_version': 'coinbase_us_perps_cde_v202602',
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'train_rows': 40_000,
            'effective_observations': 412.0,
            'symbols': ['BIP', 'ETP'],
        },
        'backtest': {'trades': 250, 'sharpe': 1.4, 'net_pnl': 12_000.0,
                     'price_pnl': 9_000.0, 'funding_pnl': 4_000.0, 'fees': 1_000.0,
                     'max_exit_participation': 0.07},
        'simulation': {
            'bootstrap': {
                'sharpe': {'n': 2000, 'median': 1.2, 'p05': 0.3, 'p95': 2.1},
                'probability_positive': 0.94,
                'risk_of_ruin': 0.01,
                'block_length': 3.2,
            },
            'cpcv': {'n': 6, 'median': 0.9, 'p05': 0.1, 'p95': 1.8},
        },
        'measurements': {'pbo': 0.12},
        'error': None,
    }


def _write_ledger(models_dir, records, live=None):
    directory = models_dir / 'promotions'
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        (directory / f"{record['version']}.json").write_text(json.dumps(record))
    if live is not None:
        (directory / 'current.json').write_text(json.dumps(live))


# ---------------------------------------------------------------------------
# Nothing promoted
# ---------------------------------------------------------------------------


def test_no_model_reports_no_model_rather_than_erroring(client, empty_models_dir):
    """A fresh install has no promoted model. That is a state, not a failure."""
    body = client.get('/model/').json()

    assert body['has_model'] is False
    assert body['live'] is None
    assert body['artifact_path'] is None
    assert body['unrecorded_artifact'] is False


def test_feature_importance_explains_its_absence(client, empty_models_dir):
    """Empty plus a reason, never a substitute table."""
    body = client.get('/model/features').json()

    assert body['features'] == []
    assert body['unavailable_reason']
    assert 'scripts.promote' in body['unavailable_reason']


def test_no_ledger_is_an_empty_history(client, empty_models_dir):
    body = client.get('/model/promotions').json()

    assert body['records'] == []
    assert body['live_version'] is None
    assert body['trials_to_date'] == 0


# ---------------------------------------------------------------------------
# With a ledger
# ---------------------------------------------------------------------------


def test_rejections_are_served_alongside_the_promotion(client, empty_models_dir):
    """The trial count is what the deflated Sharpe discounts by.

    Serving only the successes would make the surviving model look better than
    the evidence supports, which is the exact bias the deflated Sharpe exists to
    correct for.
    """
    promoted = _record('20260301T000000Z-aaa111', promoted=True)
    _write_ledger(
        empty_models_dir,
        [
            _record('20260101T000000Z-000001', promoted=False),
            _record('20260201T000000Z-000002', promoted=False),
            promoted,
        ],
        live=promoted,
    )

    body = client.get('/model/promotions').json()

    assert body['trials_to_date'] == 3
    assert len(body['records']) == 3
    assert sum(r['promoted'] for r in body['records']) == 1
    # Newest first.
    assert [r['version'] for r in body['records']] == [
        '20260301T000000Z-aaa111', '20260201T000000Z-000002', '20260101T000000Z-000001'
    ]


def test_the_live_record_is_marked_live(client, empty_models_dir):
    promoted = _record('20260301T000000Z-aaa111', promoted=True)
    _write_ledger(empty_models_dir, [promoted, _record('20260101T000000Z-b', promoted=False)],
                  live=promoted)

    body = client.get('/model/promotions').json()
    live = [r for r in body['records'] if r['is_live']]

    assert len(live) == 1
    assert live[0]['version'] == '20260301T000000Z-aaa111'


def test_failed_gates_are_named(client, empty_models_dir):
    blocked = _record('20260101T000000Z-000001', promoted=False)
    _write_ledger(empty_models_dir, [blocked])

    record = client.get('/model/promotions').json()['records'][0]

    assert record['failed_gates'] == ['cpcv_median_sharpe']
    gate = [g for g in record['gates'] if g['name'] == 'cpcv_median_sharpe'][0]
    assert gate['passed'] is False
    assert gate['value'] == -0.3
    assert gate['threshold'] == 0.5


def test_a_forced_promotion_stays_visibly_forced(client, empty_models_dir):
    """An override must not launder itself into a clean pass."""
    forced = _record('20260301T000000Z-forced', promoted=True, forced=True)
    forced['gates'][1]['passed'] = False
    _write_ledger(empty_models_dir, [forced], live=forced)

    body = client.get('/model/').json()

    assert body['live']['forced'] is True
    assert body['live']['force_reason'] == 'operator override'
    assert body['live']['failed_gates'] == ['cpcv_median_sharpe']


def test_provenance_and_decomposition_survive_the_round_trip(client, empty_models_dir):
    promoted = _record('20260301T000000Z-aaa111', promoted=True)
    _write_ledger(empty_models_dir, [promoted], live=promoted)

    live = client.get('/model/').json()['live']

    assert live['provenance']['n_features'] == 76
    assert live['provenance']['effective_observations'] == 412.0
    assert live['provenance']['uses_symbol_identity'] is False
    # Price, funding and fees kept apart: gross positive with net negative is a
    # cost problem, and no amount of retraining fixes it.
    assert live['backtest']['price_pnl'] == 9_000.0
    assert live['backtest']['funding_pnl'] == 4_000.0
    assert live['backtest']['fees'] == 1_000.0
    assert live['simulation']['bootstrap_sharpe']['p05'] == 0.3
    assert live['simulation']['probability_positive'] == 0.94


def test_an_unreadable_ledger_entry_does_not_hide_the_rest(client, empty_models_dir):
    _write_ledger(empty_models_dir, [_record('20260101T000000Z-000001', promoted=True)])
    (empty_models_dir / 'promotions' / '20260102T000000Z-bad.json').write_text('{not json')

    body = client.get('/model/promotions').json()

    assert [r['version'] for r in body['records']] == ['20260101T000000Z-000001']
    # The corrupt file still counted as an attempt, because it was one.
    assert body['trials_to_date'] == 2


def test_an_artifact_with_no_ledger_entry_is_flagged(client, empty_models_dir):
    """A model installed by hand bypassed the gates. Say so rather than rendering it as normal."""
    (empty_models_dir / 'forecast.joblib').write_bytes(b'not really a model')

    body = client.get('/model/').json()

    assert body['has_model'] is True
    assert body['live'] is None
    assert body['unrecorded_artifact'] is True


def test_an_unloadable_artifact_reports_why(client, empty_models_dir):
    (empty_models_dir / 'forecast.joblib').write_bytes(b'not really a model')

    body = client.get('/model/features').json()

    assert body['features'] == []
    assert 'cannot load' in body['unavailable_reason']


# ---------------------------------------------------------------------------
# Research summary
# ---------------------------------------------------------------------------


def test_the_research_summary_reports_unknown_on_no_evidence(client, empty_models_dir):
    """No signals means "not measured", not a grade.

    The previous implementation always produced a health rating, so a brand-new
    install displayed a verdict for every instrument on no data at all.
    """
    body = client.get('/research/summary').json()

    assert body['kpis']['health'] == 'unknown'
    assert body['kpis']['expected_net_bps'] is None
    assert body['kpis']['realised_net_bps'] is None
    assert body['kpis']['calibration_delta_bps'] is None
    for coin in body['coins']:
        assert coin['health'] == 'unknown'
        assert coin['health_reason']
        assert coin['calibration']['delta_bps'] is None


def test_the_research_features_endpoint_no_longer_invents_importances(client, empty_models_dir):
    body = client.get('/research/features/BTC').json()

    assert body['feature_importance'] == []
    assert body['importance_unavailable_reason']
