"""`scripts.promote --go-live-check`: live model_minus_market against the ledger."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.pg_writer import PgWriter
from scripts import promote as promote_script

VERSION = '20261001T050000Z'
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def models(tmp_path):
    """An installed artifact, its staged copy and its ledger entry."""
    root = tmp_path / 'models'
    (root / '.staging' / VERSION).mkdir(parents=True)
    (root / 'promotions').mkdir()
    for path in (root / 'forecast.joblib', root / '.staging' / VERSION / 'forecast.joblib'):
        path.write_bytes(b'artifact-bytes')

    def ledger(backtest: float, entry_offsets=(3,)):
        payload = {'version': VERSION, 'installed': True,
                   'gates': [{'name': 'model_minus_market', 'value': backtest}],
                   'report': {'config': {'entry_offsets': list(entry_offsets)}}}
        (root / 'promotions' / f'{VERSION}.json').write_text(json.dumps(payload))

    return root, ledger


@pytest.fixture
def database(tmp_path):
    url = f'sqlite:///{tmp_path}/serving.db'
    return url, PgWriter(database_url=url)


def write_rows(writer, *, model_edge: float, windows: int = 600, offset: int = 3,
               version=VERSION, seed: int = 1, start=NOW - timedelta(days=7)):
    """Three symbols per window. `model_edge` moves the model toward the truth."""
    rng = np.random.default_rng(seed)
    for i in range(windows):
        window_open = start + timedelta(minutes=15 * i)
        for symbol in ('BTC-USD', 'ETH-USD', 'SOL-USD'):
            outcome = int(rng.integers(0, 2))
            market = float(np.clip(0.5 + (0.15 if outcome else -0.15)
                                   + rng.normal(0, 0.12), 0.03, 0.97))
            toward = (1.0 - market) if outcome else -market
            model = float(np.clip(market + model_edge * toward, 0.02, 0.98))
            writer.write_prediction(
                symbol=symbol, window_open=window_open,
                settle_time=window_open + timedelta(minutes=15),
                offset_minutes=offset,
                decision_time=window_open + timedelta(minutes=offset),
                strike=100.0, last_price=100.0, displacement=0.0,
                baseline_probability=market, model_probability=model,
                market_probability=market, outcome=outcome, reason='edge_below_gate',
                model_version=version)


def run(root, url, days=14.0):
    return promote_script.go_live_check(days, root=root, database_url=url, now=NOW)


def test_live_matching_a_positive_backtest_passes(models, database, capsys):
    root, ledger = models
    url, writer = database
    ledger(0.01)
    write_rows(writer, model_edge=0.1)
    assert run(root, url) == 0
    assert 'verdict: PASS' in capsys.readouterr().out


def test_live_far_below_the_backtest_fails(models, database, capsys):
    root, ledger = models
    url, writer = database
    ledger(0.20)
    write_rows(writer, model_edge=0.05)
    assert run(root, url) == 1
    assert 'significantly below the backtest' in capsys.readouterr().out


def test_live_at_the_price_fails_even_against_a_zero_backtest(models, database, capsys):
    root, ledger = models
    url, writer = database
    ledger(0.0)
    write_rows(writer, model_edge=-0.02)
    assert run(root, url) == 1
    assert 'not above zero' in capsys.readouterr().out


def test_rows_outside_the_window_another_model_or_offset_are_ignored(
        models, database, capsys):
    root, ledger = models
    url, writer = database
    ledger(0.01)
    write_rows(writer, model_edge=-0.3, windows=200, start=NOW - timedelta(days=30))
    write_rows(writer, model_edge=-0.3, windows=200, version='other',
               start=NOW - timedelta(days=5), seed=2)
    write_rows(writer, model_edge=-0.3, windows=200, offset=12,
               start=NOW - timedelta(days=4), seed=3)
    write_rows(writer, model_edge=0.1, windows=400, version=None,
               start=NOW - timedelta(days=3), seed=4)
    assert run(root, url, days=14) == 0
    assert 'live rows 1,200 over 1,200 windows' in capsys.readouterr().out


def test_an_artifact_with_no_ledger_entry_fails(models, database, capsys):
    root, _ = models
    url, writer = database
    (root / 'forecast.joblib').write_bytes(b'something else')
    assert run(root, url) == 1
    assert 'no staged promotion' in capsys.readouterr().out


def test_no_live_rows_fail(models, database):
    root, ledger = models
    url, _ = database
    ledger(0.01)
    assert run(root, url) == 1


def test_the_check_writes_nothing(models, database, tmp_path):
    root, ledger = models
    url, writer = database
    ledger(0.01)
    write_rows(writer, model_edge=0.1, windows=100)
    db = tmp_path / 'serving.db'
    before = (db.read_bytes(), sorted(p.name for p in root.rglob('*')))
    run(root, url)
    assert (db.read_bytes(), sorted(p.name for p in root.rglob('*'))) == before


def test_the_cli_routes_to_the_check(monkeypatch):
    seen = {}
    monkeypatch.setattr(promote_script, 'go_live_check',
                        lambda days: seen.setdefault('days', days) and 0)
    monkeypatch.setattr('sys.argv', ['promote', '--go-live-check', '--days', '7'])
    promote_script.main()
    assert seen['days'] == 7.0
