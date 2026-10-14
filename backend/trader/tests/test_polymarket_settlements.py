"""Tests for collecting Polymarket's own settlements, an independent check on our label.

Polymarket settles on Chainlink's BTC-USD TWAP-60s; outcome `A` is up.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pandas as pd
import pytest

import scripts.collect_settlements as cs

NOW = pd.Timestamp('2026-09-16T20:00:00Z')


def _catalog(tmp_path, records):
    path = tmp_path / 'pm_catalog.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records))
    return str(path)


def _market(**over):
    m = {'symbol': 'BTC-USD', 'window_open': '2026-09-16T15:00:00+00:00',
         'result': 'A', 'status': 'closed', 'market_id': 'pm-1',
         'volume_dollars': 1234.0, 'liquidity_dollars': 99.0}
    m.update(over)
    return m


def _run(path, monkeypatch):
    monkeypatch.setenv('PM_CATALOG', path)
    return asyncio.run(cs.polymarket(None, NOW))


def test_result_A_is_up(tmp_path, monkeypatch):
    rows = _run(_catalog(tmp_path, [_market(result='A')]), monkeypatch)
    assert len(rows) == 1
    assert rows[0]['settled_up'] is True
    assert rows[0]['result'] == 'yes'


def test_result_B_is_down(tmp_path, monkeypatch):
    rows = _run(_catalog(tmp_path, [_market(result='B')]), monkeypatch)
    assert rows[0]['settled_up'] is False
    assert rows[0]['result'] == 'no'


def test_the_venue_is_recorded(tmp_path, monkeypatch):
    """The venue is recorded, so the label join can tell the two oracles apart."""
    rows = _run(_catalog(tmp_path, [_market()]), monkeypatch)
    assert rows[0]['venue'] == 'polymarket'


def test_an_unsettled_market_is_skipped(tmp_path, monkeypatch):
    assert _run(_catalog(tmp_path, [_market(status='open')]), monkeypatch) == []
    assert _run(_catalog(tmp_path, [_market(result=None)]), monkeypatch) == []


def test_the_close_is_fifteen_minutes_after_the_open(tmp_path, monkeypatch):
    rows = _run(_catalog(tmp_path, [_market()]), monkeypatch)
    delta = rows[0]['close_time'] - rows[0]['window_open']
    assert delta == pd.Timedelta(minutes=15)


def test_a_duplicate_window_is_collapsed(tmp_path, monkeypatch):
    rows = _run(_catalog(tmp_path, [_market(), _market()]), monkeypatch)
    assert len(rows) == 1


def test_a_missing_catalog_is_not_an_error(tmp_path, monkeypatch):
    assert _run(str(tmp_path / 'absent.jsonl'), monkeypatch) == []


def test_the_branch_records_that_it_answered():
    """The branch appends to `reached`, so "collected nothing" differs from "never ran"."""
    src = inspect.getsource(cs.run)
    tail = src[src.index('await polymarket('):]
    # Search up to the next branch, not a fixed slice.
    stop = tail.index('if not rows:') if 'if not rows:' in tail else len(tail)
    assert "reached.append('polymarket')" in tail[:stop]
