"""Tests that live rows record what produced them.

`model_version`, `prediction_id` and `quote_age_seconds` must be set; a missing age
reads as fresh.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.pg_writer import OrderTicket, Position, Prediction
from livetrade import artifact as live_artifact
from scripts import live
from tests import test_decision_first as fx
from tests.test_decision_first import Venue, make_quote, run


class TestModelVersion:
    def test_predictions_carry_the_model_version(self, tmp_path, monkeypatch):
        writer, _ = run(tmp_path, monkeypatch, Venue())
        with writer._session() as session:            # noqa: SLF001
            versions = {p.model_version for p in session.query(Prediction)}
        assert versions == {'m1'}

    def test_the_version_is_the_promotion_ledgers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live_artifact, 'MODELS_ROOT', tmp_path)
        installed = tmp_path / 'forecast.joblib'
        installed.write_bytes(b'artifact-bytes')
        staged = tmp_path / '.staging' / '20261011T050000Z'
        staged.mkdir(parents=True)
        (staged / 'forecast.joblib').write_bytes(b'artifact-bytes')
        assert live_artifact.artifact_version(installed) == '20261011T050000Z'
        other = tmp_path / 'elsewhere.joblib'
        other.write_bytes(b'something else')
        assert live_artifact.artifact_version(other).startswith('md5:')

    def test_a_loaded_or_reloaded_artifact_is_stamped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(live_artifact, 'MODELS_ROOT', tmp_path)
        (tmp_path / 'forecast.joblib').write_bytes(b'v2')
        staged = tmp_path / '.staging' / '20261018T050000Z'
        staged.mkdir(parents=True)
        (staged / 'forecast.joblib').write_bytes(b'v2')

        class M:
            deployable = True

            def verify(self, config):
                pass

        monkeypatch.setattr(live_artifact, 'load_live', lambda config=None: M())
        monkeypatch.setattr(live_artifact, 'config_for_artifact', lambda c, m, mode: c)
        args = argparse.Namespace(model=None, mode='live', require_gates=False,
                                  force=False)
        # `_load_artifact` is the path for both startup and `reload_if_changed`.
        model, _ = live_artifact._load_artifact(args, DEFAULT_CONFIG)
        assert model.version == '20261018T050000Z'


class TestPredictionId:
    def test_tickets_and_positions_point_at_their_prediction(self, tmp_path, monkeypatch):
        writer, _ = run(tmp_path, monkeypatch, Venue())
        with writer._session() as session:            # noqa: SLF001
            ids = {p.symbol: p.id for p in session.query(Prediction)}
            tickets = {t.symbol: t.prediction_id for t in session.query(OrderTicket)}
            positions = {p.symbol: p.prediction_id for p in session.query(Position)}
        assert None not in ids.values()
        assert tickets == ids and positions == ids

    def test_a_resend_keeps_the_same_prediction(self, tmp_path, monkeypatch):
        venue = fx.TestOneResendAfterAKill.KillOnce(fresh_ask=0.61)
        writer, _ = run(tmp_path, monkeypatch, venue)
        with writer._session() as session:            # noqa: SLF001
            prediction = session.query(Prediction).filter_by(symbol='BTC-USD').one()
            position = session.query(Position).filter_by(symbol='BTC-USD').one()
            ticket = session.query(OrderTicket).filter_by(symbol='BTC-USD').one()
        assert position.prediction_id == prediction.id == ticket.prediction_id
        assert prediction.price == pytest.approx(0.61), "the resend's price is recorded"

    def test_act_on_passes_it_through(self, tmp_path):
        from core.pg_writer import PgWriter
        from tests.test_act_on import FakeKalshi, args, decision

        writer = PgWriter(database_url=f'sqlite:///{tmp_path}/s.db')
        writer.ensure_account(100.0, mode='live')
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'x'})
        assert asyncio.run(live.act_on(args(), writer, kalshi, decision(), None,
                                       prediction_id=41)) is True
        assert writer.open_positions()[0].prediction_id == 41


class TestQuoteAgeOnTheArchive:
    def _row(self, quote, decided_at):
        captured = []

        class Store:
            def write(self, table, frame, **kw):
                captured.append(frame)

        scored = pd.DataFrame({'symbol': ['BTC-USD']})
        with mock.patch('core.datastore.ResearchStore', lambda *a, **k: Store()):
            asyncio.run(live._record_touch(
                scored, {'BTC-USD': (quote, quote.ticker)},
                pd.Timestamp('2026-10-14 12:00', tz='UTC'), 12, DEFAULT_CONFIG,
                None, decided_at=decided_at))
        return captured[0].iloc[0]

    def _quote(self, source, book_time):
        return dataclasses.replace(
            make_quote('BTC-USD', source=source), book_time=book_time,
            yes_levels=((0.59, 100.0),), no_levels=((0.40, 50.0),))

    def test_a_stream_quote_records_its_age_and_source(self):
        at = time.time() - 0.4
        row = self._row(self._quote('stream', at),
                        pd.Timestamp(at + 1.25, unit='s', tz='UTC'))
        assert row['quote_age_seconds'] == pytest.approx(1.25, abs=1e-3)
        assert row['transport'] == 'stream' and row['source'] == 'live_touch'

    def test_a_rest_quote_records_its_age_and_source(self):
        at = time.time() - 2.0
        row = self._row(self._quote('rest', at),
                        pd.Timestamp(at + 0.3, unit='s', tz='UTC'))
        assert row['quote_age_seconds'] == pytest.approx(0.3, abs=1e-3)
        assert row['transport'] == 'rest'

    def test_an_unknown_read_time_is_nan_never_zero(self):
        row = self._row(self._quote('rest', None), pd.Timestamp.now(tz='UTC'))
        assert np.isnan(row['quote_age_seconds'])

    def test_the_cycle_passes_the_decision_instant(self, tmp_path, monkeypatch):
        seen = []

        async def spy(scored, quotes, window_open, offset, config, kalshi, *,
                      decided_at=None):
            seen.append((decided_at, {s: q.book_time for s, (q, _) in quotes.items()}))

        monkeypatch.setattr(live, '_record_touch', spy)
        run(tmp_path, monkeypatch, Venue(), record_touch=True)
        decided_at, book_times = seen[0]
        assert decided_at is not None
        for at in book_times.values():
            age = live.quote_age_seconds(type('Q', (), {'book_time': at})(), decided_at)
            assert 0.0 <= age < 5.0

    def test_a_stream_quote_is_dated_by_the_connections_last_frame(self):
        from core.stream_book import BookCache
        from data_collection.kalshi_client import market_meta
        from data_collection.stream.base import BookEvent

        clock = {'now': time.time() - 1.5}
        cache = BookCache(now=lambda: clock['now'])
        cache.apply(BookEvent(venue='kalshi', market_ticker='T', kind='snapshot',
                              received=clock['now'], seq=1,
                              yes=[(0.6, 1.0)], no=[(0.3, 1.0)]))
        clock['now'] += 1.5
        quote = live.stream_quote(cache, market_meta(
            {'ticker': 'T', 'status': 'active'}), 2.0)
        assert time.time() - quote.book_time == pytest.approx(1.5, abs=0.2)
