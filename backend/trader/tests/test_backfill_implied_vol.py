"""Which endpoint serves which era, and what a wrong ladder looks like."""

from __future__ import annotations

import asyncio
import datetime as dt

from scripts import backfill_implied_vol as biv


class TestBatchCandleFallback:
    """The recent era is served ONLY by the plural `/markets/candlesticks`.

    The two per-ticker endpoints tile and then stop: historical ends
    2026-08-21, and the live per-ticker path returns nothing for a 1-hour
    strike market at any date. Live IV recording starts 2026-08-26, so
    without this fallback the rebuild stops five days short of the live era
    and there is no overlap in which to validate it against what we record.
    """

    def test_falls_through_to_the_batch_endpoint(self):
        calls = []

        class Client:
            historical_base_url = 'https://historical.example'

            async def _request(self, method, path, params=None, base=None):
                calls.append(path)
                if path == '/markets/candlesticks':
                    return {'markets': [{'ticker': 'KX-T1',
                                         'candlesticks': [{'end_period_ts': 7}]}]}
                return {'candlesticks': []}

        start = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc)
        got = asyncio.run(biv._candles(Client(), 'KX-T1', start,
                                       start + dt.timedelta(hours=1)))
        assert got == [{'end_period_ts': 7}]
        assert calls[-1] == '/markets/candlesticks'

    def test_ignores_candles_belonging_to_another_ticker(self):
        """The batch endpoint is keyed on a LIST, so a response may carry
        markets we did not ask about. Returning the first block regardless
        would silently fit one strike's quotes onto another's strike price —
        a wrong ladder that raises nothing and fits fine."""

        class Client:
            historical_base_url = 'https://historical.example'

            async def _request(self, method, path, params=None, base=None):
                if path == '/markets/candlesticks':
                    return {'markets': [{'ticker': 'KX-OTHER',
                                         'candlesticks': [{'end_period_ts': 7}]}]}
                return {'candlesticks': []}

        start = dt.datetime(2026, 9, 10, tzinfo=dt.timezone.utc)
        assert asyncio.run(biv._candles(Client(), 'KX-T1', start,
                                        start + dt.timedelta(hours=1))) == []


class TestNoDeadFallback:
    def test_the_live_per_ticker_path_is_never_requested(self):
        """It 404s for every 1-hour strike market, so trying it is one
        guaranteed-failed request per strike — ~30% of the whole backfill."""
        paths = []

        class Client:
            historical_base_url = 'https://historical.example'

            async def _request(self, method, path, params=None, base=None):
                paths.append(path)
                if path == '/markets/candlesticks':
                    return {'markets': [{'ticker': 'KX-T1', 'candlesticks': [{'x': 1}]}]}
                return {'candlesticks': []}

        start = dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc)
        asyncio.run(biv._candles(Client(), 'KX-T1', start,
                                 start + dt.timedelta(hours=1)))
        assert '/markets/KX-T1/candlesticks' not in paths
        assert paths == ['/historical/markets/KX-T1/candlesticks',
                         '/markets/candlesticks']

    def test_giving_up_on_429_is_counted_not_silent(self):
        """A throttled request returns [], which reads as 'never traded'."""

        class Client:
            historical_base_url = 'https://historical.example'

            async def _request(self, method, path, params=None, base=None):
                if path == '/markets/candlesticks':
                    return {'markets': []}
                raise RuntimeError('GET ... -> 429: too_many_requests')

        biv.THROTTLED.clear()
        start = dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc)
        assert asyncio.run(biv._candles(Client(), 'KX-T1', start,
                                        start + dt.timedelta(hours=1))) == []
        assert biv.THROTTLED['candles'] == 1


class TestAvailableTimeIsThePointInTimeField:
    """`available_time` answers "when could a decision have seen this".

    `core/book_features.py` joins fits with a backward `merge_asof` on it, so
    stamping it with the moment the backfill RAN puts every row after every
    historical decision: nothing matches, and the feature is silently always
    NaN. Provenance belongs in `source`.
    """

    def test_it_equals_the_fit_instant_not_the_fetch_time(self):
        import inspect
        src = inspect.getsource(biv.hour_rows)
        assert "row['available_time'] = pd.Timestamp(at)" in src
        assert "row['available_time'] = fetched_at" not in src

    def test_provenance_is_carried_by_source(self):
        import inspect
        assert "row['source'] = 'ladder_candles'" in inspect.getsource(biv.hour_rows)
