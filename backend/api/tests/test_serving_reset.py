"""The dashboard epoch must reach every serving query.

`account.reset_at` filters what the dashboard shows and deletes nothing. These
tests assert the shared helper, not any one query.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from controllers import serving

MODELS = ('Prediction', 'Position', 'EquityPoint', 'OrderTicket',
          'VenueSettlement', 'VenueFill', 'VenueBalance')

SOURCE = pathlib.Path(inspect.getfile(serving)).read_text()


def test_every_time_series_query_goes_through_the_helper():
    """A bare model select would keep showing pre-reset rows after a reset."""
    bare = []
    for line_no, line in enumerate(SOURCE.split('\n'), 1):
        for model in MODELS:
            if f'select({model}' in line and f'since_reset(select({model}' not in line:
                bare.append((line_no, model, line.strip()[:70]))
    assert not bare, f'unfiltered serving queries: {bare}'


def test_every_filtering_function_resolves_the_epoch():
    """Filter with an epoch you resolved, or with one you were handed.

    A helper that takes `epoch` as an argument satisfies this: the caller
    resolved it. What the guard catches is a function that filters while
    deciding the scope for itself, or not deciding it at all.
    """
    tree = ast.parse(SOURCE)
    lines = SOURCE.split(chr(10))
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != 'since_reset':
            body = chr(10).join(lines[node.lineno - 1:node.end_lineno])
            takes = [a.arg for a in node.args.args + node.args.kwonlyargs]
            if ('since_reset(' in body
                    and 'epoch = reset_epoch' not in body
                    and 'epoch' not in takes):
                missing.append(node.name)
    assert not missing, f'uses since_reset without an epoch: {missing}'


def test_every_model_has_a_named_time_column():
    """Each model names its time column, so rows filter by when they happened."""
    for model in MODELS:
        assert model in serving.RESET_COLUMN


def test_no_epoch_means_no_filter():
    from sqlalchemy import select

    from models.serving import Prediction

    query = select(Prediction)
    assert serving.since_reset(query, Prediction, None) is query


def test_an_epoch_adds_a_bound():
    import datetime as dt

    from sqlalchemy import select

    from models.serving import Prediction

    epoch = dt.datetime(2026, 9, 17, tzinfo=dt.timezone.utc)
    filtered = serving.since_reset(select(Prediction), Prediction, epoch)
    assert 'decision_time' in str(filtered)
    assert filtered is not select(Prediction)


def test_every_all_time_controller_actually_lifts_the_epoch():
    """`all_time` that is declared and never read is the defect this catches.

    The Account page serves the account of record, so it asks for the whole
    ledger. A controller that accepts the flag and still filters returns the
    epoch-scoped answer under an all-time heading, and the two scopes then show
    the same number with nothing saying why.
    """
    tree = ast.parse(SOURCE)
    lines = SOURCE.split('\n')
    inert = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = [a.arg for a in node.args.args + node.args.kwonlyargs]
        if 'all_time' not in names:
            continue
        body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        # The lift is the only correct use: resolve the epoch, then discard it.
        if 'epoch = None' not in body:
            inert.append(node.name)
    assert not inert, f'all_time declared but never lifts the epoch: {inert}'


def test_every_epoch_scoped_endpoint_forwards_all_time():
    """A controller that takes the flag is useless if its route drops it."""
    endpoints = pathlib.Path(
        inspect.getfile(serving)).parent.parent / 'endpoints' / 'serving.py'
    source = endpoints.read_text()
    tree = ast.parse(source)
    lines = source.split('\n')
    dropped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        call = [n for n in ('model_performance', 'equity_curve',
                            'venue_settlements', 'venue_daily_pnl',
                            'venue_account_state', 'venue_equity_curve')
                if f'serving.{n}(' in body]
        if call and 'all_time=all_time' not in body:
            dropped.append(node.name)
    assert not dropped, f'routes that drop all_time: {dropped}'


def test_the_calendar_is_measured_on_settlements_in_both_scopes():
    """Account value is not this strategy's result on a shared account.

    It counts whatever else the account holds: one day's balance moved -$25.64
    while the crypto markets settled -$134.10, the difference being a +$90.09
    position in another series. Across the ledger those series come to -$2,128
    and the deposits to $3,146. Both scopes therefore read `venue_settlements`,
    which holds the crypto series alone.

    Asserted on the source, because the query this guards against is
    Postgres-only and the SQLite harness cannot run it to find out.
    """
    tree = ast.parse(SOURCE)
    lines = SOURCE.split('\n')
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'venue_daily_pnl')
    body = '\n'.join(lines[fn.lineno - 1:fn.end_lineno])
    assert 'VenueBalance' not in body, (
        'the day cells must not be measured on the account balance')
    assert 'cash_change' not in body, (
        'the account-value column is gone; one measure, one label')
