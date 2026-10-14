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
    tree = ast.parse(SOURCE)
    lines = SOURCE.split('\n')
    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != 'since_reset':
            body = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            if 'since_reset(' in body and 'epoch = reset_epoch' not in body:
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
