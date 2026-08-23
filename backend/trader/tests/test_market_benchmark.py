"""The only measurement that decides anything, so its arithmetic is pinned.

Everything `scripts/evaluate.py` reports is skill against `F(x/sigma)` — a formula
we wrote. Beating it says nothing about beating the price on offer: the audit
measured a model that knows the truth exactly earning +2219% against a
baseline-priced counterparty and **zero** against an informed one.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.pg_writer import PgWriter

REPO_TRADER = '/home/cdeni/Desktop/Personal/CryptoAlgo/CryptoAlgo/backend/trader'
NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def seed(writer: PgWriter, *, n: int, market_skill: float, model_skill: float,
         seed_value: int = 0) -> None:
    """Windows where truth is known, and market/model are each `skill` away from it.

    A larger `skill` means *closer* to the truth, so the better forecaster is the
    one whose probability sits nearer the realised frequency.
    """
    rng = np.random.default_rng(seed_value)
    truth = rng.uniform(0.15, 0.85, n)
    outcome = (rng.random(n) < truth).astype(int)
    market = np.clip(0.5 + market_skill * (truth - 0.5), 0.01, 0.99)
    model = np.clip(0.5 + model_skill * (truth - 0.5), 0.01, 0.99)
    for i in range(n):
        window = NOW - timedelta(minutes=15 * (i + 2))
        writer.write_prediction(
            symbol='BTC-USD', window_open=window,
            settle_time=window + timedelta(minutes=15), offset_minutes=3,
            decision_time=window + timedelta(minutes=3), strike=100.0,
            last_price=100.0, displacement=0.0, sigma_remaining=0.001, z_score=0.0,
            baseline_probability=0.5, model_probability=float(model[i]),
            market_probability=float(market[i]), market_ask_up=float(market[i]) + 0.005,
            market_ask_down=1 - float(market[i]) + 0.005, price_source='quote',
            reason='edge_below_gate', traded=False, side=None, price=None,
            effective_cost=None, edge=None, contracts=None, model_version=None)
        writer.set_window_outcome('BTC-USD', window, settled_up=bool(outcome[i]))


def run(url: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, '-m', 'scripts.market_benchmark', '--database-url', url],
        cwd=REPO_TRADER, capture_output=True, text=True)


def test_it_refuses_when_no_quote_has_ever_been_recorded(tmp_path):
    """Which is the state today, and it must say so rather than print zeros."""
    url = f'sqlite:///{tmp_path}/empty.db'
    PgWriter(database_url=url)
    result = run(url)
    assert result.returncode == 1
    assert 'No settled window' in result.stdout
    assert 'dry-run' in result.stdout


def test_a_better_model_reads_positive_against_the_market(tmp_path):
    url = f'sqlite:///{tmp_path}/better.db'
    writer = PgWriter(database_url=url)
    seed(writer, n=300, market_skill=0.5, model_skill=0.9)
    result = run(url)
    line = next(l for l in result.stdout.splitlines() if l.strip().startswith('all'))
    model_minus_market = float(line.split()[-2])
    assert model_minus_market > 0, (
        f'the model is closer to the truth than the market and scored '
        f'{model_minus_market:+.6f}\n{result.stdout}'
    )


def test_a_worse_model_reads_negative_against_the_market(tmp_path):
    """The case that matters. A model that beats the formula and loses to the
    price must read negative here, because that is the situation the whole
    backtest apparatus cannot detect."""
    url = f'sqlite:///{tmp_path}/worse.db'
    writer = PgWriter(database_url=url)
    seed(writer, n=300, market_skill=0.9, model_skill=0.4)
    result = run(url)
    line = next(l for l in result.stdout.splitlines() if l.strip().startswith('all'))
    model_minus_market = float(line.split()[-2])
    assert model_minus_market < 0, (
        f'the market is closer to the truth and the model still scored '
        f'{model_minus_market:+.6f}\n{result.stdout}'
    )


def test_a_small_sample_is_not_a_conclusion(tmp_path):
    """300 windows cannot settle a 0.5pp question, and it must exit non-zero."""
    url = f'sqlite:///{tmp_path}/small.db'
    writer = PgWriter(database_url=url)
    seed(writer, n=300, market_skill=0.5, model_skill=0.9)
    result = run(url)
    assert result.returncode == 1, 'a 300-window sample was reported as a conclusion'
    assert 'is under the' in result.stdout
