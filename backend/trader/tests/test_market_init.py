"""Fitting the correction on top of the PRICE instead of on top of F(x/sigma).

`model_minus_market` fails at -0.00248: the model is a worse forecaster than the
quote it has to trade against. A baseline-initialised model spends its capacity
correcting a forecaster that is already behind the price, so "beat the baseline"
stops implying anything about whether the trade pays.

Initialised on the market the null inverts in the right direction: an untrained
model reproduces the PRICE, so `model_minus_market >= 0` unless the trees
actively hurt, and the residual being fitted is `logit(truth) - logit(price)` —
how the price is wrong, which is the quantity the money depends on.

This was tried once before as `refit_market_init` and overfit, its alpha
collapsing to 0.386. It needs a quote on EVERY training row to be honest, which
is what `--complete-cases` now guarantees and did not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.model import MARKET_LOGIT, attach_market_logit


def _table(market=(0.45, 0.30)):
    return pd.DataFrame({'market_probability': list(market),
                         'baseline_probability': [0.40, 0.40]})


def test_the_market_logit_is_attached():
    got = attach_market_logit(_table())
    assert MARKET_LOGIT in got.columns
    assert got[MARKET_LOGIT].iloc[0] == pytest.approx(np.log(0.45 / 0.55))


def test_a_row_without_a_quote_stays_nan_rather_than_borrowing_the_baseline():
    """A market-init model that silently fell back to the baseline would be a
    baseline-init model wearing the other one's provenance — invisible, because
    both produce identically well-formed numbers."""
    got = attach_market_logit(_table(market=(0.45, np.nan)))
    assert np.isfinite(got[MARKET_LOGIT].iloc[0])
    assert not np.isfinite(got[MARKET_LOGIT].iloc[1])


def test_a_table_with_no_quote_column_raises_rather_than_guessing():
    with pytest.raises(ValueError, match='market_probability'):
        attach_market_logit(pd.DataFrame({'baseline_probability': [0.4]}))


def test_the_config_accepts_only_the_two_sources():
    from core.config import Config
    with pytest.raises(Exception):
        Config(init_score_source='nonsense').validate()


def test_market_init_is_reachable_from_the_cli():
    """It exists in Config and in core.model, and was never wired to a flag —
    so the one fix for the gate that matters could not be run."""
    import argparse
    from scripts._common import add_data_arguments
    parser = add_data_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--init-score-source', 'market'])
    assert args.init_score_source == 'market'
