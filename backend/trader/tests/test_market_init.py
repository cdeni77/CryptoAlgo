"""Tests for fitting the correction on the market logit instead of on F(x/sigma).

An untrained market-init model reproduces the price, and the fitted residual is
how the price is wrong. It needs a quote on every training row, which
`--complete-cases` guarantees.
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
    """A row without a quote stays NaN and never borrows the baseline logit."""
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
    """`--init-score-source market` is available on the CLI."""
    import argparse
    from scripts._common import add_data_arguments
    parser = add_data_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--init-score-source', 'market'])
    assert args.init_score_source == 'market'
