"""Which offsets may open a position — measured, not assumed.

Over 70 days and 19,339 symbol-windows, per-contract edge after the measured fee
and a 0.5c half-spread, one entry per (symbol, window), 1pp gate:

    earliest offset that clears (what ran)   0.040c   t=0.10
    +9m or +12m                              1.206c   t=2.68
    +12m only                                3.304c   t=5.98

The live loop books one position per symbol-window and takes whichever offset the
clock has passed, so the earliest one that clears wins and `already_entered`
locks out the rest. In production 250 of 277 settled entries — 90% — landed at
+3m, the weakest cell, and exactly one landed at +12m.

`decision_offsets` stays (3, 6, 9, 12): every offset must keep being *scored*,
because that is the sample `market_benchmark` and the retroactive forecast test
read. What narrows is which offsets may *enter*.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import Config
from core.decide import Reason, decide

W = pd.Timestamp('2026-01-01 00:00', tz='UTC')


def row(offset: int, **over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=offset,
                baseline_probability=0.88, model_probability=0.96)
    base.update(over)
    return base


def test_every_offset_is_still_scored():
    """Narrowing entries must not narrow the measurement sample."""
    assert Config().decision_offsets == (3, 6, 9, 12)


def test_the_research_default_trades_every_scored_offset():
    """None, not (12,). `scripts.evaluate` has to be able to measure every cell,
    so the narrowing is a deployment choice rather than a library default — see
    `scripts.live --entry-offsets`."""
    assert Config().entry_offsets is None


def test_an_offset_outside_the_entry_set_refuses_by_name():
    config = Config(entry_offsets=(12,))
    decision = decide(row(3), config, bankroll=100.0)
    assert decision.reason is Reason.OFFSET_NOT_TRADED
    assert decision.contracts == 0
    assert decision.stake == 0.0


def test_the_permitted_offset_still_trades():
    config = Config()
    decision = decide(row(12), config, bankroll=100.0)
    assert decision.traded, decision.reason
    assert decision.contracts >= 1


def test_the_entry_set_must_be_a_subset_of_the_scored_offsets():
    """An entry offset that is never scored can never fire — a silent no-trade."""
    with pytest.raises(ValueError, match='entry_offsets'):
        Config(decision_offsets=(3, 6), entry_offsets=(12,))


def test_widening_the_entry_set_is_one_line():
    """The conservative fallback from the same table: 1.206c against 3.304c."""
    config = Config(entry_offsets=(9, 12))
    assert decide(row(9), config, bankroll=100.0).reason is not Reason.OFFSET_NOT_TRADED
    assert decide(row(3), config, bankroll=100.0).reason is Reason.OFFSET_NOT_TRADED


def test_the_live_script_defaults_to_entering_only_at_the_offset_that_pays():
    """The narrowing lives at the deployment boundary, not in the library.

    `scripts.evaluate` must keep measuring every offset; only the trading loop
    restricts entries. Putting it on the CLI means `docker-compose.yml` shows the
    policy the account is actually running.
    """
    from scripts.live import build_parser

    args = build_parser().parse_args([])
    assert tuple(args.entry_offsets) == (12,)


def test_the_live_entry_offsets_can_be_widened_from_the_command_line():
    from scripts.live import build_parser

    args = build_parser().parse_args(['--entry-offsets', '9', '12'])
    assert tuple(args.entry_offsets) == (9, 12)


def test_the_live_config_carries_the_entry_offsets_it_was_given():
    """The flag must reach `decide()`, not just the namespace."""
    from scripts.live import build_parser, config_from_args

    config = config_from_args(build_parser().parse_args([]))
    assert config.entry_offsets == (12,)
