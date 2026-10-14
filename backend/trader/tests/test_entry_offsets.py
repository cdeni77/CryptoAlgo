"""Tests for `entry_offsets`, which limits the offsets that may open a position.

Every offset is still scored; only entry is restricted. Live defaults to +12m.
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
    """Restricting entries does not restrict the scored offsets."""
    assert Config().decision_offsets == (3, 6, 9, 12)


def test_the_research_default_trades_every_scored_offset():
    """The library default is None, so `scripts.evaluate` can measure every offset."""
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
    """An entry offset that is never scored is refused, since it could never fire."""
    with pytest.raises(ValueError, match='entry_offsets'):
        Config(decision_offsets=(3, 6), entry_offsets=(12,))


def test_widening_the_entry_set_is_one_line():
    """The entry set can be widened, for example to (9, 12)."""
    config = Config(entry_offsets=(9, 12))
    assert decide(row(9), config, bankroll=100.0).reason is not Reason.OFFSET_NOT_TRADED
    assert decide(row(3), config, bankroll=100.0).reason is Reason.OFFSET_NOT_TRADED


def test_the_live_script_defaults_to_entering_only_at_the_offset_that_pays():
    """`scripts.live` defaults to entering only at +12m, so the deploy config shows the policy."""
    from scripts.live import build_parser

    args = build_parser().parse_args([])
    assert tuple(args.entry_offsets) == (12,)


def test_the_live_entry_offsets_can_be_widened_from_the_command_line():
    from scripts.live import build_parser

    args = build_parser().parse_args(['--entry-offsets', '9', '12'])
    assert tuple(args.entry_offsets) == (9, 12)


def test_the_live_config_carries_the_entry_offsets_it_was_given():
    """The flag reaches the Config that `decide()` reads."""
    from scripts.live import build_parser, config_from_args

    config = config_from_args(build_parser().parse_args([]))
    assert config.entry_offsets == (12,)


def test_a_refused_row_with_a_book_is_labelled_as_quoted():
    """A refused row with a real book is labelled `price_source='quote'`."""
    config = Config(entry_offsets=(12,))
    quoted = decide(row(3, ask_up=0.90, ask_down=0.12), config, bankroll=100.0)
    assert quoted.reason is Reason.OFFSET_NOT_TRADED
    assert quoted.price_source == 'quote'
    bare = decide(row(3), config, bankroll=100.0)
    assert bare.price_source == 'baseline'
