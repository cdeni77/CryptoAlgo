"""Orders go out only while the pinned model is the one loaded."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from core.config import Config
from core.decide import Reason, decide

W = pd.Timestamp('2026-10-15 12:00', tz='UTC')


def row(**over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=12,
                baseline_probability=0.50, model_probability=0.62,
                ask_up=0.52, ask_down=0.50, model_version='v2')
    base.update(over)
    return base


def test_the_pinned_model_trades():
    config = Config(require_model_version='v2')
    assert decide(row(), config, bankroll=500.0, require_quote=True).traded


def test_any_other_model_is_refused_by_name():
    config = Config(require_model_version='v2')
    for version in ('v1', None, float('nan')):
        decision = decide(row(model_version=version), config, bankroll=500.0,
                          require_quote=True)
        assert decision.reason is Reason.MODEL_NOT_PINNED
        assert decision.contracts == 0


def test_no_pin_means_no_check():
    assert decide(row(model_version='anything'), Config(), bankroll=500.0,
                  require_quote=True).traded


def test_the_flag_reaches_the_config():
    from scripts.live import build_parser, config_from_args

    args = build_parser().parse_args(['--require-model-version', 'v2'])
    config = config_from_args(args)
    assert config.require_model_version == 'v2'
    assert 'require_model_version' in config.cli_overrides


def test_the_dashboard_lists_every_refusal_reason():
    """The funnel drops any reason it does not list, so a new refusal would be
    invisible on the dashboard."""
    api = Path(__file__).resolve().parents[2] / 'api' / 'controllers' / 'serving.py'
    tree = ast.parse(api.read_text())
    funnel = next(node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, 'id', None) == 'FUNNEL_ORDER' for t in node.targets))
    listed = {element.value for element in funnel.elts}
    assert listed == {reason.value for reason in Reason}
