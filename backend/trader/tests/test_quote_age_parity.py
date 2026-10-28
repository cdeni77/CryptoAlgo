"""Quote freshness is set by the environment, not by a flag, and it decides
which rows exist at all.

`Config.max_quote_age_seconds` reads `QUOTE_MAX_AGE_SECONDS`, defaulting to
30. Compose sets it to 5 for the trader and the retrain. There is no CLI
flag, so nothing on a command line reveals which regime a run used, and a
research script started from a shell silently runs at 30 -- a six-fold
looser tolerance that roughly doubles the rows carrying a usable quote.

Measured 2026-10-28: the same walk-forward configuration scored coverage
0.130 against 0.061, trades 9,401 against 4,210, realised_edge_lcb 2.613
against 0.256 and overconfidence_abs 0.00989 against 0.03029, purely from
this. Diffing the two runs' configs showed one differing key out of a
hundred. A whole day of offset and edge-gate arms measured a regime the
account does not run, and an edge-gate result that held in 8 of 10 months
at 30s held in 4 of 10 at 5.

These tests guard the two services agreeing. They cannot guard a shell, so
read `max_quote_age_seconds` out of a run's own report.json before
believing it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
COMPOSE = ROOT / 'docker-compose.yml'
VAR = 'QUOTE_MAX_AGE_SECONDS'


def _declared() -> list[str]:
    """Every `QUOTE_MAX_AGE_SECONDS=...` value in the compose file."""
    text = COMPOSE.read_text()
    return re.findall(rf'{VAR}=(?:\$\{{{VAR}:-)?([0-9.]+)', text)


@pytest.mark.skipif(not COMPOSE.exists(), reason='no local compose')
def test_the_trader_and_the_retrain_agree_on_quote_freshness():
    """A retrain grading fresher or staler rows than the account trades is
    measuring a different policy, and nothing on the command line says so."""
    values = _declared()
    assert len(values) >= 2, (
        f'expected {VAR} on both the trader and the retrain, found '
        f'{len(values)}: a service without it falls back to the code default '
        f'of 30 and silently grades a different sample')
    assert len(set(values)) == 1, (
        f'{VAR} differs between services: {values}. The gates would grade a '
        f'policy the account does not run.')


@pytest.mark.skipif(not COMPOSE.exists(), reason='no local compose')
def test_the_deployed_quote_age_is_not_the_code_default():
    """**Fails loudly if they are ever made equal by accident.**

    They differ on purpose today: compose says 5, the code says 30. The point
    of this test is that the difference stays VISIBLE. If someone aligns them,
    delete this test and the warning in the module docstring with it -- the
    trap is gone and the docstring would then be wrong.
    """
    from core.config import Config

    deployed = float(_declared()[0])
    default = float(Config().max_quote_age_seconds)
    import os
    if os.getenv(VAR):                      # running inside the container
        assert default == deployed
    else:
        assert default != deployed, (
            'the code default now matches compose; the shell/container trap '
            'is gone, so delete this test and the docstring warning')
