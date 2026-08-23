"""The launch endpoint is the API's only way to start a process. Guard it.

Before this, `POST /jobs/{job}` had no authentication and the CORS
policy ended with `"*"`, so any page the browser had open could start a trader
script — with arbitrary arguments, in a container holding the exchange API keys.
The module name was constrained to a discovered script; nothing else was.

These tests hold the two properties that matter: the route fails closed with no
token configured, and an authenticated caller still cannot hand a script a path.
"""

from __future__ import annotations

import pytest

from security import (
    FLAG,
    MAX_ARGS,
    MAX_VALUE_LENGTH,
    VALUE,
    allowed_origins,
    validate_job_args,
)
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Failing closed
# ---------------------------------------------------------------------------


def test_launch_is_disabled_when_no_token_is_configured(client, clean_token):
    """503, not 200. The absence of a secret must not mean the absence of a lock.

    This is the direction that matters: a deployment that forgot to set the token
    should refuse to launch, not launch for anyone.
    """
    response = client.post('/jobs/scripts.baseline', json={'args': []})

    assert response.status_code == 503
    assert 'API_TOKEN' in response.json()['detail']


def test_launch_rejects_a_missing_token(client, with_token):
    response = client.post('/jobs/scripts.baseline', json={'args': []})

    assert response.status_code == 401


def test_launch_rejects_a_wrong_token(client, with_token):
    response = client.post(
        '/jobs/scripts.baseline',
        json={'args': []},
        headers={'X-API-Token': 'not-the-token'},
    )

    assert response.status_code == 401


def test_read_only_routes_stay_open(client, clean_token):
    """The dashboard polls these. Gating them buys nothing the origin policy does not."""
    assert client.get('/').status_code == 200
    assert client.get('/model/').status_code == 200


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_ordinary_arguments_pass():
    args = ['--venue', 'coinbase', '--symbols', 'BIP,ETP,SLP', '--periods', '6',
            '--as-of', '2026-06-01T00:00:00Z', '--full']

    assert validate_job_args(args) == args


def test_filesystem_paths_are_refused():
    """An authenticated caller still should not choose where a script reads or writes."""
    with pytest.raises(HTTPException) as exc:
        validate_job_args(['--store', '/etc/passwd'])
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException):
        validate_job_args(['--store', '../../../root'])


@pytest.mark.parametrize('bad', [
    '--venue; rm -rf /',      # shell metacharacters
    '--venue coinbase',       # embedded space, so two arguments smuggled as one
    '-v',                     # short flag
    '--VENUE',                # not the long lowercase form
    '$(whoami)',
    '`id`',
    'value with spaces',
])
def test_shell_shaped_arguments_are_refused(bad):
    with pytest.raises(HTTPException) as exc:
        validate_job_args([bad])
    assert exc.value.status_code == 400


def test_too_many_arguments_are_refused():
    with pytest.raises(HTTPException):
        validate_job_args(['--x'] * (MAX_ARGS + 1))


def test_an_overlong_argument_is_refused():
    with pytest.raises(HTTPException):
        validate_job_args(['a' * (MAX_VALUE_LENGTH + 1)])


def test_validation_rejects_rather_than_sanitising():
    """A silently stripped argument means the job ran with different settings.

    A research run whose parameters are not what the requester believes is worse
    than one that failed to start, because the result looks legitimate.
    """
    with pytest.raises(HTTPException):
        validate_job_args(['--venue', 'coinbase', '/etc/shadow'])


def test_blank_arguments_are_dropped_not_rejected():
    """Empty strings come from form inputs and carry no instruction."""
    assert validate_job_args(['--venue', '', 'coinbase', '  ']) == ['--venue', 'coinbase']


def test_flag_and_value_patterns_agree_with_the_examples_in_the_docstring():
    assert FLAG.match('--walk-forward-periods')
    assert FLAG.match('--venue')
    assert not FLAG.match('--Venue')
    assert not FLAG.match('-v')
    assert VALUE.match('BIP,ETP')
    assert VALUE.match('2026-06-01T00:00:00Z'), 'colons must survive the value pattern'
    assert not VALUE.match('-leading-dash')


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_wildcard_origin_is_filtered_out(monkeypatch):
    """`*` made every other entry in the list decorative."""
    monkeypatch.setenv('CORS_ALLOW_ORIGINS', 'http://localhost:3000,*')

    origins = allowed_origins()

    assert '*' not in origins
    assert origins == ['http://localhost:3000']


def test_an_all_wildcard_list_falls_back_to_localhost(monkeypatch):
    monkeypatch.setenv('CORS_ALLOW_ORIGINS', '*')

    assert allowed_origins() == ['http://localhost:3000']


def test_unset_origins_default_to_local_development(monkeypatch):
    monkeypatch.delenv('CORS_ALLOW_ORIGINS', raising=False)

    origins = allowed_origins()

    assert '*' not in origins
    assert all(o.startswith('http://localhost') or o.startswith('http://127.0.0.1')
               for o in origins)


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


def test_a_non_ascii_token_is_rejected_not_a_server_error(monkeypatch):
    """`hmac.compare_digest` raises TypeError on non-ASCII strings.

    Starlette decodes headers as latin-1, so a crafted header reached the compare
    as a str with characters outside ASCII and turned the auth check into an
    unhandled 500 rather than a 401. Comparing bytes fixes both directions — a
    non-ASCII `API_TOKEN` previously broke every request, authorised or not.
    """
    from fastapi import HTTPException

    from security import require_token

    monkeypatch.setenv('API_TOKEN', 'correct-horse')

    with pytest.raises(HTTPException) as raised:
        require_token(x_api_token='wrong-ÿ-token')

    assert raised.value.status_code == 401


def test_a_non_ascii_configured_token_is_refused_with_a_reason(monkeypatch):
    """A token no client can transmit must not fail every request mysteriously.

    HTTP header values are octets, and httpx refuses non-ASCII outright — so a
    non-ASCII `API_TOKEN` cannot be sent at all. It used to 500 inside
    `hmac.compare_digest`; then it 401'd, which is honest about the outcome but
    silent about the cause. 503 with the reason is the same treatment an unset
    token gets.
    """
    from fastapi import HTTPException

    from security import require_token

    monkeypatch.setenv('API_TOKEN', 'sécret-ključ')

    with pytest.raises(HTTPException) as raised:
        require_token(x_api_token='anything')

    assert raised.value.status_code == 503
    assert 'non-ASCII' in raised.value.detail


def test_an_ascii_token_still_authorises(monkeypatch):
    from security import require_token

    monkeypatch.setenv('API_TOKEN', 'correct-horse-battery-staple')

    require_token(x_api_token='correct-horse-battery-staple')  # must not raise


def test_the_comparison_is_constant_time():
    """Guard the mechanism behaviourally, not by grepping the source.

    The version of this test did `inspect.getsource(require_token)` and asserted
    `'compare_digest' in source` — but `getsource` includes the docstring and
    comments, both of which name `compare_digest`. Replacing the comparison with
    a plain `!=` left the prose intact and all 24 tests passing: the guard could
    not distinguish the mechanism from the sentence describing it.

    So check the bytecode instead, which contains only what executes.
    """
    import dis

    import security

    called = {
        instruction.argval
        for instruction in dis.get_instructions(security.require_token)
        if instruction.opname in ('LOAD_GLOBAL', 'LOAD_ATTR', 'LOAD_METHOD')
    }

    assert 'compare_digest' in called, (
        'require_token does not call compare_digest; a plain == leaks timing'
    )
