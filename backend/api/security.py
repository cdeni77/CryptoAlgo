"""Authentication and argument validation for the mutating API surface.

The API exposes one endpoint that starts a process — `POST /research/launch/{job}`
— in a container that holds the Coinbase API keys. Before this module, that
endpoint had no authentication and the CORS policy allowed every origin, so any
web page the browser happened to visit could start a trader script with arbitrary
arguments. The module name was constrained to a discovered script, but the
argument list was passed through verbatim.

Two independent controls, because either alone leaves a hole:

* `require_token` gates every mutating route on a shared secret, and **fails
  closed**: with no `API_TOKEN` configured, mutating routes return 503 rather
  than running unauthenticated. A default-open deployment is the failure mode
  worth designing against — nobody notices it until someone else does.
* `validate_job_args` restricts what can be passed to a launched script, because
  authentication says *who* may launch, not *what*. Even an authenticated caller
  should not be able to hand a research script an arbitrary filesystem path.

Read-only routes are deliberately left open. They serve a local dashboard, expose
no credentials, and gating them would break the frontend's polling for no gain
that the origin policy does not already provide.
"""

from __future__ import annotations

import hmac
import os
import re
from typing import Iterable, Optional

from fastapi import Header, HTTPException, status

TOKEN_ENV = 'API_TOKEN'
TOKEN_HEADER = 'X-API-Token'

# A flag: long form only. Short flags are ambiguous across scripts and there is
# no reason a dashboard needs them.
FLAG = re.compile(r'^--[a-z][a-z0-9]*(-[a-z0-9]+)*$')

# A value: alphanumerics and the punctuation that appears in real arguments —
# symbol lists, ISO timestamps, decimals, config filenames. No spaces, no shell
# metacharacters, no leading dash (which would smuggle in an unvetted flag).
VALUE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_,.:=+-]*$')

MAX_ARGS = 24
MAX_VALUE_LENGTH = 120


def token_configured() -> bool:
    return bool(os.getenv(TOKEN_ENV, '').strip())


def require_token(x_api_token: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: reject a mutating request without the shared secret.

    Compared with `hmac.compare_digest` rather than `==`, so a wrong token does
    not leak its correct prefix through response timing.
    """
    expected = os.getenv(TOKEN_ENV, '').strip()
    if expected and not expected.isascii():
        # No conforming client can send this. HTTP header values are octets and
        # httpx (among others) refuses non-ASCII outright, so a non-ASCII token
        # locks out its own holder — previously with a 500 from
        # `hmac.compare_digest`, then with a 401 that gave no clue why. Say what
        # is wrong instead of failing every request mysteriously.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f'{TOKEN_ENV} contains non-ASCII characters and cannot be sent '
                f'in an HTTP header. Set it to an ASCII value.'
            ),
        )
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f'This endpoint starts a process and is disabled until '
                f'{TOKEN_ENV} is set. Set it in the environment and send the '
                f'same value in the {TOKEN_HEADER} header.'
            ),
        )
    # Compare bytes, not str: `hmac.compare_digest` raises TypeError on strings
    # with non-ASCII characters, and Starlette decodes headers as latin-1, so a
    # crafted header turned the auth check into an unhandled 500.
    #
    # The header must be re-encoded with latin-1, the codec it was decoded with,
    # to recover the bytes the client actually sent. Encoding it as UTF-8 instead
    # produced mojibake, so a non-ASCII `API_TOKEN` still rejected its own
    # correct token — 500 became 401, which is better but still wrong.
    # Compare bytes, not str: `hmac.compare_digest` raises TypeError on strings
    # with non-ASCII characters, and Starlette decodes headers as latin-1, so a
    # crafted header turned the auth check into an unhandled 500 rather than a
    # 401. latin-1 round-trips every byte a header can carry.
    header = (x_api_token or '').strip()
    supplied = header.encode('latin-1', errors='replace')
    if not header or not hmac.compare_digest(supplied, expected.encode('latin-1')):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'Missing or invalid {TOKEN_HEADER}',
            headers={'WWW-Authenticate': TOKEN_HEADER},
        )


def validate_job_args(args: Optional[Iterable[str]]) -> list[str]:
    """Return the argument list, or raise 400 naming the offending argument.

    Rejecting rather than sanitising is deliberate. A silently stripped argument
    means the job ran with different settings than the caller asked for, and a
    research run whose parameters are not what the requester believes is worse
    than one that failed to start.
    """
    if not args:
        return []

    cleaned = [str(a).strip() for a in args if str(a).strip()]
    if len(cleaned) > MAX_ARGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Too many arguments ({len(cleaned)} > {MAX_ARGS})',
        )

    for arg in cleaned:
        if len(arg) > MAX_VALUE_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Argument longer than {MAX_VALUE_LENGTH} characters',
            )
        if arg.startswith('-'):
            if not FLAG.match(arg):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f'Not a valid flag: {arg!r}. Expected long form, '
                        f'lowercase, e.g. --venue or --walk-forward-periods.'
                    ),
                )
            continue
        if not VALUE.match(arg):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f'Not a valid value: {arg!r}. Allowed: letters, digits, and '
                    f'_ , . : = + - starting with a letter or digit.'
                ),
            )
        # Path traversal has no legitimate use here: the scripts resolve their
        # own store and config locations.
        if '..' in arg or arg.startswith('/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f'Filesystem paths are not accepted as job arguments: {arg!r}',
            )

    return cleaned


def allowed_origins() -> list[str]:
    """CORS origins, from the environment, never `*`.

    `*` with `allow_credentials=True` is rejected by browsers anyway, but the
    real problem was that its presence in the list made every other entry
    decorative — the API accepted requests from any page the browser had open.
    """
    raw = os.getenv('CORS_ALLOW_ORIGINS', '').strip()
    if not raw:
        return [
            'http://localhost:3000',
            'http://localhost:5173',
            'http://127.0.0.1:3000',
            'http://127.0.0.1:5173',
        ]
    origins = [o.strip() for o in raw.split(',') if o.strip() and o.strip() != '*']
    return origins or ['http://localhost:3000']
