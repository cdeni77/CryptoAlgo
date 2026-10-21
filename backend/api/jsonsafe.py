"""NaN-safe JSON rendering, in its own module so it can be tested without a DB.

Kept out of `app.py` because importing that module requires `DATABASE_URL` and
builds the engine, while these two are pure functions of their input. A test
that has to stand up a database to check that NaN becomes null is a test that
will be skipped.
"""

from __future__ import annotations

import math
from typing import Any

from fastapi.responses import JSONResponse

def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    **A NaN anywhere in a payload used to 500 the entire endpoint.** FastAPI's
    default encoder is `json.dumps(..., allow_nan=False)`, so one bad row
    raises `ValueError: Out of range float values are not JSON compliant` and
    the response is lost -- not the row, the whole response. Measured
    2026-10-18: 74 of 492 predictions carried `model_probability = NaN`, so
    `/predictions?limit=250` -- exactly what the Decisions page asks for --
    returned 500 while `limit=200` returned 200. The dashboard page was blank
    and the cause was invisible from the client.

    `null` rather than the `NaN` literal that `allow_nan=True` would emit:
    `NaN` is not valid JSON and `JSON.parse` rejects it, so that would move the
    failure from the server to the browser rather than fix it.

    This does not hide anything. Those rows already carry `reason` --
    `not_finite` in every one of the 74 -- so the surface still says why the
    number is absent, which is the standing rule here: a missing value is null
    with a reason, never a zero and never a substitute.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class SafeJSONResponse(JSONResponse):
    """The app-wide response class, so no endpoint can 500 on a NaN.

    Applied globally rather than at the one payload builder that was observed
    failing: every float on every surface comes from the same research store,
    and `model_probability` is not special. A fix applied only where the bug
    was found leaves the next one to be discovered by a blank page.
    """

    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))

