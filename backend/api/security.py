"""The API's cross-origin policy.

Every route is a read-only GET serving a local dashboard, so the one control
left here is which origins a browser may call it from.
"""

from __future__ import annotations

import os


def allowed_origins() -> list[str]:
    """CORS origins from `CORS_ALLOW_ORIGINS`, never `*`.

    `*` would let any page the browser has open call the API.
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
