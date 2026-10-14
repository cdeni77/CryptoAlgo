"""Small helpers shared by the live modules."""

from __future__ import annotations

import asyncio


async def _db(fn, *args, **kwargs):
    """A synchronous store call, off the event loop."""
    return await asyncio.to_thread(fn, *args, **kwargs)
