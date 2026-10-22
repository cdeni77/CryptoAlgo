import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from database import engine
from jsonsafe import SafeJSONResponse
from security import allowed_origins
from endpoints.collection import router as collection_router
from endpoints.serving import router as serving_router
from models.base import Base

# Imported for its side effect: `create_all` creates only tables whose module
# has been imported.
from models import serving as _serving_models  # noqa: F401

# Indexes and constraints added after a table first shipped. Must equal
# `backend/trader/core/pg_writer.py:MIGRATIONS`: the API and the trader own the
# same tables. `test_orm_parity.py` fails when the two lists diverge.
#
# Every statement is `CREATE INDEX IF NOT EXISTS`, which SQLite also accepts.
POSTGRES_MIGRATIONS = _serving_models.MIGRATIONS



# An arbitrary fixed key. Postgres advisory locks are global to the database,
# so nothing else in this deployment may use the same number.
_BOOTSTRAP_LOCK_KEY = 8_931_774_205_113


def run_migrations() -> None:
    """Apply the index migrations, on Postgres and SQLite alike.

    The advisory lock is Postgres-only: only there do several uvicorn workers
    race to create the same objects.
    """
    with engine.begin() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _BOOTSTRAP_LOCK_KEY},
            )
        for statement in POSTGRES_MIGRATIONS:
            connection.execute(text(statement))


def bootstrap_schema() -> None:
    """Create missing tables, then apply the migrations.

    Runs from the lifespan hook, not at import. On Postgres it holds a
    transaction-scoped advisory lock, because `create_all` is not atomic and
    several uvicorn workers run it at once.
    """
    if engine.dialect.name != "postgresql":
        Base.metadata.create_all(bind=engine)
        run_migrations()
        return

    with engine.begin() as connection:
        connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": _BOOTSTRAP_LOCK_KEY}
        )
        # Postgres DDL is transactional: all of it commits or none does, and
        # the lock releases on commit.
        Base.metadata.create_all(bind=connection)
        for statement in POSTGRES_MIGRATIONS:
            connection.execute(text(statement))


logger = logging.getLogger(__name__)

#: How long startup waits for the database before giving up.
#: A host reboot restarts every container at once through the restart policy,
#: and Postgres replays its write-ahead log before it accepts connections.
DATABASE_WAIT_SECONDS = float(os.getenv("DATABASE_WAIT_SECONDS", "120"))
DATABASE_WAIT_INTERVAL = 2.0


def wait_for_database(timeout_seconds: float = None,
                      interval_seconds: float = DATABASE_WAIT_INTERVAL) -> None:
    """Block until the database answers, then return.

    **A reboot does not honour `depends_on`.** Compose waits for the database
    to be healthy only when compose itself starts the stack; the restart
    policy that brings containers back after a reboot starts them all at once.
    Postgres then answers "the database system is starting up" until it has
    recovered.

    Treating that as fatal is what leaves nothing serving: the application
    exits, but uvicorn's reloader is the container's main process and keeps
    running, so the container stays `Up` with no application behind it and the
    proxy returns 502 until somebody restarts it by hand.

    Raises the last error once `timeout_seconds` is spent, so a database that
    is genuinely unreachable still fails loudly instead of hanging.
    """
    if timeout_seconds is None:
        timeout_seconds = DATABASE_WAIT_SECONDS
    deadline = time.monotonic() + float(timeout_seconds)
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            if attempt > 1:
                logger.info("the database answered on attempt %d", attempt)
            return
        except OperationalError as exc:
            if time.monotonic() >= deadline:
                logger.error("the database did not answer within %.0fs",
                             float(timeout_seconds))
                raise
            logger.warning("waiting for the database (attempt %d): %s",
                           attempt, str(exc).splitlines()[0][:120])
            time.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(_: FastAPI):
    wait_for_database()
    bootstrap_schema()
    yield




app = FastAPI(
    title="Quarter — 15-minute binary telemetry",
    description=(
        "Read-only telemetry for the barrier system: the current window state "
        "per symbol, the decision funnel, the paper account, and the "
        "reliability table. Every response distinguishes a measured value from "
        "an explicit null with a reason. Nothing here substitutes a plausible "
        "number for a missing one."
    ),
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
)

# Origins come from CORS_ALLOW_ORIGINS, with `*` filtered out.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def no_store(request, call_next):
    """Tell every cache this telemetry expires immediately.

    **A response with no cache headers is not an uncacheable response.** With no
    `Cache-Control`, `ETag` or `Last-Modified` -- which is what this API served
    -- a browser is free to apply HEURISTIC freshness and reuse the body without
    revalidating. The symptom is the one reported on 2026-10-18: a dashboard
    that polls on schedule, raises no error, and shows numbers that only move
    when the page is manually reloaded.

    Every route here is a live measurement of an account that changes each
    cycle, so there is no response worth caching for any length of time.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


app.include_router(serving_router)
# Data-collection health. Deliberately a separate router with no database
# dependency: it must still answer when the thing it reports on is broken.
app.include_router(collection_router)


@app.get("/")
def root():
    return {
        "service": "quarter",
        "what": "15-minute binary barrier system telemetry",
        "routes": ["/live", "/account", "/account/equity", "/predictions",
                   "/funnel", "/positions", "/model", "/model/history",
                   "/model/calibration", "/collection",
                   "/collection/freshness", "/collection/liveness"],
    }
