import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from security import TOKEN_ENV, allowed_origins, token_configured
from endpoints.jobs import router as jobs_router
from endpoints.serving import router as serving_router
from models.base import Base

# Imported for its side effect: SQLAlchemy only knows about a table once the
# module defining it has been imported, so `create_all` below would otherwise
# create nothing.
from models import serving as _serving_models  # noqa: F401

# Columns added after the tables were first created. `create_all` above only
# creates missing *tables*, so a new column on an existing table needs this.
#
# Kept in step with backend/trader/core/pg_writer.py:_run_pg_migrations — the two
# processes own the same tables from separate containers, so a column added on one
# side has to exist on the other or whichever starts second writes to a column the
# first cannot read. `backend/trader/tests/test_orm_parity.py` fails when the two
# lists diverge.
# Indexes and constraints added after a table first shipped.
#
# Kept literally identical to `backend/trader/core/pg_writer.py:MIGRATIONS` — the
# two processes own the same tables from separate containers, so an index that
# exists on one side and not the other is a query that is fast in development and
# a table scan in production. `backend/trader/tests/test_orm_parity.py` fails when
# the two lists diverge.
#
# These are all `CREATE INDEX IF NOT EXISTS`, which SQLite also understands, so
# unlike the `ADD COLUMN IF NOT EXISTS` list this replaced they are exercised by
# the test suite rather than skipped by it.
POSTGRES_MIGRATIONS = _serving_models.MIGRATIONS



# An arbitrary but fixed key. Postgres advisory locks are global to the
# database, so the only requirement is that nothing else in this deployment
# picks the same number.
_BOOTSTRAP_LOCK_KEY = 8_931_774_205_113


def run_migrations() -> None:
    """Apply the index migrations.

    Every statement is `CREATE INDEX IF NOT EXISTS`, which both Postgres and
    SQLite understand, so this no longer checks the dialect and skips itself in
    tests — the migrations the test suite runs are the migrations production
    runs. The advisory lock below is still Postgres-only, because only Postgres
    has four uvicorn workers racing to create the same tables.
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
    """Create missing tables, then add the columns `create_all` will not.

    This used to run at module import, which meant all four uvicorn workers
    (see the Dockerfile CMD) ran it simultaneously against one database.
    `create_all` checks for each table and then creates it, and those two steps
    are not atomic: two workers both see a table missing, both issue CREATE
    TABLE, and the loser dies with DuplicateTable — at import, before FastAPI
    exists to log it, so the symptom is a worker that vanishes at boot.

    Two changes fix that. It runs from the lifespan hook rather than at import,
    so a failure is a startup error with a traceback in the right place and
    importing this module no longer requires a reachable database. And on
    Postgres it holds a transaction-scoped advisory lock, so the workers
    serialise and the ones that lose the race find the work already done.
    """
    if engine.dialect.name != "postgresql":
        Base.metadata.create_all(bind=engine)
        run_migrations()
        return

    with engine.begin() as connection:
        connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": _BOOTSTRAP_LOCK_KEY}
        )
        # Postgres DDL is transactional, so the create and the ALTERs land
        # together or not at all, and the lock is released on commit.
        Base.metadata.create_all(bind=connection)
        for statement in POSTGRES_MIGRATIONS:
            connection.execute(text(statement))


@asynccontextmanager
async def lifespan(_: FastAPI):
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
)

# Origins come from CORS_ALLOW_ORIGINS, and `*` is filtered out. The previous
# list ended with "*", which made every other entry decorative: the API accepted
# credentialed requests from any page the browser had open, including the one
# endpoint that starts a process.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Token"],
)

if not token_configured():
    logging.getLogger("api").warning(
        "%s is not set: POST /jobs/{job} is disabled and returns 503. Set it to "
        "enable launching research scripts from the dashboard.",
        TOKEN_ENV,
    )

app.include_router(serving_router)
app.include_router(jobs_router)


@app.get("/")
def root():
    return {
        "service": "quarter",
        "what": "15-minute binary barrier system telemetry",
        "routes": ["/live", "/account", "/account/equity", "/predictions",
                   "/funnel", "/positions", "/model", "/model/history",
                   "/model/calibration", "/jobs"],
    }
