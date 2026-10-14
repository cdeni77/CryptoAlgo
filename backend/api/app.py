from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from security import allowed_origins
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

# Origins come from CORS_ALLOW_ORIGINS, with `*` filtered out.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(serving_router)


@app.get("/")
def root():
    return {
        "service": "quarter",
        "what": "15-minute binary barrier system telemetry",
        "routes": ["/live", "/account", "/account/equity", "/predictions",
                   "/funnel", "/positions", "/model", "/model/history",
                   "/model/calibration"],
    }
