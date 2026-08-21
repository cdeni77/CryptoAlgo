import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from security import TOKEN_ENV, allowed_origins, token_configured
from endpoints.coins import router as coins_router
from endpoints.model import router as model_router
from endpoints.paper import router as paper_router
from endpoints.research import router as research_router
from endpoints.signals import router as signals_router
from endpoints.wallet import router as wallet_router
from models.base import Base

# Ensure model modules are imported so SQLAlchemy registers tables on Base.metadata.
from models import signals as _signals_models 
from models import trade as _trade_models  
from models import wallet as _wallet_models  

# Columns added after the tables were first created. `create_all` above only
# creates missing *tables*, so a new column on an existing table needs this.
#
# Kept in step with backend/trader/core/pg_writer.py:_run_pg_migrations — the two
# processes own the same tables from separate containers, so a column added on one
# side has to exist on the other or whichever starts second writes to a column the
# first cannot read. `backend/trader/tests/test_orm_parity.py` fails when the two
# lists diverge.
POSTGRES_MIGRATIONS = (
    "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION",
    "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS sl_price DOUBLE PRECISION",
    "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS max_hold_until TIMESTAMP WITH TIME ZONE",
    "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS exit_reason VARCHAR",
    "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS funding_paid "
    "DOUBLE PRECISION NOT NULL DEFAULT 0.0",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS expected_net_bps DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS expected_price_bps DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS expected_carry_bps DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS cost_bps DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS sigma_bps DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS edge_to_risk DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS carry_share DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS participation DOUBLE PRECISION",
    "ALTER TABLE signals ADD COLUMN IF NOT EXISTS model_version VARCHAR",
    # Indexes, not columns. `create_all` only touches missing *tables*, so
    # an index added to an existing model never reaches a database that
    # already has the table. The names match what SQLAlchemy's
    # `index=True` generates (ix_<table>_<column>), so a fresh create and
    # an upgraded database end up with the same index rather than two.
    "CREATE INDEX IF NOT EXISTS ix_paper_fills_created_at "
    "ON paper_fills (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_paper_positions_is_open "
    "ON paper_positions (is_open)",
    "CREATE INDEX IF NOT EXISTS ix_paper_positions_opened_at "
    "ON paper_positions (opened_at)",
)


# An arbitrary but fixed key. Postgres advisory locks are global to the
# database, so the only requirement is that nothing else in this deployment
# picks the same number.
_BOOTSTRAP_LOCK_KEY = 8_931_774_205_113


def run_migrations() -> None:
    """Apply the additive column migrations. Postgres only, by construction.

    `ADD COLUMN IF NOT EXISTS` is Postgres syntax, so this checks the dialect
    rather than catching the failure. Guarding beats swallowing: a caught
    exception here would also hide a genuine migration failure on the deployment
    that needs these to run.
    """
    if engine.dialect.name != "postgresql":
        logging.getLogger("api").info(
            "skipping column migrations on %s: they are Postgres-specific",
            engine.dialect.name,
        )
        return
    with engine.begin() as connection:
        connection.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": _BOOTSTRAP_LOCK_KEY}
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
    title="Trading History & Market API",
    description="API for signals, paper trading telemetry, and research",
    version="0.3.0",
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
        "%s is not set: POST /research/launch is disabled. Set it to enable "
        "launching research jobs from the dashboard.",
        TOKEN_ENV,
    )

app.include_router(coins_router)
app.include_router(wallet_router)
app.include_router(signals_router)
app.include_router(paper_router)
app.include_router(research_router)
app.include_router(model_router)


@app.get("/")
def root():
    return {"message": "Trading History & Market API is running"}
