import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from database import engine
from security import TOKEN_ENV, allowed_origins, token_configured
from endpoints.coins import router as coins_router
from endpoints.paper import router as paper_router
from endpoints.research import router as research_router
from endpoints.signals import router as signals_router
from endpoints.trade import router as trades_router
from endpoints.wallet import router as wallet_router
from models.base import Base

# Ensure model modules are imported so SQLAlchemy registers tables on Base.metadata.
from models import signals as _signals_models 
from models import trade as _trade_models  
from models import wallet as _wallet_models  

Base.metadata.create_all(bind=engine)

# Idempotent column migrations for schema additions that post-date create_all().
with engine.begin() as _conn:
    # Kept in step with backend/trader/core/pg_writer.py:_run_pg_migrations —
    # the two processes own the same tables from separate containers, so a column
    # added on one side has to exist on the other or whichever starts second
    # writes to a column the first cannot read.
    for _stmt in (
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION",
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS sl_price DOUBLE PRECISION",
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS max_hold_until TIMESTAMP WITH TIME ZONE",
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS exit_reason VARCHAR",
        "ALTER TABLE paper_positions ADD COLUMN IF NOT EXISTS funding_paid "
        "DOUBLE PRECISION NOT NULL DEFAULT 0.0",
    ):
        _conn.execute(text(_stmt))

app = FastAPI(
    title="Trading History & Market API",
    description="API for trades, signals, and paper trading telemetry",
    version="0.3.0",
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

app.include_router(trades_router)
app.include_router(coins_router)
app.include_router(wallet_router)
app.include_router(signals_router)
app.include_router(paper_router)
app.include_router(research_router)


@app.get("/")
def root():
    return {"message": "Trading History & Market API is running"}
