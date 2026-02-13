from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine
from endpoints.coins import router as coins_router
from endpoints.ops import router as ops_router
from endpoints.paper import router as paper_router
from endpoints.signals import router as signals_router
from endpoints.trade import router as trades_router
from endpoints.wallet import router as wallet_router
from models.base import Base

# Ensure model modules are imported so SQLAlchemy registers tables on Base.metadata.
from models import signals as _signals_models  # noqa: F401
from models import trade as _trade_models  # noqa: F401
from models import wallet as _wallet_models  # noqa: F401

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Trading History & Market API",
    description="API for trades, signals, and paper trading telemetry",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trades_router)
app.include_router(coins_router)
app.include_router(wallet_router)
app.include_router(signals_router)
app.include_router(ops_router)
app.include_router(paper_router)


@app.get("/")
def root():
    return {"message": "Trading History & Market API is running"}
