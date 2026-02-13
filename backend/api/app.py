from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endpoints.coins import router as coins_router
from endpoints.signals import router as signals_router
from endpoints.trade import router as trades_router
from endpoints.wallet import router as wallet_router

app = FastAPI(
    title="Trading History & Market API",
    description="API for trades, signals, and coin prices",
    version="0.2.0",
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


@app.get("/")
def root():
    return {"message": "Trading History & Market API is running"}
