FROM python:3.12-slim

# Set work dir
WORKDIR /app

# Install deps (from your env—add if more)
RUN pip install --no-cache-dir pandas pandas_ta yfinance requests numpy logging joblib torch optuna coinbase-rest-client ratelimit tqdm

# Copy your scripts/files
COPY . /app

# Env vars (API keys—set at run or via .env)
ENV COINBASE_API_KEY=your_key
ENV COINBASE_API_SECRET=your_secret

# Run paper_trading.py by default
CMD ["python", "paper_trading.py"]