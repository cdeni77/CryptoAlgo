# ============================================================================
# Crypto Trading Bot — ML Pipeline + Live Signals
# ============================================================================
FROM python:3.12-slim

WORKDIR /app

# System deps for lightgbm, scipy, and aiohttp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install in two layers for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full trader codebase
COPY . .

# Create data and model directories
RUN mkdir -p /app/data /app/data/features /app/models /app/logs

# Volume mount points for persistent data
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Default entrypoint — overridden by docker-compose per service mode
ENTRYPOINT ["python"]
CMD ["--help"]