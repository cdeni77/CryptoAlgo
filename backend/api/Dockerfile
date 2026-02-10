# Use official Python slim image (smaller, secure, still has what we need)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for psycopg2 + others)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the app with uvicorn (production settings: multiple workers)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]