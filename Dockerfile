# Production Dockerfile for Baseball Fatigue ELT Lakehouse Pipeline
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies inside container
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default action: run the complete cloud ELT pipeline
ENTRYPOINT ["python", "scripts/run_cloud_elt.py"]
