FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run full pipeline to populate data lakehouse if not present
RUN python scripts/run_full_pipeline.py

EXPOSE 8080

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
