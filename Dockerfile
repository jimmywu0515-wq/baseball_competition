# Production Dockerfile for Baseball Fatigue System
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS build tools
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     curl     && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies inside container
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

# Default action: run Streamlit Coach Dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
