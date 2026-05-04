# ─── ArgusX — Production Dockerfile ──────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="ArgusX Team"
LABEL description="ArgusX LLM Security Firewall"
LABEL version="1.0.0"

# Security: run as non-root
RUN groupadd -r argusx && useradd -r -g argusx -d /app -s /sbin/nologin argusx

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .
COPY .env.example .env

# Model artifacts (copied from prototype)
COPY app/models/artifacts/ ./app/models/artifacts/

# Ownership
RUN chown -R argusx:argusx /app

USER argusx

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info", "--no-access-log"]
