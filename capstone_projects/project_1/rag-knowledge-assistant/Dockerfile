# Multi-stage build keeps the runtime image small and free of build toolchains.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first so the layer caches when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

ENV PYTHONPATH=/app/src

# Run as non-root — basic container hardening.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Healthcheck so the orchestrator can gate traffic on readiness.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
