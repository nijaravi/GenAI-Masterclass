# ============================================================
# Section 9 — Chapter 4: Dockerfile
# GenAI Decoded by Nij
#
# Build:  docker build -t techstore-api .
# Run:    docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... techstore-api
# ============================================================

# Slim Python base — smaller image, faster pulls
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first (Docker caches this layer)
# As long as requirements.txt doesn't change, this layer is reused on rebuild
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the port FastAPI/uvicorn listens on
EXPOSE 8000

# Run the production app with uvicorn (ASGI server)
# --host 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "main_production:app", "--host", "0.0.0.0", "--port", "8000"]

# ── To use the minimal app instead, change CMD to: ────────────
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Docker healthcheck — used by Railway, ECS, etc. ──────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
