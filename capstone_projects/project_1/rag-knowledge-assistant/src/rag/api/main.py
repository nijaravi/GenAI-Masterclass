"""FastAPI microservice — the deployable unit (resume: containerised FastAPI).

Endpoints:
  GET  /healthz   liveness/readiness (no heavy deps touched)
  GET  /stats     corpus size + config snapshot
  POST /query     the RAG endpoint

Production touches included here: request-id propagation into logs, a global
exception handler that never leaks stack traces to the caller, latency captured
per stage, and lifespan-based warm-up so the first user request isn't the one
that pays for model loading.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..config import settings
from ..logging_config import configure_logging, get_logger, new_request_id
from ..models import QueryRequest, QueryResponse
from .dependencies import get_pipeline

configure_logging(settings.log_level, settings.log_json)
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up heavy singletons before traffic arrives.
    get_pipeline()
    logger.info("startup complete", extra={"extra": {"env": settings.environment}})
    yield
    logger.info("shutdown")


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request, call_next):
    rid = new_request_id()
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    logger.info(
        "request",
        extra={"extra": {
            "method": request.method, "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }},
    )
    return response


@app.exception_handler(Exception)
async def unhandled(request, exc):
    # Log full detail server-side; return a safe generic message to the client.
    logger.exception("unhandled error", extra={"extra": {"path": request.url.path}})
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    p = get_pipeline()
    return {
        "vector_store": settings.vector_store,
        "chunks_indexed": p.store.count(),
        "embedding_model": (
            settings.openai_embedding_model
            if settings.embedding_provider == "openai"
            else settings.local_embedding_model
        ),
        "reranker": settings.reranker_model if settings.use_reranker else None,
        "cache_enabled": settings.cache_enabled,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    pipeline = get_pipeline()
    return pipeline.query(
        query=req.query,
        top_n=req.top_n,
        category=req.category,
        use_cache=req.use_cache,
    )
