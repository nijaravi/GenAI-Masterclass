"""FastAPI application entry point.

Wires the gateway layer (auth/rate-limit live as dependencies on the routes),
the user and admin routers, and a startup hook that builds the app state once.

Run locally:
    uvicorn backend.app.main:app --reload --port 8000
Then open http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.config import settings
from common.logging_setup import get_logger, setup_logging
from .routers import admin_routes, user_routes
from .state import get_state

setup_logging(settings.log_level)
logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_state()   # build everything once before serving traffic
    logger.info("startup complete (env=%s)", settings.environment)
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.include_router(user_routes.router)
app.include_router(admin_routes.router)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}
