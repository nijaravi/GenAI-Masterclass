"""Admin-facing endpoints (only user 1 can call these).

  GET /v1/admin/metrics  — latency p50/p95, block rate, cache-hit rate, quality
                           scores, plus any anomaly alerts that are firing
  GET /v1/admin/usage    — request counts per user
  GET /v1/admin/cost     — dollar cost per user (and total)
  GET /v1/admin/traces   — the most recent request traces (for inspection)

All of these just read and aggregate the trace store.
"""
from fastapi import APIRouter, Depends

from common.users import User
from ..gateway.auth import require_admin
from ..monitoring.anomaly import check_anomalies
from ..monitoring.metrics import cost_by_user, overall_metrics, usage_by_user
from ..state import get_state

router = APIRouter()


@router.get("/v1/admin/metrics")
def metrics(admin: User = Depends(require_admin)) -> dict:
    state = get_state()
    m = overall_metrics(state.trace_store)
    return {"metrics": m, "alerts": check_anomalies(m)}


@router.get("/v1/admin/usage")
def usage(admin: User = Depends(require_admin)) -> dict:
    state = get_state()
    return {"usage": usage_by_user(state.trace_store)}


@router.get("/v1/admin/cost")
def cost(admin: User = Depends(require_admin)) -> dict:
    state = get_state()
    return cost_by_user(state.trace_store)


@router.get("/v1/admin/traces")
def traces(admin: User = Depends(require_admin), limit: int = 20) -> dict:
    state = get_state()
    return {"traces": state.trace_store.recent(limit)}
