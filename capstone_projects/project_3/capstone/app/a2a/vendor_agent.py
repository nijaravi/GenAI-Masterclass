"""
Section 4.8 / 7.4: the counterpart on the other side of the A2A boundary -
"an agent independently built and run by another team or vendor."

Simplification note: a real A2A counterpart is a separate service reached
over HTTP, publishing its Agent Card at a well-known URL. To keep this
build easy to run (one process, no networking to reason about), the
vendor agent here is an in-process class instead - but it still models
the two things that actually matter conceptually:
  1. A published Agent Card describing what it can do.
  2. A task lifecycle - submitted -> working -> completed/failed - with
     realistic latency (asyncio.sleep), not an instant function return.
Swapping this for a real HTTP call later means changing call_external_agent()
below to an httpx call instead of an in-process await; the calling node
(app/nodes/specialists.py) doesn't need to change at all.
"""
from __future__ import annotations

import asyncio

AGENT_CARD = {
    "name": "logistics-partner-agent",
    "description": "Independently operated logistics/returns status agent "
                    "for a third-party shipping vendor. Not owned or "
                    "controlled by this platform.",
    "capabilities": ["check_return_status"],
}

# Toy "logistics system" data, keyed by order_id.
_RETURN_STATUS = {
    "ord-7712": {"status": "in_transit_to_warehouse", "carrier_eta": "2026-07-06"},
    "ord-3390": {"status": "received_pending_refund", "carrier_eta": "n/a"},
}

_TIMEOUT_SECONDS = 3.0
_consecutive_failures = 0
_CIRCUIT_BREAKER_THRESHOLD = 3


async def _vendor_task(order_id: str) -> dict:
    """Simulates the vendor's own task lifecycle: submitted -> working ->
    completed/failed. The sleep stands in for real network + processing
    latency on a system this platform doesn't control."""
    await asyncio.sleep(0.8)  # "working"
    record = _RETURN_STATUS.get(order_id)
    if record is None:
        raise LookupError(f"no return found for order_id={order_id}")
    return record


async def call_external_agent(capability: str, order_id: str) -> tuple[bool, dict | str]:
    """Returns (ok, data_or_degraded_reason). Mirrors Section 11.1's
    guardrail: a circuit breaker for repeated failures, plus a timeout so
    a slow/unavailable external agent never stalls the whole request."""
    global _consecutive_failures

    if _consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
        return False, "circuit_open"

    if capability not in AGENT_CARD["capabilities"]:
        return False, "capability_not_found"

    try:
        result = await asyncio.wait_for(_vendor_task(order_id), timeout=_TIMEOUT_SECONDS)
        _consecutive_failures = 0
        return True, result
    except (asyncio.TimeoutError, LookupError) as exc:
        _consecutive_failures += 1
        reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)
        return False, reason
