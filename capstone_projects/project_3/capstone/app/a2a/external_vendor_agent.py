"""
Section 4.8 / 7.4: the counterpart on the other side of the A2A boundary -
'an agent independently built and run by another team or vendor.' This
file simulates exactly that: a completely separate service, on its own
stack (still FastAPI here, but it could be anything), that this platform
does not own.

It implements the two things Section 4.8 says the A2A node depends on:
  1. A published Agent Card at a well-known URL describing its capabilities.
  2. A task lifecycle: submitted -> working -> completed / failed, polled
     by task_id rather than a single blocking call - the same shape a real
     A2A counterpart would expose.

Run standalone:
    python -m app.a2a.external_vendor_agent
(defaults to port 9001; app/main.py auto-launches this in DEMO_MODE so a
single `uvicorn app.main:app` command demos the whole platform.)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Logistics Partner Agent (external, independently run)")

_AGENT_CARD = json.loads((Path(__file__).parent / "agent_card.json").read_text())

_TASKS: dict[str, dict] = {}

# Toy "logistics system" data, keyed by order_id.
_RETURN_STATUS = {
    "ord-7712": {"status": "in_transit_to_warehouse", "carrier_eta": "2026-07-06"},
    "ord-3390": {"status": "received_pending_refund", "carrier_eta": "n/a"},
}


class TaskSubmission(BaseModel):
    capability: str
    input: dict


@app.get("/.well-known/agent.json")
def get_agent_card():
    return _AGENT_CARD


@app.post("/tasks")
async def submit_task(payload: TaskSubmission):
    if payload.capability not in [c["name"] for c in _AGENT_CARD["capabilities"]]:
        raise HTTPException(400, f"unknown capability: {payload.capability}")

    task_id = str(uuid.uuid4())
    _TASKS[task_id] = {"state": "submitted", "created": time.time(), "result": None}

    # Simulate independent, asynchronous work on the vendor's side.
    asyncio.create_task(_run_task(task_id, payload))
    return {"task_id": task_id, "state": "submitted"}


async def _run_task(task_id: str, payload: TaskSubmission):
    _TASKS[task_id]["state"] = "working"
    await asyncio.sleep(1.2)  # pretend the vendor is doing real work

    order_id = payload.input.get("order_id")
    record = _RETURN_STATUS.get(order_id)
    if record is None:
        _TASKS[task_id]["state"] = "failed"
        _TASKS[task_id]["result"] = {"error": f"no return found for order_id={order_id}"}
        return

    _TASKS[task_id]["state"] = "completed"
    _TASKS[task_id]["result"] = record


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = _TASKS.get(task_id)
    if task is None:
        raise HTTPException(404, "unknown task_id")
    return {"task_id": task_id, **task}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001)
