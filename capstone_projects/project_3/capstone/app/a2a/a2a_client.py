"""
Client-side half of the A2A boundary (Section 4.8):
  - discover the counterpart's capabilities via its published Agent Card
  - submit a task and track it through submitted -> working -> completed/failed
  - carry a timeout and a circuit breaker so a slow/unavailable external
    agent never stalls the whole request (Section 11.1)

This talks to app/a2a/external_vendor_agent.py over plain HTTP using the
same shape a real A2A SDK call would have. Kept dependency-light (httpx
only) rather than pulling in an early-stage A2A client library, since the
protocol surface used here - Agent Card + task polling - is the stable
part of the spec.
"""
from __future__ import annotations

import asyncio
import time

import httpx

from app import config


class CircuitBreaker:
    """Trivial consecutive-failure breaker, process-local. Section 11.1:
    'a circuit breaker for repeated failures.' After `threshold` consecutive
    failures it stays open for `cooldown_seconds`, short-circuiting calls
    without even hitting the network."""

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 30.0):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at > self.cooldown_seconds:
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self):
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold:
            self._opened_at = time.time()


_breaker = CircuitBreaker()


class A2AResult:
    def __init__(self, ok: bool, data: dict | None, degraded_reason: str | None = None):
        self.ok = ok
        self.data = data
        self.degraded_reason = degraded_reason


async def call_external_agent(capability: str, task_input: dict) -> A2AResult:
    if _breaker.is_open():
        return A2AResult(False, None, degraded_reason="circuit_open")

    base_url = config.EXTERNAL_AGENT_URL
    timeout = config.A2A_TIMEOUT_SECONDS

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # 1. Discover capabilities via the published Agent Card.
            card_resp = await client.get(f"{base_url}/.well-known/agent.json")
            card_resp.raise_for_status()
            card = card_resp.json()
            capability_names = [c["name"] for c in card.get("capabilities", [])]
            if capability not in capability_names:
                _breaker.record_failure()
                return A2AResult(False, None, degraded_reason="capability_not_found")

            # 2. Submit the task with the minimum data needed (Section 13:
            #    'sends the minimum data needed for the task, not full
            #    user/account records').
            submit_resp = await client.post(
                f"{base_url}/tasks", json={"capability": capability, "input": task_input}
            )
            submit_resp.raise_for_status()
            task_id = submit_resp.json()["task_id"]

            # 3. Poll the task lifecycle: submitted -> working -> completed/failed.
            deadline = time.time() + timeout
            while time.time() < deadline:
                poll_resp = await client.get(f"{base_url}/tasks/{task_id}")
                poll_resp.raise_for_status()
                task = poll_resp.json()
                if task["state"] == "completed":
                    _breaker.record_success()
                    return A2AResult(True, task["result"])
                if task["state"] == "failed":
                    _breaker.record_failure()
                    return A2AResult(False, task["result"], degraded_reason="task_failed")
                await asyncio.sleep(0.3)

            _breaker.record_failure()
            return A2AResult(False, None, degraded_reason="timeout")

    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        _breaker.record_failure()
        return A2AResult(False, None, degraded_reason=f"unreachable: {exc.__class__.__name__}")
