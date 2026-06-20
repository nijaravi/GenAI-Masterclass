"""Gateway: per-user rate limiting (fixed window).

Each user gets N requests per time window (defaults: 30 per 60s). We keep a
small in-memory record of when each user's window started and how many requests
they've made in it. Over the limit -> HTTP 429.

In-memory means it resets on restart and isn't shared across replicas; in
production you'd back this with Redis (one counter key per user with a TTL). The
calling code wouldn't change.
"""
import time

from fastapi import HTTPException

from common.config import settings
from common.logging_setup import get_logger

logger = get_logger(__name__)

# user_id -> (window_start_epoch, count_in_window)
_windows: dict[str, tuple[float, int]] = {}


def check_rate_limit(user_id: str) -> None:
    now = time.time()
    window = settings.rate_limit_window_seconds
    limit = settings.rate_limit_requests

    start, count = _windows.get(user_id, (now, 0))
    if now - start >= window:
        # window expired -> start a fresh one
        start, count = now, 0

    if count >= limit:
        retry_in = int(window - (now - start))
        logger.warning("rate limit hit for %s", user_id)
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({limit}/{window}s); retry in {retry_in}s")

    _windows[user_id] = (start, count + 1)
