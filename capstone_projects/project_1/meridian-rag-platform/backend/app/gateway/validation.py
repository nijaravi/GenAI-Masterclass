"""Gateway: request validation.

Pydantic already validates the request body shape and field lengths (see
AskRequest in common/types.py). This module holds the extra, business-level
checks that don't belong on the model — kept here so the "validation" step of
the gateway is a real, visible thing rather than scattered around.
"""
from fastapi import HTTPException

from common.types import AskRequest

ALLOWED_CATEGORIES = {"hr", "it", "security", "engineering"}


def validate_ask(req: AskRequest) -> None:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if req.category is not None and req.category not in ALLOWED_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"category must be one of {sorted(ALLOWED_CATEGORIES)}")
