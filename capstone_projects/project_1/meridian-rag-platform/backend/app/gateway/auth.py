"""Gateway: authentication.

Every request must carry an `X-API-Key` header. We look the key up against the
five seeded users; unknown keys get a 401. `require_admin` is a second dependency
for the admin-only endpoints (only user 1 is an admin).

These are FastAPI dependencies, so a route just declares `user = Depends(...)`
and the check runs before the handler.
"""
from fastapi import Header, HTTPException

from common.users import User, get_user_by_key


def require_user(x_api_key: str = Header(default="")) -> User:
    user = get_user_by_key(x_api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return user


def require_admin(x_api_key: str = Header(default="")) -> User:
    user = require_user(x_api_key)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="admin access required")
    return user
