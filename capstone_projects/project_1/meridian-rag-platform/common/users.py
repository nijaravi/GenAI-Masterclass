"""The 5 onboarded users (plus their API keys and roles).

In a real system this would be a users table + a real auth provider. For this
learning project we hard-code five users so the whole auth / rate-limit / cost
attribution story is easy to follow and demo. User 1 is also the admin.

The API key is what the frontend sends in the `X-API-Key` header; the backend
looks the user up from it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    api_key: str
    is_admin: bool = False


# Keys are obviously fake and committed on purpose — this is a demo.
USERS = [
    User("u1", "Aisha (Admin)", "key-aisha-001", is_admin=True),
    User("u2", "Ben", "key-ben-002"),
    User("u3", "Carlos", "key-carlos-003"),
    User("u4", "Diya", "key-diya-004"),
    User("u5", "Erik", "key-erik-005"),
]

# Fast lookups built once at import.
_BY_KEY = {u.api_key: u for u in USERS}
_BY_ID = {u.user_id: u for u in USERS}


def get_user_by_key(api_key: str) -> User | None:
    return _BY_KEY.get(api_key)


def get_user_by_id(user_id: str) -> User | None:
    return _BY_ID.get(user_id)
