"""
Single source of truth for configuration. This platform makes real LLM
calls - there's no offline/demo model to fall back to - so OPENAI_API_KEY
is required, not optional. We check for it here, once, with a clear error
message, rather than letting every node hit a confusing auth error on its
first API call.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# CrewAI tries to phone home tracing spans on every kickoff() by default,
# which adds latency and noisy error logs when it can't reach the
# collector. Off by default; set CREWAI_DISABLE_TELEMETRY=false in .env
# if you want it.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "1) cp .env.example .env\n"
        "2) put your key in .env\n"
        "This platform makes real LLM calls end to end - there's no demo "
        "mode to fall back to."
    )

MAX_HOPS = 4  # Section 6: guards against runaway multi-hop loops

# Section 10: role-aware routing map. A route only fires for roles listed
# here - enforced at the agent layer as a UX control, on top of (not
# instead of) whatever server-side authorization a real backend would add.
ROLE_ALLOWED_ROUTES = {
    "customer": {"rag", "external_agent", "final"},
    "client": {"mcp", "external_agent", "final"},
    "developer": {"rag", "coder", "final"},
}
