"""
Central configuration. Every other module reads settings from here instead
of calling os.getenv() directly, so there is exactly one place that knows
how to interpret the environment.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


DEMO_MODE: bool = _bool("DEMO_MODE", True)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "orchestrator-platform")

CHECKPOINTER_DB_URL: str = os.getenv("CHECKPOINTER_DB_URL", "")

EXTERNAL_AGENT_URL: str = os.getenv("EXTERNAL_AGENT_URL", "http://localhost:9001")
A2A_TIMEOUT_SECONDS: float = float(os.getenv("A2A_TIMEOUT_SECONDS", "6"))
A2A_MAX_RETRIES: int = int(os.getenv("A2A_MAX_RETRIES", "1"))

MAX_HOPS: int = 4  # matches Section 6 (hop_count guard) of the design doc

# Effective mode is "live" only if DEMO_MODE is off AND a key is actually present.
# This stops a misconfigured .env from crashing the app at request time -
# it just quietly falls back to demo behaviour and logs why.
LIVE_LLM: bool = (not DEMO_MODE) and bool(OPENAI_API_KEY)
LIVE_PINECONE: bool = (not DEMO_MODE) and bool(PINECONE_API_KEY)
