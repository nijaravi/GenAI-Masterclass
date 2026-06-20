"""Cost tracking: turn token counts into dollars.

Each provider charges per million tokens, with different input vs output prices.
We keep a small price table and compute a per-request cost from the token counts
the provider returned. That cost is stored on every trace, which is what powers
the admin "cost per user" view.

Prices are approximate USD per 1M tokens and easy to update in one place.
"""
from common.logging_setup import get_logger

logger = get_logger(__name__)

# (input_price_per_1M, output_price_per_1M) in USD. Approximate.
PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # The stub has no real cost; price it like the mini model so the demo
    # dashboard shows non-zero numbers to look at.
    "stub-model": (0.15, 0.60),
}


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    cost = (prompt_tokens / 1_000_000) * in_price + \
           (completion_tokens / 1_000_000) * out_price
    return round(cost, 6)
