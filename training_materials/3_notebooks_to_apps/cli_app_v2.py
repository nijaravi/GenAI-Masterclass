# ask_v2.py
# ============================================================
# V2: CLI with retry logic + token/cost tracking
# ============================================================

import time
from openai import OpenAI
from openai import RateLimitError, APITimeoutError
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# Initialize client (ensure OPENAI_API_KEY is set)
client = OpenAI()

# Pricing per 1M tokens (USD)
PRICING = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
}

def cli_v2(model="gpt-4o-mini"):
    """CLI with retry logic and cost tracking."""

    # Take input from CLI
    question = input("Ask your question: ")
    messages = [{"role": "user", "content": question}]
    
    # Retry logic
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10,
                timeout=30
            )
            
            # Print response
            print("\nResponse:\n")
            print(response.choices[0].message.content)
            
            # Check if truncated
            if response.choices[0].finish_reason == "length":
                print("\n⚠️  Response was truncated.")
            
            # Cost tracking
            usage = response.usage
            pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
            
            cost = (
                usage.prompt_tokens * pricing["input"] +
                usage.completion_tokens * pricing["output"]
            ) / 1_000_000
            
            print(f"\n{'─' * 50}")
            print(
                f"Model: {model} | Tokens: {usage.total_tokens} "
                f"(in {usage.prompt_tokens}, out {usage.completion_tokens}) | "
                f"Cost: ~${cost:.6f}"
            )
            return
        
        except (RateLimitError, APITimeoutError) as e:
            wait = 2 ** attempt
            print(f"⚠️  {type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    print("❌ Failed after 3 retries.")


if __name__ == "__main__":
    cli_v2()