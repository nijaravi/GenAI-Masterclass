# ask_v3.py
# ============================================================
# V3: CLI with streaming + retry + cost tracking (partial)
# ============================================================

import time
from openai import OpenAI
from openai import RateLimitError, APITimeoutError

client = OpenAI()

# Pricing per 1M tokens (USD)
PRICING = {
    "gpt-4o":      {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini": {"input": 0.15,  "output": 0.60},
}


def cli_v2(question, model="gpt-4o-mini"):
    """Non-streaming fallback with retry + cost tracking."""
    messages = [{"role": "user", "content": question}]
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                timeout=30
            )
            
            print("\nResponse:\n")
            print(response.choices[0].message.content)

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
            print(f"⚠️ {type(e).__name__}. Retrying in {wait}s...")
            time.sleep(wait)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    print("❌ Failed after retries.")


def cli_v3(model="gpt-4o-mini", stream=True):
    """Streaming CLI."""
    
    question = input("Ask your question: ")
    messages = [{"role": "user", "content": question}]
    
    try:
        if stream:
            print("\nResponse:\n")
            
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                stream=True
            )

            collected = []

            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    collected.append(content)

            full_text = "".join(collected)

            # Rough estimation (since streaming doesn't return usage)
            est_tokens = len(full_text) // 4 + len(question) // 4

            print(f"\n\n{'─' * 50}")
            print(f"Model: {model} | ~{est_tokens} tokens (estimated)")

        else:
            cli_v2(question, model)

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("Streaming mode enabled 🚀")
    cli_v3()