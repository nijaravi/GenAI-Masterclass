#!/usr/bin/env python3
"""
ask.py — A CLI tool to query LLMs from the terminal.

GenAI Decoded by Nij — Section 3: From Notebooks to Apps

Usage:
    python ask.py "What is Hadoop?"
    python ask.py "Explain RAG" --model gpt-4o
    python ask.py "Review this code" --system "You are a senior Python engineer"
    python ask.py --no-stream "Give me a one-word answer: yes or no?"
    echo "Summarize this" | python ask.py --stdin
"""

import argparse
import sys
import os
import time
from openai import OpenAI, RateLimitError, APITimeoutError, BadRequestError
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# ============================================================
# CONFIG
# ============================================================
DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
TIMEOUT = 60

# Pricing per 1M tokens (update if prices change)
PRICING = {
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


# ============================================================
# CORE FUNCTIONS
# ============================================================
def get_client():
    """Create OpenAI client. Reads API key from environment."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    return OpenAI(timeout=TIMEOUT)


def estimate_cost(model, prompt_tokens, completion_tokens):
    """Estimate cost in dollars."""
    prices = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (prompt_tokens * prices["input"] + completion_tokens * prices["output"]) / 1_000_000


def call_with_retry(client, messages, model, temperature, max_tokens, stream=False):
    """Call the API with exponential backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
        except RateLimitError:
            wait = 2 ** attempt
            print(f"\n⚠️  Rate limited. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...",
                  file=sys.stderr)
            time.sleep(wait)
        except APITimeoutError:
            wait = 2 ** attempt
            print(f"\n⚠️  Timeout. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...",
                  file=sys.stderr)
            time.sleep(wait)
        except BadRequestError as e:
            print(f"\n❌ Bad request: {e}", file=sys.stderr)
            sys.exit(1)
    
    print(f"\n❌ Failed after {MAX_RETRIES} retries.", file=sys.stderr)
    sys.exit(1)


def ask_streaming(client, messages, model, temperature, max_tokens):
    """Send prompt with streaming — print tokens as they arrive."""
    stream = call_with_retry(client, messages, model, temperature, max_tokens, stream=True)
    
    collected = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            collected.append(delta.content)
    
    print()
    return "".join(collected)


def ask_standard(client, messages, model, temperature, max_tokens):
    """Send prompt without streaming — wait for full response."""
    response = call_with_retry(client, messages, model, temperature, max_tokens, stream=False)
    
    content = response.choices[0].message.content or ""
    print(content)
    
    if response.choices[0].finish_reason == "length":
        print("\n⚠️  Response was truncated (hit max_tokens).", file=sys.stderr)
    
    usage = response.usage
    cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    print(f"\n{'─' * 50}")
    print(f"Tokens: {usage.total_tokens} (in {usage.prompt_tokens}, out {usage.completion_tokens}) "
          f"| Cost: ${cost:.6f} | Model: {model}")
    
    return content


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ask an LLM a question from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ask.py "What is Hadoop?"
  python ask.py "Explain RAG" --model gpt-4o
  python ask.py "Review this code" --system "You are a senior engineer"
  echo "some text" | python ask.py --stdin "Summarize this"
"""
    )
    parser.add_argument("question", nargs="?", help="Your question or prompt")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--system", "-s", default=None,
                        help="System prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.3,
                        help="Temperature 0-2 (default: 0.3)")
    parser.add_argument("--max-tokens", type=int, default=1000,
                        help="Max response tokens (default: 1000)")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming (wait for full response)")
    parser.add_argument("--stdin", action="store_true",
                        help="Read additional input from stdin")
    
    args = parser.parse_args()
    
    # Build the prompt
    prompt = args.question or ""
    
    if args.stdin or (not sys.stdin.isatty() and not prompt):
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            prompt = f"{prompt}\n\n{stdin_data}" if prompt else stdin_data
            print("**********************")
            print(stdin_data)
            print("-----------------")
            print(prompt)
            print("-----------------")
    
    if not prompt:
        parser.print_help()
        sys.exit(1)
    
    # Build messages
    client = get_client()
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    
    # Call the API
    if args.no_stream:
        ask_standard(client, messages, args.model, args.temperature, args.max_tokens)
    else:
        content = ask_streaming(client, messages, args.model, args.temperature, args.max_tokens)
        print(f"\n{'─' * 50}")
        print(f"Model: {args.model} | Streamed {len(content.split())} words")


if __name__ == "__main__":
    main()
