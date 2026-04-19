# ask_v4.py
# ============================================================
# V4: CLI with system prompt + streaming + config options
# ============================================================

from openai import OpenAI


from dotenv import load_dotenv
import os
load_dotenv(override=True)

client = OpenAI()


def cli_v4():
    """Full-featured CLI with system prompt support."""

    # ===== CLI INPUTS =====
    question = input("Ask your question: ")

    use_system = input("Add system prompt? (y/n): ").strip().lower()
    system_prompt = None
    if use_system == "y":
        system_prompt = input("Enter system prompt: ")

    model = input("Model (default gpt-4o-mini): ").strip() or "gpt-4o-mini"

    temp_input = input("Temperature (default 0.3): ").strip()
    temperature = float(temp_input) if temp_input else 0.3

    stream_input = input("Streaming? (y/n, default y): ").strip().lower()
    stream = False if stream_input == "n" else True

    # ===== MESSAGE BUILD =====
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    # ===== EXECUTION =====
    try:
        if stream:
            print("\nResponse:\n")

            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                stream=True
            )

            for chunk in response_stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)

            print(f"\n\n{'─' * 50}")
            print(f"Model: {model} | temp: {temperature}")
            if system_prompt:
                preview = system_prompt[:60]
                print(f"System: {preview}..." if len(system_prompt) > 60 else f"System: {system_prompt}")

        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )

            print("\nResponse:\n")
            print(response.choices[0].message.content)

            usage = response.usage
            print(f"\n{'─' * 50}")
            print(f"Model: {model} | Tokens: {usage.total_tokens} | temp: {temperature}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 V4 CLI — System Prompt Enabled")
    print("═" * 60)
    cli_v4()