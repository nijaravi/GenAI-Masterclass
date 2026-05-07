# ask_v1.py
# ============================================================
# V1: Simplest possible CLI — user input via terminal prompt
# ============================================================

from openai import OpenAI

from dotenv import load_dotenv
# import os
load_dotenv(override=True)

# Initialize client (make sure OPENAI_API_KEY is set in env)
client = OpenAI()

def cli_v1():
    """The simplest possible LLM CLI with user input."""
    
    # Take input from CLI
    question = input("Ask your question: ")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
        max_tokens=200,
    )

    print("\nResponse:\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    cli_v1()
    print("\n💡 This works but has zero error handling, no cost tracking, no options.")