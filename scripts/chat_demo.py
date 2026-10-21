#!/usr/bin/env python3
"""Phase 0: call a local LLM through the OpenAI-compatible chat API.

The same code works against Ollama, vLLM, or a hosted model: only the three
environment variables change (see .env.example).
"""

import os
import sys
import time

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("LLM_API_KEY", "not-needed-for-local"),
)
model = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")

question = " ".join(sys.argv[1:]) or "What is a KV cache? Answer in one sentence."

start = time.perf_counter()
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "Answer in the same language as the question."},
        {"role": "user", "content": question},
    ],
    temperature=0,
)
elapsed = time.perf_counter() - start

print(resp.choices[0].message.content)
print(f"\n[{model}] {resp.usage.prompt_tokens} prompt tok, "
      f"{resp.usage.completion_tokens} completion tok, {elapsed:.2f}s")
