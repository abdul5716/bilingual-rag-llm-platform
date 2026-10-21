#!/usr/bin/env python3
"""Phase 0 environment check.

Prints what is available on this machine so you can see what each phase needs.
Uses only the standard library so it runs in a bare virtualenv.
Exit code is non-zero only if a *required* check fails (Python version, chat endpoint).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request

OK, WARN, FAIL = "[ ok ]", "[warn]", "[FAIL]"


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 10)
    print(f"{OK if ok else FAIL} Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")
    return ok


def check_gpu() -> bool:
    out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
        ]
    )
    if out is None:
        print(f"{WARN} No working NVIDIA GPU/driver found (needed for phases 3 and 5)")
        return False
    for line in out.splitlines():
        print(f"{OK} GPU: {line}")
    return True


def check_binary(name: str, needed_for: str) -> None:
    path = shutil.which(name)
    if path:
        print(f"{OK} {name}: {path}")
    else:
        print(f"{WARN} {name} not found (needed for {needed_for})")


def check_package(name: str, needed_for: str) -> None:
    if importlib.util.find_spec(name) is not None:
        print(f"{OK} python package: {name}")
    else:
        print(f"{WARN} python package '{name}' not installed (needed for {needed_for})")


def check_llm_endpoint() -> bool:
    """Send one tiny chat request to the configured OpenAI-compatible server."""
    base = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK."}],
            "max_tokens": 5,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'x')}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 - report any failure, this is a diagnostic tool
        print(f"{FAIL} chat completion at {base} with model '{model}' failed: {e}")
        return False
    print(f"{OK} chat completion works at {base} (model {model}): {text.strip()!r}")
    return True


def check_disk(min_free_gb: int = 50) -> None:
    free_gb = shutil.disk_usage(".").free / 1e9
    tag = OK if free_gb >= min_free_gb else WARN
    print(f"{tag} free disk: {free_gb:.0f} GB (model weights are large; {min_free_gb}+ GB advised)")


def main() -> int:
    required_ok = check_python()
    check_gpu()
    check_disk()
    print()
    for binary, phase in [
        ("git", "all phases"),
        ("docker", "Qdrant / Compose (phases 1, 7)"),
        ("ollama", "optional host install; we run Ollama in Docker instead"),
        ("uv", "fast, reproducible environments (optional)"),
    ]:
        check_binary(binary, phase)
    print()
    llm_ok = check_llm_endpoint()
    print()
    for package, phase in [
        ("torch", "embeddings, training"),
        ("sentence_transformers", "phase 1"),
        ("qdrant_client", "phase 1"),
        ("ragas", "phase 2"),
        ("mcp", "phase 4"),
        ("peft", "phase 5"),
        ("trl", "phase 5"),
        ("vllm", "phase 3 (install in its own venv)"),
    ]:
        check_package(package, phase)
    return 0 if (required_ok and llm_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
