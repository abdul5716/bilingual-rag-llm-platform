# Bilingual RAG and LLM Engineering Platform

A personal learning project: an Arabic–English knowledge assistant built from the ground up
to learn the modern LLM engineering stack.

- Multilingual embeddings, hybrid vector retrieval, and reranking
- A locally served open-source LLM (Ollama and vLLM, behind one OpenAI-compatible interface)
- RAG evaluation (retrieval metrics plus RAGAS) with regression gates
- An MCP server exposing document-search tools
- LoRA/QLoRA fine-tuning with PEFT and TRL
- vLLM serving benchmarks (latency, throughput, GPU memory)
- An agentic layer that uses the MCP tools

**The plan lives in [`docs/ROADMAP.md`](docs/ROADMAP.md)**, including the review of the
original plan, the phase order, deliverables, and acceptance criteria.

## Status

| Phase | Topic | Status |
|-------|-------|--------|
| 0 | Foundations and environment | environment done; tokenisation/embedding exercises left |
| 1 | Bilingual RAG baseline | not started |
| 2 | Evaluation and regression gates | not started |
| 3 | vLLM serving and benchmarks | not started |
| 4 | MCP server | not started |
| 5 | LoRA / QLoRA | not started |
| 6 | Agentic layer | not started |
| 7 | Production hygiene | not started |
| 8 | Write-up | not started |

No results are reported yet. Numbers will be added to `reports/` only after they are measured.

## Getting started

```bash
python3 -m venv .venv-app && source .venv-app/bin/activate
pip install -e ".[dev]"
make check-env      # shows which GPU, tools, and packages, and tests a chat call
make test
python scripts/chat_demo.py "Hello"   # needs a local server; see docs/ENVIRONMENT.md
```

Training and vLLM use separate virtualenvs on purpose; see `docs/ROADMAP.md`, section 3.

Copy `.env.example` to `.env` for local settings. Never commit `.env`, model weights, or
large datasets.
