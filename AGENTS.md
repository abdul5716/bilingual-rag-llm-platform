# Project context for AI assistants

This is a personal learning project: a **Bilingual (Arabic–English) RAG and LLM Engineering
Platform**. The owner is learning generative AI and wants to become an expert agentic AI
engineer, and wants to put this project on a resume. Help them **build and learn** each
component: explain concepts, then build, and don't just dump finished code.

## Source of truth
- `docs/ROADMAP.md` has the reviewed plan, phase order, deliverables, and acceptance
  criteria. Follow it unless the user changes it.
- `README.md` has the phase status table. Update it as phases progress.

## Phase order (decided)
0 Foundations/env, 1 RAG baseline (no framework first), 2 Evaluation and regression gates,
3 vLLM serving and benchmarks, 4 MCP server, 5 LoRA/QLoRA, 6 Agentic layer,
7 Production hygiene, 8 Write-up.

## Environment (checked 2026-10-22)
- NVIDIA L4, 23 GB VRAM, driver 580; 30 GB RAM; about 150 GB free disk.
- Python 3.10.12, Docker available. `ollama` and `uv` were **not installed**.
- One GPU: don't run vLLM/Ollama and QLoRA training at the same time.
- Use separate virtualenvs: `.venv-app`, `.venv-train`, `.venv-vllm`.
- Inside Cursor's sandbox `nvidia-smi` may fail even though the GPU works. Run GPU commands
  with sandbox disabled.

## Open decisions
- Corpus: Arabic/English Wikipedia subset vs public bilingual policy documents
  (the roadmap leans toward policy documents because of the `lookup_policy` tool).
- Whether Ollama already exists elsewhere (the user mentioned an "existing Ollama
  deployment").
- Not yet done: first git commit, remote/GitHub repo.

## Ground rules
- Resume honesty: only report numbers that were actually measured; write them to `reports/`.
- Never commit `.env`, model weights, or large datasets.
- Keep the LLM client behind an OpenAI-compatible interface so Ollama, vLLM, and a judge
  model are swappable by config.
- Freeze the test split; don't tune on it. Split data by document, not by question.
