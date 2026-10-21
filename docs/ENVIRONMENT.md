# Environment (recorded 2026-10-22)

Everything here was read from the machine, not assumed. Re-run `make check-env` to verify.

## Hardware and OS
| Item | Value |
|------|-------|
| GPU | NVIDIA L4, 23034 MiB VRAM |
| NVIDIA driver | 580.178.04 (max CUDA supported by driver: 13.0) |
| CUDA toolkit (`nvcc`) | not installed on host. Not needed: PyTorch and vLLM wheels bundle their own CUDA runtime |
| RAM / disk | 30 GB / about 149 GB free |
| OS / kernel | Ubuntu 22.04.5 LTS, 6.8.0-1066-aws |
| sudo | not available, so anything needing root is run via Docker or user-space installs |

## Software
| Item | Version |
|------|---------|
| Python | 3.10.12 (system) |
| Docker | 29.3.1, with the `nvidia` runtime (GPU containers work) |
| `uv` | not installed (optional) |
| Ollama | 0.40.8, **running as a Docker container** (see below) |
| `.venv-app` packages | openai 3.36.0, pydantic 2.14.1, httpx 0.28.1, pytest 9.1.1, ruff 0.17.1, tokenizers 0.23.3, huggingface_hub 2.2.0, sentence-transformers 6.2.0, transformers 5.20.2, numpy 2.2.6 |
| `.venv-train`, `.venv-vllm` | not created yet (Phases 5 and 3) |
| torch (in `.venv-app`) | 2.14.1+cu130, `torch.cuda.is_available()` is True |

## Models
| Model | Where | Details |
|-------|-------|---------|
| `qwen2.5:3b-instruct` | Ollama | 3.1B params, Q4_K_M (4-bit GGUF), 2.2 GB, native context 32768, tools capability, 100% on GPU. Licence: Qwen Research Licence (check terms before any non-research use) |
| `BAAI/bge-m3` | sentence-transformers (HF cache) | 1024-dim embeddings, max 8192 tokens, runs on the GPU |

## Ollama in Docker
```bash
docker run -d --gpus all --restart unless-stopped \
  -v ollama:/root/.ollama -p 127.0.0.1:11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull qwen2.5:3b-instruct
```
- Model weights live in the named Docker volume `ollama`, not in the repo.
- Bound to `127.0.0.1` only, so it is not exposed on the network.
- OpenAI-compatible endpoint: `http://localhost:11434/v1`.
- Before Phase 3 (vLLM) or Phase 5 (QLoRA), free the GPU: `docker stop ollama`.
  Ollama unloads idle models after about 5 minutes by default.

## Verified
- `make check-env` passes, including a real chat completion against the local server.
- `scripts/chat_demo.py` answers in English and Arabic through the OpenAI client.
- First request was slow (about 48 s, cold model load). Later requests took well under a
  second. This is a single informal observation, not a benchmark.

## Notes
- Inside Cursor's sandbox `nvidia-smi` and Docker may fail. Run GPU/Docker commands with the
  sandbox disabled.
