# Roadmap: Bilingual RAG and LLM Engineering Platform

This is the reviewed and extended version of the original four-part plan. The original
components (LoRA/QLoRA, vLLM, MCP, RAG evaluation) are all kept. What changed is the
**order**, a **missing foundation** (the RAG system itself), and an **agentic layer** on
top, because the long-term goal is agentic AI engineering.

---

## 1. Review of the original plan

### What was good
- Every component has a concrete deliverable. Keep that habit.
- It plans an apples-to-apples vLLM vs Ollama comparison and notes that vLLM compatibility
  depends on GPU, versions, and VRAM.
- It asks for regression tests, not just a one-off evaluation.

### Gaps and changes

| # | Issue | Change |
|---|-------|--------|
| 1 | **There is no RAG system in the plan.** All four items assume a RAG app exists. | Added Phase 1: corpus, chunking, embeddings, vector store, hybrid retrieval, reranking, generation with citations. Built **without a framework first**. |
| 2 | **Evaluation comes last.** You can't tell whether fine-tuning or vLLM helped without a test set. | Evaluation moved to Phase 2, right after the baseline. Every later phase is judged with it. |
| 3 | **No data plan.** "Curated Arabic–English QA dataset" is the hardest part of fine-tuning and eval. | Added a data section: sources, licences, synthetic generation plus manual review, train/dev/test split by *document* to avoid leakage. |
| 4 | **Fine-tuning goal is vague.** Fine-tuning rarely adds knowledge; RAG does that. | Define the target behaviour: grounded answers, citations, answering in the question's language, abstaining when evidence is missing. Measure exactly that. |
| 5 | **Arabic-specific work is missing.** | Normalisation (diacritics, alef/ya/ta-marbuta variants, tatweel), Arabic chunking, cross-lingual retrieval (ar→en, en→ar), judge-quality caveats. |
| 6 | **Eval relies only on LLM-judged metrics (RAGAS).** These are noisy, slow, and judge-dependent, especially in Arabic. | Add deterministic retrieval metrics (recall@k, MRR, nDCG), language-match rate, citation accuracy, and abstention accuracy. RAGAS or TruLens is the second layer. |
| 7 | **Regression checks are underspecified.** | Two tiers: a fast smoke set on every PR, and the full set before a release. Thresholds are relative to a stored baseline, with bootstrap confidence intervals. |
| 8 | **MCP "authentication" is hand-wavy.** stdio MCP servers have no network auth. | Cover both transports. Local stdio: least privilege and an allow-list. HTTP: bearer token or OAuth, per-tool authorisation, rate limits, audit log. Also cover prompt injection through retrieved documents. |
| 9 | **No agent layer.** MCP tools are only useful if something decides when to call them. | Added Phase 6: tool-calling loop, query rewriting/decomposition, corrective retrieval, LangGraph, and **agent evaluation** (tool-call accuracy, trajectories). |
| 10 | **No observability or packaging.** | Added tracing (Langfuse or Phoenix via OpenTelemetry), FastAPI service, Docker Compose, CI, experiment tracking. |
| 11 | **No hardware budget.** One 23 GB GPU can't hold vLLM, QLoRA training, an embedder, and a reranker at once. | Explicit VRAM budget (section 3) and separate virtual environments, since vLLM and the training stack often conflict on `torch`/`transformers` versions. |
| 12 | **Ollama is assumed to exist.** It is not installed on this machine. | Phase 0 installs it (Docker image or the official installer) and records versions. |
| 13 | **Resume claims aren't tied to evidence.** | Section 8: each resume claim maps to an artifact and a measured number. |

---

## 2. Target architecture

```
                        +-------------------+
   user / client  --->  |  FastAPI service  |  <--- traces ---> Langfuse / Phoenix
                        +---------+---------+
                                  |
                        +---------v---------+
                        |   Agent / RAG     |  query rewrite, decomposition,
                        |   orchestrator    |  corrective retrieval, citations
                        +----+---------+----+
                             |         |
              OpenAI-compatible API    | MCP client
                             |         |
                  +----------v--+   +--v----------------------+
                  | vLLM/Ollama |   | MCP server              |
                  | (base or    |   | search_documents        |
                  |  LoRA)      |   | get_document_details    |
                  +-------------+   | lookup_policy           |
                                    +----------+--------------+
                                               |
                          +--------------------v----------------+
                          | Retriever: bge-m3 dense (+ sparse)  |
                          | -> reranker -> Qdrant / metadata DB |
                          +-------------------------------------+

 Offline:  ingestion -> normalise -> chunk -> embed -> index
           eval harness (retrieval metrics + RAGAS + regression gate)
           training (QLoRA, separate env) -> adapter -> served by vLLM
```

**Rule:** the LLM client talks to an OpenAI-compatible endpoint. Ollama, vLLM, and a hosted
judge model are then interchangeable by changing config, which is what makes fair
benchmarks and A/B evals possible.

---

## 3. Environment and hardware budget

Detected on this machine: **NVIDIA L4, 23 GB VRAM**, driver 580, 30 GB RAM, 149 GB free
disk, Python 3.10, Docker 29. `ollama` and `uv` are not installed.

Implications:
- **Ada architecture (L4)** supports bf16 and FP8, so it is fine for modern vLLM.
- Approximate VRAM (weights only, before KV cache):
  - 7–8B in bf16: about 15–16 GB. Fits, with limited KV cache.
  - 7–8B in 4-bit (AWQ/GPTQ/NF4): about 5–6 GB.
  - bge-m3 embedder: about 1–2 GB. bge-reranker-v2-m3: about 1–2 GB.
- QLoRA of a 7–8B model fits in 24 GB with a small batch, gradient checkpointing, and
  sequence length of about 1–2k. **Stop vLLM/Ollama while training.**
- Start with a 1.5B–3B model for fast iteration, then repeat the key experiments on a 7–8B.
- Use **separate virtualenvs**: `.venv-app` (RAG, MCP, eval), `.venv-train`
  (transformers/PEFT/TRL/bitsandbytes), `.venv-vllm` (vLLM). This is deliberate.
- Prefer `uv` or `pyenv` for Python 3.11+ if a library requires it. Pin everything with a
  lock file and record `pip freeze` with every result.

Model candidates (verify the current state of each on Hugging Face before choosing):
- LLM: Qwen2.5/Qwen3 instruct family (strong multilingual and Arabic), Llama 3.x instruct,
  Gemma, or Arabic-focused models such as Jais/ALLaM if a size that fits is available.
- Embeddings: `BAAI/bge-m3` (multilingual, dense + sparse), with `multilingual-e5` as a
  comparison.
- Reranker: `BAAI/bge-reranker-v2-m3`.
- Judge for RAGAS: should be **at least as strong as the generator**. A small local judge
  gives unreliable scores, especially in Arabic. Options are a larger quantised local model,
  or a hosted model used sparingly with cached results. Record which judge produced each
  score.

---

## 4. Data plan (decide this first)

1. **Corpus**: a bounded bilingual domain of about 200–500 documents. Options:
   - Arabic and English Wikipedia articles on one topic area (CC BY-SA, attribute it).
   - Public bilingual policy or government documents with permissive licences. This
     suits the `lookup_policy` tool well.
   - Don't use private or employer data.
2. **Existing QA sets for sanity checks and training signal**: TyDi QA, XQuAD, MLQA, MKQA,
   Arabic SQuAD/ARCD. Check each licence.
3. **Your own golden set** (this is the one that matters):
   - 150–300 questions, manually reviewed, each with the gold passage(s) and a reference
     answer.
   - Stratify: ar→ar, en→en, ar→en, en→ar, multi-passage, and **unanswerable** questions
     (the correct behaviour is to abstain).
   - Generate candidates with an LLM, then review every one by hand. Note the
     reviewed/accepted rate; it is a good interview talking point.
4. **Splits**: split by *document*, not by question. Train/dev/test must not share source
   documents, or fine-tuning results will be inflated. Freeze the test set; do not tune on
   it.
5. Keep dataset cards (source, licence, creation method, known limitations) in `data/`.

---

## 5. Phases

Each phase has: **learn**, **build**, **deliverable**, and **done when**.

### Phase 0: Foundations and environment (week 1)
- **Learn**: how transformers tokenise (Arabic tokenisation cost), embeddings and cosine
  similarity, what a KV cache is, what the OpenAI chat API looks like.
- **Build**: repo scaffold (done), virtualenvs, `scripts/check_env.py`, Ollama running one
  small model, a 20-line script calling it through the OpenAI client.
- **Deliverable**: `docs/ENVIRONMENT.md` with versions of the driver, CUDA, Python, torch,
  and each model.
- **Done when**: `make check-env` is green and a chat completion works against a local server.

### Phase 1: Bilingual RAG baseline (weeks 1–3)
- **Learn**: chunking trade-offs, dense vs sparse vs hybrid retrieval, reciprocal rank
  fusion, reranking, prompt grounding and citation formats.
- **Build** (no LangChain or LlamaIndex yet, so you understand every moving part):
  1. Ingestion and Arabic/English normalisation (with unit tests).
  2. Chunking with metadata (doc id, language, section, offsets).
  3. bge-m3 embeddings into Qdrant (Docker). Add BM25 or bge-m3 sparse for hybrid, and the
     reranker.
  4. Generation with a grounded prompt, **citations**, and abstention when evidence is
     insufficient. Answer in the question's language.
  5. CLI and a minimal FastAPI `/ask` endpoint.
- **Deliverable**: working `ask` CLI, config-driven (YAML) so chunk size, k, embedder, and
  prompt version are all swappable.
- **Done when**: you can answer Arabic and English questions with citations, and a
  retrieval ablation (dense vs hybrid vs hybrid+rerank) exists, even if only qualitative
  at this point.
- **Pitfalls**: PDF extraction that reverses Arabic text, mixed RTL/LTR, chunks that split
  mid-sentence, and embeddings of un-normalised text.

### Phase 2: Evaluation harness and regression gates (weeks 3–5)
*(Moved up from your item 4. Everything after this is measured with it.)*
- **Learn**: retrieval metrics (recall@k, MRR, nDCG), RAGAS metrics (context precision,
  context recall, faithfulness, answer relevancy), LLM-as-judge bias and variance,
  bootstrap confidence intervals.
- **Build**:
  1. Golden set (section 4) in a versioned JSONL.
  2. Layer 1, deterministic and cheap: recall@k, MRR, nDCG, language-match rate, citation
     validity (does the cited chunk exist and contain the claim support?), abstention
     accuracy on unanswerable questions.
  3. Layer 2, RAGAS (or TruLens): context precision, context recall, faithfulness, answer
     relevancy. Temperature 0, cache judge calls, record the judge model and version.
  4. Run ablations: chunk size, embedder, k, reranker on/off, prompt v1 vs v2.
  5. **Regression gate**: store a baseline JSON; a script compares a new run to it per
     metric and per language slice, and fails if a drop exceeds a tolerance **beyond the
     noise** (use bootstrap CI or repeated runs). Two tiers: smoke (about 30 questions, every
     PR) and full (before release).
- **Deliverable**: `eval/run_eval.py`, `reports/baseline.md`, and `make eval-smoke` /
  `make eval-full`.
- **Done when**: deliberately breaking something (a worse prompt, k=1, a wrong embedder)
  makes the gate fail, and reverting it makes the gate pass.
- **Pitfalls**: judge scoring Arabic poorly, so spot-check 20–30 judged samples by hand
  and report judge-vs-human agreement. Tuning on the test split. Reporting one number
  without variance.

### Phase 3: vLLM serving and benchmarking (weeks 5–7)
- **Learn**: continuous batching, PagedAttention and KV-cache sizing, prefix caching,
  `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs`, quantisation (AWQ,
  GPTQ, FP8), TTFT vs TPOT vs end-to-end latency, tail latency.
- **Build**:
  1. Validate the environment with a small model first, then the 7–8B.
  2. Serve with vLLM (OpenAI-compatible API) and point the RAG app at it by config only.
  3. Benchmark harness (`bench/`): concurrency sweep (1, 2, 4, 8, 16, 32), fixed prompt
     and output length distributions drawn from **real RAG prompts** (long context, short
     answer), warm-up, and enough requests for stable p50/p95/p99.
  4. Metrics: TTFT, inter-token latency/TPOT, end-to-end latency, output tokens/s, requests/s,
     and peak GPU memory (`pynvml`, sampled during the run).
  5. Same model, same quantisation, same prompts against Ollama. If formats differ (GGUF
     vs HF), say so in the report.
  6. Optional: bf16 vs AWQ vs FP8, prefix caching on/off, and served LoRA adapters
     (`--enable-lora`).
- **Deliverable**: serve config/script, `bench/` harness, `reports/serving_benchmark.md`
  with tables and plots, and exact versions and flags.
- **Done when**: anyone can reproduce your numbers with one command, and the report states
  conditions, caveats, and what you'd change.
- **Pitfalls**: comparing different quantisations, tiny prompts that hide prefill cost, no
  warm-up, reporting averages only, and leaving other processes on the GPU.

### Phase 4: MCP server (weeks 7–8)
- **Learn**: MCP concepts (tools, resources, prompts), the stdio and Streamable HTTP
  transports, JSON Schema validation, and why tool output is an injection surface.
- **Build** with the official MCP Python SDK:
  1. Tools: `search_documents(query, language?, top_k)`, `get_document_details(doc_id)`,
     `lookup_policy(policy_id | query)`. Validate with Pydantic: bounds on `top_k`, query
     length caps, and an enum for language.
  2. Read-only, least-privilege tools. No arbitrary file or shell access.
  3. Structured errors (not-found, invalid input, rate-limited) that the model can recover from.
  4. Auth: for HTTP transport, bearer token or OAuth plus per-tool scopes. For stdio,
     document the trust model. Rate limiting.
  5. **Audit log**: structured JSON per call (timestamp, caller, tool, args hash, latency,
     status), with no secrets or full document bodies in logs.
  6. Treat retrieved text as untrusted: delimit it, tell the model it is data, strip or flag
     instruction-like content, and include a prompt-injection test document in the corpus.
  7. Tests with the MCP Inspector and a Python MCP client; a demo where the LLM calls the
     tools end to end.
- **Deliverable**: `mcp_server/`, `examples/mcp_client_demo.py`, tests, and a short threat
  model in `docs/MCP_SECURITY.md`.
- **Done when**: invalid input, an unauthorised call, and a prompt-injection document each
  produce the intended safe behaviour in a test.

### Phase 5: LoRA / QLoRA fine-tuning (weeks 8–10)
- **Learn**: why low-rank adapters work, the rank/alpha/target-modules/dropout
  hyperparameters, NF4 and double quantisation, gradient checkpointing, chat templates and
  loss masking (train on the assistant tokens only), and overfitting on small data.
- **Build**:
  1. Training data: RAG-formatted examples (question + retrieved context -> grounded,
     cited answer in the right language, plus abstention examples). Train/dev/test split by
     document (section 4).
  2. Baseline first: measure the untuned model with the Phase 2 harness.
  3. QLoRA with Transformers + PEFT + TRL (`SFTTrainer`) + bitsandbytes on a 1.5–3B model,
     then on a 7–8B. Log loss curves, with seeds and configs in version control.
  4. Compare base vs LoRA vs QLoRA: quality (Phase 2 metrics on the **frozen test split**),
     peak training memory, training time, adapter size.
  5. Optionally ablate rank (8/16/64) and data size.
  6. Serve the adapter through vLLM (`--enable-lora`) or merge it, and re-run the RAG eval
     end to end. This links Phases 2, 3, and 5.
- **Deliverable**: `train/` scripts and configs, adapter checkpoint (hosted on the Hugging
  Face Hub, not committed to git), and `reports/finetune_eval.md` with before/after tables.
- **Done when**: you can explain, with numbers, what improved, what did not, and why.
  **A null or negative result, reported honestly, is still a valid and respectable
  outcome.**
- **Pitfalls**: training on the test documents, training on prompt tokens, a wrong chat
  template, comparing a tuned model with a different prompt than the base, and
  catastrophic forgetting of English or Arabic.

### Phase 6: Agentic layer (weeks 10–12)
*(New. This is what moves the project from "RAG app" toward "agentic AI engineer".)*
- **Learn**: the tool-calling loop, ReAct vs plan-and-execute, structured outputs, state
  machines (LangGraph), failure modes (looping, wrong tool, over-calling), guardrails, and
  trajectory-level evaluation.
- **Build**:
  1. **Write the agent loop by hand first**: LLM + MCP tools + max-steps + error recovery.
     vLLM needs tool-calling flags (`--enable-auto-tool-choice` with the right
     `--tool-call-parser` for your model). Check the model supports it.
  2. Agentic RAG techniques: query rewriting, decomposition for multi-hop questions,
     retrieve-then-grade-then-retry (corrective RAG), and language routing.
  3. Re-implement the same flow in **LangGraph** and compare complexity, debuggability, and
     behaviour with the hand-written version.
  4. Guardrails: step and token budgets, tool allow-list, input/output validation, and a
     human-approval hook for any hypothetical write tool.
  5. **Agent evaluation**: tool-call accuracy (right tool, right args), task success,
     steps per task, cost/latency per task, and a small set of adversarial cases
     (injection, ambiguous query, tool failure).
- **Deliverable**: `agent/` (hand-written and LangGraph versions), an agent eval set and
  report, and a short design note comparing the two.
- **Done when**: multi-hop and cross-lingual questions that the single-shot RAG fails on
  are measurably better, and the gate covers agent metrics too.

### Phase 7: Production hygiene (weeks 12–13)
- **Build**: tracing of retrieval, LLM, and tool spans (Langfuse or Arize Phoenix, via
  OpenTelemetry); experiment tracking (MLflow or W&B, or versioned JSON in git); a
  Docker Compose stack (app, Qdrant, vLLM, MCP server); GitHub Actions running lint,
  unit tests, and the smoke eval (a CPU-only or mocked-LLM path so CI doesn't need a GPU); a
  model/prompt/config **version stamp** on every response and eval run.
- **Deliverable**: `docker-compose.yml`, `.github/workflows/ci.yml`, a tracing dashboard
  screenshot, and a `make release-check` that runs the full gate.

### Phase 8: Write-up (week 13+)
- README with the architecture diagram, result tables, a 2-minute demo GIF/video, "what I
  would do next", and honest limitations. Optionally a blog post on the vLLM benchmark or
  the Arabic eval findings. Fill in the resume bullets using real numbers (section 8).

---

## 6. Suggested repository layout (created as each phase starts)

```
configs/            YAML for retrieval, generation, serving, training, eval
data/               raw/, processed/, golden/ (small files and cards only; large data via DVC or the Hub)
src/bilingual_rag/  ingestion/, retrieval/, generation/, llm_client/, service/
mcp_server/         tools, schemas, auth, audit logging
agent/              hand-written loop, LangGraph version
eval/               metrics, RAGAS runner, regression gate, baselines/
bench/              vLLM/Ollama benchmark harness
train/              QLoRA scripts and configs (own environment)
reports/            baseline.md, serving_benchmark.md, finetune_eval.md
docs/               ROADMAP.md, ENVIRONMENT.md, MCP_SECURITY.md, dataset cards
scripts/            check_env.py, helpers
tests/              unit and integration tests
```

Never commit model weights, `.env`, API keys, or large datasets (the `.gitignore` already
excludes them).

---

## 7. Timeline (about 13 weeks part-time; adjust freely)

| Weeks | Phase |
|-------|-------|
| 1 | 0. Foundations |
| 1–3 | 1. RAG baseline |
| 3–5 | 2. Evaluation and regression gates |
| 5–7 | 3. vLLM serving and benchmarks |
| 7–8 | 4. MCP server |
| 8–10 | 5. LoRA / QLoRA |
| 10–12 | 6. Agentic layer |
| 12–13 | 7. Production hygiene, then 8. Write-up |

Reordering rationale: vLLM comes before fine-tuning so you can serve the adapter, and MCP
comes before the agent so the agent has real tools to call.

---

## 8. Resume integrity checklist

Only claim what you can point to. Mapping:

| Resume claim | Evidence in repo |
|--------------|------------------|
| Arabic–English assistant, multilingual embeddings, vector retrieval, local LLM | Phase 1 code, ablation table |
| RAG evaluation with RAGAS, retrieval and answer-quality metrics | `reports/baseline.md`, `eval/` |
| Regression checks for prompt and retrieval changes | gate script plus a CI run that fails on a deliberately bad change |
| MCP server exposing document-search tools | `mcp_server/`, demo, tests |
| LoRA/QLoRA with PEFT | `train/`, adapter on the Hub, before/after report |
| vLLM latency, throughput, GPU memory | `reports/serving_benchmark.md` |

Resume bullet template (fill with **your measured** numbers; delete anything you did not do):

> Built a bilingual (Arabic–English) RAG assistant with bge-m3 embeddings, hybrid retrieval
> and reranking over a **N**-document corpus, served by a local open-source LLM; improved
> context recall from **X** to **Y** and faithfulness from **A** to **B** on a
> **N**-question manually reviewed test set (RAGAS).
> Added a CI regression gate that blocks changes dropping retrieval/answer metrics beyond
> a bootstrap-CI tolerance.
> Built an MCP server (**3** tools, schema validation, token auth, audit logging) and an
> agent loop that uses it.
> Fine-tuned **MODEL** with QLoRA (PEFT/TRL) and evaluated it against the base model;
> benchmarked vLLM vs Ollama on an NVIDIA L4 (**X** tokens/s, p95 TTFT **Y** ms at
> concurrency **C**, **Z** GB peak VRAM).

---

## 9. Learning resources (official docs first)

- Hugging Face: Transformers, PEFT, TRL, and bitsandbytes docs; the LLM course.
- vLLM docs (serving, benchmarks, quantisation, LoRA, tool calling).
- Model Context Protocol docs and the MCP Python SDK.
- RAGAS and TruLens docs; Hamel Husain's writing on LLM evals.
- Qdrant docs (hybrid search). The BGE-M3 paper.
- LangGraph docs; Anthropic's "Building effective agents".
- The QLoRA paper (Dettmers et al.) and the LoRA paper (Hu et al.).

Library APIs change quickly. Check the current docs and release notes before pinning
versions, and treat this roadmap as the plan, not as a source of exact flags or versions.
