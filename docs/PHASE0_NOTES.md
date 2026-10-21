# Phase 0 learning notes

All numbers below were measured on 2026-10-22 with the scripts in `scripts/`. They come from
5 hand-written sentence pairs, so treat them as illustrations, not benchmarks.

## 1. Tokenisation: Arabic costs more tokens (`scripts/token_cost.py`)

LLMs read **tokens**, not words. Context limits, KV-cache memory, latency, and API cost all
scale with token count.

| Tokenizer | EN tokens | AR tokens | AR/EN |
|-----------|-----------|-----------|-------|
| Qwen2.5 (our LLM) | 60 | 107 | 1.78 |
| bge-m3 (our embedder) | 78 | 94 | 1.21 |
| GPT-2 (English-centric) | 62 | 309 | 4.98 |

Takeaways:
- With Qwen2.5, the same content costs about 1.8x more tokens in Arabic. An Arabic RAG
  prompt is longer, slower to prefill, and uses more KV cache than the English one.
  This is why Phase 3 benchmarks must use Arabic prompts too, not only English ones.
- The ratio depends heavily on the tokenizer: a multilingual vocabulary (bge-m3) is much
  closer to parity than an English-centric one (GPT-2).
- Qwen's Arabic tokens are often sub-word fragments (`ن`, `مو`, `ذ`, `ج` for "نموذج").
- **Diacritics** (tashkeel) turn the 1-token word `كتب` into 6 tokens, and in bge-m3 the
  embedding similarity between the two forms was only 0.79. That is the concrete reason
  Phase 1 normalises Arabic text before chunking and embedding.

## 2. Embeddings and cosine similarity (`scripts/embedding_demo.py`)

- bge-m3 gives 1024-dimensional vectors and accepts up to 8192 tokens.
- With normalised vectors, cosine similarity is the dot product.
- Arabic query -> English sentence: 5 of 5 correct top-1 matches. Average similarity was
  0.87 for translation pairs vs 0.43 for mismatched pairs. This is what makes cross-lingual
  retrieval (ar->en, en->ar) possible with one embedder.
- An off-topic query ("capital of Japan") still got a nearest neighbour, with score 0.31.
  Nearest-neighbour search **always returns something**, so a RAG system needs a score
  threshold or a grounding check to abstain. This is a Phase 1 and Phase 2 design point.
- Scores are not comparable across models, so any threshold has to be tuned per embedder.

## 3. KV cache

During generation, each new token attends to all previous tokens. The attention **keys and
values** of past tokens never change, so the server stores them (the KV cache) instead of
recomputing them for each new token. That makes decoding much cheaper, but it uses GPU
memory that grows with sequence length and with the number of concurrent requests.

Size per token = 2 (K and V) x layers x KV heads x head_dim x bytes per value.
Computed from the real model configs, fp16 cache:

| Model | Layers | KV heads | Head dim | KV per token | 4096 tokens | 32 seqs x 4096 |
|-------|--------|----------|----------|--------------|-------------|----------------|
| Qwen2.5-3B-Instruct | 36 | 2 | 128 | 36 KiB | 144 MiB | 4.5 GiB |
| Qwen2.5-7B-Instruct | 28 | 4 | 128 | 56 KiB | 224 MiB | 7.0 GiB |

Why this matters later:
- Grouped-query attention (few KV heads versus query heads) is why these models have small
  caches. This is a calculation, not a measurement, and servers add overhead.
- On the 23 GB L4, a 7B model in bf16 takes about 15 GB of weights, leaving only a few GB for
  KV cache. That caps concurrency, which is the core Phase 3 trade-off.
- Arabic's higher token count (section 1) multiplies KV memory per request.
- vLLM's PagedAttention manages this cache in blocks. Prefix caching reuses it across
  requests that share a prompt prefix, such as a RAG system prompt.

## 4. OpenAI chat API shape (`scripts/chat_demo.py`)

A request is `model` plus a list of `messages` with roles `system`, `user` and `assistant`.
The response has `choices[0].message.content` and a `usage` block with prompt and completion
token counts. Ollama and vLLM both implement this, so the client only changes `base_url`.
Setting `temperature=0` makes output close to deterministic, which matters for evaluation.
