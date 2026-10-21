#!/usr/bin/env python3
"""Phase 0 exercise: how many tokens does the same sentence cost in English vs Arabic?

Why it matters: context window, KV-cache memory, latency, and cost all scale with *tokens*,
not characters or words. If Arabic needs more tokens per sentence, every Arabic RAG prompt
is longer, slower, and uses more KV cache than its English twin.

Only tokenizer files are downloaded (small), not model weights.
"""

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TOKENIZERS = {
    "Qwen2.5 (our LLM)": "Qwen/Qwen2.5-3B-Instruct",
    "bge-m3 (our embedder)": "BAAI/bge-m3",
    "GPT-2 (English-centric)": "openai-community/gpt2",
}

# Hand-written parallel sentences (same meaning in both languages).
PAIRS = [
    (
        "The company must keep customer data for at least five years.",
        "يجب على الشركة الاحتفاظ ببيانات العملاء لمدة خمس سنوات على الأقل.",
    ),
    (
        "Employees are entitled to thirty days of paid annual leave.",
        "يحق للموظفين الحصول على إجازة سنوية مدفوعة الأجر مدتها ثلاثون يوماً.",
    ),
    (
        "Retrieval-augmented generation grounds language model answers in source documents.",
        "يعتمد التوليد المعزز بالاسترجاع على ربط إجابات نموذج اللغة بالمستندات المصدرية.",
    ),
    (
        "Please submit the application form before the end of the month.",
        "يرجى تقديم نموذج الطلب قبل نهاية الشهر.",
    ),
    (
        "The committee approved the new data protection policy yesterday.",
        "وافقت اللجنة على سياسة حماية البيانات الجديدة أمس.",
    ),
]

# Same Arabic word with and without diacritics (tashkeel): why we normalise in Phase 1.
DIACRITICS = ("كتب", "كَتَبَ")


def _byte_decoder() -> dict[str, int]:
    """Invert the GPT-2 byte->unicode table used by byte-level BPE tokenizers."""
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, n = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(keep, chars, strict=True)}


_BYTE_DECODER = _byte_decoder()


def readable(piece: str) -> str:
    """Show a byte-level BPE token as text. A piece may be a partial UTF-8 character."""
    raw = bytes(_BYTE_DECODER[ch] for ch in piece)
    return raw.decode("utf-8", errors="replace").replace(" ", "_")


def load(repo: str) -> Tokenizer:
    return Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))


def main() -> None:
    tokenizers = {name: load(repo) for name, repo in TOKENIZERS.items()}

    for name, tok in tokenizers.items():
        en = [len(tok.encode(e).ids) for e, _ in PAIRS]
        ar = [len(tok.encode(a).ids) for _, a in PAIRS]
        print(f"\n== {name}")
        print(f"{'pair':>4} {'EN tok':>7} {'AR tok':>7} {'AR/EN':>6}")
        for i, (e, a) in enumerate(zip(en, ar, strict=True), 1):
            print(f"{i:>4} {e:>7} {a:>7} {a / e:>6.2f}")
        print(f"{'sum':>4} {sum(en):>7} {sum(ar):>7} {sum(ar) / sum(en):>6.2f}")

    qwen = tokenizers["Qwen2.5 (our LLM)"]
    print("\n== How Qwen splits sentence 4 (tokens shown as pieces)")
    for label, text in (("EN", PAIRS[3][0]), ("AR", PAIRS[3][1])):
        print(f"{label}: {[readable(t) for t in qwen.encode(text).tokens]}")

    print("\n== Diacritics change the token sequence (Qwen)")
    for text in DIACRITICS:
        enc = qwen.encode(text)
        print(f"{text!r}: {len(enc.ids)} tokens {[readable(t) for t in enc.tokens]}")


if __name__ == "__main__":
    main()
