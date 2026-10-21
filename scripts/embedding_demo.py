#!/usr/bin/env python3
"""Phase 0 exercise: embeddings and cosine similarity with bge-m3.

An embedding model maps text to a vector so that texts with similar *meaning* land close
together, even across languages. Retrieval is then "find the nearest vectors to the query".

Cosine similarity = dot(a, b) / (|a| |b|). With normalised vectors it is just the dot
product, which is why vector databases often store normalised embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from token_cost import PAIRS  # same hand-written parallel sentences

MODEL = "BAAI/bge-m3"


def main() -> None:
    model = SentenceTransformer(MODEL, device="cuda")
    print(f"model: {MODEL}, dim={model.get_sentence_embedding_dimension()}, "
          f"max_seq_length={model.max_seq_length}")

    english = [e for e, _ in PAIRS]
    arabic = [a for _, a in PAIRS]
    # normalize_embeddings=True -> unit vectors, so dot product == cosine similarity
    en_vecs = model.encode(english, normalize_embeddings=True)
    ar_vecs = model.encode(arabic, normalize_embeddings=True)

    # Rows: Arabic queries. Columns: English documents. The diagonal is the true translation.
    sim = ar_vecs @ en_vecs.T
    np.set_printoptions(precision=2, suppress=True)
    print("\nCosine similarity, Arabic query (row) vs English sentence (column):")
    print(sim)

    top1 = sim.argmax(axis=1)
    hits = int((top1 == np.arange(len(PAIRS))).sum())
    print(f"\nAr->En top-1 correct: {hits}/{len(PAIRS)}  (tiny hand-made set, not a benchmark)")
    diag = np.diag(sim).mean()
    off = (sim.sum() - np.trace(sim)) / (sim.size - len(PAIRS))
    print(f"mean similarity: translation pairs {diag:.2f} vs mismatched pairs {off:.2f}")

    # Un-normalised Arabic: same word with and without diacritics.
    plain, marked = model.encode(["كتب", "كَتَبَ"], normalize_embeddings=True)
    print(f"\nsame word with/without diacritics: cosine = {float(plain @ marked):.2f}")

    # A question with no answer in the corpus: nearest neighbour still has a score.
    # Lesson for Phase 1/2: a top-1 hit always exists, so you need a threshold or a judge
    # to decide when to abstain.
    q = model.encode(["What is the capital of Japan?"], normalize_embeddings=True)[0]
    scores = en_vecs @ q
    best = int(scores.argmax())
    print(f"\nOff-topic query best match score: {scores[best]:.2f} -> {english[best]!r}")


if __name__ == "__main__":
    main()
