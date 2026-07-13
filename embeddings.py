"""
embeddings.py
-------------
STEP 4 of the pipeline: generate embeddings for every chunk (text, table,
image-description) via Gemini's embedding model, and STEP 5: store them
together in ONE shared vector store (a single numpy matrix + metadata list).

Because every chunk -- regardless of whether it came from body text, a
transcribed table, or a vision-generated figure description -- is embedded
with the SAME model into the SAME vector space, retrieval can naturally pull
back a mix of modalities for a single query.
"""

import json
import time
from dataclasses import asdict
from typing import List

import numpy as np
from google import genai
from google.genai import types

from config import (
    require_api_key,
    GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    DATA_DIR,
    STORE_DIR,
)
from chunking import Chunk


def make_client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


def embed_texts(client: genai.Client, texts: List[str], task_type: str,
                 batch_size: int = 8, max_retries: int = 3) -> List[List[float]]:
    """Embed a list of texts in batches. task_type is one of
    'RETRIEVAL_DOCUMENT' (for indexing chunks) or 'RETRIEVAL_QUERY' (for queries)."""
    all_vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                all_vectors.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                last_err = e
                print(f"  [embed] batch {i}-{i+len(batch)} attempt {attempt} failed: {e}")
                time.sleep(2 * attempt)
        else:
            raise RuntimeError(f"Embedding failed for batch starting at {i}: {last_err}")
    return all_vectors


def build_vector_store(chunks: List[Chunk]) -> dict:
    client = make_client()
    print(f"[embeddings] Embedding {len(chunks)} chunks "
          f"(model={GEMINI_EMBEDDING_MODEL}, dim={EMBEDDING_DIM}) ...")

    texts = [c.content for c in chunks]
    vectors = embed_texts(client, texts, task_type="RETRIEVAL_DOCUMENT")

    matrix = np.array(vectors, dtype=np.float32)
    # L2-normalize once so retrieval can use a plain dot product as cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    matrix = matrix / norms

    np.save(STORE_DIR / "vectors.npy", matrix)
    metadata = [asdict(c) for c in chunks]
    (STORE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[embeddings] Stored matrix {matrix.shape} -> {STORE_DIR/'vectors.npy'}")
    print(f"[embeddings] Stored metadata -> {STORE_DIR/'metadata.json'}")

    # modality breakdown for sanity
    from collections import Counter
    counts = Counter(c.modality for c in chunks)
    print(f"[embeddings] Modality breakdown in shared vector store: {dict(counts)}")

    return {"matrix": matrix, "metadata": metadata}


if __name__ == "__main__":
    chunks_raw = json.loads((DATA_DIR / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_raw]
    build_vector_store(chunks)
