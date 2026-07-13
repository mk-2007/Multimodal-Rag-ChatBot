"""
retrieval.py
------------
STEP 6 of the pipeline: given a user query, embed it with the SAME Gemini
embedding model (task_type=RETRIEVAL_QUERY) and find the top-k nearest chunks
in the shared vector store via cosine similarity (a plain dot product, since
vectors were pre-normalized when the store was built).

Because the store mixes text/table/figure chunks in one space, results for a
single query can naturally span modalities.
"""

import json
from dataclasses import dataclass
from typing import List

import numpy as np
from google import genai
from google.genai import types

from config import require_api_key, GEMINI_EMBEDDING_MODEL, EMBEDDING_DIM, STORE_DIR, TOP_K


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    modality: str
    page: int
    heading_context: str
    content: str
    source_ref: str
    image_path: str


def make_client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


class VectorStore:
    def __init__(self, store_dir=STORE_DIR):
        self.matrix = np.load(store_dir / "vectors.npy")
        self.metadata = json.loads((store_dir / "metadata.json").read_text())
        assert self.matrix.shape[0] == len(self.metadata), (
            "Vector count and metadata count mismatch -- rebuild the store."
        )

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K) -> List[RetrievedChunk]:
        qv = np.array(query_vector, dtype=np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        scores = self.matrix @ qv  # cosine similarity (both sides normalized)
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append(
                RetrievedChunk(
                    score=float(scores[idx]),
                    chunk_id=meta["chunk_id"],
                    modality=meta["modality"],
                    page=meta["page"],
                    heading_context=meta["heading_context"],
                    content=meta["content"],
                    source_ref=meta["source_ref"],
                    image_path=meta.get("image_path") or "",
                )
            )
        return results


def embed_query(client: genai.Client, query: str) -> np.ndarray:
    resp = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return np.array(resp.embeddings[0].values, dtype=np.float32)


def retrieve(query: str, top_k: int = TOP_K, client: genai.Client = None,
             store: VectorStore = None) -> List[RetrievedChunk]:
    client = client or make_client()
    store = store or VectorStore()
    qvec = embed_query(client, query)
    return store.search(qvec, top_k=top_k)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What BLEU score did the Transformer achieve on English-to-German?"
    results = retrieve(query)
    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r.score:.3f}] ({r.modality}) {r.source_ref} - {r.heading_context[:60]}")
        print(f"    {r.content[:150]}...")
        print()
