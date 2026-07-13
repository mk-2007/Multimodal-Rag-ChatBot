"""
main.py
-------
Orchestrates the full Multimodal RAG pipeline end-to-end:

  1. extract.py     - pull text/headings, table page images, figure images from the PDF
  2. vision.py       - Gemini Vision describes figures & transcribes tables
  3. chunking.py      - unify everything into modality-tagged chunks
  4. embeddings.py    - embed all chunks into ONE shared vector store (Gemini embeddings)
  5. retrieval.py     - embed a query, cosine-search the shared store
  6. generate.py      - Gemini answers grounded ONLY in the retrieved context

Run:
    export GEMINI_API_KEY="your-key-here"
    python3 main.py                  # full pipeline + demo queries
    python3 main.py --skip-build     # reuse existing store, just run demo queries
    python3 main.py --query "..."    # ask a single custom question after building
"""

import argparse
import json
import time

from config import require_api_key, DATA_DIR, STORE_DIR, OUTPUT_DIR
from extract import run_extraction
from vision import run_vision_pass
from chunking import run_chunking, Chunk
from embeddings import build_vector_store, make_client as make_embed_client
from retrieval import retrieve, VectorStore, make_client as make_retrieval_client
from generate import generate_answer, make_client as make_generate_client

# Sample queries deliberately chosen to each hit a DIFFERENT modality in the
# shared vector store, to demonstrate true multimodal retrieval:
DEMO_QUERIES = [
    # -> should retrieve from a TABLE chunk (Table 2, BLEU scores)
    "What BLEU score did the Transformer (big) model achieve on the WMT 2014 "
    "English-to-German translation task, and how does it compare to GNMT + RL?",

    # -> should retrieve from a FIGURE chunk (Figure 1, architecture diagram)
    "Describe the overall encoder-decoder architecture of the Transformer as "
    "shown in the model diagram -- what are the main blocks and how do they connect?",

    # -> should retrieve from a TEXT chunk (Section 3.2.2)
    "What is multi-head attention and why is it used instead of a single "
    "attention function?",
]


def build_pipeline():
    t0 = time.time()
    require_api_key()

    print("\n" + "=" * 70)
    print("STEP 1/5: EXTRACTION")
    print("=" * 70)
    manifest = run_extraction()

    print("\n" + "=" * 70)
    print("STEP 2/5: VISION UNDERSTANDING (figures + tables)")
    print("=" * 70)
    manifest = run_vision_pass(manifest)

    print("\n" + "=" * 70)
    print("STEP 3/5: CHUNKING")
    print("=" * 70)
    chunks = run_chunking(manifest)

    print("\n" + "=" * 70)
    print("STEP 4/5: EMBEDDING INTO SHARED VECTOR STORE")
    print("=" * 70)
    build_vector_store(chunks)

    print(f"\n[main] Pipeline build complete in {time.time() - t0:.1f}s")


def run_demo(queries=None, save_path=None):
    queries = queries or DEMO_QUERIES
    embed_client = make_retrieval_client()
    gen_client = make_generate_client()
    store = VectorStore()

    results_log = []
    print("\n" + "=" * 70)
    print("STEP 5/5: RETRIEVAL + GROUNDED GENERATION (demo queries)")
    print("=" * 70)

    for q in queries:
        print(f"\n{'#' * 70}\nQUERY: {q}\n{'#' * 70}")
        retrieved = retrieve(q, client=embed_client, store=store)

        print("\n-- Retrieved context (top matches, mixed modalities) --")
        for r in retrieved:
            print(f"  [{r.score:.3f}] modality={r.modality:<7} source={r.source_ref:<12} "
                  f"page={r.page:<3} :: {r.heading_context[:55]}")

        answer = generate_answer(q, retrieved, client=gen_client)
        print("\n-- Grounded answer --")
        print(answer)

        results_log.append({
            "query": q,
            "retrieved_context": [
                {
                    "score": r.score, "modality": r.modality, "source_ref": r.source_ref,
                    "page": r.page, "heading_context": r.heading_context,
                    "content": r.content, "image_path": r.image_path,
                }
                for r in retrieved
            ],
            "answer": answer,
        })

    save_path = save_path or (OUTPUT_DIR / "demo_results.json")
    save_path.write_text(json.dumps(results_log, indent=2))
    print(f"\n[main] Wrote full query+context+answer log -> {save_path}")

    md_path = OUTPUT_DIR / "demo_results.md"
    write_markdown_report(results_log, md_path)
    print(f"[main] Wrote human-readable report -> {md_path}")

    return results_log


def write_markdown_report(results_log, path):
    lines = ["# Multimodal RAG Demo Results", "",
             "Source document: *Attention Is All You Need* (Vaswani et al., 2017)", ""]
    for i, r in enumerate(results_log, start=1):
        lines.append(f"## Query {i}: {r['query']}")
        lines.append("")
        lines.append("**Retrieved context (shared vector store, mixed modalities):**")
        lines.append("")
        for c in r["retrieved_context"]:
            lines.append(f"- `[{c['score']:.3f}]` **{c['modality']}** | `{c['source_ref']}` "
                          f"| page {c['page']} | _{c['heading_context'][:70]}_")
        lines.append("")
        lines.append("**Grounded answer:**")
        lines.append("")
        lines.append(r["answer"])
        lines.append("")
        lines.append("---")
        lines.append("")
    path.write_text("\n".join(lines))


def run_image_query_and_save(image_path, question=None, save_path=None):
    from image_query import run_image_query, make_client as make_iq_client
    from retrieval import VectorStore

    client = make_iq_client()
    store = VectorStore()

    print("\n" + "=" * 70)
    print("IMAGE QUERY: describe -> retrieve -> generate (dual-grounded)")
    print("=" * 70)
    result = run_image_query(image_path, question, client=client, store=store)

    print("\n-- Grounded answer --")
    print(result["answer"])

    save_path = save_path or (OUTPUT_DIR / "image_query_result.json")
    save_path.write_text(json.dumps(result, indent=2))
    print(f"\n[main] Wrote image query result -> {save_path}")

    md_path = save_path.with_suffix(".md")
    write_image_query_markdown(result, md_path)
    print(f"[main] Wrote human-readable report -> {md_path}")

    return result


def write_image_query_markdown(result, path):
    lines = ["# Multimodal RAG - Image Query Result", "",
             "Source document: *Attention Is All You Need* (Vaswani et al., 2017)", ""]
    lines.append(f"**Query image:** `{result['image_path']}`")
    lines.append("")
    if result.get("user_question"):
        lines.append(f"**Question asked:** {result['user_question']}")
        lines.append("")
    lines.append("**Stage 1 - image described for search:**")
    lines.append("")
    lines.append(f"> {result['search_description']}")
    lines.append("")
    lines.append("**Retrieved context (shared vector store, mixed modalities):**")
    lines.append("")
    for c in result["retrieved_context"]:
        lines.append(f"- `[{c['score']:.3f}]` **{c['modality']}** | `{c['source_ref']}` "
                      f"| page {c['page']} | _{c['heading_context'][:70]}_")
    lines.append("")
    lines.append("**Grounded answer (dual-grounded: image + retrieved context):**")
    lines.append("")
    lines.append(result["answer"])
    lines.append("")
    path.write_text("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG over 'Attention Is All You Need'")
    parser.add_argument("--skip-build", action="store_true",
                         help="Skip extraction/vision/embedding, reuse existing store")
    parser.add_argument("--query", type=str, default=None,
                         help="Run a single custom TEXT query instead of the demo set")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to an image to use as the query (optionally combine with --query)")
    args = parser.parse_args()

    if not args.skip_build:
        build_pipeline()

    if args.image:
        run_image_query_and_save(args.image, question=args.query)
    else:
        queries = [args.query] if args.query else None
        run_demo(queries=queries)