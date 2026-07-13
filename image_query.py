"""
image_query.py
--------------
Extends the pipeline to accept an IMAGE as the query, not just text.

Why this is needed: the shared vector store was built with gemini-embedding-001,
which is text-only -- it cannot embed an image directly into the same space.
So an image query needs a bridge step before it can reuse retrieve()/generate().

The bridge, in two stages:

  1. DESCRIBE: Gemini Vision looks at your image and writes a short, search-
     oriented description of it (what it shows, its structure/components).
     This reuses the exact same client call pattern as vision.py's figure
     description step during indexing -- consistent with how Figures 1-5
     were themselves turned into text before being embedded.

  2. RETRIEVE + GENERATE: that description becomes the query text, fed into
     the EXACT SAME retrieve() from retrieval.py -- no special-casing needed
     downstream, because by this point it's just a normal text query.

  3. (Bonus) DUAL GROUNDING: unlike a normal text query, your original image
     bytes are ALSO attached to the final generate_content call alongside the
     retrieved paper context -- so Gemini can directly compare what's in your
     image against what the paper's figures/tables/text actually say, rather
     than reasoning only from a text summary of your image.
"""

import json

from google import genai
from google.genai import types

from config import require_api_key, GEMINI_VISION_MODEL, GEMINI_GENERATION_MODEL
from retrieval import retrieve, VectorStore, RetrievedChunk
from generate import format_context, SYSTEM_INSTRUCTION

DESCRIBE_FOR_SEARCH_PROMPT = """Look at this image and write a concise (2-4 sentence) \
search-oriented description of what it shows: its subject, structure, and any labels, \
numbers, or components visible. This description will be used to search a paper's \
indexed content for related material, so focus on concrete, matchable details rather \
than artistic interpretation."""


def make_client() -> genai.Client:
    return genai.Client(api_key=require_api_key())


def describe_image_for_search(client: genai.Client, image_path: str) -> str:
    """Stage 1: turn the query image into search-oriented text."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            DESCRIBE_FOR_SEARCH_PROMPT,
        ],
    )
    return response.text.strip()


def generate_answer_with_image(
    client: genai.Client,
    user_question: str,
    image_path: str,
    chunks: list[RetrievedChunk],
) -> str:
    """Stage 3: final answer grounded in BOTH the retrieved paper context AND
    the actual query image (dual grounding), not just a text description of it."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    mime_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"

    context_str = format_context(chunks)
    prompt_text = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"The user has ALSO attached an image with their question -- it is included below. "
        f"Ground your answer in the retrieved CONTEXT below, and explicitly relate it to "
        f"what is shown in the attached image where relevant.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"QUESTION: {user_question}\n\nANSWER:"
    )
    response = client.models.generate_content(
        model=GEMINI_GENERATION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt_text,
        ],
    )
    return response.text.strip()


def run_image_query(image_path: str, user_question: str = None,
                     client: genai.Client = None, store: VectorStore = None) -> dict:
    """Full pipeline for an image-based query: describe -> retrieve -> generate."""
    client = client or make_client()
    store = store or VectorStore()

    print(f"[image_query] Describing query image for search: {image_path}")
    search_description = describe_image_for_search(client, image_path)
    print(f"[image_query]   -> \"{search_description[:120]}...\"")

    # If the user also typed a question, combine it with the image description
    # for retrieval -- gives the embedding both "what's in the image" and
    # "what the user actually wants to know" to match against.
    retrieval_query = (
        f"{search_description} {user_question}" if user_question else search_description
    )

    print("[image_query] Retrieving related chunks from the shared vector store ...")
    retrieved = retrieve(retrieval_query, client=client, store=store)
    for r in retrieved:
        print(f"  [{r.score:.3f}] modality={r.modality:<7} source={r.source_ref:<12} "
              f"page={r.page:<3} :: {r.heading_context[:55]}")

    print("[image_query] Generating grounded answer (with image attached) ...")
    question_for_answer = user_question or "What does this image show, and how does it relate to the paper?"
    answer = generate_answer_with_image(client, question_for_answer, image_path, retrieved)

    result = {
        "image_path": image_path,
        "search_description": search_description,
        "user_question": user_question,
        "retrieved_context": [
            {"score": r.score, "modality": r.modality, "source_ref": r.source_ref,
             "page": r.page, "heading_context": r.heading_context, "content": r.content,
             "image_path": r.image_path}
            for r in retrieved
        ],
        "answer": answer,
    }
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 image_query.py <path_to_image> [optional question]")
    img_path = sys.argv[1]
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
    result = run_image_query(img_path, question)
    print("\n--- ANSWER ---")
    print(result["answer"])
