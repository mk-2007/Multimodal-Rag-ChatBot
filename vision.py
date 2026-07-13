"""
vision.py
---------
STEP 2 of the pipeline: multimodal understanding of visual elements via Gemini's
vision capability.

Two distinct visual content types are handled with different prompts:

  * FIGURES (architecture diagram, attention mechanism diagrams, attention
    visualization plots) -> asked for a rich natural-language DESCRIPTION that
    captures structure, components, and relationships (since a figure's value in
    a RAG system is "what it shows / means", not literal pixel content).

  * TABLES (rendered page images of Table 1-4) -> asked for a faithful
    TRANSCRIPTION into a markdown table, since tables contain exact numbers
    that must be retrievable precisely (e.g. BLEU scores).

Both go through the same Gemini multimodal model. Results are cached to
data/vision_cache.json so re-runs don't re-spend API calls.
"""

import json
import time

from google import genai
from google.genai import types

from config import (
    require_api_key,
    GEMINI_VISION_MODEL,
    DATA_DIR,
)


def make_client() -> genai.Client:
    """Create a Gemini client using the API key from the GEMINI_API_KEY env var."""
    return genai.Client(api_key=require_api_key())

FIGURE_PROMPT = """You are analyzing a figure from the paper "Attention Is All You Need" \
(the Transformer architecture paper). Look at this image carefully and produce a detailed, \
self-contained description that would let someone answer questions about it WITHOUT seeing \
the image. Cover:

1. What the figure depicts (its title/subject, if visible).
2. The main components/blocks and how they connect or relate to each other (top-to-bottom \
or left-to-right flow, arrows, groupings).
3. Any labels, symbols, or annotations that carry meaning (e.g. Q/K/V, N×, Add & Norm, \
specific words/tokens shown, colors representing different attention heads).
4. The key takeaway or purpose this figure illustrates in the context of the paper.

Be precise and factual. Do not speculate beyond what is visibly shown. Write 150-300 words \
of plain prose (no markdown headers)."""

TABLE_PROMPT = """You are looking at a page image from the paper "Attention Is All You Need" \
that contains a data table. Transcribe the table FAITHFULLY into GitHub-flavored markdown \
table syntax, preserving every row, column header, and numeric value exactly as shown \
(including footnote markers if present). Also transcribe the table's caption text.

Respond in this exact structure:

CAPTION: <the exact caption text, e.g. "Table 2: The Transformer achieves better BLEU scores...">

MARKDOWN:
<the full markdown table>

NOTES: <1-3 sentences on what the table shows / how to interpret it, e.g. what the columns mean>
"""


def _call_vision(client, image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )
    return response.text.strip()


def run_vision_pass(manifest: dict, cache_path=None, max_retries: int = 3) -> dict:
    """Mutates and returns manifest with 'description' added to figures and
    'markdown' filled in for tables, using Gemini Vision."""
    client = make_client()

    cache_path = cache_path or (DATA_DIR / "vision_cache.json")
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    def cached_call(key, image_path, prompt):
        if key in cache:
            return cache[key]
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = _call_vision(client, image_path, prompt)
                cache[key] = result
                cache_path.write_text(json.dumps(cache, indent=2))
                return result
            except Exception as e:
                last_err = e
                print(f"  [vision] attempt {attempt} failed for {key}: {e}")
                time.sleep(2 * attempt)
        raise RuntimeError(f"Vision call failed for {key} after {max_retries} attempts: {last_err}")

    print(f"[vision] Describing {len(manifest['figures'])} figures ...")
    for fig in manifest["figures"]:
        print(f"  -> {fig['figure_id']} ({fig['caption'][:60]}...)")
        desc = cached_call(fig["figure_id"], fig["image_path"], FIGURE_PROMPT)
        fig["description"] = desc

    print(f"[vision] Transcribing {len(manifest['tables'])} tables ...")
    for tbl in manifest["tables"]:
        print(f"  -> {tbl['table_id']}")
        raw = cached_call(tbl["table_id"], tbl["image_path"], TABLE_PROMPT)
        tbl["vision_transcription"] = raw
        # Pull out the MARKDOWN: section for downstream chunking, keep rest as notes
        md_part = raw
        if "MARKDOWN:" in raw:
            md_part = raw.split("MARKDOWN:", 1)[1]
            if "NOTES:" in md_part:
                md_part = md_part.split("NOTES:", 1)[0]
        tbl["markdown"] = md_part.strip()

    out_path = DATA_DIR / "extracted_content_with_vision.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[vision] Wrote enriched manifest -> {out_path}")
    return manifest


if __name__ == "__main__":
    manifest_path = DATA_DIR / "extracted_content.json"
    manifest = json.loads(manifest_path.read_text())
    run_vision_pass(manifest)
