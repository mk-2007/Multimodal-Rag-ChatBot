"""
chunking.py
-----------
STEP 3 of the pipeline: turn the enriched manifest (text blocks + vision-described
figures + vision-transcribed tables) into a flat list of retrieval-ready "chunks".

Each chunk is modality-tagged (text / table / figure) but structurally IDENTICAL
(same fields) so they can all be embedded into and retrieved from ONE shared
vector store -- this is what makes the RAG "multimodal": the retriever doesn't
care whether a chunk originated from body text, a table, or a figure.

Chunk schema:
{
  "chunk_id": str,
  "modality": "text" | "table" | "figure",
  "page": int,
  "heading_context": str,      # nearest preceding heading, for text chunks
  "content": str,              # the actual text passed to the embedding model
  "source_ref": str,           # e.g. "figure_1" / "table_2" / "page 5 text"
  "image_path": str | None,    # set for table/figure chunks (for optional display)
}
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

from config import DATA_DIR, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    chunk_id: str
    modality: str
    page: int
    heading_context: str
    content: str
    source_ref: str
    image_path: Optional[str] = None


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Simple sliding-window splitter on sentence boundaries, capped at max_chars."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk with overlap tail of previous chunk
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_text_chunks(text_blocks: List[dict]) -> List[Chunk]:
    """Group body-text blocks under their nearest preceding heading, then
    split into size-bounded chunks, carrying heading context for retrieval
    quality and citation."""
    chunks: List[Chunk] = []
    current_heading = "Document start"
    buffer = ""
    buffer_page = None
    counter = 0

    def flush():
        nonlocal buffer, buffer_page, counter
        if not buffer.strip():
            return
        for piece in _split_long_text(buffer.strip(), CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS):
            counter += 1
            chunks.append(
                Chunk(
                    chunk_id=f"text_{counter}",
                    modality="text",
                    page=buffer_page or 0,
                    heading_context=current_heading,
                    content=f"[Section: {current_heading}] {piece}",
                    source_ref=f"page {buffer_page} body text",
                )
            )
        buffer = ""

    for block in text_blocks:
        if block["is_heading"]:
            flush()
            current_heading = block["text"]
            buffer_page = block["page"]
            continue
        # Skip obvious noise: emails, arXiv footer, page numbers, references list entries
        if re.match(r"^\d+$", block["text"]):
            continue
        if buffer_page is None:
            buffer_page = block["page"]
        buffer += " " + block["text"]

    flush()
    return chunks


def build_table_chunks(tables: List[dict]) -> List[Chunk]:
    chunks = []
    for t in tables:
        content = (
            f"[{t['caption']}]\n{t.get('markdown', '').strip()}\n"
            f"{_extract_notes(t.get('vision_transcription', ''))}"
        ).strip()
        chunks.append(
            Chunk(
                chunk_id=t["table_id"],
                modality="table",
                page=t["page"],
                heading_context=t["caption"],
                content=content,
                source_ref=t["table_id"],
                image_path=t.get("image_path"),
            )
        )
    return chunks


def _extract_notes(vision_text: str) -> str:
    if "NOTES:" in vision_text:
        return "Notes: " + vision_text.split("NOTES:", 1)[1].strip()
    return ""


def build_figure_chunks(figures: List[dict]) -> List[Chunk]:
    chunks = []
    for fig in figures:
        content = f"[{fig['caption']}]\n{fig.get('description', '').strip()}"
        chunks.append(
            Chunk(
                chunk_id=fig["figure_id"],
                modality="figure",
                page=fig["page"],
                heading_context=fig["caption"],
                content=content,
                source_ref=fig["figure_id"],
                image_path=fig.get("image_path"),
            )
        )
    return chunks


def run_chunking(manifest: dict) -> List[Chunk]:
    text_chunks = build_text_chunks(manifest["text_blocks"])
    table_chunks = build_table_chunks(manifest["tables"])
    figure_chunks = build_figure_chunks(manifest["figures"])

    all_chunks = text_chunks + table_chunks + figure_chunks
    print(f"[chunking] text={len(text_chunks)} table={len(table_chunks)} "
          f"figure={len(figure_chunks)} total={len(all_chunks)}")

    out_path = DATA_DIR / "chunks.json"
    out_path.write_text(json.dumps([asdict(c) for c in all_chunks], indent=2))
    print(f"[chunking] Wrote chunks -> {out_path}")
    return all_chunks


if __name__ == "__main__":
    manifest_path = DATA_DIR / "extracted_content_with_vision.json"
    manifest = json.loads(manifest_path.read_text())
    run_chunking(manifest)
