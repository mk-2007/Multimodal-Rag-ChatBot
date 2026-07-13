"""
extract.py
----------
STEP 1 of the pipeline: content extraction from the source PDF.

Pulls out three distinct content types so each can be processed appropriately:
  1. Body text + headings  -> via PyMuPDF (font-size heuristics to spot headings)
  2. Tables                -> this specific arXiv PDF has broken inter-word spacing
                               in its text layer (confirmed via testing: pdfplumber's
                               table detector returns garbled, space-less text on the
                               table pages). Rather than fight a bad text layer, we
                               render each table's page as a PNG and let Gemini's
                               VISION model transcribe + interpret it in the next step.
  3. Figures/diagrams      -> Figures 1 & 2 are embedded raster images (extracted
                               directly via PyMuPDF). Figures 3-5 (attention
                               visualizations) are vector line-drawings, not raster
                               images, so they can't be pulled out the same way --
                               we render their full pages as PNGs instead.

Output: a single JSON manifest (data/extracted_content.json) describing everything
found, plus the actual image files in images/.
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import fitz  # PyMuPDF
import pdfplumber

from config import SOURCE_PDF, DATA_DIR, IMAGES_DIR, require_source_pdf


@dataclass
class TextBlock:
    page: int
    text: str
    is_heading: bool
    font_size: float


@dataclass
class TableItem:
    page: int
    table_id: str
    markdown: str
    caption: str = ""
    image_path: str = ""


@dataclass
class FigureItem:
    page: int
    figure_id: str
    image_path: str
    caption: str = ""


# ---------------------------------------------------------------------------
# 1. Text + headings extraction (PyMuPDF, font-size based heading heuristic)
# ---------------------------------------------------------------------------
def extract_text_and_headings(doc: fitz.Document) -> List[TextBlock]:
    blocks_out: List[TextBlock] = []

    # First pass: collect all font sizes to determine the "body" baseline size
    all_sizes = []
    for page in doc:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        all_sizes.append(round(span["size"], 1))
    if not all_sizes:
        return blocks_out
    body_size = max(set(all_sizes), key=all_sizes.count)  # most common size = body text

    for page_idx, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            block_text_parts = []
            max_span_size = 0.0
            is_bold_block = False
            for line in b.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    txt = span["text"]
                    line_text += txt
                    max_span_size = max(max_span_size, span["size"])
                    if "bold" in span.get("font", "").lower():
                        is_bold_block = True
                block_text_parts.append(line_text)
            block_text = " ".join(t.strip() for t in block_text_parts if t.strip())
            block_text = re.sub(r"\s+", " ", block_text).strip()
            if not block_text:
                continue
            # Heuristic: a heading is noticeably larger than body text, OR
            # bold + short line that looks like a numbered section title.
            looks_like_heading_number = bool(
                re.match(r"^(\d+(\.\d+)*)\s+[A-Z]", block_text)
            )
            is_heading = (
                max_span_size > body_size + 0.5
                or (is_bold_block and len(block_text) < 80 and looks_like_heading_number)
            ) and len(block_text) < 120

            blocks_out.append(
                TextBlock(
                    page=page_idx,
                    text=block_text,
                    is_heading=is_heading,
                    font_size=round(max_span_size, 1),
                )
            )
    return blocks_out


# ---------------------------------------------------------------------------
# 2. Table extraction
# ---------------------------------------------------------------------------
# Known table caption pages, located by searching for the (space-less) literal
# "TableN:" pattern that this PDF's text layer actually contains -- confirmed
# via inspection: Table1 on page 6, Table2 on page 8, Table3 on page 9, Table4
# on page 10. We render each of these pages to a PNG; the actual transcription
# and structuring into markdown happens later via Gemini Vision (vision.py),
# since that is far more reliable than this PDF's broken text-layer spacing.
TABLE_CAPTION_PATTERN = re.compile(r"Table\s*(\d+)\s*:\s*", re.IGNORECASE)


def find_table_pages(pdf_path: str) -> List[dict]:
    """Locate pages containing a 'Table N:' caption, tolerant of missing spaces."""
    found = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            m2 = re.search(r"Table\s*(\d+)\s*:", raw_text)
            if m2:
                # Grab a short caption snippet right after the match for reference
                start = m2.start()
                snippet = raw_text[start:start + 140]
                found.append({"page": page_idx, "table_num": int(m2.group(1)), "caption_snippet": snippet})
    return found


def render_table_pages(pdf_path: str, out_dir) -> List[TableItem]:
    table_locations = find_table_pages(pdf_path)
    doc = fitz.open(pdf_path)
    tables_out: List[TableItem] = []
    for loc in table_locations:
        page = doc[loc["page"] - 1]
        pix = page.get_pixmap(dpi=220)  # high-res render for accurate vision transcription
        fname = f"table_{loc['table_num']}_page{loc['page']}.png"
        fpath = out_dir / fname
        pix.save(str(fpath))
        tables_out.append(
            TableItem(
                page=loc["page"],
                table_id=f"table_{loc['table_num']}",
                markdown="",  # filled in later by vision.py after transcription
                caption=f"Table {loc['table_num']} (see rendered page image {fname})",
                image_path=str(fpath),
            )
        )
    doc.close()
    return tables_out


# ---------------------------------------------------------------------------
# 3. Figure/diagram extraction
# ---------------------------------------------------------------------------
# Figures 1 (architecture) and 2 (attention mechanism diagrams) are embedded
# raster images and can be pulled out directly. Figures 3-5 (the attention-
# visualization word-alignment plots) are drawn as vector line graphics
# directly on the page -- there is no embedded raster image to extract, so we
# render those full pages to PNG instead and treat them the same way in the
# vision step.
FIGURE_CAPTION_PATTERN = re.compile(r"Figure\s*(\d+)\s*:\s*([^\n]+)")
VECTOR_FIGURE_PAGES = {13: 3, 14: 4, 15: 5}  # page_num -> figure_num for vector-drawn figures


def extract_figures(doc: fitz.Document, out_dir) -> List[FigureItem]:
    figures_out: List[FigureItem] = []
    fig_counter = 0
    seen_figure_nums = set()

    # --- 3a. Embedded raster images (Figures 1 & 2) ---
    for page_idx, page in enumerate(doc, start=1):
        page_text = page.get_text()
        image_list = page.get_images(full=True)
        if not image_list:
            continue

        caption_match = FIGURE_CAPTION_PATTERN.search(page_text)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            # Skip tiny images (icons/artifacts), keep meaningful diagrams
            if pix.width < 80 or pix.height < 80:
                continue
            if pix.n - pix.alpha >= 4:  # CMYK -> convert to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)

            fig_counter += 1
            fname = f"figure_{fig_counter}_page{page_idx}.png"
            fpath = out_dir / fname
            pix.save(str(fpath))

            if caption_match:
                fig_num = int(caption_match.group(1))
                cap = f"Figure {fig_num}: {caption_match.group(2).strip()}"
                seen_figure_nums.add(fig_num)
            else:
                cap = f"Uncaptioned figure on page {page_idx}"

            figures_out.append(
                FigureItem(
                    page=page_idx,
                    figure_id=f"figure_{fig_counter}",
                    image_path=str(fpath),
                    caption=cap,
                )
            )

    # --- 3b. Vector-drawn figures (Figures 3, 4, 5): render full page ---
    for page_idx, fig_num in VECTOR_FIGURE_PAGES.items():
        if fig_num in seen_figure_nums:
            continue
        page = doc[page_idx - 1]
        page_text = page.get_text()
        caption_match = FIGURE_CAPTION_PATTERN.search(page_text)
        cap = (
            f"Figure {fig_num}: {caption_match.group(2).strip()}"
            if caption_match
            else f"Figure {fig_num} (attention visualization, page {page_idx})"
        )
        pix = page.get_pixmap(dpi=200)
        fig_counter += 1
        fname = f"figure_{fig_counter}_page{page_idx}.png"
        fpath = out_dir / fname
        pix.save(str(fpath))
        figures_out.append(
            FigureItem(
                page=page_idx,
                figure_id=f"figure_{fig_counter}",
                image_path=str(fpath),
                caption=cap,
            )
        )

    return figures_out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_extraction():
    require_source_pdf()
    print(f"[extract] Opening {SOURCE_PDF}")
    doc = fitz.open(str(SOURCE_PDF))

    print("[extract] Extracting text + headings ...")
    text_blocks = extract_text_and_headings(doc)
    print(f"[extract]   -> {len(text_blocks)} text blocks "
          f"({sum(b.is_heading for b in text_blocks)} detected as headings)")

    print("[extract] Rendering table pages for vision transcription ...")
    tables = render_table_pages(str(SOURCE_PDF), IMAGES_DIR)
    print(f"[extract]   -> {len(tables)} tables located: "
          f"{[t.table_id for t in tables]}")

    print("[extract] Extracting figures/diagrams ...")
    figures = extract_figures(doc, IMAGES_DIR)
    print(f"[extract]   -> {len(figures)} figures saved to {IMAGES_DIR}")

    manifest = {
        "text_blocks": [asdict(b) for b in text_blocks],
        "tables": [asdict(t) for t in tables],
        "figures": [asdict(f) for f in figures],
    }
    out_path = DATA_DIR / "extracted_content.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[extract] Wrote manifest -> {out_path}")
    return manifest


if __name__ == "__main__":
    run_extraction()
