"""
config.py
---------
Central configuration for the Multimodal RAG pipeline.

The Gemini API key is loaded ONLY from an environment variable (optionally
populated from a local, untracked .env file). It is never hardcoded here.

Set it before running, e.g.:
    export GEMINI_API_KEY="your-key-here"
Or create a .env file (see .env.example) — it's loaded automatically below.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file, if present, into os.environ
except ImportError:
    pass  # python-dotenv is optional; plain `export` still works without it

# ---- API key -----------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def require_api_key() -> str:
    """Fail fast with a clear message if the key isn't set."""
    if not GEMINI_API_KEY:
        sys.exit(
            "\n[CONFIG ERROR] Environment variable GEMINI_API_KEY is not set.\n"
            "Set it first, e.g.:\n"
            "    export GEMINI_API_KEY='your-key-here'   (Linux/Mac)\n"
            "    setx GEMINI_API_KEY \"your-key-here\"     (Windows)\n"
            "Then re-run this script.\n"
        )
    return GEMINI_API_KEY

# ---- Models --------------------------------------------------------------
# Vision + text generation model (multimodal understanding & grounded answers)
GEMINI_VISION_MODEL = "gemini-3.1-flash-lite"
GEMINI_GENERATION_MODEL = "gemini-3.1-flash-lite"

# Embedding model (shared embedding space for text, table, and image-caption content)
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # output_dimensionality we request from the embedding model

# ---- Paths -----------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
STORE_DIR = ROOT / "store"
OUTPUT_DIR = ROOT / "output"

for d in (DATA_DIR, IMAGES_DIR, STORE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Source PDF --------------------------------------------------------
# Put the paper's PDF in the project's data/ folder (or set PDF_PATH env var
# to point anywhere else). We do NOT ship the PDF itself in this repo.
_env_pdf = os.environ.get("PDF_PATH")
if _env_pdf:
    SOURCE_PDF = Path(_env_pdf)
else:
    # Look for any PDF dropped into data/, fall back to the expected filename
    candidates = list(DATA_DIR.glob("*.pdf"))
    SOURCE_PDF = candidates[0] if candidates else DATA_DIR / "1706_03762v7.pdf"


def require_source_pdf() -> Path:
    if not SOURCE_PDF.exists():
        sys.exit(
            f"\n[CONFIG ERROR] Source PDF not found at: {SOURCE_PDF}\n"
            f"Put the paper's PDF file into the '{DATA_DIR}' folder\n"
            f"(any .pdf there will be picked up), or set the PDF_PATH env var:\n"
            f"    export PDF_PATH=/path/to/1706_03762v7.pdf\n"
        )
    return SOURCE_PDF

# Chunking parameters
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

# Retrieval
TOP_K = 5
