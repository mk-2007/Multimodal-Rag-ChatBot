# Multimodal RAG — "Attention Is All You Need"

A retrieval-augmented generation pipeline that indexes the Transformer paper
across **three modalities** — body text, tables, and figures/diagrams — into
one shared Gemini embedding space, then answers questions grounded strictly
in retrieved context.

## Architecture

```
PDF
 │
 ├─► extract.py    ── PyMuPDF: body text + heading detection (font-size heuristic)
 │                  ── PyMuPDF: render table pages (1,2,3,4) to PNG
 │                  ── PyMuPDF: extract Fig.1/2 raster images + render Fig.3-5 pages
 │
 ├─► vision.py      ── Gemini Vision describes each figure (structure/meaning)
 │                  ── Gemini Vision transcribes each table page -> markdown
 │
 ├─► chunking.py    ── unifies text/table/figure content into one Chunk schema
 │
 ├─► embeddings.py  ── Gemini embedding model embeds ALL chunks into ONE
 │                     shared vector store (numpy matrix + metadata.json)
 │
 ├─► retrieval.py   ── embeds the query with the same model, cosine search
 │
 └─► generate.py    ── Gemini answers using ONLY the retrieved context
```

Why this design:
- **Tables**: this specific arXiv PDF's text layer has broken inter-word
  spacing on the table pages (confirmed by testing — pdfplumber's table/word
  detector returns garbled, space-less text there). Rather than fight it,
  table pages are rendered as images and transcribed by Gemini Vision, which
  is both more robust and directly uses the vision model as required.
- **Figures**: Figures 1 & 2 are embedded raster images (extracted directly).
  Figures 3-5 (the attention-visualization plots) are vector line-drawings
  with no embedded raster image, so their pages are rendered to PNG instead.
  Both paths feed the same vision-description step.
- **Shared vector space**: every chunk — regardless of modality — is embedded
  by the same model into the same space, so a single query can retrieve a
  mix of text/table/figure chunks.

## Setup

```bash
cd multimodal_rag

# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide your API key -- either method works:
#    (a) plain environment variable
export GEMINI_API_KEY="your-key-here"
#    (b) or copy .env.example -> .env and fill it in; it's loaded automatically
cp .env.example .env && nano .env
```

Get a key at https://ai.google.dev/gemini-api/docs/api-key if you don't have one.
The key is never hardcoded anywhere in the source — `config.py` only ever reads
it from the environment (optionally populated from your local `.env`, which is
gitignored and never shipped with a real value).

**Put the source PDF in the `data/` folder** — e.g. `data/1706_03762v7.pdf`.
Any single `.pdf` dropped there is picked up automatically. Alternatively point
to it anywhere on disk with:
```bash
export PDF_PATH=/path/to/1706_03762v7.pdf
```
The PDF itself is not shipped in this repo — you provide it.

## Usage

```bash
# Full pipeline: extract -> vision -> chunk -> embed -> demo queries
python3 main.py

# Reuse an already-built vector store, just run the demo queries again
python3 main.py --skip-build

# Ask your own question after building
python3 main.py --query "What optimizer and learning rate schedule was used?"
```

Each stage can also be run/inspected individually:
```bash
python3 extract.py      # -> data/extracted_content.json, images/*.png
python3 vision.py       # -> data/extracted_content_with_vision.json
python3 chunking.py     # -> data/chunks.json
python3 embeddings.py   # -> store/vectors.npy, store/metadata.json
python3 retrieval.py "What is multi-head attention?"
python3 generate.py     # demo single Q&A
```

## Outputs

- `output/demo_results.json` — full query + retrieved context + answer log
- `output/demo_results.md` — the same, formatted for human reading
- `images/` — extracted figure PNGs and rendered table-page PNGs
- `store/` — the shared embedding matrix + chunk metadata (the vector store)

## Demo queries (one per modality)

1. **Table-grounded**: BLEU scores for Transformer (big) vs GNMT+RL → pulls
   from the Table 2 chunk (vision-transcribed).
2. **Figure-grounded**: encoder-decoder architecture description → pulls from
   the Figure 1 chunk (vision-described).
3. **Text-grounded**: what is multi-head attention → pulls from body-text
   chunks in Section 3.2.2.

## Notes / limitations

- `gemini-embedding-001` is used for embeddings (`output_dimensionality=768`),
  and `gemini-3.1-flash-lite` for both vision description and grounded generation
  — change these in `config.py` if you want a different model tier.
- The vector store is a simple in-memory numpy matrix with cosine similarity
  (fine at this paper's scale — a few dozen chunks). Swap in a real vector DB
  (e.g. FAISS, Chroma, pgvector) for larger corpora.
- Vision calls are cached in `data/vision_cache.json` so re-running the
  pipeline doesn't re-spend API calls on unchanged images.
