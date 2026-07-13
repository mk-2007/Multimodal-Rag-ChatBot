"""
backend/app.py
---------------
A thin FastAPI layer over the existing pipeline modules (retrieval.py,
generate.py, image_query.py). No RAG logic lives here -- this only:
  1. Loads the vector store + a Gemini client once at startup
  2. Exposes two endpoints: text query and image query
  3. Serves the static frontend and the source images referenced in results

Run from the project root:
    uvicorn backend.app:app --reload
Then open http://127.0.0.1:8000
"""

import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `import retrieval, generate, ...` from project root

from config import STORE_DIR, IMAGES_DIR, GEMINI_API_KEY  # noqa: E402
from retrieval import VectorStore, retrieve, make_client as make_retrieval_client  # noqa: E402
from generate import generate_answer, make_client as make_generate_client  # noqa: E402
from image_query import run_image_query, make_client as make_image_client  # noqa: E402

app = FastAPI(title="Multimodal RAG - Attention Is All You Need")

_state = {"store": None, "client": None, "ready": False, "error": None}


@app.on_event("startup")
def load_store():
    try:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Set it in your environment or .env file, "
                "then restart the server."
            )
        _state["client"] = make_generate_client()
        _state["store"] = VectorStore()
        _state["ready"] = True
    except Exception as e:
        # Don't crash the server -- surface the problem to the frontend instead,
        # since the most common causes (missing key, store not built yet) are
        # fixable without a restart.
        _state["error"] = str(e)
        _state["ready"] = False


def _require_ready():
    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=(
                _state["error"]
                or "Vector store not loaded. Run `python3 main.py` once to build it first."
            ),
        )


def _serialize_chunk(c):
    has_image = bool(c.image_path) and Path(c.image_path).exists()
    return {
        "score": round(c.score, 3),
        "chunk_id": c.chunk_id,
        "modality": c.modality,
        "page": c.page,
        "heading_context": c.heading_context,
        "content": c.content,
        "source_ref": c.source_ref,
        "image_url": f"/api/asset/{Path(c.image_path).name}" if has_image else None,
    }


@app.post("/api/query")
def api_query(payload: dict):
    _require_ready()
    query = (payload or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is empty.")

    try:
        embed_client = make_retrieval_client()
        chunks = retrieve(query, client=embed_client, store=_state["store"])
        answer = generate_answer(query, chunks, client=_state["client"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "answer": answer,
        "sources": [_serialize_chunk(c) for c in chunks],
    }


@app.post("/api/image-query")
async def api_image_query(file: UploadFile = File(...), question: str = Form("")):
    _require_ready()

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        client = make_image_client()
        result = run_image_query(
            tmp_path,
            question.strip() or None,
            client=client,
            store=_state["store"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "answer": result["answer"],
        "search_description": result["search_description"],
        "sources": [
            {
                "score": round(c["score"], 3),
                "chunk_id": c.get("chunk_id", c["source_ref"]),
                "modality": c["modality"],
                "page": c["page"],
                "heading_context": c["heading_context"],
                "content": c["content"],
                "source_ref": c["source_ref"],
                "image_url": (
                    f"/api/asset/{Path(c['image_path']).name}"
                    if c.get("image_path") and Path(c["image_path"]).exists()
                    else None
                ),
            }
            for c in result["retrieved_context"]
        ],
    }


@app.get("/api/status")
def api_status():
    return {"ready": _state["ready"], "error": _state["error"]}


@app.get("/api/asset/{filename}")
def api_asset(filename: str):
    # Guard against path traversal -- only ever serve a bare filename from IMAGES_DIR
    safe_name = Path(filename).name
    path = IMAGES_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path)


# Serve the static frontend LAST so /api/* routes above take precedence.
FRONTEND_DIR = ROOT / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
