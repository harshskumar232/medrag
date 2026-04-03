"""
MedRAG — FastAPI Backend
========================
Serves the frontend + all API routes.
"""

import uuid
import json
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
from parser import parse_document, chunk_text
from rag import embed_and_store, retrieve, delete_doc_chunks, get_all_chunks_for_doc, collection_count, reset_collection
from agents import run_summary_agent, run_symptom_agent, run_rag_chat

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedRAG API",
    description="Clinical Intelligence Suite — RAG + AI Agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document registry (persists to JSON file)
REGISTRY_FILE = config.BASE_DIR / "data" / "registry.json"
doc_registry: dict[str, dict] = {}


def load_registry():
    global doc_registry
    if REGISTRY_FILE.exists():
        try:
            doc_registry = json.loads(REGISTRY_FILE.read_text())
            print(f"[Registry] Loaded {len(doc_registry)} documents")
        except Exception:
            doc_registry = {}


def save_registry():
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(doc_registry, indent=2))


@app.on_event("startup")
async def startup():
    load_registry()
    # Pre-load embedding model in background
    from rag import get_embedder, get_collection
    get_embedder()
    get_collection()
    print("[MedRAG] Server ready ✓")


# ── Serve Frontend ────────────────────────────────────────────────────────────
if config.FRONTEND_DIR.exists():
    @app.get("/", response_class=FileResponse)
    async def serve_frontend():
        return FileResponse(config.FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


# ── Request / Response Models ─────────────────────────────────────────────────

class PasteRequest(BaseModel):
    title: str
    text: str
    chunk_size: int = config.CHUNK_SIZE
    chunk_overlap: int = config.CHUNK_OVERLAP

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    top_k: int = config.TOP_K
    api_key: str = ""

class SummaryRequest(BaseModel):
    doc_type: str = "patient_record"
    output_format: str = "structured"
    focus_areas: str = ""
    custom_text: str = ""
    doc_ids: list[str] = []
    api_key: str = ""

class SymptomRequest(BaseModel):
    symptoms: list[str] = []
    age: Optional[int] = None
    sex: str = "M"
    vitals: dict = {}
    description: str = ""
    medical_history: str = ""
    api_key: str = ""
    top_k: int = config.TOP_K


# ── Document Routes ───────────────────────────────────────────────────────────

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
):
    """Upload and index a file."""
    doc_id = str(uuid.uuid4())
    filename = file.filename or f"document_{doc_id}"
    safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ")
    save_path = config.UPLOAD_DIR / f"{doc_id}_{safe_name}"

    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Parse
    try:
        text = parse_document(save_path, filename)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse document: {e}")

    if not text.strip():
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "No text could be extracted from this document")

    # Chunk + embed
    chunks = chunk_text(text, filename, doc_id, chunk_size, chunk_overlap)
    stored = embed_and_store(chunks)

    # Register
    doc_info = {
        "id": doc_id,
        "name": filename,
        "type": filename.rsplit(".", 1)[-1].lower(),
        "size": len(content),
        "chunks": stored,
        "uploaded_at": datetime.now().isoformat(),
        "file_path": str(save_path),
    }
    doc_registry[doc_id] = doc_info
    save_registry()

    return {"success": True, "document": doc_info}


@app.post("/api/documents/paste")
async def paste_document(req: PasteRequest):
    """Index pasted text."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    doc_id = str(uuid.uuid4())
    filename = req.title or f"Note_{doc_id[:8]}"

    chunks = chunk_text(req.text, filename, doc_id, req.chunk_size, req.chunk_overlap)
    stored = embed_and_store(chunks)

    doc_info = {
        "id": doc_id,
        "name": filename,
        "type": "txt",
        "size": len(req.text.encode()),
        "chunks": stored,
        "uploaded_at": datetime.now().isoformat(),
        "file_path": None,
    }
    doc_registry[doc_id] = doc_info
    save_registry()

    return {"success": True, "document": doc_info}


@app.get("/api/documents")
async def list_documents():
    """List all indexed documents."""
    return {
        "documents": list(doc_registry.values()),
        "total_chunks": collection_count(),
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document and its chunks from the vector store."""
    if doc_id not in doc_registry:
        raise HTTPException(404, "Document not found")

    doc = doc_registry[doc_id]

    # Remove file
    if doc.get("file_path"):
        Path(doc["file_path"]).unlink(missing_ok=True)

    # Remove from ChromaDB
    deleted = delete_doc_chunks(doc_id)

    # Remove from registry
    del doc_registry[doc_id]
    save_registry()

    return {"success": True, "chunks_deleted": deleted, "doc_name": doc["name"]}


# ── Chat Route ────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """RAG-grounded chat endpoint."""
    api_key = req.api_key or config.ANTHROPIC_API_KEY
    if not api_key or not api_key.startswith("sk-ant-"):
        raise HTTPException(400, "Valid Anthropic API key required")

    # Retrieve relevant chunks
    sources = retrieve(req.query, top_k=req.top_k)

    # Get AI response
    reply = run_rag_chat(
        query=req.query,
        context_chunks=sources,
        chat_history=req.history,
        api_key=api_key,
    )

    return {
        "reply": reply,
        "sources": sources,
        "chunks_searched": collection_count(),
    }


# ── Agent Routes ──────────────────────────────────────────────────────────────

@app.post("/api/agents/summary")
async def summary_agent(req: SummaryRequest):
    """Medical Summary Agent endpoint."""
    api_key = req.api_key or config.ANTHROPIC_API_KEY
    if not api_key or not api_key.startswith("sk-ant-"):
        raise HTTPException(400, "Valid Anthropic API key required")

    # Build context
    if req.custom_text.strip():
        context = req.custom_text
    elif req.doc_ids:
        all_texts = []
        for doc_id in req.doc_ids:
            chunks = get_all_chunks_for_doc(doc_id)
            all_texts.extend(chunks)
        context = "\n\n---\n\n".join(all_texts)
    elif doc_registry:
        # Use all docs
        all_texts = []
        for doc_id in list(doc_registry.keys())[:5]:  # max 5 docs
            chunks = get_all_chunks_for_doc(doc_id)
            all_texts.extend(chunks[:10])  # max 10 chunks per doc
        context = "\n\n---\n\n".join(all_texts)
    else:
        raise HTTPException(400, "No documents indexed and no custom text provided")

    result = run_summary_agent(
        context=context,
        doc_type=req.doc_type,
        output_format=req.output_format,
        focus_areas=req.focus_areas,
        api_key=api_key,
    )

    return result


@app.post("/api/agents/symptom")
async def symptom_agent(req: SymptomRequest):
    """Symptom Checker Agent endpoint."""
    api_key = req.api_key or config.ANTHROPIC_API_KEY
    if not api_key or not api_key.startswith("sk-ant-"):
        raise HTTPException(400, "Valid Anthropic API key required")

    if not req.symptoms and not req.description:
        raise HTTPException(400, "Provide at least one symptom or description")

    # Get relevant doc context
    query = " ".join(req.symptoms) + " " + req.description
    doc_hits = retrieve(query, top_k=req.top_k)
    doc_context = "\n\n".join([h["text"] for h in doc_hits]) if doc_hits else ""

    result = run_symptom_agent(
        symptoms=req.symptoms,
        age=req.age,
        sex=req.sex,
        vitals=req.vitals,
        description=req.description,
        medical_history=req.medical_history,
        doc_context=doc_context,
        api_key=api_key,
    )

    result["sources"] = doc_hits
    return result


# ── Stats & Reset ──────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    return {
        "documents": len(doc_registry),
        "total_chunks": collection_count(),
        "embedding_model": config.EMBEDDING_MODEL,
        "claude_model": config.CLAUDE_MODEL,
        "upload_dir": str(config.UPLOAD_DIR),
        "chroma_dir": str(config.CHROMA_DIR),
    }


@app.delete("/api/reset")
async def reset():
    """Wipe everything — documents, vectors, registry."""
    # Delete uploaded files
    for doc in doc_registry.values():
        if doc.get("file_path"):
            Path(doc["file_path"]).unlink(missing_ok=True)

    doc_registry.clear()
    save_registry()
    reset_collection()

    return {"success": True, "message": "All data cleared"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
