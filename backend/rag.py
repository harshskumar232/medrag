"""
RAG Pipeline
------------
- Real semantic embeddings via sentence-transformers (runs locally)
- ChromaDB for persistent vector storage
- Cosine similarity retrieval
"""

import uuid
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K

# ── Singletons (loaded once at startup) ──────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.Client] = None
_collection = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[RAG] Embedding model loaded ✓")
    return _embedder


def get_collection():
    global _chroma_client, _collection
    if _chroma_client is None:
        # Use in-memory ChromaDB for cloud deployment compatibility
        _chroma_client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _chroma_client.get_or_create_collection(
            name="medrag_docs",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[RAG] ChromaDB (in-memory) ready ✓")
    return _collection


# ── Core Operations ───────────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict]) -> int:
    """Embed a list of chunk dicts and store in ChromaDB. Returns count stored."""
    if not chunks:
        return 0

    embedder = get_embedder()
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "doc_name": c["doc_name"],
            "doc_id": c["doc_id"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    print(f"[RAG] Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # Upsert (handles re-indexing same doc gracefully)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[RAG] Stored {len(chunks)} chunks ✓")
    return len(chunks)


def retrieve(query: str, top_k: int = TOP_K, doc_ids: Optional[list[str]] = None) -> list[dict]:
    """
    Semantic search over the vector store.
    Returns list of {text, doc_name, doc_id, score} dicts.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    where_filter = None
    if doc_ids:
        where_filter = {"doc_id": {"$in": doc_ids}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "doc_name": results["metadatas"][0][i]["doc_name"],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "score": round(1 - results["distances"][0][i], 4),  # cosine sim
        })

    return hits


def delete_doc_chunks(doc_id: str) -> int:
    """Remove all chunks belonging to a document from ChromaDB."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_all_chunks_for_doc(doc_id: str) -> list[str]:
    """Return all chunk texts for a document (for summary agent)."""
    collection = get_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"]
    )
    # Sort by chunk_index
    paired = sorted(
        zip(results["documents"], results["metadatas"]),
        key=lambda x: x[1].get("chunk_index", 0)
    )
    return [text for text, _ in paired]


def collection_count() -> int:
    return get_collection().count()


def reset_collection():
    """Wipe the entire ChromaDB collection."""
    global _chroma_client, _collection
    if _chroma_client:
        try:
            _chroma_client.delete_collection("medrag_docs")
        except Exception:
            pass
        _collection = _chroma_client.get_or_create_collection(
            name="medrag_docs",
            metadata={"hnsw:space": "cosine"}
        )
