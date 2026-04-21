"""
ChromaDB vector store: chunk documents, embed with OpenAI, persist.
Exposes retrieve(query, top_k) -> List[Dict] used by retrieval_eval.
"""
import os
import re
from typing import List, Dict
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
_raw_collection = os.getenv("CHROMA_COLLECTION", "kb")
CHROMA_COLLECTION = _raw_collection if len(_raw_collection) >= 3 else f"{_raw_collection}_col"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DOCS_DIR = Path(__file__).parent / "docs"

CHUNK_SIZE = 400      # characters
CHUNK_OVERLAP = 80


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=ef)


def _slugify(source: str) -> str:
    """Convert source path to a ChromaDB-safe ID prefix (alphanumeric + dots + dashes only)."""
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", source)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def make_chunk_id(source: str, idx: int) -> str:
    """Canonical chunk ID used by both vector_store and synthetic_gen."""
    return f"{_slugify(source)}__chunk_{idx:03d}"


def _chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into overlapping chunks, return list of {id, text, source}."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"id": make_chunk_id(source, idx), "text": chunk_text, "source": source})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_documents(docs_dir: Path = DOCS_DIR, force: bool = False) -> List[Dict]:
    """
    Read all .txt files in docs_dir, chunk them, upsert to ChromaDB.
    Returns the full list of chunks for reference.
    Returns cached list if collection already has docs and force=False.
    """
    collection = _get_collection()

    if not force and collection.count() > 0:
        print(f"[VectorStore] Collection '{CHROMA_COLLECTION}' already has {collection.count()} chunks. Skipping ingest.")
        return []

    all_chunks: List[Dict] = []
    for doc_path in sorted(docs_dir.glob("*.txt")):
        raw = doc_path.read_text(encoding="utf-8")
        # derive a short source key from the first "Source:" line
        source_match = re.search(r"Source:\s*(\S+)", raw)
        source = source_match.group(1) if source_match else doc_path.stem
        chunks = _chunk_text(raw, source)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[VectorStore] No documents found to ingest.")
        return []

    collection.upsert(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    print(f"[VectorStore] Ingested {len(all_chunks)} chunks from {docs_dir}.")
    return all_chunks


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Query ChromaDB for top_k most relevant chunks.
    Returns list of {id, text, source, distance}.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"id": doc_id, "text": doc, "source": meta.get("source", ""), "distance": dist})
    return hits


if __name__ == "__main__":
    ingest_documents(force=True)
    sample = retrieve("Quy trình xin nghỉ phép như thế nào?", top_k=3)
    for h in sample:
        print(h["id"], "|", h["text"][:80])
