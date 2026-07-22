"""
Vector Store — ChromaDB operations for storing and searching document embeddings.
"""

import chromadb
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_DIR, CHROMA_COLLECTION_NAME

# Global client
_client = None
_collection = None


def get_collection():
    """Get or initialize ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection ready: {CHROMA_COLLECTION_NAME} ({_collection.count()} docs)")
    return _collection


def add_chunks(chunks, embeddings: list[list[float]]):
    """Add chunks with pre-computed embeddings to ChromaDB."""
    collection = get_collection()

    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "page_num": c.page_num,
            "chunk_index": c.chunk_index,
            "doc_type": c.doc_type,
            "ocr_confidence": c.ocr_confidence if c.ocr_confidence is not None else -1.0,
            "quality_score": c.quality_score,
            "char_count": c.char_count,
        }
        for c in chunks
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    logger.info(f"Stored {len(ids)} chunks in ChromaDB (total: {collection.count()})")


def search(query_embedding: list[float], top_k: int = 20, where: dict = None) -> list[dict]:
    """Search ChromaDB for similar chunks using embedding vector."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    formatted = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": max(0.0, 1.0 - results["distances"][0][i]),
            })

    return formatted


def get_document_count() -> int:
    """Get total number of chunks in the collection."""
    return get_collection().count()


def reset_collection():
    """Delete and recreate the collection."""
    global _client, _collection
    if _client:
        _client.delete_collection(CHROMA_COLLECTION_NAME)
        _collection = None
        logger.info("Collection reset")
        get_collection()
