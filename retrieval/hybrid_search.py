"""
Hybrid Search — Combines dense vector search + BM25 sparse search.
Uses Reciprocal Rank Fusion (RRF) for score merging.
"""

from loguru import logger

from indexing.embedder import embed_query
from indexing.vector_store import search as vector_search
from indexing.bm25_index import bm25_index

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT, RETRIEVAL_TOP_K


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    dense_weight: float = HYBRID_DENSE_WEIGHT,
    sparse_weight: float = HYBRID_SPARSE_WEIGHT,
    k: int = 60,
) -> list[dict]:
    """Merge dense and sparse results using Reciprocal Rank Fusion."""
    scores = {}
    result_map = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (k + rank + 1)
        result_map[doc_id] = result

    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight / (k + rank + 1)
        if doc_id not in result_map:
            result_map[doc_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, fused_score in ranked:
        result = result_map[doc_id].copy()
        result["fused_score"] = fused_score
        results.append(result)

    return results


def hybrid_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Execute hybrid search combining dense vector + BM25 sparse."""
    logger.info(f"Hybrid search: '{query[:80]}...' (top-{top_k})")

    # Dense search via ChromaDB
    query_embedding = embed_query(query)
    dense_results = vector_search(query_embedding, top_k=top_k)
    logger.debug(f"  Dense: {len(dense_results)} results")

    # Sparse search via BM25
    sparse_results = bm25_index.search(query, top_k=top_k)
    logger.debug(f"  Sparse: {len(sparse_results)} results")

    # Fuse results
    if dense_results and sparse_results:
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
    elif dense_results:
        fused = dense_results
    elif sparse_results:
        fused = sparse_results
    else:
        fused = []

    fused = fused[:top_k]
    logger.info(f"  Fused: {len(fused)} candidates")

    return fused
