"""
Cross-Encoder Re-Ranker — Re-ranks retrieval candidates for higher precision.
Uses local cross-encoder model (MS-MARCO) for accurate relevance scoring.
"""

from loguru import logger
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RERANKER_MODEL, RETRIEVAL_TOP_N

# Global model (lazy loaded)
_reranker = None


def get_reranker() -> CrossEncoder:
    """Get or initialize the cross-encoder re-ranker."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading re-ranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Re-ranker loaded")
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int = RETRIEVAL_TOP_N) -> list[dict]:
    """Re-rank candidates using cross-encoder for high-precision relevance scoring."""
    if not candidates:
        return []

    if len(candidates) <= top_n:
        # Still score them for relevance
        reranker = get_reranker()
        pairs = [[query, c["text"]] for c in candidates]
        scores = reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
            c["relevance_score"] = max(0.0, min(1.0, (float(scores[i]) + 5) / 10))
        return candidates

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(scores[i])

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    top_results = ranked[:top_n]

    # Normalize scores to 0-1
    if top_results:
        max_score = max(r["rerank_score"] for r in top_results)
        min_score = min(r["rerank_score"] for r in top_results)
        score_range = max_score - min_score if max_score != min_score else 1.0
        for result in top_results:
            result["relevance_score"] = max(0.0, min(1.0, (result["rerank_score"] - min_score) / score_range))

    if top_results:
        logger.info(f"Re-ranked: {len(candidates)} -> top-{len(top_results)} "
                     f"(best={top_results[0]['relevance_score']:.3f})")

    return top_results
