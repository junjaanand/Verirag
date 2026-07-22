"""
BM25 Index — Sparse keyword-based retrieval using BM25.
"""

import pickle
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


class BM25Index:
    """BM25 sparse retrieval index."""

    def __init__(self):
        self.index = None
        self.chunks = []  # Store chunk data alongside index
        self.save_path = DATA_DIR / "bm25_index.pkl"

    def build(self, chunks):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self.index = BM25Okapi(tokenized)
        logger.info(f"📑 BM25 index built with {len(chunks)} chunks")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search BM25 index and return top-k results."""
        if self.index is None or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        # Get top-k indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for rank, idx in enumerate(ranked):
            chunk = self.chunks[idx]
            results.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "source_file": chunk.source_file,
                    "page_num": chunk.page_num,
                    "chunk_index": chunk.chunk_index,
                    "doc_type": chunk.doc_type,
                    "ocr_confidence": chunk.ocr_confidence if chunk.ocr_confidence else -1.0,
                    "quality_score": chunk.quality_score,
                    "char_count": chunk.char_count,
                },
                "bm25_score": float(scores[idx]),
                "relevance_score": float(scores[idx]) / max(float(max(scores)), 1e-6),  # Normalize
            })

        return results

    def save(self):
        """Save index to disk."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump({"index": self.index, "chunks": self.chunks}, f)
        logger.info(f"💾 BM25 index saved to {self.save_path}")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if self.save_path.exists():
            with open(self.save_path, "rb") as f:
                data = pickle.load(f)
            self.index = data["index"]
            self.chunks = data["chunks"]
            logger.info(f"📂 BM25 index loaded ({len(self.chunks)} chunks)")
            return True
        return False


# Global instance
bm25_index = BM25Index()
