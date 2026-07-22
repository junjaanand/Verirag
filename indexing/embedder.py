"""
Embedding Generator — Creates dense vector embeddings using Sentence Transformers.
Uses local MiniLM-L6-v2 model (384-dim) for high-quality embeddings.
"""

from loguru import logger
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL

# Global model instance (lazy loaded)
_model = None


def get_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Model loaded (dim={_model.get_sentence_embedding_dimension()})")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_model()
    logger.info(f"Embedding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_model()
    embedding = model.encode(query)
    return embedding.tolist()
