"""
VeriRAG Configuration — All configurable parameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# API KEYS
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma_db"
LOGS_DIR = DATA_DIR / "logs"
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# ============================================================
# DOCUMENT INGESTION
# ============================================================
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md"}
OCR_CONFIDENCE_THRESHOLD = 70  # Minimum Tesseract confidence to accept (0-100)

# ============================================================
# CHUNKING
# ============================================================
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap between chunks
MIN_CHUNK_SIZE = 50       # minimum chunk size to keep

# ============================================================
# EMBEDDING
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformers model
EMBEDDING_DIMENSION = 384              # Output dimension

# ============================================================
# VECTOR STORE
# ============================================================
CHROMA_COLLECTION_NAME = "verirag_documents"

# ============================================================
# RETRIEVAL
# ============================================================
HYBRID_DENSE_WEIGHT = 0.7     # Weight for dense (vector) search
HYBRID_SPARSE_WEIGHT = 0.3    # Weight for sparse (BM25) search
RETRIEVAL_TOP_K = 20          # Candidates for re-ranking
RETRIEVAL_TOP_N = 5           # Final results after re-ranking

# ============================================================
# RE-RANKING
# ============================================================
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================
# SELF-CORRECTION
# ============================================================
SUFFICIENCY_RELEVANCE_THRESHOLD = 0.6   # Min avg relevance to pass
SUFFICIENCY_MIN_CHUNKS = 2              # Min relevant chunks needed
MAX_REQUERY_ATTEMPTS = 3                # Max correction loop retries

# Confidence score weights
CONFIDENCE_WEIGHT_RETRIEVAL = 0.4
CONFIDENCE_WEIGHT_SUFFICIENCY = 0.3
CONFIDENCE_WEIGHT_AGREEMENT = 0.2
CONFIDENCE_WEIGHT_OCR = 0.1

# Confidence levels
CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MEDIUM_THRESHOLD = 0.5

# Contradiction detection
NLI_ENTAILMENT_THRESHOLD = 0.7
NLI_CONTRADICTION_THRESHOLD = 0.5

# ============================================================
# LLM (GROQ)
# ============================================================
LLM_MODEL_MAIN = "llama-3.3-70b-versatile"      # Main generation
LLM_MODEL_FAST = "llama-3.1-8b-instant"          # Quick tasks (decomposition, expansion)
LLM_TEMPERATURE = 0.1                             # Low temp for factual responses
LLM_MAX_TOKENS = 1024                             # Max response tokens

# ============================================================
# NLI MODEL (Contradiction Detection)
# ============================================================
NLI_MODEL = "facebook/bart-large-mnli"

# ============================================================
# EVALUATION
# ============================================================
EVAL_TEST_QUESTIONS_PATH = BASE_DIR / "evaluation" / "test_questions.json"

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = "INFO"
PIPELINE_LOG_FILE = LOGS_DIR / "pipeline.jsonl"
