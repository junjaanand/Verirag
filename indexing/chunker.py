"""
Semantic Chunker — Splits documents into chunks for embedding.
Sentence-boundary aware with configurable size and overlap.
"""

import re
from dataclasses import dataclass
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: str
    text: str
    source_file: str
    page_num: int
    chunk_index: int
    doc_type: str
    ocr_confidence: float | None
    quality_score: float
    char_count: int


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def create_chunks(
    text: str,
    source_file: str,
    page_num: int,
    doc_type: str,
    ocr_confidence: float | None,
    quality_score: float,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Create chunks from text using sentence-boundary aware splitting.
    
    Args:
        text: Input text to chunk
        source_file: Source filename for metadata
        page_num: Source page number
        doc_type: Document type
        ocr_confidence: OCR confidence (None for text PDFs)
        quality_score: Document quality score
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        start_index: Starting chunk index
    
    Returns:
        List of Chunk objects
    """
    if not text or estimate_tokens(text) < MIN_CHUNK_SIZE:
        return []

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    chunk_index = start_index

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence exceeds chunk_size, finalize current chunk
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                chunk_id=f"{source_file}_{page_num}_{chunk_index}",
                text=chunk_text,
                source_file=source_file,
                page_num=page_num,
                chunk_index=chunk_index,
                doc_type=doc_type,
                ocr_confidence=ocr_confidence,
                quality_score=quality_score,
                char_count=len(chunk_text),
            ))
            chunk_index += 1

            # Keep overlap sentences
            overlap_tokens = 0
            overlap_start = len(current_chunk)
            for i in range(len(current_chunk) - 1, -1, -1):
                overlap_tokens += estimate_tokens(current_chunk[i])
                if overlap_tokens >= chunk_overlap:
                    overlap_start = i
                    break
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if estimate_tokens(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(Chunk(
                chunk_id=f"{source_file}_{page_num}_{chunk_index}",
                text=chunk_text,
                source_file=source_file,
                page_num=page_num,
                chunk_index=chunk_index,
                doc_type=doc_type,
                ocr_confidence=ocr_confidence,
                quality_score=quality_score,
                char_count=len(chunk_text),
            ))

    return chunks


def chunk_document(processed_doc) -> list[Chunk]:
    """
    Chunk an entire ProcessedDocument into a list of Chunks.
    Processes page by page to maintain page-level metadata.
    """
    all_chunks = []
    chunk_index = 0

    ocr_conf = None
    if processed_doc.quality.ocr_confidence is not None:
        ocr_conf = processed_doc.quality.ocr_confidence

    for page in processed_doc.pages:
        page_chunks = create_chunks(
            text=page["text"],
            source_file=processed_doc.file_name,
            page_num=page["page_num"],
            doc_type=processed_doc.doc_type,
            ocr_confidence=ocr_conf,
            quality_score=processed_doc.quality.quality_score,
            start_index=chunk_index,
        )
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    logger.info(f"✂️ Chunked {processed_doc.file_name}: {len(all_chunks)} chunks")
    return all_chunks
