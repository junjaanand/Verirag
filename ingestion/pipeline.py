"""
Ingestion Pipeline — Orchestrates the full document ingestion flow.
Detect → Extract/OCR → Normalize → Quality Score
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from ingestion.format_detector import detect_format, FileInfo
from ingestion.pdf_extractor import extract_text_from_pdf
from ingestion.ocr_engine import ocr_pdf, ocr_single_image
from ingestion.text_normalizer import normalize_text, read_plain_text
from ingestion.quality_scorer import score_quality, QualityReport

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SUPPORTED_EXTENSIONS


@dataclass
class ProcessedDocument:
    """A fully processed document ready for chunking."""
    file_path: str
    file_name: str
    doc_type: str           # "text_pdf", "scanned_pdf", "image", "plain_text"
    pages: list[dict]       # [{page_num, text}]
    full_text: str          # All pages concatenated
    quality: QualityReport
    total_pages: int
    total_chars: int


def process_single_document(file_path: str) -> ProcessedDocument:
    """
    Process a single document through the full ingestion pipeline.
    
    Flow: Detect → Extract/OCR → Normalize → Quality Score
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {Path(file_path).name}")
    logger.info(f"{'='*60}")

    # Step 1: Detect format
    file_info = detect_format(file_path)

    # Step 2: Extract text
    pages = []
    ocr_confidence = None

    if file_info.doc_type == "text_pdf":
        result = extract_text_from_pdf(file_path)
        for page in result.pages:
            pages.append({
                "page_num": page.page_num,
                "text": normalize_text(page.text),
            })

    elif file_info.doc_type == "scanned_pdf":
        result = ocr_pdf(file_path)
        ocr_confidence = result.avg_confidence
        for page in result.pages:
            pages.append({
                "page_num": page.page_num,
                "text": normalize_text(page.text),
            })

    elif file_info.doc_type == "image":
        result = ocr_single_image(file_path)
        ocr_confidence = result.avg_confidence
        for page in result.pages:
            pages.append({
                "page_num": page.page_num,
                "text": normalize_text(page.text),
            })

    elif file_info.doc_type == "plain_text":
        text = read_plain_text(file_path)
        pages.append({
            "page_num": 1,
            "text": text,
        })

    # Step 3: Combine all text
    full_text = "\n\n".join(p["text"] for p in pages if p["text"])

    # Step 4: Quality scoring
    quality = score_quality(
        text=full_text,
        page_count=file_info.page_count,
        doc_type=file_info.doc_type,
        ocr_confidence=ocr_confidence,
    )

    return ProcessedDocument(
        file_path=file_path,
        file_name=file_info.file_name,
        doc_type=file_info.doc_type,
        pages=pages,
        full_text=full_text,
        quality=quality,
        total_pages=len(pages),
        total_chars=len(full_text),
    )


def process_directory(directory: str) -> list[ProcessedDocument]:
    """
    Process all supported documents in a directory.
    
    Returns list of ProcessedDocuments sorted by quality score.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all supported files
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))

    if not files:
        logger.warning(f"No supported files found in {directory}")
        return []

    logger.info(f"\n📁 Found {len(files)} documents in {directory}")

    # Process each file
    documents = []
    for file_path in sorted(files):
        try:
            doc = process_single_document(str(file_path))
            documents.append(doc)
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path.name}: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📋 INGESTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total documents: {len(documents)}")
    for doc in documents:
        logger.info(
            f"  {'✅' if doc.quality.quality_score > 0.5 else '⚠️'} "
            f"{doc.file_name} | "
            f"type={doc.doc_type} | "
            f"quality={doc.quality.quality_score:.2f} | "
            f"chars={doc.total_chars}"
        )

    return documents
