"""
Format Detector — Detects file type and classifies documents.
Determines if a document needs OCR or has extractable text.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

import fitz  # PyMuPDF


@dataclass
class FileInfo:
    """Result of file format detection."""
    file_path: str
    file_name: str
    file_type: str          # "pdf", "image", "text"
    doc_type: str           # "text_pdf", "scanned_pdf", "image", "plain_text"
    needs_ocr: bool
    page_count: int


def detect_file_type(file_path: str) -> str:
    """Detect file type from extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        return "image"
    elif ext in {".txt", ".md"}:
        return "text"
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def check_pdf_has_text(file_path: str, min_chars_per_page: int = 30) -> tuple[bool, int]:
    """
    Check if a PDF has extractable text.
    Returns (has_text, page_count).
    A page is considered to have text if it contains more than min_chars_per_page characters.
    """
    doc = fitz.open(file_path)
    page_count = len(doc)
    pages_with_text = 0

    for page in doc:
        text = page.get_text().strip()
        if len(text) > min_chars_per_page:
            pages_with_text += 1

    doc.close()

    # If more than 50% of pages have text, it's a text PDF
    has_text = (pages_with_text / max(page_count, 1)) > 0.5
    return has_text, page_count


def detect_format(file_path: str) -> FileInfo:
    """
    Detect the format of a document and classify it.
    
    Returns FileInfo with:
    - doc_type: "text_pdf", "scanned_pdf", "image", or "plain_text"
    - needs_ocr: whether OCR is required
    """
    file_path = str(file_path)
    file_name = Path(file_path).name

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = detect_file_type(file_path)

    if file_type == "pdf":
        has_text, page_count = check_pdf_has_text(file_path)
        if has_text:
            doc_type = "text_pdf"
            needs_ocr = False
            logger.info(f"📄 {file_name}: Text PDF ({page_count} pages)")
        else:
            doc_type = "scanned_pdf"
            needs_ocr = True
            logger.info(f"🖼️ {file_name}: Scanned PDF ({page_count} pages) — OCR needed")

    elif file_type == "image":
        doc_type = "image"
        needs_ocr = True
        page_count = 1
        logger.info(f"🖼️ {file_name}: Image file — OCR needed")

    elif file_type == "text":
        doc_type = "plain_text"
        needs_ocr = False
        page_count = 1
        logger.info(f"📝 {file_name}: Plain text file")

    else:
        raise ValueError(f"Unknown file type: {file_type}")

    return FileInfo(
        file_path=file_path,
        file_name=file_name,
        file_type=file_type,
        doc_type=doc_type,
        needs_ocr=needs_ocr,
        page_count=page_count,
    )
