"""
PDF Text Extractor — Extracts text from text-based PDFs using PyMuPDF.
Preserves page-level mapping for citation purposes.
"""

from dataclasses import dataclass
from loguru import logger

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_num: int
    text: str
    char_count: int


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    file_path: str
    pages: list[PageContent]
    total_chars: int
    total_pages: int


def extract_text_from_pdf(file_path: str) -> PDFExtractionResult:
    """
    Extract text from a text-based PDF using PyMuPDF.
    
    Returns text per page with character counts.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        # Clean up common PDF artifacts
        text = text.replace("\x00", "")  # Remove null bytes
        
        pages.append(PageContent(
            page_num=page_num,
            text=text,
            char_count=len(text),
        ))

    doc.close()

    total_chars = sum(p.char_count for p in pages)
    logger.info(f"📄 Extracted {total_chars} chars from {len(pages)} pages: {file_path}")

    return PDFExtractionResult(
        file_path=file_path,
        pages=pages,
        total_chars=total_chars,
        total_pages=len(pages),
    )
