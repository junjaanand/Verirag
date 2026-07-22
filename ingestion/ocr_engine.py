"""
OCR Engine — Handles OCR for scanned PDFs and images using Tesseract.
Includes confidence scoring per page.
"""

from dataclasses import dataclass
from pathlib import Path
from loguru import logger

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import sys

# Set Tesseract path for Windows
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TESSERACT_PATH, OCR_CONFIDENCE_THRESHOLD

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


@dataclass
class OCRPageResult:
    """OCR result for a single page."""
    page_num: int
    text: str
    confidence: float  # 0-100
    char_count: int


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    file_path: str
    pages: list[OCRPageResult]
    avg_confidence: float
    total_chars: int


def ocr_image(image: Image.Image) -> tuple[str, float]:
    """
    Run Tesseract OCR on a single image.
    Returns (text, confidence).
    """
    # Get detailed data with confidence scores
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract text and confidence
    words = []
    confidences = []

    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf > 0 and word.strip():  # Skip empty/low-confidence entries
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    avg_conf = sum(confidences) / max(len(confidences), 1)

    return text, avg_conf


def ocr_pdf(file_path: str) -> OCRResult:
    """
    OCR a scanned PDF by converting each page to an image and running Tesseract.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        # Convert PDF page to image (300 DPI for good OCR quality)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        # Run OCR
        text, confidence = ocr_image(image)

        pages.append(OCRPageResult(
            page_num=page_num,
            text=text,
            confidence=confidence,
            char_count=len(text),
        ))

        logger.debug(f"  Page {page_num}: {len(text)} chars, confidence: {confidence:.1f}%")

    doc.close()

    avg_confidence = sum(p.confidence for p in pages) / max(len(pages), 1)
    total_chars = sum(p.char_count for p in pages)

    logger.info(f"🔍 OCR complete: {file_path} | {len(pages)} pages | avg confidence: {avg_confidence:.1f}%")

    return OCRResult(
        file_path=file_path,
        pages=pages,
        avg_confidence=avg_confidence,
        total_chars=total_chars,
    )


def ocr_single_image(file_path: str) -> OCRResult:
    """
    OCR a single image file (PNG/JPG).
    """
    image = Image.open(file_path)
    text, confidence = ocr_image(image)

    page = OCRPageResult(
        page_num=1,
        text=text,
        confidence=confidence,
        char_count=len(text),
    )

    logger.info(f"🔍 OCR complete: {file_path} | confidence: {confidence:.1f}%")

    return OCRResult(
        file_path=file_path,
        pages=[page],
        avg_confidence=confidence,
        total_chars=len(text),
    )
