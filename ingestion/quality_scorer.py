"""
Quality Scorer — Assigns quality scores (0-1) to processed documents.
Based on OCR confidence, text density, and formatting consistency.
"""

import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class QualityReport:
    """Quality assessment for a document."""
    quality_score: float        # Overall score (0-1)
    ocr_confidence: float       # OCR confidence (0-1), None for text PDFs
    text_density: float         # Characters per page ratio (0-1)
    formatting_score: float     # Formatting consistency (0-1)
    special_char_ratio: float   # Ratio of special/garbage characters


def calculate_text_density(text: str, page_count: int) -> float:
    """
    Calculate text density score.
    Good documents have 500-3000 chars per page.
    """
    if page_count == 0:
        return 0.0

    chars_per_page = len(text) / page_count

    if chars_per_page < 50:
        return 0.1  # Almost empty
    elif chars_per_page < 200:
        return 0.4  # Sparse
    elif chars_per_page < 500:
        return 0.7  # Light
    elif chars_per_page <= 3000:
        return 1.0  # Good density
    else:
        return 0.8  # Very dense (might have extraction artifacts)


def calculate_formatting_score(text: str) -> float:
    """
    Assess formatting consistency.
    Well-formatted text has proper sentences, paragraphs, and punctuation.
    """
    if not text:
        return 0.0

    scores = []

    # Check for proper sentence endings
    sentences = re.split(r'[.!?]', text)
    avg_sentence_len = sum(len(s.strip()) for s in sentences if s.strip()) / max(len(sentences), 1)
    if 20 < avg_sentence_len < 300:
        scores.append(1.0)
    elif 10 < avg_sentence_len < 500:
        scores.append(0.6)
    else:
        scores.append(0.3)

    # Check for word-like tokens (vs garbage)
    words = text.split()
    if words:
        real_words = sum(1 for w in words if re.match(r'^[a-zA-Z0-9]+', w))
        word_ratio = real_words / len(words)
        scores.append(min(word_ratio * 1.2, 1.0))
    else:
        scores.append(0.0)

    # Check for excessive special characters
    if text:
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?-()"\'/&')
        special_ratio = special_count / len(text)
        scores.append(max(1.0 - (special_ratio * 5), 0.0))
    else:
        scores.append(0.0)

    return sum(scores) / len(scores)


def calculate_special_char_ratio(text: str) -> float:
    """Calculate ratio of non-standard characters."""
    if not text:
        return 0.0
    special = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?-()"\'/&')
    return special / len(text)


def score_quality(
    text: str,
    page_count: int,
    doc_type: str,
    ocr_confidence: float = None,
) -> QualityReport:
    """
    Calculate overall quality score for a processed document.
    
    Score components:
    - OCR confidence (if applicable): 40% weight
    - Text density: 30% weight
    - Formatting consistency: 30% weight
    """
    text_density = calculate_text_density(text, page_count)
    formatting_score = calculate_formatting_score(text)
    special_char_ratio = calculate_special_char_ratio(text)

    # Normalize OCR confidence to 0-1
    if ocr_confidence is not None:
        ocr_norm = ocr_confidence / 100.0
    else:
        ocr_norm = None

    # Calculate weighted score
    if ocr_norm is not None:
        # For OCR'd documents: OCR confidence matters most
        quality_score = (
            0.4 * ocr_norm +
            0.3 * text_density +
            0.3 * formatting_score
        )
    else:
        # For text PDFs/plain text: no OCR factor
        quality_score = (
            0.5 * text_density +
            0.5 * formatting_score
        )

    # Penalize high special character ratio
    if special_char_ratio > 0.1:
        quality_score *= (1.0 - special_char_ratio)

    quality_score = max(0.0, min(1.0, quality_score))

    logger.info(
        f"📊 Quality: {quality_score:.2f} | "
        f"density={text_density:.2f}, format={formatting_score:.2f}, "
        f"ocr={'N/A' if ocr_norm is None else f'{ocr_norm:.2f}'}"
    )

    return QualityReport(
        quality_score=quality_score,
        ocr_confidence=ocr_norm,
        text_density=text_density,
        formatting_score=formatting_score,
        special_char_ratio=special_char_ratio,
    )
