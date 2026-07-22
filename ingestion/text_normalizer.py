"""
Text Normalizer — Cleans and normalizes extracted text.
Fixes encoding, whitespace, and special characters.
"""

import re
import unicodedata
from loguru import logger


def normalize_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Steps:
    1. Fix encoding (ensure valid UTF-8)
    2. Normalize unicode characters
    3. Remove null bytes and control characters
    4. Collapse multiple whitespace
    5. Fix common OCR artifacts
    6. Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # 1. Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # 2. Remove null bytes and control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Fix common OCR artifacts
    text = text.replace("|", "I")       # Common OCR mistake
    text = text.replace("}", ")")       # Bracket confusion
    text = text.replace("{", "(")
    text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # Smart quotes to standard
    text = re.sub(r'[\u2018\u2019\u201a]', "'", text)  # Smart apostrophes

    # 4. Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # 5. Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 7. Final strip
    text = text.strip()

    return text


def read_plain_text(file_path: str) -> str:
    """Read and normalize a plain text file."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            return normalize_text(text)
        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"Could not decode file with any encoding: {file_path}")
