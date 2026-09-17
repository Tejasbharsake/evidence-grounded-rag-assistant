"""
Text extraction and cleaning.

Supports PDF (via PyMuPDF), TXT and Markdown. Returns a list of
(page_number, text) tuples so page numbers survive into chunk metadata.
TXT/MD files have no real pages, so they are treated as a single
"page 1" document (page number preserved as 1 for citation purposes).
"""
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF


class ExtractionError(Exception):
    """Raised for empty, corrupt, or unsupported files."""


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters without
    altering the semantic content of the text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> List[Tuple[int, str]]:
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ExtractionError(f"Corrupt or unreadable PDF: {e}")

    if doc.page_count == 0:
        raise ExtractionError("PDF has zero pages.")

    pages = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        text = clean_text(text)
        if text:
            pages.append((i, text))
    doc.close()

    if not pages:
        raise ExtractionError("PDF contains no extractable text (possibly scanned/empty).")
    return pages


def extract_text_file(path: Path) -> List[Tuple[int, str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise ExtractionError(f"Could not read text file: {e}")

    text = clean_text(raw)
    if not text:
        raise ExtractionError("File is empty after cleaning.")
    return [(1, text)]


def extract_document(path: Path) -> List[Tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return extract_text_file(path)
    else:
        raise ExtractionError(f"Unsupported file type: {suffix}")
