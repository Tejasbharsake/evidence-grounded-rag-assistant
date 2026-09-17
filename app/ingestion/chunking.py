"""
Defensible chunking strategy.

Strategy: fixed-size character windows with overlap, but we snap chunk
boundaries to sentence ends where possible so we don't cut a claim (and
therefore a citation) in the middle of a sentence. This is a standard,
explainable approach for RAG over prose-heavy research documents (see
README for the full chunk-size/overlap rationale).
"""
import re
import hashlib
from typing import List, Tuple

from app.models import Chunk
from app import config

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_END.split(text)
    return [s for s in sentences if s.strip()]


def chunk_page_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Greedily pack sentences into windows of ~chunk_size characters,
    carrying the last `overlap` characters forward into the next chunk
    so context isn't lost at a boundary."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(current.strip())
            # carry overlap tail forward
            tail = current[-overlap:] if overlap > 0 else ""
            current = (tail + " " + sentence).strip()
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    # Fallback: if a single sentence is longer than chunk_size (e.g. no
    # punctuation), hard-split it so nothing is silently dropped.
    final = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                final.append(c[i:i + chunk_size])
    return final


def make_chunk_id(document_name: str, page_number: int, index: int, text: str) -> str:
    h = hashlib.sha1(f"{document_name}:{page_number}:{index}:{text[:50]}".encode()).hexdigest()[:10]
    return f"{document_name}_p{page_number}_c{index}_{h}"


def chunk_document(document_name: str, pages: List[Tuple[int, str]]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for page_number, page_text in pages:
        page_chunks = chunk_page_text(page_text)
        for idx, chunk_text in enumerate(page_chunks):
            cid = make_chunk_id(document_name, page_number, idx, chunk_text)
            all_chunks.append(
                Chunk(
                    chunk_id=cid,
                    document_name=document_name,
                    page_number=page_number,
                    text=chunk_text,
                )
            )
    return all_chunks
