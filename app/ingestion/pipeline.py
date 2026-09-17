"""
Top-level ingestion pipeline: validates a file, extracts text, chunks it,
embeds the chunks, and writes both the vector index and the metadata store.
Returns a structured result so the UI can show clear success/error status
per file (never crashes the whole batch on one bad file).
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from app import config
from app.ingestion.extract import extract_document, ExtractionError
from app.ingestion.chunking import chunk_document
from app.ingestion.store import init_db, is_duplicate, save_document, get_document_stats
from app.retrieval.embeddings import embed_texts
from app.retrieval.vector_store import add as faiss_add
from app.retrieval.bm25_index import rebuild_bm25


@dataclass
class IngestResult:
    file_name: str
    status: str  # "success" | "duplicate" | "error"
    message: str = ""
    chunk_count: int = 0
    page_count: int = 0


def validate_file(path: Path, size_bytes: int) -> Optional[str]:
    """Returns an error message, or None if the file is valid to attempt ingestion."""
    if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        return f"Unsupported file type '{path.suffix}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}"
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return f"File exceeds max upload size of {config.MAX_UPLOAD_SIZE_MB} MB."
    if size_bytes == 0:
        return "File is empty."
    return None


def ingest_file(path: Path) -> IngestResult:
    init_db()
    document_name = path.name

    try:
        raw_bytes = path.read_bytes()
    except Exception as e:
        return IngestResult(document_name, "error", f"Could not read file: {e}")

    err = validate_file(path, len(raw_bytes))
    if err:
        return IngestResult(document_name, "error", err)

    if is_duplicate(document_name, raw_bytes):
        return IngestResult(document_name, "duplicate", "Identical document already ingested.")

    try:
        pages = extract_document(path)
    except ExtractionError as e:
        return IngestResult(document_name, "error", str(e))
    except Exception as e:
        return IngestResult(document_name, "error", f"Unexpected extraction failure: {e}")

    chunks = chunk_document(document_name, pages)
    if not chunks:
        return IngestResult(document_name, "error", "No chunkable text found after cleaning.")

    try:
        vectors = embed_texts([c.text for c in chunks])
        row_ids = save_document(document_name, raw_bytes, chunks)
        faiss_add(vectors, row_ids)
        rebuild_bm25()
    except Exception as e:
        return IngestResult(document_name, "error", f"Indexing failed: {e}")

    return IngestResult(
        document_name, "success",
        f"Ingested {len(chunks)} chunks across {len(pages)} pages.",
        chunk_count=len(chunks),
        page_count=len(pages),
    )


def ingest_paths(paths: List[Path]) -> List[IngestResult]:
    return [ingest_file(p) for p in paths]


def current_stats():
    init_db()
    return get_document_stats()
