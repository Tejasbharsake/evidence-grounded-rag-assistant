"""
SQLite metadata storage.

Stores document-level info (for duplicate detection + ingestion status)
and chunk-level metadata (document name, page number, chunk id, text).
The FAISS index only stores vectors + an integer row id; this store maps
that row id back to human-readable citation metadata.
"""
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from app import config
from app.models import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_name TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT UNIQUE NOT NULL,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (document_name) REFERENCES documents(document_name)
);
"""


@contextmanager
def get_conn():
    Path(config.SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)  # idempotent (CREATE TABLE IF NOT EXISTS) — guarantees
                                 # schema exists even if a caller reads before ingesting.
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def content_hash(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def is_duplicate(document_name: str, raw_bytes: bytes) -> bool:
    """A document is a duplicate if the same name+content hash already exists."""
    h = content_hash(raw_bytes)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE document_name = ?",
            (document_name,),
        ).fetchone()
        if row is None:
            return False
        return row["content_hash"] == h


def save_document(document_name: str, raw_bytes: bytes, chunks: List[Chunk]) -> List[int]:
    """Persists document + chunk metadata. Returns the FAISS row ids
    (SQLite autoincrement ids) assigned to each chunk, in order."""
    h = content_hash(raw_bytes)
    page_count = len({c.page_number for c in chunks}) or 1
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (document_name, content_hash, page_count, chunk_count) "
            "VALUES (?, ?, ?, ?)",
            (document_name, h, page_count, len(chunks)),
        )
        row_ids = []
        for c in chunks:
            cur = conn.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, document_name, page_number, text) "
                "VALUES (?, ?, ?, ?)",
                (c.chunk_id, c.document_name, c.page_number, c.text),
            )
            row_ids.append(cur.lastrowid)
        return row_ids


def get_chunk_by_row_id(row_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chunks WHERE row_id = ?", (row_id,)).fetchone()
        return dict(row) if row else None


def get_all_chunks() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM chunks ORDER BY row_id").fetchall()
        return [dict(r) for r in rows]


def get_document_stats() -> Dict[str, Any]:
    with get_conn() as conn:
        docs = conn.execute("SELECT * FROM documents").fetchall()
        total_pages = sum(d["page_count"] for d in docs)
        return {
            "document_count": len(docs),
            "total_pages": total_pages,
            "documents": [dict(d) for d in docs],
        }


def clear_all():
    with get_conn() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
