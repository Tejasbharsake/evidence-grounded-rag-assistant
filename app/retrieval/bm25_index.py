"""
BM25 lexical index — the "retrieval improvement" for this project
(hybrid lexical + semantic retrieval). Kept in memory and rebuilt
whenever the corpus changes (called from the ingestion pipeline). This
is cheap enough to rebuild fully on each ingest for a knowledge base of
this size (tens of documents), which keeps the implementation simple and
correct rather than adding incremental-index complexity.
"""
import re
import threading
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from app.ingestion.store import get_all_chunks

_lock = threading.RLock()
_bm25: BM25Okapi = None
_row_ids: List[int] = []
_tokenized_corpus: List[List[str]] = []

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def rebuild_bm25():
    global _bm25, _row_ids, _tokenized_corpus
    with _lock:
        rows = get_all_chunks()
        _row_ids = [r["row_id"] for r in rows]
        _tokenized_corpus = [tokenize(r["text"]) for r in rows]
        _bm25 = BM25Okapi(_tokenized_corpus) if _tokenized_corpus else None


def search(query: str, top_k: int) -> List[Tuple[int, float]]:
    with _lock:
        if _bm25 is None:
            rebuild_bm25()
        if _bm25 is None or not _row_ids:
            return []
        scores = _bm25.get_scores(tokenize(query))
        max_score = max(scores) if len(scores) else 0.0
        ranked = sorted(zip(_row_ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        # normalize to 0-1 so it can be combined with cosine similarity
        return [(rid, (s / max_score if max_score > 0 else 0.0)) for rid, s in ranked]
