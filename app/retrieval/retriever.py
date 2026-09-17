"""
Hybrid retrieval pipeline: combines FAISS semantic similarity with BM25
lexical scoring (the retrieval improvement required by the assessment),
then removes duplicate/near-duplicate chunk text (a second retrieval
improvement: duplicate-context removal) before returning the top-k.

combined_score = alpha * semantic_score + (1 - alpha) * lexical_score
"""
from typing import List, Dict
from difflib import SequenceMatcher

from app import config
from app.models import RetrievedChunk
from app.ingestion.store import get_chunk_by_row_id
from app.retrieval.embeddings import embed_query
from app.retrieval.vector_store import search as faiss_search
from app.retrieval.bm25_index import search as bm25_search


def _is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold


def retrieve(query: str, top_k: int = config.TOP_K, document_filter: str = None) -> List[RetrievedChunk]:
    """Metadata filtering: optional `document_filter` restricts results
    to a single source document name."""
    semantic_hits = dict(faiss_search(embed_query(query), top_k=top_k * 4))
    lexical_hits = dict(bm25_search(query, top_k=top_k * 4))

    all_row_ids = set(semantic_hits) | set(lexical_hits)
    candidates: List[RetrievedChunk] = []

    for row_id in all_row_ids:
        meta = get_chunk_by_row_id(row_id)
        if not meta:
            continue
        if document_filter and meta["document_name"] != document_filter:
            continue
        sem = semantic_hits.get(row_id, 0.0)
        lex = lexical_hits.get(row_id, 0.0)
        combined = config.HYBRID_ALPHA * sem + (1 - config.HYBRID_ALPHA) * lex
        candidates.append(
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                document_name=meta["document_name"],
                page_number=meta["page_number"],
                text=meta["text"],
                semantic_score=round(sem, 4),
                lexical_score=round(lex, 4),
                combined_score=round(combined, 4),
            )
        )

    candidates.sort(key=lambda c: c.combined_score, reverse=True)

    # Duplicate-context removal: drop near-identical chunk text, keeping
    # the higher-scored occurrence.
    deduped: List[RetrievedChunk] = []
    for c in candidates:
        if any(_is_near_duplicate(c.text, kept.text) for kept in deduped):
            continue
        deduped.append(c)
        if len(deduped) >= top_k:
            break

    return deduped
