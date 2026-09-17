"""
FAISS vector store wrapper. Uses IndexIDMap over a flat inner-product
index (vectors are pre-normalized, so inner product == cosine similarity).
The integer ids used here are the SQLite `chunks.row_id` values, so a
FAISS hit maps directly back to citation metadata with no extra lookup
table required.
"""
import threading
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss

from app import config

_lock = threading.Lock()
_index = None
DIM = 384  # all-MiniLM-L6-v2 output dimension


def _new_index():
    flat = faiss.IndexFlatIP(DIM)
    return faiss.IndexIDMap(flat)


def load_or_create() -> faiss.Index:
    global _index
    if _index is not None:
        return _index
    if Path(config.FAISS_INDEX_PATH).exists():
        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    else:
        _index = _new_index()
    return _index


def save():
    if _index is not None:
        Path(config.FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(_index, str(config.FAISS_INDEX_PATH))


def add(vectors: np.ndarray, ids: List[int]):
    with _lock:
        index = load_or_create()
        index.add_with_ids(vectors, np.array(ids, dtype="int64"))
        save()


def search(query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    with _lock:
        index = load_or_create()
        if index.ntotal == 0:
            return []
        scores, ids = index.search(query_vector.reshape(1, -1), min(top_k, index.ntotal))
        results = []
        for i, s in zip(ids[0], scores[0]):
            if i == -1:
                continue
            results.append((int(i), float(s)))
        return results


def reset():
    global _index
    _index = _new_index()
    save()
