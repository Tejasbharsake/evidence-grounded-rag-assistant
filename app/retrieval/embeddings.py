"""
Local, free embedding model wrapper (sentence-transformers/all-MiniLM-L6-v2
by default). Loaded once and cached at module level so repeated calls
(ingestion + every query) don't reload the model from disk.
"""
from typing import List
import numpy as np

from app import config

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
