"""
Shared test fixtures.

Unit tests mock the embedding model with a deterministic, hash-based fake
embedder. This keeps the test suite fully offline and fast (no model
download required in CI), while still exercising the real chunking,
storage, FAISS indexing, BM25, security, and generation-validation logic.
Retrieval-quality itself (whether semantically similar text scores higher)
is not something this fake embedder can validate — that's an integration
concern, documented in the README as requiring the real model.
"""
import hashlib
import shutil
import numpy as np
import pytest

from app import config


def _fake_embed(texts):
    """Deterministic 384-dim vector per text, derived from its hash.
    Not semantically meaningful, but stable and normalized so cosine
    similarity math and shapes behave exactly like the real thing."""
    vectors = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        v = rng.normal(size=384).astype("float32")
        v = v / (np.linalg.norm(v) + 1e-8)
        vectors.append(v)
    return np.array(vectors, dtype="float32") if vectors else np.zeros((0, 384), dtype="float32")


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect all persistent storage to a temp dir per test, and reset
    the in-memory FAISS/BM25 singletons so tests don't leak state."""
    kb_dir = tmp_path / "kb"
    sqlite_dir = tmp_path / "sqlite"
    kb_dir.mkdir()
    sqlite_dir.mkdir()

    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", kb_dir)
    monkeypatch.setattr(config, "SQLITE_PATH", sqlite_dir / "metadata.db")
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", sqlite_dir / "faiss.index")

    import app.retrieval.vector_store as vs
    vs._index = None
    import app.retrieval.bm25_index as bm25
    bm25._bm25 = None
    bm25._row_ids = []
    bm25._tokenized_corpus = []

    yield

    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    fake_query = lambda t: _fake_embed([t])[0]

    import app.retrieval.embeddings as emb
    monkeypatch.setattr(emb, "embed_texts", _fake_embed)
    monkeypatch.setattr(emb, "embed_query", fake_query)

    # Several modules import these names directly (`from ... import embed_texts`),
    # which binds a separate reference that patching the source module does not
    # affect. Patch each importing module's local reference too.
    import app.retrieval.retriever as retriever
    monkeypatch.setattr(retriever, "embed_query", fake_query)

    import app.ingestion.pipeline as pipeline
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
