from pathlib import Path

from app.ingestion.pipeline import ingest_file
from app.retrieval.retriever import retrieve
from app import config


def _ingest_sample(tmp_path, monkeypatch, name, text):
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", tmp_path)
    f = tmp_path / name
    f.write_text(text)
    return ingest_file(f)


def test_retrieve_returns_chunks_after_ingestion(tmp_path, monkeypatch):
    result = _ingest_sample(
        tmp_path, monkeypatch, "doc1.txt",
        "Self-attention lets transformers relate distant tokens directly. " * 15,
    )
    assert result.status == "success"

    hits = retrieve("How does self-attention work?", top_k=3)
    assert len(hits) > 0
    assert hits[0].document_name == "doc1.txt"
    assert hits[0].chunk_id


def test_retrieve_empty_index_returns_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", tmp_path)
    hits = retrieve("anything at all", top_k=3)
    assert hits == []


def test_metadata_filter_restricts_to_document(tmp_path, monkeypatch):
    _ingest_sample(tmp_path, monkeypatch, "docA.txt", "Content about RAG pipelines. " * 15)
    _ingest_sample(tmp_path, monkeypatch, "docB.txt", "Content about agentic systems. " * 15)

    hits = retrieve("pipelines", top_k=5, document_filter="docA.txt")
    assert all(h.document_name == "docA.txt" for h in hits)


def test_duplicate_context_removal(tmp_path, monkeypatch):
    # Same paragraph appears in two different files -> should not both
    # appear as near-identical entries in the top results.
    text = "Vector search finds nearest neighbors efficiently at scale. " * 10
    _ingest_sample(tmp_path, monkeypatch, "dup1.txt", text)
    _ingest_sample(tmp_path, monkeypatch, "dup2.txt", text)

    hits = retrieve("vector search nearest neighbors", top_k=10)
    texts = [h.text for h in hits]
    assert len(texts) == len(set(texts))
