from pathlib import Path
import pytest

from app.ingestion.extract import extract_document, extract_text_file, ExtractionError
from app.ingestion.pipeline import ingest_file, validate_file, current_stats
from app import config


def test_extract_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Hello world. This is a test document about transformers.")
    pages = extract_text_file(f)
    assert pages[0][0] == 1
    assert "transformers" in pages[0][1]


def test_extract_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ExtractionError):
        extract_document(f)


def test_extract_unsupported_type(tmp_path):
    f = tmp_path / "file.docx"
    f.write_text("content")
    with pytest.raises(ExtractionError):
        extract_document(f)


def test_validate_file_rejects_bad_extension(tmp_path):
    f = tmp_path / "x.docx"
    f.write_bytes(b"data")
    err = validate_file(f, 10)
    assert err is not None


def test_validate_file_rejects_empty(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"")
    err = validate_file(f, 0)
    assert "empty" in err.lower()


def test_ingest_txt_file_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text(
        "Transformers use self-attention. " * 30 +
        "This second part discusses retrieval augmented generation in depth. " * 20
    )
    result = ingest_file(f)
    assert result.status == "success"
    assert result.chunk_count > 0

    stats = current_stats()
    assert stats["document_count"] == 1


def test_duplicate_document_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", tmp_path)
    f = tmp_path / "doc.txt"
    f.write_text("Some repeated content about embeddings and vector search. " * 10)
    r1 = ingest_file(f)
    r2 = ingest_file(f)
    assert r1.status == "success"
    assert r2.status == "duplicate"


def test_corrupt_pdf_handled_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KNOWLEDGE_BASE_DIR", tmp_path)
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"%PDF-1.4 not a real pdf body")
    result = ingest_file(f)
    assert result.status == "error"
