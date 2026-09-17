from app.ingestion.chunking import chunk_page_text, chunk_document, make_chunk_id


def test_chunk_page_text_respects_rough_size():
    text = ("This is a sentence about transformers. " * 60).strip()
    chunks = chunk_page_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        assert len(c) <= 200 * 1.6  # allow some slack for sentence boundaries


def test_chunk_page_text_empty():
    assert chunk_page_text("") == []


def test_chunk_document_preserves_page_numbers():
    pages = [(1, "First page content about RAG. " * 20), (2, "Second page content about agents. " * 20)]
    chunks = chunk_document("doc.pdf", pages)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}
    assert all(c.document_name == "doc.pdf" for c in chunks)


def test_chunk_ids_are_unique():
    pages = [(1, "Repeated sentence here. " * 40)]
    chunks = chunk_document("doc.pdf", pages)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_make_chunk_id_deterministic():
    id1 = make_chunk_id("doc.pdf", 1, 0, "some text")
    id2 = make_chunk_id("doc.pdf", 1, 0, "some text")
    assert id1 == id2
