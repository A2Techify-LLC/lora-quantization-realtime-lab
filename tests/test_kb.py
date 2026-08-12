from pathlib import Path

from texas_education_kb.chunking import chunks_from_documents
from texas_education_kb.schema import RawDocument
from texas_education_kb.store import LocalRetrievalStore


def test_chunks_preserve_source_metadata():
    document = RawDocument(
        source_id="tea-test",
        title="TEA Test",
        text="Graduation requirements apply to Texas public high school students.",
        topics=["graduation"],
    )

    chunks = chunks_from_documents([document], max_chars=120, overlap_chars=10)

    assert len(chunks) == 1
    assert chunks[0].source_id == "tea-test"
    assert chunks[0].topics == ["graduation"]


def test_local_store_returns_cited_retrieval_result():
    document = RawDocument(
        source_id="special-ed",
        title="Special Education Guidance",
        text="Special education services may be delivered through an individualized education program.",
        topics=["special_education"],
    )
    chunks = chunks_from_documents([document])
    store = LocalRetrievalStore(chunks)

    citations = store.search("How are special education services delivered?", top_k=1)

    assert citations
    assert citations[0].source_id == "special-ed"
    assert "individualized education program" in citations[0].excerpt


def test_sample_source_file_exists():
    assert Path("sources/sample_local_sources.yaml").exists()
