"""Tests for the ingestion pipeline (no real PDFs needed)."""

from langchain_core.documents import Document

import gpu_dev_assist.data_ingestion as ingestion_module
from gpu_dev_assist.data_ingestion import DataIngestion

TEXT = (
    "The oneAPI programming model supports single-source compilation, so device code "
    "can be defined right next to the call site in the host code. "
) * 3


def fake_load_all_documents(_data_dir):
    return [
        Document(page_content=TEXT, metadata={"source_file": "fake.pdf", "page": 0}),
        Document(
            page_content=TEXT.replace("single-source compilation", "shared local memory"),
            metadata={"source_file": "fake.pdf", "page": 1},
        ),
    ]


def make_ingestion(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion_module, "load_all_documents", fake_load_all_documents)
    return DataIngestion(persist_directory=str(tmp_path / "vector_store"))


def test_ingestion_is_idempotent(tmp_path, monkeypatch):
    ingestion = make_ingestion(tmp_path, monkeypatch)

    ingestion.initiate_data_ingestion()
    first_count = ingestion.vector_store.collection.count()
    ingestion.initiate_data_ingestion()

    assert first_count > 0
    assert ingestion.vector_store.collection.count() == first_count


def test_stored_chunks_keep_source_and_page(tmp_path, monkeypatch):
    ingestion = make_ingestion(tmp_path, monkeypatch)
    ingestion.initiate_data_ingestion()

    stored = ingestion.vector_store.collection.get(include=["metadatas"])
    for meta in stored["metadatas"]:
        assert meta["source_file"] == "fake.pdf"
        assert meta["page"] in (0, 1)