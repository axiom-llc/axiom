"""Tests for apex.core.rag.store — uses a real in-memory ChromaDB instance, no Gemini."""
import pytest
import tempfile
import os

from apex.core.rag.config import load_config
from apex.core.rag import store


@pytest.fixture()
def cfg(tmp_path):
    """Config pointing to a temp ChromaDB directory."""
    store._get_client.cache_clear()
    yield load_config(
        gemini_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        collection_name="test_col",
    )
    store._get_client.cache_clear()


def _fake_embedding(dim: int = 8) -> list[float]:
    import random
    v = [random.random() for _ in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _upsert_doc(cfg, doc_id: str, n_chunks: int = 3):
    chunks = [f"chunk {i} of {doc_id}" for i in range(n_chunks)]
    embeddings = [_fake_embedding() for _ in range(n_chunks)]
    store.upsert(chunks, embeddings, doc_id, cfg)
    return chunks, embeddings


class TestUpsertAndList:
    def test_upsert_appears_in_list(self, cfg):
        _upsert_doc(cfg, "doc_a")
        assert "doc_a" in store.list_documents(cfg)

    def test_multiple_docs_listed(self, cfg):
        _upsert_doc(cfg, "doc_a")
        _upsert_doc(cfg, "doc_b")
        docs = store.list_documents(cfg)
        assert "doc_a" in docs and "doc_b" in docs

    def test_list_returns_unique_ids(self, cfg):
        _upsert_doc(cfg, "dup")
        _upsert_doc(cfg, "dup")  # re-upsert same doc
        assert store.list_documents(cfg).count("dup") == 1

    def test_upsert_idempotent(self, cfg):
        _upsert_doc(cfg, "doc_a", n_chunks=3)
        _upsert_doc(cfg, "doc_a", n_chunks=3)
        stats = store.collection_stats(cfg)
        assert stats["total_chunks"] == 3  # not 6


class TestDelete:
    def test_delete_removes_chunks(self, cfg):
        _upsert_doc(cfg, "to_delete", n_chunks=4)
        n = store.delete_document("to_delete", cfg)
        assert n == 4
        assert "to_delete" not in store.list_documents(cfg)

    def test_delete_nonexistent_returns_zero(self, cfg):
        assert store.delete_document("ghost", cfg) == 0

    def test_delete_only_targets_doc(self, cfg):
        _upsert_doc(cfg, "keep")
        _upsert_doc(cfg, "remove")
        store.delete_document("remove", cfg)
        assert "keep" in store.list_documents(cfg)


class TestQuery:
    def test_score_threshold_filters_low_scores(self, cfg):
        # A single 2-D-style unit vector on axis 0; an orthogonal query vector
        # has cosine similarity 0, so a high threshold must exclude it while
        # a low/negative threshold must include it.
        chunk_embedding = [1.0, 0.0, 0.0, 0.0]
        orthogonal_query = [0.0, 1.0, 0.0, 0.0]
        store.upsert(["axis-aligned chunk"], [chunk_embedding], "doc_orth", cfg)

        strict_cfg = load_config(
            gemini_api_key=cfg.gemini_api_key,
            chroma_path=cfg.chroma_path,
            collection_name=cfg.collection_name,
            score_threshold=0.9,
        )
        assert store.store_query(orthogonal_query, strict_cfg) == []

        lenient_cfg = load_config(
            gemini_api_key=cfg.gemini_api_key,
            chroma_path=cfg.chroma_path,
            collection_name=cfg.collection_name,
            score_threshold=-1.0,
        )
        results = store.store_query(orthogonal_query, lenient_cfg)
        assert len(results) == 1
        assert results[0]["metadata"]["doc_id"] == "doc_orth"
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-6)

    def test_query_returns_sorted_by_score_descending(self, cfg):
        store.upsert(["exact match"], [[1.0, 0.0, 0.0, 0.0]], "doc_high", cfg)
        store.upsert(["partial match"], [[0.7, 0.7, 0.0, 0.0]], "doc_low", cfg)
        lenient_cfg = load_config(
            gemini_api_key=cfg.gemini_api_key,
            chroma_path=cfg.chroma_path,
            collection_name=cfg.collection_name,
            score_threshold=-1.0,
        )
        results = store.store_query([1.0, 0.0, 0.0, 0.0], lenient_cfg)
        assert [r["metadata"]["doc_id"] for r in results] == ["doc_high", "doc_low"]
        assert results[0]["score"] >= results[1]["score"]

    def test_query_empty_collection_returns_empty(self, cfg):
        assert store.store_query([1.0, 0.0, 0.0, 0.0], cfg) == []

    def test_query_respects_top_k(self, cfg):
        for i in range(5):
            store.upsert([f"chunk {i}"], [_fake_embedding()], f"doc_{i}", cfg)
        capped_cfg = load_config(
            gemini_api_key=cfg.gemini_api_key,
            chroma_path=cfg.chroma_path,
            collection_name=cfg.collection_name,
            top_k=2,
            score_threshold=-1.0,
        )
        results = store.store_query(_fake_embedding(), capped_cfg)
        assert len(results) <= 2


class TestCollectionStats:
    def test_stats_reflect_documents_and_chunks(self, cfg):
        _upsert_doc(cfg, "doc_a", n_chunks=2)
        _upsert_doc(cfg, "doc_b", n_chunks=3)
        stats = store.collection_stats(cfg)
        assert stats["total_chunks"] == 5
        assert set(stats["documents"]) == {"doc_a", "doc_b"}
