"""Integration tests for apex.core.rag.pipeline.

Gemini API calls (embedder + generator) are mocked.
ChromaDB runs against a real temp directory.
"""
import pytest
from unittest.mock import patch, MagicMock

from apex.core.rag.config import load_config
from apex.core.rag import store, pipeline


@pytest.fixture()
def cfg(tmp_path):
    store._get_client.cache_clear()
    yield load_config(
        gemini_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        collection_name="pipe_test",
    )
    store._get_client.cache_clear()


def _fake_embedding(dim: int = 8) -> list[float]:
    v = [0.1] * dim
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


@pytest.fixture(autouse=True)
def mock_gemini():
    """Patch embedder and generator at the function boundary.

    Patching at this level (rather than the underlying genai client) avoids
    SDK-version sensitivity and eliminates triple-nesting bugs where the mock
    return value gets wrapped again by the caller.
    """
    fake_embed = _fake_embedding()

    def fake_embed_texts(texts, config):
        return [fake_embed for _ in texts]

    def fake_embed_query(query, config):
        return fake_embed

    mock_response = MagicMock()
    mock_response.text = "Mocked answer from context."

    with (
        patch("rag.embedder.embed_texts", side_effect=fake_embed_texts),
        patch("rag.embedder.embed_query", side_effect=fake_embed_query),
        patch("rag.generator.genai") as mock_gen,
    ):
        mock_gen.Client.return_value.models.generate_content.return_value = mock_response
        yield


class TestIngest:
    def test_empty_text_returns_zero_chunks(self, cfg):
        result = pipeline.ingest("", doc_id="empty", config=cfg)
        assert result == {"doc_id": "empty", "chunks_stored": 0}

    def test_basic_ingest(self, cfg):
        text = "word " * 100
        result = pipeline.ingest(text, doc_id="test_doc", config=cfg)
        assert result["doc_id"] == "test_doc"
        assert result["chunks_stored"] > 0

    def test_ingest_idempotent(self, cfg):
        text = "word " * 100
        r1 = pipeline.ingest(text, doc_id="idem", config=cfg)
        pipeline.ingest(text, doc_id="idem", config=cfg)  # re-ingest same doc
        r2 = pipeline.ingest(text, doc_id="idem2", config=cfg)
        stats = store.collection_stats(cfg)
        # "idem" should not have doubled; total = chunks_for_idem + chunks_for_idem2
        assert stats["total_chunks"] == r1["chunks_stored"] + r2["chunks_stored"]
        assert set(stats["documents"]) == {"idem", "idem2"}

    def test_sentence_strategy(self, cfg):
        text = "First sentence here. Second sentence there. Third sentence now."
        result = pipeline.ingest(text, doc_id="sent", config=cfg, strategy="sentences")
        assert result["chunks_stored"] >= 1

    def test_metadata_stored(self, cfg):
        pipeline.ingest(
            "some content here now",
            doc_id="meta_doc",
            config=cfg,
            metadata={"source": "unit_test"},
        )
        assert "meta_doc" in store.list_documents(cfg)

    def test_reingest_shorter_document_removes_stale_chunks(self, cfg):
        long_text = "word " * (cfg.chunk_size * 3)
        first = pipeline.ingest(long_text, doc_id="replace", config=cfg)
        assert first["chunks_stored"] > 1
        second = pipeline.ingest("short document", doc_id="replace", config=cfg)
        assert second["chunks_stored"] == 1
        assert store.collection_stats(cfg)["total_chunks"] == 1

    def test_invalid_strategy_raises(self, cfg):
        with pytest.raises(ValueError, match="strategy"):
            pipeline.ingest("text", doc_id="bad", config=cfg, strategy="invalid")


class TestIngestFile:
    def test_ingest_file_uses_filename_as_doc_id(self, cfg, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("content of the sample file " * 10)
        result = pipeline.ingest_file(str(f), cfg)
        assert result["doc_id"] == "sample.txt"
        assert result["chunks_stored"] > 0

    def test_ingest_directory_processes_txt_and_md(self, cfg, tmp_path):
        (tmp_path / "a.txt").write_text("text content " * 20)
        (tmp_path / "b.md").write_text("markdown content " * 20)
        (tmp_path / "c.py").write_text("# ignored")
        results = pipeline.ingest_directory(str(tmp_path), cfg)
        doc_ids = [r["doc_id"] for r in results]
        assert doc_ids == sorted(doc_ids)
        assert "a.txt" in doc_ids
        assert "b.md" in doc_ids
        assert not any("c.py" in d for d in doc_ids)

    def test_ingest_directory_uses_relative_paths_for_duplicate_basenames(self, cfg, tmp_path):
        left = tmp_path / "left"
        right = tmp_path / "right"
        left.mkdir()
        right.mkdir()
        (left / "same.txt").write_text("left content")
        (right / "same.txt").write_text("right content")
        doc_ids = [r["doc_id"] for r in pipeline.ingest_directory(str(tmp_path), cfg)]
        assert "left/same.txt" in doc_ids
        assert "right/same.txt" in doc_ids


class TestQuery:
    def test_query_response_shape(self, cfg):
        pipeline.ingest("relevant content about the topic", doc_id="q_doc", config=cfg)
        result = pipeline.query("what is the topic?", cfg)
        assert "answer" in result
        assert "sources" in result
        assert "chunks" in result
        assert "chunk_count" in result

    def test_query_no_documents_returns_graceful_answer(self, cfg):
        result = pipeline.query("question with no docs", cfg)
        assert "answer" in result
        assert isinstance(result["sources"], list)

    def test_query_chunk_count_matches_chunks_length(self, cfg):
        pipeline.ingest("some searchable content here", doc_id="cc_doc", config=cfg)
        result = pipeline.query("searchable content", cfg)
        assert result["chunk_count"] == len(result["chunks"])


def test_retrieval_uses_canonical_rag_implementations():
    from rag import pipeline as canonical_pipeline, store as canonical_store
    assert pipeline.ingest is canonical_pipeline.ingest
    assert store.upsert is canonical_store.upsert
