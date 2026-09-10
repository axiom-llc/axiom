"""Tests for apex.core.rag.chunker — pure functions, no mocks needed."""
import pytest
from apex.core.rag.chunker import chunk_fixed, chunk_sentences


class TestChunkFixed:
    def test_basic_returns_list(self):
        result = chunk_fixed("word " * 100, chunk_size=20, overlap=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_returns_empty(self):
        assert chunk_fixed("", chunk_size=10, overlap=2) == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_fixed("   \n\t  ", chunk_size=10, overlap=2) == []

    def test_single_word(self):
        result = chunk_fixed("hello", chunk_size=10, overlap=2)
        assert result == ["hello"]

    def test_overlap_produces_shared_tokens(self):
        words = [str(i) for i in range(20)]
        text = " ".join(words)
        chunks = chunk_fixed(text, chunk_size=10, overlap=3)
        # Last tokens of chunk[0] should appear at start of chunk[1]
        tail = chunks[0].split()[-3:]
        head = chunks[1].split()[:3]
        assert tail == head

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_fixed("a b c", chunk_size=5, overlap=5)

    def test_no_empty_chunks(self):
        result = chunk_fixed("a " * 50, chunk_size=10, overlap=2)
        assert all(c.strip() for c in result)

    def test_chunk_size_respected(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_fixed(text, chunk_size=20, overlap=0)
        for c in chunks:
            assert len(c.split()) <= 20


class TestChunkSentences:
    def test_basic(self):
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_sentences(text, chunk_size=20)
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)

    def test_empty_returns_empty(self):
        assert chunk_sentences("", chunk_size=10) == []

    def test_does_not_exceed_chunk_size(self):
        # Each sentence is 5 words; chunk_size=10 → max 2 sentences per chunk
        sentences = ["One two three four five."] * 10
        text = " ".join(sentences)
        chunks = chunk_sentences(text, chunk_size=10)
        for c in chunks:
            assert len(c.split()) <= 10

    def test_single_sentence(self):
        result = chunk_sentences("Hello world.", chunk_size=50)
        assert result == ["Hello world."]

    def test_oversized_sentence_is_split(self):
        result = chunk_sentences(" ".join(["word"] * 25) + ".", chunk_size=10)
        assert [len(chunk.split()) for chunk in result] == [10, 10, 5]

    @pytest.mark.parametrize("chunk_size", [0, -1])
    def test_invalid_chunk_size_raises(self, chunk_size):
        with pytest.raises(ValueError):
            chunk_sentences("hello", chunk_size=chunk_size)
