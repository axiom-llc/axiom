"""Deterministic word-window and sentence-aware text chunking."""
from __future__ import annotations

import re


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into word-bounded chunks with a fixed word overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[index : index + chunk_size])
        for index in range(0, len(words), step)
        if words[index : index + chunk_size]
    ]


def chunk_sentences(text: str, chunk_size: int) -> list[str]:
    """Group sentences without exceeding chunk_size words; split oversized sentences."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            chunks.extend(chunk_fixed(sentence, chunk_size, 0))
            continue
        if current and count + len(words) > chunk_size:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks
