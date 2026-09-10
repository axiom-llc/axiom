"""Resolve and validate in-process RAG configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    chroma_path: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    embedding_model: str
    generation_model: str

    def requires_api_key(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for this operation. "
                "Set it in the environment or pass gemini_api_key to load_config()."
            )


def load_config(**overrides) -> Config:
    def get(field: str, env_var: str, default, cast=str):
        value = overrides[field] if field in overrides else os.environ.get(env_var, default)
        return cast(value)

    api_key = (
        str(overrides["gemini_api_key"])
        if "gemini_api_key" in overrides
        else os.environ.get("GEMINI_API_KEY", "")
    )
    config = Config(
        gemini_api_key=api_key,
        chroma_path=get("chroma_path", "RAG_CHROMA_PATH", "~/.rag/chroma"),
        collection_name=get("collection_name", "RAG_COLLECTION", "documents"),
        chunk_size=get("chunk_size", "RAG_CHUNK_SIZE", 512, int),
        chunk_overlap=get("chunk_overlap", "RAG_CHUNK_OVERLAP", 64, int),
        top_k=get("top_k", "RAG_TOP_K", 5, int),
        score_threshold=get("score_threshold", "RAG_SCORE_THRESHOLD", 0.4, float),
        embedding_model=get("embedding_model", "RAG_EMBEDDING_MODEL", "gemini-embedding-2"),
        generation_model=get("generation_model", "RAG_GENERATION_MODEL", "gemini-3.5-flash-lite"),
    )
    if config.chunk_size <= 0:
        raise ValueError("RAG_CHUNK_SIZE must be greater than 0")
    if config.chunk_overlap < 0 or config.chunk_overlap >= config.chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be >= 0 and less than RAG_CHUNK_SIZE")
    if config.top_k <= 0:
        raise ValueError("RAG_TOP_K must be greater than 0")
    if not -1.0 <= config.score_threshold <= 1.0:
        raise ValueError("RAG_SCORE_THRESHOLD must be between -1 and 1")
    if not config.collection_name:
        raise ValueError("RAG_COLLECTION must not be empty")
    return config
