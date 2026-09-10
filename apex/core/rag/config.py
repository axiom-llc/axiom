"""Use canonical RAG validation while preserving APEX model defaults."""
import os

from rag.config import Config, load_config as _load_config


def load_config(**overrides) -> Config:
    overrides.setdefault("embedding_model", os.environ.get("RAG_EMBEDDING_MODEL", "gemini-embedding-2"))
    overrides.setdefault("generation_model", os.environ.get("RAG_GENERATION_MODEL", "gemini-3.5-flash-lite"))
    return _load_config(**overrides)
