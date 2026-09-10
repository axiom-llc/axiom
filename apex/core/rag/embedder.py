"""Gemini text embedding adapter for retrieval documents and queries."""
from google import genai
from google.genai import types

from apex.core.rag.config import Config

_BATCH_LIMIT = 100


def _client(config: Config) -> genai.Client:
    return genai.Client(api_key=config.gemini_api_key)


def _embedding_2(model: str) -> bool:
    return model.rsplit("/", 1)[-1] == "gemini-embedding-2"


def _document_text(text: str) -> str:
    return f"title: none | text: {text}"


def _query_text(query: str) -> str:
    return f"task: search result | query: {query}"


def _content(text: str) -> types.Content:
    return types.Content(parts=[types.Part.from_text(text=text)])


def embed_texts(texts: list[str], config: Config) -> list[list[float]]:
    """Embed document texts and return one vector per input text."""
    if not texts:
        return []
    config.requires_api_key()
    client = _client(config)
    embeddings: list[list[float]] = []

    for index in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[index : index + _BATCH_LIMIT]
        if _embedding_2(config.embedding_model):
            response = client.models.embed_content(
                model=config.embedding_model,
                contents=[_content(_document_text(text)) for text in batch],
            )
        else:
            response = client.models.embed_content(
                model=config.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
        batch_embeddings = [list(item.values) for item in (response.embeddings or [])]
        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(batch_embeddings)} vector(s) for {len(batch)} document(s)"
            )
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_query(query: str, config: Config) -> list[float]:
    """Embed one retrieval query."""
    config.requires_api_key()
    client = _client(config)
    if _embedding_2(config.embedding_model):
        response = client.models.embed_content(
            model=config.embedding_model,
            contents=_query_text(query),
        )
    else:
        response = client.models.embed_content(
            model=config.embedding_model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
    if not response.embeddings:
        raise RuntimeError("Embedding API returned no query embedding")
    return list(response.embeddings[0].values)
